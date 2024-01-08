from functools import partial
import inspect
import json
import tensorflow as tf

import tensorflow_datasets as tfds
#Don't use GPU for dataloading
tf.config.set_visible_devices([], "GPU")
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union
from .dataset_transformations import RLDS_TRAJECTORY_MAP_TRANSFORMS
from typing import Any, Dict, List, Union, Tuple, Optional
import tree
import hashlib
import pickle
import torch
import robomimic.utils.torch_utils as TorchUtils
from .dataset_transformations import RLDS_TRAJECTORY_MAP_TRANSFORMS
import robomimic.data.common_transformations as CommonTransforms
import robomimic.utils.data_utils as DataUtils
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
import tqdm
from torch.utils.data import DataLoader




from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def subsample(traj: dict, subsample_length: int) -> dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["actions"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def apply_trajectory_transforms(dataset, train, subsample_length = None, num_parallel_calls: int = tf.data.AUTOTUNE):
    if train and subsample_length is not None:
        
        #TODO CHANGE THIS TO FOLLOW SAME APPROACH AS REVERB
        dataset = dataset.filter(
            lambda x: tf.shape(x["actions"])[0] >= subsample_length
        )

        dataset = dataset.traj_map(
            partial(subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )
    return dataset


class RLDSTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_iterator, try_to_use_cuda=True):
        self.dataset_iterator = dataset_iterator
        self.device = TorchUtils.get_torch_device(try_to_use_cuda)
        self.keys = ['obs', 'goal_obs', 'actions']

    def __iter__(self):
        for batch in self.dataset_iterator.as_numpy_iterator():
            torch_batch = {}
            for key in self.keys:
                if key in batch.keys():
                    torch_batch[key] = DataUtils.tree_map(
                        batch[key],
                        map_fn=lambda x: torch.tensor(x).to(self.device)
                    )
            yield torch_batch 
        

def decode_images(
    obs: dict,
):
    for key in obs["obs"]:
        if "image" in key:
            image = obs["obs"][key]
            assert image.dtype == tf.string
            image_decoded = tf.io.decode_image(
                    image, expand_animations=False, dtype=tf.uint8
                )
            obs["obs"][key] = image_decoded
    return obs

    

def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    
    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        decode_images,
        num_parallel_calls,
    )
    return dataset

def get_obs_action_metadata(
    builder: DatasetBuilder, dataset: tf.data.Dataset, keys: List[str],
    load_if_exists=True
) -> Dict[str, Dict[str, List[float]]]:
    # get statistics file path --> embed unique hash that catches if dataset info changed
    data_info_hash = hashlib.sha256(
        (str(builder.info) + str(keys)).encode("utf-8")
    ).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"obs_action_stats_{data_info_hash}.pkl"
    )

    # check if stats already exist and load, otherwise compute
    if tf.io.gfile.exists(path) and load_if_exists:
        print(f"Loading existing statistics for normalization from {path}.")
        with tf.io.gfile.GFile(path, "rb") as f:
            metadata = pickle.load(f)
    else:
        print("Computing obs/action statistics for normalization...")
        eps_by_key = {key: [] for key in keys}

        i, n_samples = 0, 50
        dataset_iter = dataset.as_numpy_iterator()
        for _ in tqdm.tqdm(range(n_samples)):
            episode = next(dataset_iter)
            i = i + 1
            for key in keys:
                eps_by_key[key].append(DataUtils.index_nested_dict(episode, key))
        eps_by_key = {key: np.concatenate(values) for key, values in eps_by_key.items()}
    
        metadata = {}        
        # breakpoint()
        for key in keys:
            # #print(key)
            # #print(eps_by_key[key])
            # breakpoint()
            if "image" not in key:
                metadata[key] = {
                    "mean": eps_by_key[key].mean(0),
                    "std": eps_by_key[key].std(0),
                    "max": eps_by_key[key].max(0),
                    "min": eps_by_key[key].min(0),
                }
            else:
                metadata[key] = {
                    "mean": np.frombuffer(eps_by_key[key], dtype=np.uint8).mean(0),
                    "std": np.frombuffer(eps_by_key[key], dtype=np.uint8).std(0),
                    "max": np.frombuffer(eps_by_key[key], dtype=np.uint8).max(0),
                    "min": np.frombuffer(eps_by_key[key], dtype=np.uint8).min(0),
                }
        # breakpoint()
        # with tf.io.gfile.GFile(path, "wb") as f:
        #     pickle.dump(metadata, f)
        logging.info("Done!")

    return metadata


def make_dataset_from_rlds(
    config:dict,
    train: bool,
    shuffle: bool = True,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    
    data_info = config.train.data[0]
    name = data_info['name']
    data_dir = data_info['path']
    builder = tfds.builder(name, data_dir=data_dir)


    

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

    metadata_keys = [k for k in config.train.action_keys]
    if config.all_obs_keys is not None:
        metadata_keys.extend([f'observation/{k}' 
            for k in config.all_obs_keys])
        
    normalization_metadata = get_obs_action_metadata(
            builder,
            dataset,
            keys=metadata_keys,
            load_if_exists=True#False
        )

    if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
        if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'] is not None:
            dataset = dataset.traj_map(
                partial(
                    RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'],
                    config=config
                ),
                num_parallel_calls
            )

    if normalization_metadata is not None:
        dataset = dataset.traj_map(
            partial(
                CommonTransforms.normalize_obs_and_actions,
                config=config,
                metadata=normalization_metadata,
            ),
            num_parallel_calls,
        )
    if config.train.action_keys != None:
        dataset = dataset.traj_map(
            partial(
                CommonTransforms.concatenate_action_transform,
                action_keys=config.train.action_keys
            ),
            num_parallel_calls,
        )
    
    if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
        if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'] is not None:
            dataset = dataset.traj_map(
                partial(
                    RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'],
                    config=config
                ),
                num_parallel_calls
            )

    return builder, dataset, normalization_metadata


def make_single_dataset(
    config: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = None,
    shuffle_buffer_size=1000,
) -> dl.DLataset:
    
    if traj_transform_kwargs is None:
        traj_transform_kwargs = {
            "subsample_length": config.train.seq_length + config.train.frame_stack - 1
        }

    builder, dataset, normalization_metdata = make_dataset_from_rlds(
        config=config,
        train=train,
    )



    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset)

    # this seems to reduce memory usage without affecting speed

    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat().batch(config.train.batch_size)
    dataset = dataset.with_ram_budget(1)


    pytorch_dataset = RLDSTorchDataset(dataset)


    return builder, pytorch_dataset, normalization_metdata

