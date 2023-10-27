from typing import Any, Callable, Dict, Sequence, Union, List, Optional, Tuple
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#Don't use GPU for dataloading
tf.config.set_visible_devices([], "GPU")
import tqdm
import logging
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
from collections import OrderedDict
from functools import partial
import numpy as np
import hashlib
import json
import pickle
import torch

import robomimic.utils.torch_utils as TorchUtils
from .dataset_transformations import RLDS_TRAJECTORY_MAP_TRANSFORMS
import robomimic.data.common_transformations as CommonTransforms
import robomimic.utils.data_utils as DataUtils


class RLDSTorchDataset:
    def __init__(self, dataset_iterator, try_to_use_cuda=True):
        self.dataset_iterator = dataset_iterator
        self.device = TorchUtils.get_torch_device(try_to_use_cuda)
        self.keys = ['obs', 'goal_obs', 'actions']

    def __iter__(self):
        for batch in self.dataset_iterator:
            torch_batch = {}
            for key in self.keys:
                if key in batch.keys():
                    torch_batch[key] = DataUtils.tree_map(
                        batch[key],
                        map_fn=lambda x: torch.tensor(x).to(self.device)
                    )
            yield torch_batch 
        

def get_action_normalization_stats_rlds(obs_action_metadata, config):
    action_config = config.train.action_config
    normal_keys = [key for key in config.train.action_keys
        if action_config[key].get('normalization', None) == 'normal']
    min_max_keys = [key for key in config.train.action_keys
        if action_config[key].get('normalization', None) == 'min_max']

    stats = OrderedDict()   
    for key in config.train.action_keys:
        if key in normal_keys:
            normal_stats = {
                'scale': obs_action_metadata[key]['std'].reshape(1, -1),
                'offset': obs_action_metadata[key]['mean'].reshape(1, -1)
            }
            stats[key] = normal_stats
        elif key in min_max_keys:
            min_max_range = obs_action_metadata[key]['max'] - obs_action_metadata[key]['min'] 
            min_max_stats = {
                'scale': (min_max_range / 2).reshape(1, -1),
                'offset': (obs_action_metadata[key]['min'] + min_max_range / 2).reshape(1, -1)
            }
            stats[key] = min_max_stats
        else:
            identity_stats = {
                'scale': np.ones_like(obs_action_metadata[key]['std']).reshape(1, -1),
                'offset': np.zeros_like(obs_action_metadata[key]['mean']).reshape(1, -1)
            }
            stats[key] = identity_stats
    return stats


def get_obs_normalization_stats_rlds(obs_action_metadata, config):
    stats = OrderedDict() 
    for key, obs_action_stats in obs_action_metadata.items():
        feature_type, feature_key = key.split('/')
        if feature_type != 'observation':
            continue
        stats[feature_key] = {
            'mean': obs_action_stats['mean'][None],
            'std': obs_action_stats['std'][None],
        }
    return stats
 

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

        i, n_samples = 0, 500
        dataset_iter = dataset.as_numpy_iterator()
        for _ in tqdm.tqdm(range(n_samples)):
            episode = next(dataset_iter)
            i = i + 1
            for key in keys:
                eps_by_key[key].append(DataUtils.index_nested_dict(episode, key))
        eps_by_key = {key: np.concatenate(values) for key, values in eps_by_key.items()}
    
        metadata = {}        
        for key in keys:
            metadata[key] = {
                "mean": eps_by_key[key].mean(0),
                "std": eps_by_key[key].std(0),
                "max": eps_by_key[key].max(0),
                "min": eps_by_key[key].min(0),
            }
        with tf.io.gfile.GFile(path, "wb") as f:
            pickle.dump(metadata, f)
        logging.info("Done!")

    return metadata


def decode_dataset(
    dataset: tf.data.Dataset
    ):

    #Decode images
    dataset = dataset.frame_map(
        DataUtils.decode_images
    )
    return dataset


def apply_common_transforms(
    dataset: tf.data.Dataset,
    config: dict,
    *,
    train: bool,
    obs_action_metadata: Optional[dict] = None,
    ):

    #Normalize observations and actions
    if obs_action_metadata is not None:
        dataset = dataset.map(
            partial(
                CommonTransforms.normalize_obs_and_actions,
                config=config,
                metadata=obs_action_metadata,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    #Relabel goals
    if config.train.goal_mode == 'last' or config.train.goal_mode == 'uniform':
        dataset = dataset.map(
            partial(
                CommonTransforms.relabel_goals_transform,
                goal_mode=config.goal_mode
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    #Concatenate actions
    if config.train.action_keys != None:
        dataset = dataset.map(
            partial(
                CommonTransforms.concatenate_action_transform,
                action_keys=config.train.action_keys
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    #Get a random subset of length frame_stack + seq_length - 1
    dataset = dataset.map(
        partial(
            CommonTransforms.random_dataset_sequence_transform_v2,
            frame_stack=config.train.frame_stack,
            seq_length=config.train.seq_length,
            pad_frame_stack=config.train.pad_frame_stack,
            pad_seq_length=config.train.pad_seq_length
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    #augmentation? #chunking?
    
    return dataset

def decode_trajectory(builder, obs_keys, episode):
    steps = episode
    new_steps = dict()
    new_steps['action_dict'] = dict()
    new_steps['observation'] = dict()
    for key in steps["action_dict"]:
        new_steps['action_dict'][key] = builder.info.features["steps"]['action_dict'][
                        key
                    ].decode_batch_example(steps["action_dict"][key])
    for key in obs_keys:
        new_steps['observation'][key] = builder.info.features["steps"]['observation'][
                        key
                    ].decode_batch_example(steps["observation"][key])
    return new_steps

def make_dataset(
    config: dict,
    train: bool = True,
    shuffle: bool = True,
    resize_size: Optional[Tuple[int, int]] = None,
    normalization_metadata: Optional[Dict] = None,
    **kwargs,
) -> tf.data.Dataset:
   
    data_info = config.train.data[0]
    name = data_info['name']
    data_dir = data_info['path']

    builder = tfds.builder(name, data_dir=data_dir)

    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle,
        num_parallel_reads=8)
    if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
        if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'] is not None:
            dataset = dataset.map(partial(
                RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'],
                config=config),
            )
    metadata_keys = [k for k in config.train.action_keys]
    if config.all_obs_keys is not None:
        metadata_keys.extend([f'observation/{k}' 
            for k in config.all_obs_keys])
    if normalization_metadata is None:
        normalization_metadata = get_obs_action_metadata(
            builder,
            dataset,
            keys=metadata_keys,
            load_if_exists=True#False
        )
    dataset = apply_common_transforms(
        dataset,
        config=config,
        train=train,
        obs_action_metadata=normalization_metadata,
        **kwargs,
    )
    if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
        if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'] is not None:
            dataset = dataset.map(partial(
                RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'],
                config=config),
    )
    dataset = decode_dataset(dataset)
    dataset = dataset.repeat().batch(config.train.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.as_numpy_iterator()
    dataset = RLDSTorchDataset(dataset)

    return builder, dataset, normalization_metadata


