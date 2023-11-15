
from typing import Any, Dict, List, Union, Tuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#Don't use GPU for dataloading
tf.config.set_visible_devices([], "GPU")
import rlds
import reverb
from rlds import transformations
import tree

import abc
import dataclasses
from functools import partial
from collections import OrderedDict
from rlds import rlds_types
from PIL import Image
import torch
import hashlib
import json
import pickle
import torch
import tqdm
import logging
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

import robomimic.utils.torch_utils as TorchUtils
from .dataset_transformations import RLDS_TRAJECTORY_MAP_TRANSFORMS
import robomimic.data.common_transformations as CommonTransforms
import robomimic.utils.data_utils as DataUtils
import robomimic.utils.tensorflow_utils as TensorflowUtils

# Transformation definitions
def _features_to_tensor_spec(
        feature: tfds.features.FeatureConnector
) -> tf.TensorSpec:
    """Converts a tfds Feature into a TensorSpec."""

    def _get_feature_spec(nested_feature: tfds.features.FeatureConnector):
        if isinstance(nested_feature, tf.DType):
            return tf.TensorSpec(shape=(), dtype=nested_feature)
        else:
            return nested_feature.get_tensor_spec()

    # FeaturesDict can sometimes be a plain dictionary, so we use tf.nest to
    # make sure we deal with the nested structure.
    return tf.nest.map_structure(_get_feature_spec, feature)


def _encoded_feature(feature: Optional[tfds.features.FeatureConnector],
                                         image_encoding: Optional[str],
                                         tensor_encoding: Optional[tfds.features.Encoding]):
    """Adds encoding to Images and/or Tensors."""
    def _apply_encoding(feature: tfds.features.FeatureConnector,
                                            image_encoding: Optional[str],
                                            tensor_encoding: Optional[tfds.features.Encoding]):
        if image_encoding and isinstance(feature, tfds.features.Image):
            return tfds.features.Image(
                    shape=feature.shape,
                    dtype=feature.dtype,
                    use_colormap=feature.use_colormap,
                    encoding_format=image_encoding)
        if tensor_encoding and isinstance(
                feature, tfds.features.Tensor) and feature.dtype != tf.string:
            return tfds.features.Tensor(
                    shape=feature.shape, dtype=feature.dtype, encoding=tensor_encoding)
        return feature

    if not feature:
        return None
    return tf.nest.map_structure(
            lambda x: _apply_encoding(x, image_encoding, tensor_encoding), feature)


@dataclasses.dataclass
class RLDSSpec(metaclass=abc.ABCMeta):
    """Specification of an RLDS Dataset.

    It is used to hold a spec that can be converted into a TFDS DatasetInfo or
    a `tf.data.Dataset` spec.
    """
    observation_info: Optional[tfds.features.FeatureConnector] = None
    action_info: Optional[tfds.features.FeatureConnector] = None
    reward_info: Optional[tfds.features.FeatureConnector] = None
    discount_info: Optional[tfds.features.FeatureConnector] = None
    step_metadata_info: Optional[tfds.features.FeaturesDict] = None
    episode_metadata_info: Optional[tfds.features.FeaturesDict] = None

    def step_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        step = {}
        if self.observation_info:
            step[rlds_types.OBSERVATION] = _features_to_tensor_spec(
                    self.observation_info)
        if self.action_info:
            step['action_dict'] = _features_to_tensor_spec(
                    self.action_info)
        if self.discount_info:
            step[rlds_types.DISCOUNT] = _features_to_tensor_spec(
                    self.discount_info)
        if self.reward_info:
            step[rlds_types.REWARD] = _features_to_tensor_spec(
                    self.reward_info)
        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step[k] = _features_to_tensor_spec(v)

        step[rlds_types.IS_FIRST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_LAST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_TERMINAL] = tf.TensorSpec(shape=(), dtype=bool)
        return step

    def episode_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        episode = {}
        episode[rlds_types.STEPS] = tf.data.DatasetSpec(
                element_spec=self.step_tensor_spec())
        if self.episode_metadata_info:
            for k, v in self.episode_metadata_info.items():
                episode[k] = _features_to_tensor_spec(v)
        return episode

    def to_dataset_config(
            self,
            name: str,
            image_encoding: Optional[str] = None,
            tensor_encoding: Optional[tfds.features.Encoding] = None,
            citation: Optional[str] = None,
            homepage: Optional[str] = None,
            description: Optional[str] = None,
            overall_description: Optional[str] = None,
    ) -> tfds.rlds.rlds_base.DatasetConfig:
        """Obtains the DatasetConfig for TFDS from the Spec."""
        return tfds.rlds.rlds_base.DatasetConfig(
                name=name,
                description=description,
                overall_description=overall_description,
                homepage=homepage,
                citation=citation,
                observation_info=_encoded_feature(self.observation_info, image_encoding, tensor_encoding),
                action_info=_encoded_feature(self.action_info, image_encoding, tensor_encoding),
                reward_info=_encoded_feature(self.reward_info, image_encoding, tensor_encoding),
                discount_info=_encoded_feature(self.discount_info, image_encoding, tensor_encoding),
                step_metadata_info=_encoded_feature(self.step_metadata_info, image_encoding, tensor_encoding),
                episode_metadata_info=_encoded_feature(self.episode_metadata_info, image_encoding, tensor_encoding))

    def to_features_dict(self):
        """Returns a TFDS FeaturesDict representing the dataset config."""
        step_config = {
                rlds_types.IS_FIRST: tf.bool,
                rlds_types.IS_LAST: tf.bool,
                rlds_types.IS_TERMINAL: tf.bool,
        }

        if self.observation_info:
            step_config[rlds_types.OBSERVATION] = self.observation_info
        if self.action_info:
            step_config[rlds_types.ACTION] = self.action_info
        if self.discount_info:
            step_config[rlds_types.DISCOUNT] = self.discount_info
        if self.reward_info:
            step_config[rlds_types.REWARD] = self.reward_info

        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step_config[k] = v

        if self.episode_metadata_info:
            return tfds.features.FeaturesDict({
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
                    **self.episode_metadata_info,
            })
        else:
            return tfds.features.FeaturesDict({
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
            })

RLDS_SPEC = RLDSSpec
TENSOR_SPEC = Union[tf.TensorSpec, Dict[str, tf.TensorSpec]]


@dataclasses.dataclass
class TrajectoryTransform(metaclass=abc.ABCMeta):
    """Specification the TrajectoryTransform applied to a dataset of episodes.

    A TrajectoryTransform is a set of rules transforming a dataset
    of RLDS episodes to a dataset of trajectories.
    This involves three distinct stages:
    - An optional `episode_to_steps_map_fn(episode)` is called at the episode
        level, and can be used to select or modify steps.
        - Augmentation: an `episode_key` could be propagated to `steps` for
            debugging.
        - Selection: Particular steps can be selected.
        - Stripping: Features can be removed from steps. Prefer using `step_map_fn`.
    - An optional `step_map_fn` is called at the flattened steps dataset for each
        step, and can be used to featurize a step, e.g. add/remove features, or
        augument images
    - A `pattern` leverages DM patterns to set a rule of slicing an episode to a
        dataset of overlapping trajectories.

    Importantly, each TrajectoryTransform must define a `expected_tensor_spec`
    which specifies a nested TensorSpec of the resulting dataset. This is what
    this TrajectoryTransform will produce, and can be used as an interface with
    a neural network.
    """
    episode_dataset_spec: RLDS_SPEC
    episode_to_steps_fn_dataset_spec: RLDS_SPEC
    steps_dataset_spec: Any
    pattern: reverb.structured_writer.Pattern
    episode_to_steps_map_fn: Any
    expected_tensor_spec: TENSOR_SPEC
    step_map_fn: Optional[Any] = None

    def get_for_cached_trajectory_transform(self):
        """Creates a copy of this traj transform to use with caching.

        The returned TrajectoryTransfrom copy will be initialized with the default
        version of the `episode_to_steps_map_fn`, because the effect of that
        function has already been materialized in the cached copy of the dataset.
        Returns:
            trajectory_transform: A copy of the TrajectoryTransform with overridden
                `episode_to_steps_map_fn`.
        """
        traj_copy = dataclasses.replace(self)
        traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
        traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
        return traj_copy

    def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
        """Applies this TrajectoryTransform to the dataset of episodes."""

        # Convert the dataset of episodes to the dataset of steps.
        steps_dataset = episodes_dataset.map(
                self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).flat_map(lambda x: x)

        return self._create_pattern_dataset(steps_dataset)

    def transform_steps_rlds_dataset(
            self, steps_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Applies this TrajectoryTransform to the dataset of episode steps."""

        return self._create_pattern_dataset(steps_dataset)

    def create_test_dataset(
            self,
    ) -> tf.data.Dataset:
        """Creates a test dataset of trajectories.

        It is guaranteed that the structure of this dataset will be the same as
        when flowing real data. Hence this is a useful construct for tests or
        initialization of JAX models.
        Returns:
            dataset: A test dataset made of zeros structurally identical to the
                target dataset of trajectories.
        """
        zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

        return tf.data.Dataset.from_tensors(zeros)

    def _create_pattern_dataset(
            self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Create PatternDataset from the `steps_dataset`."""
        config = create_structured_writer_config('temp', self.pattern)

        # Further transform each step if the `step_map_fn` is provided.
        if self.step_map_fn:
            steps_dataset = steps_dataset.map(self.step_map_fn)
        pattern_dataset = reverb.PatternDataset(
                input_dataset=steps_dataset,
                configs=[config],
                respect_episode_boundaries=True,
                is_end_of_episode=lambda x: x[rlds_types.IS_LAST])
        return pattern_dataset


class TrajectoryTransformBuilder(object):
    """Facilitates creation of the `TrajectoryTransform`."""

    def __init__(self,
                dataset_spec: RLDS_SPEC,
                episode_to_steps_map_fn=lambda e: e[rlds_types.STEPS],
                step_map_fn=None,
                pattern_fn=None,
                expected_tensor_spec=None):
        self._rds_dataset_spec = dataset_spec
        self._steps_spec = None
        self._episode_to_steps_map_fn = episode_to_steps_map_fn
        self._step_map_fn = step_map_fn
        self._pattern_fn = pattern_fn
        self._expected_tensor_spec = expected_tensor_spec

    def build(self,
              validate_expected_tensor_spec: bool = True) -> TrajectoryTransform:
        """Creates `TrajectoryTransform` from a `TrajectoryTransformBuilder`."""

        if validate_expected_tensor_spec and self._expected_tensor_spec is None:
            raise ValueError('`expected_tensor_spec` must be set.')

        episode_ds = zero_episode_dataset_from_spec(self._rds_dataset_spec)

        steps_ds = episode_ds.flat_map(self._episode_to_steps_map_fn)

        episode_to_steps_fn_dataset_spec = self._rds_dataset_spec

        if self._step_map_fn is not None:
            steps_ds = steps_ds.map(self._step_map_fn)

        zeros_spec = transformations.zeros_from_spec(steps_ds.element_spec)    # pytype: disable=wrong-arg-types

        ref_step = reverb.structured_writer.create_reference_step(zeros_spec)

        pattern = self._pattern_fn(ref_step)

        steps_ds_spec = steps_ds.element_spec

        target_tensor_structure = create_reverb_table_signature(
                'temp_table', steps_ds_spec, pattern)

        if (validate_expected_tensor_spec and
                self._expected_tensor_spec != target_tensor_structure):
            raise RuntimeError(
                    'The tensor spec of the TrajectoryTransform doesn\'t '
                    'match the expected spec.\n'
                    'Expected:\n%s\nActual:\n%s\n' %
                    (str(self._expected_tensor_spec).replace('TensorSpec',
                                                                                                     'tf.TensorSpec'),
                     str(target_tensor_structure).replace('TensorSpec', 'tf.TensorSpec')))

        return TrajectoryTransform(
                episode_dataset_spec=self._rds_dataset_spec,
                episode_to_steps_fn_dataset_spec=episode_to_steps_fn_dataset_spec,
                steps_dataset_spec=steps_ds_spec,
                pattern=pattern,
                episode_to_steps_map_fn=self._episode_to_steps_map_fn,
                step_map_fn=self._step_map_fn,
                expected_tensor_spec=target_tensor_structure)

def zero_episode_dataset_from_spec(rlds_spec: RLDS_SPEC):
    """Creates a zero valued dataset of episodes for the given RLDS Spec."""

    def add_steps(episode, step_spec):
        episode[rlds_types.STEPS] = transformations.zero_dataset_like(
                tf.data.DatasetSpec(step_spec))
        if 'fake' in episode:
            del episode['fake']
        return episode

    episode_without_steps_spec = {
            k: v
            for k, v in rlds_spec.episode_tensor_spec().items()
            if k != rlds_types.STEPS
    }

    if episode_without_steps_spec:
        episodes_dataset = transformations.zero_dataset_like(
                tf.data.DatasetSpec(episode_without_steps_spec))
    else:
        episodes_dataset = tf.data.Dataset.from_tensors({'fake': ''})

    episodes_dataset_with_steps = episodes_dataset.map(
            lambda episode: add_steps(episode, rlds_spec.step_tensor_spec()))
    return episodes_dataset_with_steps


def create_reverb_table_signature(table_name: str, steps_dataset_spec,
                                  pattern: reverb.structured_writer.Pattern) -> reverb.reverb_types.SpecNest:
    config = create_structured_writer_config(table_name, pattern)
    reverb_table_spec = reverb.structured_writer.infer_signature(
            [config], steps_dataset_spec)
    return reverb_table_spec


def create_structured_writer_config(table_name: str,
                                    pattern: reverb.structured_writer.Pattern) -> Any:
    config = reverb.structured_writer.create_config(
            pattern=pattern, table=table_name, conditions=[])
    return config

def n_step_pattern_builder(n: int) -> Any:
    """Creates trajectory of length `n` from all fields of a `ref_step`."""

    def transform_fn(ref_step):
        traj = {}
        for key in ref_step:
            if isinstance(ref_step[key], dict):
                transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n:], ref_step[key])
                traj[key] = transformed_entry
            else:
                traj[key] = ref_step[key][-n:]

        return traj

    return transform_fn











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
            # print(key)
            # print(eps_by_key[key])
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

    # builder_episodic_dataset = builder.as_dataset(split=split, decoders={"steps": tfds.decode.SkipDecoding()}).enumerate().map(broadcast_metadata_rlds)

    builder_episodic_dataset = builder.as_dataset(split=split)
    # episodes = list(iter(builder_episodic_dataset))

    rlds_spec = RLDSSpec(
        observation_info=builder.info.features['steps']['observation'],
        action_info=builder.info.features['steps']['action_dict'],
    )

    
    # hack
    import dlimp as dl
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
    
    dataset = builder_episodic_dataset
    
        
    def episode_to_steps_map_fn(traj: Dict[str, Any]) -> Dict[str, Any]:
        return traj

    def step_map_fn(traj: Dict[str, Any]) -> Dict[str, Any]:
        # Apply pre-transforms
        if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'] is not None:
                traj = RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['pre'](traj, config=config)
                
        # Normalizing observations and actions
        if normalization_metadata is not None:
            traj = CommonTransforms.normalize_obs_and_actions(traj, config, normalization_metadata)

        # Relabel goals (Doesn't work for now)
        if config.train.goal_mode == 'last' or config.train.goal_mode == 'uniform':
            traj = CommonTransforms.relabel_goals_transform(traj, goal_mode=config.goal_mode)
            
        # Concatenate actions
        if config.train.action_keys != None:
            traj = CommonTransforms.concatenate_action_transform(traj, action_keys=config.train.action_keys)
        
        # Apply post-transforms
        if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            if RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'] is not None:
                traj = RLDS_TRAJECTORY_MAP_TRANSFORMS[name]['post'](traj, config=config)

        return traj

        
    
    trajectory_transform = TrajectoryTransformBuilder(
        rlds_spec,
        #   episode_to_steps_map_fn=episode_to_steps_map_fn,
        step_map_fn=step_map_fn,
        pattern_fn=n_step_pattern_builder(config.train.seq_length + config.train.frame_stack - 1)
    ).build(validate_expected_tensor_spec=False)
    
    trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(dataset)
    
    # combined_dataset = tf.data.Dataset.sample_from_datasets([trajectory_dataset])
    # combined_dataset = combined_dataset.batch(2)
    # combined_dataset_it = iter(combined_dataset)
    # example = next(combined_dataset_it)
    # x = Image.fromarray(example['obs']['exterior_image_1_left'].numpy()[0][0])
    
    dataset = trajectory_dataset
    # shuffle, repeat, pre-fetch, batch
    # dataset = dataset.cache()         # optionally keep full dataset in memory
    dataset = dataset.shuffle(1000)    # set shuffle buffer size
    dataset = dataset.repeat().batch(config.train.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.as_numpy_iterator()
    dataset = RLDSTorchDataset(dataset)

    return builder, dataset, normalization_metadata



