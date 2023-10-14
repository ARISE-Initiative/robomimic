import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from typing import Any, Callable, Dict, Sequence, Union
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.data_utils import *


def add_next_obs(traj: Dict[str, Any], pad: bool = True) -> Dict[str, Any]:
    """
    Given a trajectory with a key "observation", add the key "next_observation". If pad is False, discards the last
    value of all other keys. Otherwise, the last transition will have "observation" == "next_observation".
    """
    if not pad:
        traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
        traj_truncated["next_observation"] = tf.nest.map_structure(
            lambda x: x[1:], traj["observation"]
        )
        return traj_truncated
    else:
        traj["next_observation"] = tf.nest.map_structure(
            lambda x: tf.concat((x[1:], x[-1:]), axis=0), traj["observation"]
        )
        return traj


def normalize_obs_and_actions(traj, config, metadata):
    '''
    For now, only normalize appropriate action keys
    '''
    action_config = config.train.action_config
    normal_keys = [key for key in config.train.action_keys
        if key in action_config.keys() and action_config[key].get('normalization', None) == 'normal']
    
    min_max_keys = [key for key in config.train.action_keys
        if key in action_config.keys() and action_config[key].get('normalization', None) == 'min_max']
    
    for key in normal_keys:
        map_nested_dict_index(
            traj,
            key,
            lambda x: (x - metadata[key]["mean"]) / metadata[key]["std"]
        )        

    for key in min_max_keys:
        map_nested_dict_index(
            traj,
            key,
            lambda x: tf.clip_by_value(2 * (x - metadata[key]["min"])
                / (metadata[key]["max"] - metadata[key]["min"]) - 1,
                -1,
                1)
        )
    
    return traj


def random_dataset_sequence_transform(traj, sequence_length):
    '''
    Extract a random subsequence of the data given sequence_length given keys
    '''
    traj_len = len(traj["action"])
    index_in_demo = tf.cast(tf.random.uniform(shape=[])
         * tf.cast(traj_len, dtype=tf.float32), dtype=tf.int32)
    last_index = tf.math.minimum(traj_len, index_in_demo + sequence_length)
    seq_end_pad = tf.math.maximum(0, index_in_demo + sequence_length - traj_len)
    padding = [0, seq_end_pad]
    keys = ["observation", "action", "action_dict", "goal"]

    def random_sequence_func(x):
        sequence = x[index_in_demo: last_index]
        padding = tf.repeat([x[0]], repeats=[seq_end_pad], axis=0)
        return tf.concat((sequence, padding), axis=0)

    traj = dl.transforms.selective_tree_map(
        traj,
        match=keys,
        map_fn=random_sequence_func
    ) 
    return traj


def random_dataset_sequence_transform_v2(traj, frame_stack, seq_length,
    pad_frame_stack, pad_seq_length):
    '''
    Extract a random subsequence of the data given sequence_length given keys
    '''
    traj_len = tf.shape(traj["action"])[0]
    seq_begin_pad, seq_end_pad = 0, 0
    if pad_frame_stack:
        seq_begin_pad = frame_stack - 1
    if pad_seq_length:
        seq_end_pad = seq_length - 1
    index_in_demo = tf.random.uniform(shape=[],
            maxval=traj_len + seq_end_pad - (seq_length - 1), 
          dtype=tf.int32)
    pad_mask = tf.concat((tf.repeat(0, repeats=seq_begin_pad),
                        tf.repeat(1, repeats=traj_len), 
                        tf.repeat(0, repeats=seq_end_pad)), axis=0)[:, None]
    traj['pad_mask'] = tf.cast(pad_mask, dtype=tf.bool)
    keys = ["observation", "action", "action_dict", "goal"]

    def random_sequence_func(x):
        begin_padding = tf.repeat([x[0]], repeats=[seq_begin_pad], axis=0)
        end_padding = tf.repeat([x[-1]], repeats=[seq_end_pad], axis=0)
        sequence = tf.concat((begin_padding, x, end_padding), axis=0)
        return sequence[index_in_demo: index_in_demo + seq_length + frame_stack - 1]

    traj = dl.transforms.selective_tree_map(
        traj,
        match=keys,
        map_fn=random_sequence_func
    )
    return traj

 

def relabel_goals_transform(traj, goal_mode):
    traj_len = len(traj["action"])

    if goal_mode == "last":
        goal_idxs = tf.ones(traj_len) * (traj_len - 1)
        goal_idxs = tf.cast(goal_idxs, tf.int32)
    elif goal_mode == "uniform":
        rand = tf.random.uniform([traj_len])
        low = tf.cast(tf.range(traj_len) + 1, tf.float32)
        high = tf.cast(traj_len, tf.float32)
        goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    traj["goal_observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs), traj["observation"]
    )
    return traj


def concatenate_action_transform(traj, action_keys):
    '''
    Concatenates the action_keys
    '''
    traj["action"] = tf.concat(
        list(index_nested_dict(traj, key) for key in action_keys),
        axis=-1
    )

    return traj


def frame_stack_transform(traj, num_frames):
    '''
    Stacks the previous num_frame-1 frames with the current frame
    Converts the trajectory into size
    traj_len x num_frames x ...
    '''
    traj_len = len(traj["action"])

    #Pad beginning of observation num_frames times:
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.concat((tf.repeat([x[0]], repeats=[num_frames], axis=0)
            , x), axis=0)
        , traj["observation"])
    
    def stack_func(x):
        indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(num_frames)
        return tf.gather(x, indices)

    #Concatenate and clip to original size
    traj["observation"] = tf.nest.map_structure(
        stack_func,
        traj["observation"]
    )

    return traj

