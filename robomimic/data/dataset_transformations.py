from typing import Any, Callable, Dict, Sequence, Union
import tensorflow as tf


def r2d2_dataset_pre_transform(traj: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    keep_keys = [
        "observation",
        "action",
        "action_dict",
        "language_instruction",
        "is_terminal",
        "is_last",
        "_traj_index",
    ]
    traj = {k: v for k, v in traj.items() if k in keep_keys}
    return traj


def r2d2_dataset_post_transform(traj: Dict[str, Any]) -> Dict[str, Any]:
    #Set obs key
    traj['obs'] = traj['observation']

    #Set actions key
    traj['actions'] = traj['action']
    
    #Use one goal per sequence
    if 'goal_observation' in traj.keys():
        traj['goal_obs'] = traj['goal_observation'][0]
    return traj


def robomimic_dataset_pre_transform(traj: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    keep_keys = [
        "observation",
        "action",
        "action_dict",
        "language_instruction",
        "is_terminal",
        "is_last",
        "_traj_index",
    ]
    traj = {k: v for k, v in traj.items() if k in keep_keys}
    return traj


def robomimic_dataset_post_transform(traj: Dict[str, Any]) -> Dict[str, Any]:
    new_traj = dict()
    #Set obs key
    traj['obs'] = traj['observation']

    #Set actions key
    traj['actions'] = traj['action']

    #Use one goal per sequence
    if 'goal_observation' in traj.keys():
        new_traj['goal_obs'] = traj['goal_observation'][0]    

    keep_keys = ['obs',
                'goal_obs',
                'actions',
                'action_dict']
    traj = {k: v for k, v in traj.items() if k in keep_keys} 
    return traj


RLDS_TRAJECTORY_MAP_TRANSFORMS = {
    'r2d2': {
        'pre': r2d2_dataset_pre_transform,
        'post': r2d2_dataset_post_transform,
    },
    'robomimic_dataset': {
        'pre': robomimic_dataset_pre_transform,
        'post': robomimic_dataset_post_transform,
    }
}

