from typing import Any, Callable, Dict, Sequence, Union
import robomimic.utils.tensorflow_utils as TensorflowUtils
import tensorflow as tf


def r2d2_dataset_pre_transform(traj: Dict[str, Any], 
    config: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    keep_keys = [
        'observation',
        'action',
        'is_first',
        'is_last',
        'is_terminal'
    ]
    ac_keys = ['cartesian_position', 'cartesian_velocity']
    new_traj = {k: v for k, v in traj.items() if k in keep_keys}
    new_traj['action_dict'] = {
        'gripper_position': traj['action_dict']['gripper_position']
    }
    for key in ac_keys:
        in_action = traj['action_dict'][key]
        if len(traj['action_dict'][key].shape) == 2:
            pos = traj['action_dict'][key][:, :3]
            rot = traj['action_dict'][key][:, 3:6]
        else:
            pos = traj['action_dict'][key][:3]
            rot = traj['action_dict'][key][3:6]
        
        rot_6d = TensorflowUtils.euler_angles_to_rot_6d(
            rot, convention="XYZ",
        ) 
        if key == 'cartesian_position':
            prefix = 'abs_'
        else:
            prefix = 'rel_'

        new_traj['action_dict'].update({
            prefix + 'pos': pos,
            prefix + 'rot_euler': rot,
            prefix + 'rot_6d': rot_6d
        })
    return new_traj


def r2d2_dataset_post_transform(traj: Dict[str, Any], 
    config: Dict[str, Any]) -> Dict[str, Any]:
  
    new_traj = {'observation': {}}
    for key in config.all_obs_keys:
        nested_keys = key.split('/')
        value = traj['observation']
        assign = new_traj['observation']
        for i, nk in enumerate(nested_keys):
            if i == len(nested_keys) - 1:
                assign[nk] = value[nk]
                break
            value = value[nk]
            if nk not in assign.keys():
               assign[nk] = dict()
            assign = assign[nk]
    #Set obs key
    new_traj['obs'] = new_traj['observation']

    #Set actions key
    new_traj['actions'] = traj['action']
    
    new_traj['is_first'] = traj['is_first']
    new_traj['is_last'] = traj['is_last']
    new_traj['is_terminal'] = traj['is_terminal']
    new_traj['action_dict'] = traj['action_dict']
    
    #Use one goal per sequence
    if 'goal_observation' in traj.keys():
        new_traj['goal_obs'] = traj['goal_observation'][0]
    keep_keys = ['obs',
                'goal_obs',
                'actions',
                'action_dict',
                'is_first',
                'is_last',
                'is_terminal']
    new_traj = {k: v for k, v in new_traj.items() if k in keep_keys}
    return new_traj


def robomimic_dataset_pre_transform(traj: Dict[str, Any],
    config: Dict[str, Any]) -> Dict[str, Any]:
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


def robomimic_dataset_post_transform(traj: Dict[str, Any],
    config: Dict[str, Any]) -> Dict[str, Any]:
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
    'r2_d2': {
        'pre': r2d2_dataset_pre_transform,
        'post': r2d2_dataset_post_transform,
    },
    'robomimic_dataset': {
        'pre': robomimic_dataset_pre_transform,
        'post': robomimic_dataset_post_transform,
    }
}

