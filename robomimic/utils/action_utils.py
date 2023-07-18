
from typing import Union, Sequence, Dict, Optional, Tuple

from copy import deepcopy
from collections import OrderedDict
import functools

import numpy as np


def action_dict_to_vector(
        action_dict: Dict[str, np.ndarray], 
        action_keys: Optional[Sequence[str]]=None) -> np.ndarray:
    if action_keys is None:
        action_keys = list(action_dict.keys())
    actions = [action_dict[k] for k in action_keys]

    action_vec = np.concatenate(actions, axis=-1)
    return action_vec


def vector_to_action_dict(
        action: np.ndarray, 
        action_shapes: Dict[str, Tuple[int]],
        action_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    action_dict = dict()
    start_idx = 0
    for key in action_keys:
        this_act_shape = action_shapes[key]
        this_act_dim = np.prod(this_act_shape)
        end_idx = start_idx + this_act_dim
        action_dict[key] = action[...,start_idx:end_idx].reshape(
            action.shape[:-1]+this_act_shape)
        start_idx = end_idx
    return action_dict
