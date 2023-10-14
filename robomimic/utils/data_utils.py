import numpy as np
import tensorflow as tf
from functools import partial
from typing import Any, Callable, Dict, Sequence, Union


def index_nested_dict(d: Dict[str, Any], index: int):
    """
    Indexes a nested dictionary with backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    return d


def set_nested_dict_index(d: Dict[str, Any], index: int, value):
    """
    Sets an index in a nested dictionary with a value
    Indexes have backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices[:-1]:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    d[indices[-1]] = value


def map_nested_dict_index(d: Dict[str, Any], index: int, map_func):
    """
    Maps an index in a nested dictionary with a function
    Indexes have backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices[:-1]:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    d[indices[-1]] = map_func(d[indices[-1]])


def tree_map(
    x: Dict[str, Any],
    map_fn: Callable,
    *,
    _keypath: str = "",
) -> Dict[str, Any]:

    if not isinstance(x, dict):
        out = map_fn(x)
        return out
    out = {}
    for key in x.keys():
        if isinstance(x[key], dict):
            out[key] = tree_map(
                x[key], map_fn, _keypath=_keypath + key + "/"
            )
        else:
            out[key] = map_fn(x[key])
    return out


def selective_tree_map(
    x: Dict[str, Any],
    match: Union[str, Sequence[str], Callable[[str, Any], bool]],
    map_fn: Callable,
    *,
    _keypath: str = "",
) -> Dict[str, Any]:
    """Maps a function over a nested dictionary, only applying it leaves that match a criterion.

    Args:
        x (Dict[str, Any]): The dictionary to map over.
        match (str or Sequence[str] or Callable[[str, Any], bool]): If a string or list of strings, `map_fn` will only
        be applied to leaves whose key path contains one of `match`. If a function, `map_fn` will only be applied to
        leaves for which `match(key_path, value)` returns True.
        map_fn (Callable): The function to apply.
    """
    if not callable(match):
        if isinstance(match, str):
            match = [match]
        match_fn = lambda keypath, value: any([s in keypath for s in match])
    else:
        match_fn = match

    out = {}
    for key in x:
        if isinstance(x[key], dict):
            out[key] = selective_tree_map(
                x[key], match_fn, map_fn, _keypath=_keypath + key + "/"
            )
        elif match_fn(_keypath + key, x[key]):
            out[key] = map_fn(x[key])
        else:
            out[key] = x[key]
    return out


def decode_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    if isinstance(match, str):
        match = [match]

    def match_fn(keypath, value):
        image_in_keypath = any([s in keypath for s in match])
        return image_in_keypath

    return selective_tree_map(
        x,
        match=match_fn,
        map_fn=partial(tf.io.decode_image, expand_animations=False),
    )

