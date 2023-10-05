import numpy as np
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
