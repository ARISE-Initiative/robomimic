"""
Set of general purpose utility functions for easier interfacing with Python API
"""
import inspect
from copy import deepcopy
import robomimic.macros as Macros


def get_class_init_kwargs(cls):
    """
    Helper function to return a list of all valid keyword arguments (excluding "self") for the given @cls class.

    Args:
        cls (object): Class from which to grab __init__ kwargs

    Returns:
        list: All keyword arguments (excluding "self") specified by @cls __init__ constructor method
    """
    return list(inspect.signature(cls.__init__).parameters.keys())[1:]


def extract_subset_dict(dic, keys, copy=False):
    """
    Helper function to extract a subset of dictionary key-values from a current dictionary. Optionally (deep)copies
    the values extracted from the original @dic if @copy is True.

    Args:
        dic (dict): Dictionary containing multiple key-values
        keys (Iterable): Specific keys to extract from @dic. If the key doesn't exist in @dic, then the key is skipped
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys

    Returns:
        dict: Extracted subset dictionary containing only the specified @keys and their corresponding values
    """
    subset = {k: dic[k] for k in keys if k in dic}
    return deepcopy(subset) if copy else subset


def extract_class_init_kwargs_from_dict(cls, dic, copy=False, verbose=False):
    """
    Helper function to return a dictionary of key-values that specifically correspond to @cls class's __init__
    constructor method, from @dic which may or may not contain additional, irrelevant kwargs.

    Note that @dic may possibly be missing certain kwargs as specified by cls.__init__. No error will be raised.

    Args:
        cls (object): Class from which to grab __init__ kwargs that will be be used as filtering keys for @dic
        dic (dict): Dictionary containing multiple key-values
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys
        verbose (bool): If True (or if macro DEBUG is True), then will print out mismatched keys

    Returns:
        dict: Extracted subset dictionary possibly containing only the specified keys from cls.__init__ and their
            corresponding values
    """
    # extract only relevant kwargs for this specific backbone
    cls_keys = get_class_init_kwargs(cls)
    subdic = extract_subset_dict(
        dic=dic,
        keys=cls_keys,
        copy=copy,
    )

    # Run sanity check if verbose or debugging
    if verbose or Macros.DEBUG:
        keys_not_in_cls = [k for k in dic if k not in cls_keys]
        keys_not_in_dic = [k for k in cls_keys if k not in list(dic.keys())]
        if len(keys_not_in_cls) > 0:
            print(f"Warning: For class {cls.__name__}, got unknown keys: {keys_not_in_cls} ")
        if len(keys_not_in_dic) > 0:
            print(f"Warning: For class {cls.__name__}, got missing keys: {keys_not_in_dic} ")

    return subdic