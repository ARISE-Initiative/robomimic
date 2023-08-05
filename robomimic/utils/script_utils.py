"""
Collection of miscellaneous utility tools
"""

def deep_update(d, u):
    """
    Copied from https://stackoverflow.com/a/3233356
    """
    import collections
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d