"""
Basic config class - provides a convenient way to work with nested
dictionaries (by exposing keys as attributes) and to save / load from jsons.

Based on addict: https://github.com/mewwts/addict
"""

import json
import copy
import contextlib
from copy import deepcopy


class Config(dict):

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__key_locked', False)  # disallow adding new keys
        object.__setattr__(__self, '__all_locked', False)  # disallow both key and value update
        object.__setattr__(__self, '__do_not_lock_keys', False)  # cannot be key-locked
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def lock(self):
        """
        Lock the config. Afterwards, new keys cannot be added to the
        config, and the values of existing keys cannot be modified.
        """
        object.__setattr__(self, '__all_locked', True)
        if self.key_lockable:
            object.__setattr__(self, '__key_locked', True)

        for k in self:
            if isinstance(self[k], Config):
                self[k].lock()

    def unlock(self):
        """
        Unlock the config. Afterwards, new keys can be added to the
        config, and the values of existing keys can be modified.
        """
        object.__setattr__(self, '__all_locked', False)
        object.__setattr__(self, '__key_locked', False)

        for k in self:
            if isinstance(self[k], Config):
                self[k].unlock()

    def _get_lock_state_recursive(self):
        """
        Internal helper function to get the lock state of all sub-configs recursively.
        """
        lock_state = {"__all_locked": self.is_locked, "__key_locked": self.is_key_locked}
        for k in self:
            if isinstance(self[k], Config):
                assert k not in ["__all_locked", "__key_locked"]
                lock_state[k] = self[k]._get_lock_state_recursive()
        return lock_state

    def _set_lock_state_recursive(self, lock_state):
        """
        Internal helper function to set the lock state of all sub-configs recursively.
        """
        lock_state = deepcopy(lock_state)
        object.__setattr__(self, '__all_locked', lock_state.pop("__all_locked"))
        object.__setattr__(self, '__key_locked', lock_state.pop("__key_locked"))
        for k in lock_state:
            if isinstance(self[k], Config):
                self[k]._set_lock_state_recursive(lock_state[k])

    def _get_lock_state(self):
        """
        Retrieves the lock state of this config.

        Returns:
            lock_state (dict): a dictionary with an "all_locked" key that is True
                if both key and value updates are locked and False otherwise, and
                a "key_locked" key that is True if only key updates are locked (value
                updates still allowed) and False otherwise
        """
        return {
            "all_locked": self.is_locked,
            "key_locked": self.is_key_locked
        }

    def _set_lock_state(self, lock_state):
        """
        Sets the lock state for this config.

        Args:
            lock_state (dict): a dictionary with an "all_locked" key that is True
                if both key and value updates should be locked and False otherwise, and
                a "key_locked" key that is True if only key updates should be locked (value
                updates still allowed) and False otherwise
        """
        if lock_state["all_locked"]:
            self.lock()
        if lock_state["key_locked"]:
            self.lock_keys()

    @contextlib.contextmanager
    def unlocked(self):
        """
        A context scope for modifying a Config object. Within the scope,
        both keys and values can be updated. Upon leaving the scope,
        the initial level of locking is restored.
        """
        lock_state = self._get_lock_state()
        self.unlock()
        yield
        self._set_lock_state(lock_state)

    @contextlib.contextmanager
    def values_unlocked(self):
        """
        A context scope for modifying a Config object. Within the scope,
        only values can be updated (new keys cannot be created). Upon 
        leaving the scope, the initial level of locking is restored.
        """
        lock_state = self._get_lock_state()
        self.unlock()
        self.lock_keys()
        yield
        self._set_lock_state(lock_state)

    def lock_keys(self):
        """
        Lock this config so that new keys cannot be added.
        """
        if not self.key_lockable:
            return
        object.__setattr__(self, '__key_locked', True)
        for k in self:
            if isinstance(self[k], Config):
                self[k].lock_keys()

    def unlock_keys(self):
        """
        Unlock this config so that new keys can be added.
        """
        object.__setattr__(self, '__key_locked', False)
        for k in self:
            if isinstance(self[k], Config):
                self[k].unlock_keys()

    @property
    def is_locked(self):
        """
        Returns True if the config is locked (no key or value updates allowed).
        """
        return object.__getattribute__(self, '__all_locked')

    @property
    def is_key_locked(self):
        """
        Returns True if the config is key-locked (no key updates allowed).
        """
        return object.__getattribute__(self, '__key_locked')

    def do_not_lock_keys(self):
        """
        Calling this function on this config indicates that key updates should be 
        allowed even when this config is key-locked (but not when it is completely
        locked). This is convenient for attributes that contain kwargs, where there
        might be a variable type and number of arguments contained in the sub-config.
        """
        object.__setattr__(self, '__do_not_lock_keys', True)

    @property
    def key_lockable(self):
        """
        Returns true if this config is key-lockable (new keys cannot be inserted in a 
        key-locked lock level).
        """
        return not object.__getattribute__(self, '__do_not_lock_keys')

    def __setattr__(self, name, value):
        if self.is_locked:
            raise RuntimeError("This config has been locked - cannot set attribute '{}' to {}".format(name, value))

        if hasattr(Config, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        elif not hasattr(self, name) and self.is_key_locked:
            raise RuntimeError("This config is key-locked - cannot add key '{}'".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        super(Config, self).__setitem__(name, value)
        p = object.__getattribute__(self, '__parent')
        key = object.__getattribute__(self, '__key')
        if p is not None:
            p[key] = self

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            # We return Config instance instead of cls instance to ensure all sub-configs are not a top-level class
            return Config(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(Config._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __repr__(self):
        json_string = json.dumps(self.to_dict(), indent=4)
        return json_string

    def __getitem__(self, name):
        if name not in self:
            if object.__getattribute__(self, '__all_locked') or object.__getattribute__(self, '__key_locked'):
                 raise RuntimeError("This config has been locked and '{}' is not in this config".format(name))
            return Config(__parent=self, __key=name)
        return super(Config, self).__getitem__(name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        """
        Update this config using another config or nested dictionary.
        """
        if self.is_locked:
            raise RuntimeError('Cannot update - this config has been locked')
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if self.is_key_locked and k not in self:
                raise RuntimeError("Cannot update - this config has been key-locked and key '{}' does not exist".format(k))
            if (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def dump(self, filename=None):
        """
        Dumps the config to a json.

        Args:
            filename (str): if not None, save to json file.

        Returns:
            json_string (str): json string representation of
                this config
        """
        json_string = json.dumps(self.to_dict(), indent=4)
        if filename is not None:
            f = open(filename, "w")
            f.write(json_string)
            f.close()
        return json_string