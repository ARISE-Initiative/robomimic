# Getting Familiar with Configs

TODO: what parts should go here vs. in modules/configs?

## Config Example

The simple config example script at `examples/simple_config.py` shows how the `Config` object can easily be instantiated and modified safely with different levels of locking. We reproduce certain portions of the script. First, we can create a `Config` object and call `lock` when we think we won't need to change it anymore.

```python
from robomimic.config.base_config import Config

# create config
config = Config()

# add nested attributes to the config
config.train.batch_size = 100
config.train.learning_rate = 1e-3
config.algo.actor_network_size = [1000, 1000]
config.lock()  # prevent accidental changes
```

Now, when we try to add a new key (or modify the value of an existing key), the config will throw an error.

```python
# the config is locked --- cannot add new keys or modify existing keys
try:
    config.train.optimizer = "Adam"
except RuntimeError as e:
    print(e)
```

However, the config can be safely modified using appropriate contexts.

```python
# values_unlocked scope allows modifying values of existing keys, but not adding keys
with config.values_unlocked():
    config.train.batch_size = 200
print("batch_size={}".format(config.train.batch_size))

# unlock config within the scope, allowing new keys to be inserted
with config.unlocked():
    config.test.num_eval = 10

# verify that the config remains locked outside of the scope
assert config.is_locked
assert config.test.is_locked
```

Finally, the config can also be updated by using external dictionaries - this is helpful for loading config jsons.

```python
# update this config with external config from a dict
ext_config = {
    "train": {
        "learning_rate": 1e-3
    },
    "algo": {
        "actor_network_size": [1000, 1000]
    }
}
with config.values_unlocked():
    config.update(ext_config)

print(config)
```

Please see the [Config documentation](../modules/configs.html) for more information on Config objects.