"""
An example for creating and using the custom Config object.
"""

from robomimic.config.base_config import Config

if __name__ == "__main__":

    # create config
    config = Config()

    config.train.batch_size = 100
    config.train.learning_rate = 1e-3
    config.algo.actor_network_size = [1000, 1000]
    config.lock()  # prevent accidental changes

    # access config
    print("batch_size={}".format(config.train.batch_size))

    # the config is locked --- cannot add new keys or modify existing keys
    try:
        config.train.optimizer = "Adam"
    except RuntimeError as e:
        print(e)

    # values_unlocked scope allows modifying values of existing keys, but not adding keys
    with config.values_unlocked():
        config.train.batch_size = 200
    print("batch_size={}".format(config.train.batch_size))

    # allow adding new keys to the config
    with config.unlocked():
        config.test.num_eval = 10

    assert config.is_locked
    assert config.test.is_locked

    # read external config from a dict
    ext_config = {
        "train": {"learning_rate": 1e-3},
        "algo": {"actor_network_size": [1000, 1000]},
    }
    with config.values_unlocked():
        config.update(ext_config)

    print(config)
