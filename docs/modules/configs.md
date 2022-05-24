# Configs

Configs are instances of the `Config` class defined in `config/config.py`. The implementation is largely based on [addict](https://github.com/mewwts/addict), which makes working with nested dictionaries convenient. At their core, configs are essentially nested dictionaries (similar to loaded json objects), with an easy way to access and set nested keys by using attributes. See the example below:

```python
from robomimic.config import Config

# normal way to create nested dictionaries and read values
c = dict()
c["experiment"] = dict()
c["experiment"]["save"] = dict()
c["experiment"]["save"]["enabled"] = True
print("save enabled: {}".format(c["experiment"]["save"]["enabled"]))

# can use dot syntax instead of key lookup to create nested dictionaries
c = Config()
c.experiment.save.enabled = True

# can also use dot syntax to access attributes
print("save enabled: {}".format(c.experiment.save.enabled))
```

It's easy to go back and forth between `Config` objects and jsons as well, which is convenient when saving config objects to disk (this happens when generating new config jsons for training, and when saving the config in a model checkpoint), and loading configs from jsons.

```python
# dump config as a json string
json_string = c.dump()

# dump config to a json file
c.dump(filename="c.json")

# load config from json
import json
json_dic = json.load(json_string)
c = Config(json_dic)
```

All algorithm config classes (one class per algorithm type) are subclasses of the `BaseConfig` class (defined in `config/base_config.py`), which is a subclass of the general `Config` class. The algorithm config classes are kept in a global registry (more details on this later). The `generate_config_templates.py` script uses the functionality demonstrated above to easily generate template config jsons for each algorithm, by instantiating the default config object per class. **This script should be run any time changes are made to default settings, or new config settings are added.** We reproduce the code snippet below.

```python
import os
import json
import robomimic
from robomimic.config import get_all_registered_configs

# store template config jsons in this directory
target_dir = os.path.join(robomimic.__path__[0], "exps/templates/")

# iterate through registered algorithm config classes
all_configs = get_all_registered_configs()
for algo_name in all_configs:
    # make config class for this algorithm
    c = all_configs[algo_name]()
    # dump to json
    json_path = os.path.join(target_dir, "{}.json".format(algo_name))
    c.dump(filename=json_path)
```

We now go over the general structure of algorithm config classes by dicusssing the `BaseConfig` class, which has more specific functionality than the general `Config` class for use with this repository. 

## Config Factory

The `BaseConfig` class has a property and classmethod called `ALGO_NAME` that must be filled out by all subclasses -- this should correspond to the algorithm name for each config, and should match the algorithm name that is passed to `register_algo_factory_func` at the top of each algorithm implementation file. For example, for `BCConfig`, `ALGO_NAME = "bc"`. This property is important in order to make sure that all configs that subclass `BaseConfig` get registered into the `REGISTERED_CONFIGS` global registry (`ALGO_NAME` is used as a key to register the class into the registry). This also allows the `config_factory` function to easily create the appropriate config class for an algorithm -- **this is the standard entry point for creating config objects in the codebase**. While config jsons can also be used to load configs during training (as discussed in the Quick Start section of the documentation), the `config_factory` function is still used to create an initial config object, which is then updated (see the example below).

```python
import json
from robomimic.config import config_factory

# base config for algorithm with default values
config = config_factory("bc")

# update defaults with config json
with open("/path/to/config.json", "r") as f:
    ext_config_json = json.load(f)
config.update(ext_config_json)
```

At test-time, when loading a model from a checkpoint, the config is restored by reading the json string from the checkpoint, and then using it to instantiate the config. An example is below (modified slightlyt from `config_from_checkpoint` in `utils/file_utils.py`)

```python
import robomimic.utils.file_utils as FileUtils
ckpt_dict = FileUtils.load_dict_from_checkpoint("path/to/ckpt.pth")
config_json = ckpt_dict["config"]
config = config_factory(ckpt_dict["algo_name"], dic=json.loads(config_json))
```

## Config Structure

The `BaseConfig` class (and all subclasses) have 4 important sections that need to be filled out -- each is a method of the config class. See `config/base_config.py` (and the relevant algorithm config files like `config/bc_config.py`) for details on the specific settings under each section.

- `experiment_config(self)`
  - This function populates the `config.experiment` attribute of the config, which has several experiment settings such as the name of the training run, whether to do logging, whether to save models (and how often), whether to render videos, and whether to do rollouts (and how often). The `BaseConfig` class has a default implementation that usually doesn't need to be overriden.
- `train_config(self)`
  - This function populates the `config.train` attribute of the config, which has several settings related to the training process, such as the dataset to use for training, and how the data loader should load the data. The `BaseConfig` class has a default implementation that usually doesn't need to be overriden.
- `algo_config(self)`
  - This function populates the `config.algo` attribute of the config, and is given to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` argument to the constructor. Any parameter that an algorithm needs to determine its training and test-time behavior should be populated here. This function should be implemented by every subclass.
- `observation_config(self)`
  - This function populates the `config.observation` attribute of the config, and is given to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `obs_config` argument to the constructor. This portion of the config is used to specify what observation modalities should be used by the networks for training, and how the observation modalities should be encoded by the networks. While the `BaseConfig` class has a default implementation that usually doesn't need to be overriden, certain algorithm configs may choose to, in order to have seperate configs for different networks in the algorithm. 

## Config Locking

To prevent accidental config modification, each `Config` object implements a two-level locking mechanism: _key-locked_ and _all-locked_. A `Config` object can be put into the _key-locked_ state by calling `config.lock_keys()`. Under the _key-locked_  state, a `Config` object does not allow adding new keys but only changing the values of existing keys. A `Config` object can be put into the _all-locked_ state by calling `config.lock()`. In this state, the object does not allow adding new keys nor changing existing values.

All `Config` objects that inherit the constructor of `BaseConfig` are _key-locked_ by default. Upon finishing constructing the object, it is recommended to put the config object into the _all-locked_ state by calling `config.lock()`. `Config` objects also implement context manager functions that allow convenient and controlled config modification within a scope.

- `value_unlocked()`
  - This context scope allows values of existing keys to be modified if the `Config` object is _all-locked_.
- `unlocked()`
  - This context scope allows new keys to be added as well as values of existing keys to be modified, regardless of the locking state.

## Minimum Example

Please see the [config tutorial](../tutorials/configs.html) for more information on how to use the Config object, and examples on how the locking mechanism works.
