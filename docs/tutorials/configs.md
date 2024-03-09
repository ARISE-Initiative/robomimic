# Configuring and Launching Training Runs

Robomimic uses a centralized [configuration system](../modules/configs.html) to specify (hyper)parameters at all levels. Below we walk through two ways to configure and launching training runs.


#### Best practices
<div class="admonition warning">
<p class="admonition-title">Warning! Do not modify default configs!</p>

Do not directly modify the default configs such as `config/bc_config.py`, especially if using the codebase with version control (e.g. git). Modifying these files modifies the default settings, and it’s easy to forget that these changes were made, or unintentionally commit these changes so that they become the new defaults.

</div>


Please see the [Config documentation](../modules/configs.html) for more information on Config objects, and the [hyperparameter scan tutorial](../tutorials/hyperparam_scan.html) for configuring hyperparameter sweeps.

#### 1. Using a config json (preferred)

The preferred way to specify training parameters is to pass a config json to the main training script `train.py` via the `--config` argument. The dataset can be specified by setting the `data` attribute of the `train` section of the config json, or specified via the `--dataset` argument. The example below runs a default template json for the BC algorithm. **This is the preferred way to launch training runs.**

```sh
$ python train.py --config ../exps/templates/bc.json --dataset ../../tests/assets/test.hdf5
```

Please see the [hyperparameter helper docs](./advanced.html#using-the-hyperparameter-helper-to-launch-runs) to see how to easily generate json configs for launching training runs.

#### 2. Constructing a config object in code

Another way to launch a training run is to make a default config (with a line like `config = config_factory(algo_name="bc")`), modify the config in python code, and then call the train function, like in the `examples/train_bc_rnn.py` script.

```python
import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bc")

# set config attributes here that you would like to update
config.experiment.name = "bc_rnn_example"
config.train.data = "/path/to/dataset.hdf5"
config.train.output_dir = "/path/to/desired/output_dir"
config.train.batch_size = 256
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)
```
