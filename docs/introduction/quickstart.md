# Getting Started

This section discusses how to get started with the robomimic repository, by providing examples of how to train and evaluate models.

## Training Models

This section discusses how models can be trained.

**Note:** These examples [require robosuite](./installation.html#robosuite) to be installed, but they can run without robosuite by disabling rollouts in `robomimic/configs/base_config.py`, `robomimic/exps/templates/bc.json`, and `examples/train_bc_rnn.py`. 

### Run a quick example

To see a quick example of a training run, along with the outputs, run the `train_bc_rnn.py` script in the `examples` folder (the `--debug` flag is used to ensure the training run only takes a few seconds).

```sh
$ python train_bc_rnn.py --debug
```

The default dataset used is the one in `tests/assets/test.hdf5` and the default directory where results are saved for the example training run is in `tests/tmp_model_dir`. Both can be overridden by passing arguments to the above script. 

**Warning:** If you are using the default dataset (and rollouts are enabled), please make sure that robosuite is on the `offline_study` branch of robosuite.

After the script finishes, you can check the training outputs in the output directory (`tests/tmp_model_dir/bc_rnn_example` by default). See the "Viewing Training Results" section below for more information on interpreting the output.

### Ways to launch training runs

In this section, we describe the different ways to launch training runs.

#### Using a config json (preferred)

One way is to use the `train.py` script, and pass a config json via the `--config` argument. The dataset can be specified by setting the `data` attribute of the `train` section of the config json, or specified via the `--dataset` argument. The example below runs a default template json for the BC algorithm. **This is the preferred way to launch training runs.**

```sh
$ python train.py --config ../exps/templates/bc.json --dataset ../../tests/assets/test.hdf5
```

Please see the [hyperparameter helper docs](./advanced.html#using-the-hyperparameter-helper-to-launch-runs) to see how to easily generate json configs for launching training runs.

#### Constructing a config object in code

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

#### Directly modifying the config class source code (avoid this)

Technically, a third way to launch a training run is to directly modify the relevant `Config` classes (such as `config/bc_config.py` and `config/base_config.py`) and then run `train.py` but **this is not recommended**, especially if using the codebase with version control (e.g. git). Modifying these files modifies the default settings, and it's easy to forget that these changes were made, or unintentionally commit these changes so that they become the new defaults. For this reason, **we recommend never modifying the config classes directly, unless you are modifying an algorithm and adding new config keys**. 

To learn more about the `Config` class, read the [Configs documentation](../modules/configs.html), or look at the source code.


## Viewing Training Results

This section discusses how to view and interpret the results of training runs.

### Logs, Models, and Rollout Videos

Training runs will output results to the directory specified by `config.train.output_dir`, under a folder with the experiment name (specified by `config.experiment.name`). This folder contains a directory named by a timestamp (e.g. `20210708174935`) for every training run with this same name, and within that directory, there should be three folders - `logs`, `models`, and `videos`. 

The `logs` directory will contain everything printed to stdout in `log.txt` (only if `config.experiment.logging.terminal_output_to_txt` is set to `True`), and a `tb` folder containing tensorboard logs (only if `config.experiment.logging.log_tb` is set to True). You can visualize the tensorboard results by using a command like the below, and then opening the link printed on the terminal in a web browser. The tensorboard logs have convenient sections for rollout evaluations, quantities logged during training, quantities logged during validation, and timing statistics for different parts of the training process (in minutes).

```sh
$ tensorboard --logdir /path/to/output/dir --bind_all
```

The `models` directory contains saved model checkpoints. These can be used by the `run_trained_agent.py` script (more on this below). The `config.experiment.save` portion of the config controls if and when models are saved during training.

The `videos` directory contains evaluation rollout videos collected during training, when evaluating trained models in the environment (only if `config.experiment.render_video` is set to `True`). The `config.experiment.rollout` portion of the config controls how often rollouts happen, and how many happen.

### Evaluating Trained Policies

Saved policy checkpoints in the `models` directory can be evaluated using the `run_trained_agent.py` script. The below example can be used to evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video. The agentview and wrist camera images are used to render video frames.

```sh
$ python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --video_path /path/to/output.mp4 --camera_names agentview robot0_eye_in_hand 
```

The 50 agent rollouts can also be written to a new dataset hdf5.

```sh
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5 --dataset_obs 
```

Instead of storing the observations, which can consist of high-dimensional images, they can be excluded by omitting the `--dataset_obs` flag. The observations can be extracted using the `dataset_states_to_obs.hdf5` script (see the Datasets documentation for more information on this).

```sh
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5
```
