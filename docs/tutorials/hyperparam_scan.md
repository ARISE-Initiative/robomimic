# Running Hyperparameter Scans

We provide the `ConfigGenerator` class under `utils/hyperparam_utils.py` to easily set and sweep over hyperparameters.
**This is the preferred way to launch multiple training runs using the repository.** 
Follow the steps below for running your own hyperparameter scan:
1. [Create Base Config json](#step-1-create-base-config-json)
2. [Create Config Generator](#step-2-create-config-generator)
3. [Set Hyperparameter Values](#step-3-set-hyperparameter-values)
4. [Run Hyperparameter Helper Script](#step-4-run-hyperparameter-helper-script)

## Step 1: Create Base Config json
The first step is to start with a base config json. A common choice is to copy one of the templates in `exps/templates` (such as `exps/templates/bc.json`) into a new folder (where additional config jsons will be generated).

```sh
$ cp ../exps/templates/bc.json /tmp/gen_configs/base.json
```

<div class="admonition tip">
<p class="admonition-title">Relevant settings in base json file</p>

Sections of the config that are not involved in the scan and that do not differ from the default values in the template can also be omitted, if desired.

</div>

We modify `/tmp/gen_configs/base.json`, adding a base experiment name (`"bc_rnn_hyper"`) and specified the dataset path (`"/tmp/test_v15.hdf5"`).

```sh
$ cat /tmp/gen_configs/base.json
```

<details>
  <summary><b>Click to see output</b></summary>
<p>

```json
{
    "algo_name": "bc",
    "experiment": {
        "name": "bc_rnn_hyper",
        "validate": true,
        "save": {
            "enabled": true,
            "every_n_seconds": null,
            "every_n_epochs": 50,
            "epochs": [],
            "on_best_validation": false,
            "on_best_rollout_return": false,
            "on_best_rollout_success_rate": true
        },
        "epoch_every_n_steps": 100,
        "validation_epoch_every_n_steps": 10,
        "rollout": {
            "enabled": true,
            "n": 50,
            "horizon": 400,
            "rate": 50,
            "warmstart": 0,
            "terminate_on_success": true
        }
    },
    "train": {
        "data": "/tmp/test_v15.hdf5",
        "output_dir": "../bc_trained_models",
        "num_data_workers": 0,
        "hdf5_cache_mode": "all",
        "hdf5_use_swmr": true,
        "hdf5_normalize_obs": false,
        "hdf5_filter_key": null,
        "seq_length": 1,
        "goal_mode": null,
        "cuda": true,
        "batch_size": 100,
        "num_epochs": 2000,
        "seed": 1
    },
    "algo": {
        "optim_params": {
            "policy": {
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 0.1,
                    "epoch_schedule": []
                },
                "regularization": {
                    "L2": 0.0
                }
            }
        },
        "actor_layer_dims": [
            1024,
            1024
        ],
        "gmm": {
            "enabled": false,
            "num_modes": 5,
            "min_std": 0.0001,
            "std_activation": "softplus",
            "low_noise_eval": true
        },
        "rnn": {
            "enabled": false,
            "horizon": 10,
            "hidden_dim": 400,
            "rnn_type": "LSTM",
            "num_layers": 2
        }
    }
}
```

</p>
</details>

## Step 2: Create Config Generator

The next step is create a `ConfigGenerator` object which procedurally generates new configs (one config per unique hyperparameter combination).
We provide an example in `scripts/hyperparam_helper.py` and for the remainder of this tutorial we will follow this script step-by-step.

First, we define a function `make_generator` that creates a `ConfigGenerator` object.
After this, our next step will be to set hyperparameter values.

```python
import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )
    
    # next: set and sweep over hyperparameters
    generator.add_param(...) # set / sweep hp1
    generator.add_param(...) # set / sweep hp2
    generator.add_param(...) # set / sweep hp3
    ...
    
    return generator

def main(args):

    # make config generator
    generator = make_generator(
      config_file=args.config, # base config file from step 1
      script_file=args.script  # explained later in step 4
    )

    # generate jsons and script
    generator.generate()
...
```

## Step 3: Set Hyperparameter Values

Next, we use the `generator.add_param` function to set hyperparameter values, which takes the following arguments:
- `key`: (string) full name of config key to sweep
- `name`: (string) shorthand name for this key
- `values`: (list) values to sweep for this key
- `value_names` (list) (optional) shorthand names associated for each value in `values`
- `group`: (integer) hp group identifier. hps with same group are swept together. hps with different groups are swept as a cartesian product 

### Set fixed values
Going back to our example, we first set hyperparameters that are fixed single values.
We could have modified our base json file directly but we opted to set it in the generator function instead.

In this case, we would like to run the BC-RNN algorithm with an RNN horizon of 10. This requires setting `config.train.seq_length = 10` and `config.algo.rnn.enabled = True`.

```python
    # use RNN with horizon 10
    generator.add_param(
        key="algo.rnn.enabled",
        name="", 
        group=0, 
        values=[True],
    )
    generator.add_param(
        key="train.seq_length", 
        name="", 
        group=0, 
        values=[10], 
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="", 
        group=0, 
        values=[10], 
    )
```

<div class="admonition tip">
<p class="admonition-title">Empty hyperparameter names</p>

Leaving `name=""` ensures that the experiment name is not determined by these parameter values.
Only do this if you are sweeping over a single value!

</div>

<div class="admonition tip">
<p class="admonition-title">wandb logging</p>

If you would like to log and view results on wandb, enable wandb logging in the hyperparameter generator:
```python
generator.add_param(
    key="experiment.logging.log_wandb",
    name="", 
    group=-1, 
    values=[True],
)
```

</div>

### Define hyperparameter scan values
Now we define our scan - we could like to sweep the following:
- policy learning rate in [1e-3, 1e-4]
- whether to use a GMM policy or not
- whether to use an RNN dimension of 400 with an MLP of size (1024, 1024) or an RNN dimension of 1000 with an empty MLP

Notice that the learning rate goes in `group` 1, the GMM enabled parameter goes in `group` 2, and the RNN dimension and MLP layer dims both go in `group` 3. 

<div class="admonition tip">
<p class="admonition-title">Sweeping hyperparameters together</p>

We set the RNN dimension and MLP layer dims in the same group to ensure that the parameters change together (RNN dimension 400 always occurs with MLP layer dims (1024, 1024), and RNN dimension 1000 always occurs with an empty MLP).

</div>

```python
    # LR - 1e-3, 1e-4
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial", 
        name="plr", 
        group=1, 
        values=[1e-3, 1e-4], 
    )

    # GMM y / n
    generator.add_param(
        key="algo.gmm.enabled", 
        name="gmm", 
        group=2, 
        values=[True, False], 
        value_names=["t", "f"],
    )

    # RNN dim 400 + MLP dims (1024, 1024) vs. RNN dim 1000 + empty MLP dims ()
    generator.add_param(
        key="algo.rnn.hidden_dim", 
        name="rnnd", 
        group=3, 
        values=[
            400, 
            1000,
        ], 
    )
    generator.add_param(
        key="algo.actor_layer_dims", 
        name="mlp", 
        group=3, 
        values=[
            [1024, 1024], 
            [],
        ], 
        value_names=["1024", "0"],
    )
```

## Step 4: Run Hyperparameter Helper Script
Finally, we run the hyperparameter helper script (which contains the function we defined above).

```sh
$ python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
```

All generated configs have been added to `/tmp/gen_configs`, along with a helpful bash script that can be used to launch your training runs.

```sh
$ cat /tmp/gen_configs/out.sh

#!/bin/bash

python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.001_gmm_t_rnnd_400_mlp_1024.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.001_gmm_t_rnnd_1000_mlp_0.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.001_gmm_f_rnnd_400_mlp_1024.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.001_gmm_f_rnnd_1000_mlp_0.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.0001_gmm_t_rnnd_400_mlp_1024.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.0001_gmm_t_rnnd_1000_mlp_0.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.0001_gmm_f_rnnd_400_mlp_1024.json
python train.py --config /tmp/gen_configs/bc_rnn_hyper_plr_0.0001_gmm_f_rnnd_1000_mlp_0.json
```

<div class="admonition tip">
<p class="admonition-title">Meta information</p>

For each generated config file you will find a `meta` section that contains hyperparameter names, values, and other metadata information. This `meta` section is generated automatically, and you should NOT need to edit or modify it.

</div>
