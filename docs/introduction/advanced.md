# Advanced Features

This section discusses some advanced features of **robomimic**.

## Using the Hyperparameter Helper to launch runs

While copying an algorithm's template json from `exps/templates` and modifying it manually is a perfectly valid way to run experiments, we also provide the `hyperparam_helper.py` script to easily generate config jsons to use with the `train.py` script. **This is the preferred way to launch training runs using the repository.** It also makes hyperparameter scans a breeze. We'll walk through an example below, by reproducing sections of the `hyperparam_helper.py` script.

The first step is to start with a base config json. A common choice is to copy one of the templates in `exps/templates` (such as `exps/templates/bc.json`) into a new folder (where additional config jsons will be generated). 

```sh
$ cp ../exps/templates/bc.json /tmp/gen_configs/base.json
```

Sections of the config that are not involved in the scan and that do not differ from the default values in the template can also be omitted, if desired. For instance, in the example below, we don't need the `config.algo.gaussian`,  `config.algo.vae`, and `config.observation` portions (since we don't sweep over them, and we didn't want to set them to anything other than the default values), so we deleted them. We also added a base experiment name (`"bc_rnn_hyper"`) and specified the dataset path (`"/tmp/test.hdf5"`).

```sh
$ cat /tmp/gen_configs/base.json
```

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
        "data": "/tmp/test.hdf5",
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

The next step is to define a function that returns a `ConfigGenerator`. In our example, we would like to  run the BC-RNN algorithm with an RNN horizon of 10. This requires setting `config.train.seq_length = 10` and `config.algo.rnn.enabled = True` -- we could have modified our base json file directly (as mentioned above) but we opted to set it in the generator function below. The first three calls to `add_param` do exactly this. Leaving `name=""` ensures that the experiment name is not determined by these parameter values.

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

Now we define our scan - we could like to sweep the policy learning rate in [1e-3, 1e-4], whether to use a GMM policy or not, and whether to use an RNN dimension of 400 with an MLP of size (1024, 1024) or an RNN dimension of 1000 with an empty MLP. Notice that the learning rate goes in `group` 1, the GMM enabled parameter goes in `group` 2, and the RNN dimension and MLP layer dims both go in `group` 3. 

The `group` argument specifies which arguments should be modified together. The hyperparameter script will generate a training run for each hyperparameter setting in the cartesian product between all groups. Thus, putting the RNN dimension and MLP layer dims in the same group ensures that the parameters change together (RNN dimension 400 always occurs with MLP layer dims (1024, 1024), and RNN dimension 1000 always occurs with an empty MLP). Finally, notice the use of the `value_names` argument  -- by default, the generated config will have an experiment name consisting of the base name under `config.experiment.name` already present in the base json, and then the `name` specified for each parameter, along with the string representation of the selected value in `values`, but `value_names` allows you to override this with a custom string for the corresponding value. 

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

    return generator
```

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
