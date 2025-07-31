# Training Diffusion Policy

This tutorial shows how to train a Diffusion Policy model (introduced in [this paper](https://arxiv.org/pdf/2303.04137v4)).

<div class="admonition note">
<p class="admonition-title">Note: Understand how to launch training runs and view results first!</p>

Before trying to train a Diffusion Policy, it might be useful to read the following tutorials:
- [how to launch training runs](./configs.html)
- [how to view training results](./viewing_results.html)
- [how to launch multiple training runs efficiently](./hyperparam_scan.html) 

</div>

A template with tuned parameters for Diffusion Policy is defined in `robomimic/exps/templates/diffusion_policy.json`.

#### 1. Using default configurations

The easiest way to train a Diffusion Policy model is to pass the default template json to the main training script `train.py` via the `--config` argument. The dataset can be specified by setting the `data` attribute of the `train` section of the config json, or specified via the `--dataset` argument.  You may find that your data has different rollout horizon lengths, observation modalities, or other incompatibilities with the default template.  In this scenario, we suggest defining custom parameters as described in (2).

```sh
$ python train.py --config ../exps/templates/diffusion_policy.json --dataset /path/to/dataset.hdf5
```

#### 2. Defining custom parameters

If you want to modify the default Diffusion Policy parameters, do not directly modify the default config (`config/diffusion_policy_config.py`) or template (`exps/templates/diffusion_policy.json`). Instead, you can create a copy of `robomimic/exps/templates/diffusion_policy.json` and store it in a new directory on your computer. Set this as the base file for `scripts/hyperparam_helper.py` and define custom settings as described [here](./hyperparam_scan.html).  This is particularly useful when running a sweep over hyperparameters; **it is the prefered way to launch multiple training runs**. 

Optionally, you can modify the default template in python code or directly set the appropriate keys in your copy of the config file.  This code snippet below highlights useful parameters to tune for Diffusion Policy. To see all Diffusion Policy settings, refer to `config/diffusion_policy_config.py`.

```python
# make sure diffusion policy is enabled
config.algo_name = "diffusion_policy"

# useful config attributes to modify for diffusion policy
## horizon parameters
config.algo.horizon.observation_horizon = 2                 # number of observation frames to condition the action denoising
config.algo.horizon.prediction_horizon = 16                 # number of actions to predict
config.algo.horizon.action_horizon = 8                      # among predicted, number of actions to use during rollout
## noise scheduler: ddpm or ddim
config.algo.ddpm.enabled = True
config.algo.ddim.enabled = False
```
