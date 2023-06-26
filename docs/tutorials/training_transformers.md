# Training Transformers

This tutorial shows how to train a transformer policy network.

<div class="admonition note">
<p class="admonition-title">Note: Understand how to launch training runs and view results first!</p>

Before trying to train a transformer, it might be useful to read the following tutorials:
- [how to launch training runs](./configs.html)
- [how to view training results](./viewing_results.html)
- [how to launch multiple training runs efficiently](./hyperparam_scan.html) 

</div>

A template with tuned parameters for transformer based policy networks is defined in `robomimic/config/default_templates/bc_transformer.json`.

#### 1. Using default configurations

The easiest way to train a transformer policy network is to pass the default template json to the main training script `train.py` via the `--config` argument. The dataset can be specified by setting the `data` attribute of the `train` section of the config json, or specified via the `--dataset` argument.  You may find that your data has different rollout horizon lengths, observation modalities, or other incompatibilities with the default template.  In this scenario, we suggest defining custom parameters as described in (2).

```sh
$ python train.py --config ../config/default_templates/bc_transformer.json --dataset /path/to/dataset.hdf5
```

#### 2. Defining custom parameters

If you want to modify the default transformer parameters, do not directly modify the default config (`config/bc_config.py`) or template (`config/default_templates/bc_transformer.json`).  Instead, you can create a copy of `robomimic/config/default_templates/bc_transformer.json` and store it in a new directory on your computer.  Set this as the base file for `scripts/hyperparam_helper.py` and define custom settings as described [here](./hyperparam_scan.html).  This is particularly useful when running a sweep over hyperparameters; **it is the prefered way to launch multiple training runs**. 

Optionally, you can modify the default template in python code and then call the train function, as shown in the code snippet below.  This code highlights useful parameters to tune for transformers and modifications that should be made when training a policy on image observations (default is low-dimensional observations).  To see all transformer policy settings, refer to `config/bc_config.py`.

```python
import json
import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# load json with default transformer parameters
ext_cfg = json.load(open('robomimic/config/default_templates/bc_transformer.json', 'r'))
#generate base behavioral cloning config
config = config_factory(ext_cfg["algo_name"])

with config.values_unlocked():
    # update config with default transformer parameters
    config.update(ext_cfg)

    # set config attributes for training
    config.experiment.name = "bc_transformer_example"
    config.train.data = "/path/to/dataset.hdf5"
    config.train.output_dir = "/path/to/output_dir/"

    # set horizon based on length of demonstrations (can be obtained with scripts/get_dataset_info.py)
    config.experiment.rollout.horizon = 700

    # set config attributes for image modalities
    config.experiment.save.every_n_epochs = 20
    config.experiment.rollout.rate = 20
    config.experiment.epoch_every_n_steps = 500
    config.experiment.validation_epoch_every_n_steps = 50

    config.train.batch_size = 16
    config.train.num_epochs = 600
    config.train.num_data_workers = 2

    # Keep low-dimensional states for robot arm position/orientation
    config.observation.modalities.obs.low_dim = [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos"
                    ]
    # Add images to observation, names may depend on your dataset naming convention
    config.observation.modalities.obs.rgb = [
                        "agentview_image",
                        "robot0_eye_in_hand_image"
                    ]

    #set config attributes for transformer
    config.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
    config.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
    config.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
    config.algo.transformer.context_length = 10                   # length of (s, a) seqeunces to feed to transformer
    config.train.frame_stack = 10                                 # shoule be same as context length
    config.train.seq_length = 1                                   # length of (s, a) seqeunces for transformer to predict

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)
```

