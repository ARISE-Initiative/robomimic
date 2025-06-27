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

Optionally, you can modify the default template in python code or directly set the appropriate keys in your copy of the config file.  This code snippet below highlights useful parameters to tune for transformers.  To see all transformer policy settings, refer to `config/bc_config.py`.

```python
# make sure transformer is enabled
config.algo.transformer.enabled = True

# useful config attributes to modify for transformers
config.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
config.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
config.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
config.algo.transformer.context_length = 10                   # length of (s, a) sub-seqeunces to feed to transformer
config.algo.transformer.pred_future_acs = False               # shift action prediction forward to predict future actions instead of past actions
config.train.frame_stack = 10                                 # length of sub-sequence to observe: (s_{t-1}, a_{t-1}), (s_{t-2}, a_{t-2}), ..., (s_{t-9}, a_{t-9})
config.train.seq_length = 1                                   # length of sub-seqeunce to predict: (s_{t}, a_{t})
```

