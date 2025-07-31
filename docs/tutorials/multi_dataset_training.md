# Multi-Dataset Training

This tutorial shows how to train a model on multiple datasets simultaneously.

<div class="admonition note">
<p class="admonition-title">Note: Understand how to launch training runs first!</p>

Before trying to train on multiple datasets, it might be useful to read the following tutorials:
- [how to launch training runs](./configs.html)
- [how to view training results](./viewing_results.html)

</div>

#### 1. Overview

Robomimic supports training on multiple datasets simultaneously. This is useful when you want to:
- Train a single model on multiple tasks
- Combine datasets with different qualities (e.g., expert and suboptimal demonstrations)
- Balance data from different sources

Each dataset can have its own weight for sampling, and you can control whether these weights are normalized by dataset size or not.

#### 2. Configuring Multi-Dataset Training

To train on multiple datasets, you need to specify a list of dataset configurations in your config file. Each dataset configuration is a dictionary with the following keys:

```python
config.train.data = [
    {
        "path": "/path/to/dataset1.hdf5",      # (required) path to the hdf5 file
        "demo_limit": 100,                     # (optional) limit number of demos to use
        "weight": 1.0,                         # (optional) weight for sampling, defaults to 1.0
        "eval": True,                          # (optional) whether to evaluate on this dataset's env
        "lang": "make coffee",                 # (optional) language instruction for the dataset
        "key": "coffee"                        # (optional) key for naming eval videos
    },
    {
        "path": "/path/to/dataset2.hdf5",
        "weight": 2.0,                         # this dataset will be sampled twice as often
    }
]
```

Additionally, you can control how the weights are used with the `normalize_weights_by_ds_size` setting:

```python
config.train.normalize_weights_by_ds_size = False  # default
```

#### 3. Understanding Weight Normalization

The `normalize_weights_by_ds_size` setting controls how dataset weights affect sampling:

- When `False` (default):
  - Raw weights are used directly
  - Larger datasets will naturally be sampled more often when assigned the same weight
  - Example: If dataset A has 1000 samples and dataset B has 100 samples, with equal weights (1.0), you'll see roughly 10 samples from A for every 1 sample from B

- When `True`:
  - Weights are normalized by dataset size
  - Equal weights result in balanced sampling regardless of dataset size
  - Example: If dataset A has 1000 samples and dataset B has 100 samples, with equal weights (1.0), you'll see roughly equal numbers of samples from both datasets

#### 4. Example Configuration

Here's a complete example showing how to train a BC model on two datasets with different weights:

```python
import robomimic
from robomimic.config import config_factory

# create BC config
config = config_factory(algo_name="bc")

# configure datasets
config.train.data = [
    {
        "path": "expert_demos.hdf5",
        "weight": 2.0,                    # sample expert demos more frequently
    },
    {
        "path": "suboptimal_demos.hdf5",
        "weight": 1.0,
    }
]

# normalize weights by dataset size for balanced sampling
config.train.normalize_weights_by_ds_size = True

# other training settings...
config.train.batch_size = 100
config.train.num_epochs = 1000
```

#### 5. Best Practices

1. **Weight Selection**:
   - Use higher weights for higher-quality data
   - Consider using `normalize_weights_by_ds_size=True` when datasets have very different sizes
   - Start with equal weights and adjust based on performance

2. **Dataset Compatibility**:
   - Ensure all datasets have compatible observation and action spaces
   - Use consistent preprocessing across datasets

3. **Evaluation**:
   - Use the `eval` flag to control which environments to evaluate on
   - Set descriptive `key` values for clear video naming