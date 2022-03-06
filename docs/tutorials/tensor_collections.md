# Operations over Tensor Collections

This section highlights some important utility functions and classes used in the codebase for working with
collections of tensors.

## TensorUtils

Most models in **robomimic** operate on nested tensor dictionaries, both for input and training labels. We provide a suite of utilities to work with these dictionaries in `robomimic.utils.tensor_utils`. For example, given a numpy dictionary of observations:
```python
import numpy as np

x = {
    'image': np.random.randn(3, 224, 224),
    'proprio': {
        'eef_pos': np.random.randn(3),
        'eef_rot': np.random.randn(3)
    }
}
```

For example, we can use `robomimic.utils.tensor_utils` to convert them to pytorch tensors, add a batch dimension, and send them to GPU:

```python
import torch
import robomimic.utils.tensor_utils as TensorUtils

# Converts all numpy arrays in nested dictionary or list or tuple to torch tensors
x = TensorUtils.to_tensor(x)  

# add a batch dimension to all tensors in the dict
x = TensorUtils.to_batch(x)

# send all nested tensors to GPU (if available)
x = TensorUtils.to_device(x, torch.device("cuda:0"))
```

The library also supports nontrivial shape operations on the nested dict. For example:

```python
# create a new dimension at dim=1 and expand the dimension size to 10
x = TensorUtils.unsqueeze_expand_at(x, size=10, dim=1)  
# x["rgb"].shape == torch.Size([1, 10, 3, 224, 224])

# repeat the 0-th dimension 10 times
x = TensorUtils.repeat_by_expand_at(x, repeats=10, dim=0)  
# x["rgb"].shape == torch.Size([10, 10, 3, 224, 224])

# gather the sequence dimension (dim=1) by some index
x = TensorUtils.gather_sequence(x_seq, indices=torch.arange(10)) 
# x["rgb"].shape == torch.Size([10, 3, 224, 224])
```

In addition, `map_tensor` allows applying an arbitrary function to all tensors in a nested dictionary or list of tensors and returns the same nested structure.
```python
x = TensorUtils.map_tensor(x, your_func)
```

The complete documentation of `robomimic.utils.tensor_utils.py` is available [here](../api/robomimic.utils.html#module-robomimic.utils.tensor_utils).


## ObsUtils

`robomimic.utils.obs_utils` implements a suite of utility functions to preprocess different observation modalities such as images and functions to determine types of observations in order to create suitable encoder network architectures. Below we list the important functions.

- **initialize_obs_utils_with_obs_specs(obs_modality_specs)**
    
    This function initialize a global registry of mapping between observation key names and observation modalities e.g. which ones are low-dimensional, and which ones are rgb images). For example, given an `obs_modality_specs` of the following format:
    ```python
    {
        "obs": {
            "low_dim": ["robot0_eef_pos", "robot0_eef_quat"],
            "rgb": ["agentview_image", "robot0_eye_in_hand"],
        }
        "goal": {
            "low_dim": ["robot0_eef_pos"],
            "rgb": ["agentview_image"]
        }
    }

    ```
    The function will create a mapping between observation names such as `'agentview_image'` and observation modalities such as `'rgb'`. The registry is stored in `OBS_MODALITIES_TO_KEYS` and can be accessed globally. Utility functions such as `key_is_obs_modality()` rely on this global registry to determine observation modalities.

- **process_obs(obs_dict)**
    
    Preprocess a dictionary of observations to be fed to a neural network. For example, image observations will be casted to `float` format, rescaled to `[0-1]`, and axis-transposed to `[C, H, W]` format.

- **unprocess_obs(obs_dict)**

    Revert the preprocessing transformation applied to observations by `process_obs`. Useful for converting images back to `uint8` for efficient storage.

- **normalize_obs(obs_dict, obs_normalization_stats)**

    Normalize observations by computing the mean observation and std of each observation (in each dimension and observation key), and normalizing unit mean and variance in each dimension.