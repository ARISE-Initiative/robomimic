# TensorUtils

Most Robomimic's models operate on nested tensor dictionaries, both for input and training labels. We provide a suite of utility to work with these dictionaries in `robomimic.utils.tensor_utils`. For example, given a numpy dictionary of observations:
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

The library also supports nontrivial shape operations on the nest dict. For example:

```python
# create a new dimension at dim=1 and expand the dimension size to 10
x = TensorUtils.unsqueeze_expand_at(x, size=10, dim=1)  
# x["image"].shape == torch.Size([1, 10, 3, 224, 224])

# repeat the 0-th dimension 10 times
x = TensorUtils.repeat_by_expand_at(x, repeats=10, dim=0)  
# x["image"].shape == torch.Size([10, 10, 3, 224, 224])

# gather the sequence dimension (dim=1) by some index
x = TensorUtils.gather_sequence(x_seq, indices=torch.arange(10)) 
# x["image"].shape == torch.Size([10, 3, 224, 224])
```

In addition, `map_tensor` allows applying arbitrary function to all tensors in a nested dictionary or list of tensors and return the same nested structure.
```python
x = TensorUtils.map_tensor(x, your_func)
```

The complete documentation of `robomimic.utils.tensor_utils.py` is available [here](../api/robomimic.utils.html#module-robomimic.utils.tensor_utils).