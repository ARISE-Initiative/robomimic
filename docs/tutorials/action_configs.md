# Action Configuration

This tutorial shows how to configure action spaces and normalization in robomimic, particularly useful for tasks with structured action spaces like robot manipulation.

<div class="admonition note">
<p class="admonition-title">Note: Understand how to launch training runs first!</p>

Before diving into action configuration, it might be useful to read the following tutorials:
- [how to launch training runs](./configs.html)
- [how to view training results](./viewing_results.html)

</div>

#### 1. Overview

Robomimic supports structured action spaces where different action components can be treated differently. This is particularly useful for:
- Robot manipulation tasks with different action components (e.g., end-effector position and rotation)
- Actions that require different normalization schemes
- Combining multiple action outputs with different physical meanings

#### 2. Action Configuration Structure

The action configuration consists of two main components:

1. `action_keys`: List of action components to use
2. `action_config`: Dictionary specifying how each action component should be processed

Here's the basic structure:

```python
config.train.action_keys = ["action/eef_pos", "action/eef_rot"]  # order matters!

config.train.action_config = {
    "action/eef_pos": {
        "normalization": "min_max",    # normalize to [-1, 1]
    },
    "action/eef_rot": {
        "normalization": None,                  # no normalization
        "format": "rot_6d",                     # rotation format in dataset / prediction
        "convert_at_runtime": "rot_axis_angle"  # convert rotation format at runtime
    }
}
```

#### 3. Supported Normalization Methods

Robomimic supports several normalization methods for action components:

1. `None`: No normalization
   - Uses unit scale and zero offset
   - Useful when actions are already in desired range

2. `"min_max"`: Min-max normalization
   - Scales actions to range [-1, 1]
   - Useful for bounded action components like positions
   - Handles numerical stability with small ranges
   ```python
   "normalization": "min_max"
   ```

3. `"gaussian"`: Gaussian normalization
   - Normalizes to zero mean and unit variance
   - Useful for unbounded action components
   ```python
   "normalization": "gaussian"
   ```

#### 4. Supported Rotation Conversions

Currently, robomimic supports converting 6D rotations (as proposed in [this paper](https://arxiv.org/abs/1812.07035)) to either axis-angle or Euler formats during rollouts. Below are example action configurations that convert the action key `action/eef_rot` from 6D to each format:

1. 6D to axis-angle
```python
config.train.action_config = {
    "action/eef_rot": {
        "normalization": None,                  # no normalization
        "format": "rot_6d",                     # 6D rotation format in dataset / prediction
        "convert_at_runtime": "rot_axis_angle"  # convert rotation format to axis-angle at runtime
    }
}
```

2. 6D to axis-angle
```python
config.train.action_config = {
    "action/eef_rot": {
        "normalization": None,                  # no normalization
        "format": "rot_6d",                     # 6D rotation format in dataset / prediction
        "convert_at_runtime": "rot_euler"       # convert rotation format to Euler at runtime
    }
}
```

If `"convert_at_runtime"` is not provided, 6D actions are converted to axis-angle format by default.

#### 5. Example Configurations

Here are some common use cases:

##### 5.1 Robot End-Effector Control

```python
# Configure end-effector position and rotation actions
config.train.action_keys = ["action/eef_pos", "action/eef_rot"]
config.train.action_config = {
    "action/eef_pos": {
        "normalization": "min_max",     # normalize position to [-1, 1]
    },
    "action/eef_rot": {
        "normalization": "gaussian",    # normalize rotation with zero mean, unit variance
    }
}
```

##### 5.2 Mixed Action Spaces

```python
# Configure position, rotation, and gripper actions
config.train.action_keys = ["action/eef_pos", "action/eef_rot", "action/gripper"]
config.train.action_config = {
    "action/eef_pos": {
        "normalization": "min_max",
    },
    "action/eef_rot": {
        "normalization": "gaussian",
    },
    "action/gripper": {
        "normalization": None,          # gripper already in [-1, 1]
    }
}
```

#### 6. Best Practices

1. **Action Component Order**:
   - Order in `action_keys` determines concatenation order
   - Keep order consistent across training and deployment
   - Document the expected order in your configs

2. **Normalization Selection**:
   - Use `min_max` for bounded values (e.g., positions, normalized vectors)
   - Use `gaussian` for unbounded values or when distribution matters
   - Use `None` when values are already properly scaled

3. **Numerical Stability**:
   - In `min_max` implementation, ranges smaller than 1e-4 are not scaled
   - In `gaussian` implementation, distributions with stddev smaller than 1e-6 are not scaled

4. **Diffusion Policy Compatibility**:
   - When using Diffusion Policy, ensure actions are normalized to [-1, 1]
   - Use `min_max` normalization or pre-normalize your data
   - The policy will check if actions are in the correct range

#### 7. Implementation Details

The normalization process:
1. Computes statistics (min, max, mean, std) across the entire dataset
2. Applies the specified normalization method to each action component
3. Concatenates the normalized components in the order specified by `action_keys`

For `min_max` normalization:
```python
# Normalizes to range [-0.999999, 0.999999] for numerical stability
scale = (input_max - input_min) / (output_max - output_min)
offset = input_min - scale * output_min
normalized_action = (raw_action - offset) / scale
```

For `gaussian` normalization:
```python
# Normalizes to zero mean, unit variance
normalized_action = (raw_action - mean) / (std + epsilon)
``` 