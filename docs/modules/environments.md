# Environments

**robomimic** interfaces trained policy models with standard simulated environments such as [robosuite](https://robosuite.ai/) and [OpenAI-Gym](https://gym.openai.com/) through a standardized wrapper class. The base wrapper is located at `envs.env_base`. The standard entry point for creating an environment is through the `create_env_from_metadata` function located at `robomimic.utils.env_utils`:

```python
def create_env_from_metadata(
    env_meta,
    render=False, 
    render_offscreen=False, 
    use_image_obs=False, 
):
    """
    Create environment.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.
    """
    if env_name is None:
        env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]

    env = create_env(
        env_type=env_type,
        env_name=env_name,  
        render=render, 
        render_offscreen=render_offscreen, 
        use_image_obs=use_image_obs, 
        **env_kwargs,
    )
    return env
```

`env_meta` is a dictionary containing three keys: `'env_name'`, `'type'`, and `'env_kwargs'`. 
- `env_name` specifies the unique identifier of the environment. For example, in `EnvGym`, `env_name` is the full environment name such as `Hopper-v2`. 
- `type` is an enum defined in `robomimicenvs.env_base.EnvType` that specifies the type of environment. `type` is mainly used by the `robomimic.utils.env_utils.get_env_class()` function to look up the correct environment wrapper class.
- `env_kwargs` specifies the keyword args that are required to initialize an environment. The `env_kwargs` will be passed to the constructor of the wrapped environment as keyword arguments through the `robomimic.utils.env_utils.create_env()` helper function.

Although it is possible to manually specify the `env_meta` dictionary, the **robomimic** training pipeline reads the `env_meta` from the hdf5 dataset as an attribute. Please refer to the [Dataset section](../datasets/overview.html#dataset-structure) for more details on where the metadata is stored, and the `robomimic.utils.file_utils.get_env_metadata_from_dataset` function to see how it is loaded from the dataset at run-time.


## Initialize an Environment from a Dataset
The demonstration dataset file should contain all necessary information to construct an environment. Here is standalone example for initializing a `EnvRobosuite` environment instance by reading environment metadata from the a dataset.

```python
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

env_meta = FileUtils.get_env_metadata_from_dataset("path/to/the/dataset.hdf5")

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    env_name=env_name, 
    render=False, 
    render_offscreen=False,
    use_image_obs=False, 
)
```

The repo offers a simple utility tool `robomimic/scripts/get_dataset_info.py` to view the environment metadata included in a dataset. For example:
```bash
$ python robomimic/scripts/get_dataset_info.py --dataset path/to/the/dataset.hdf5

...

==== Env Meta ====
{
    "env_name": "Lift",
    "type": 1,
    "env_kwargs": {
        "has_renderer": false,
        "has_offscreen_renderer": true,
        "ignore_done": true,
        "use_object_obs": true,
        "use_camera_obs": true,
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [
                0.05,
                0.05,
                0.05,
                0.5,
                0.5,
                0.5
            ],
            "output_min": [
                -0.05,
                -0.05,
                -0.05,
                -0.5,
                -0.5,
                -0.5
            ],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [
                0,
                300
            ],
            "damping_limits": [
                0,
                10
            ],
            "position_limits": null,
            "orientation_limits": null,
            "uncouple_pos_ori": true,
            "control_delta": true,
            "interpolation": null,
            "ramp_ratio": 0.2
        },
        "robots": [
            "Panda"
        ],
        "camera_depths": false,
        "camera_heights": 84,
        "camera_widths": 84,
        "reward_shaping": false,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand"
        ],
        "render_gpu_device_id": 0
    }
}

...

```

When training a policy using `robomimic/scripts/train.py`, this environment metadata is extracted to instantiate an environment for policy evaluation (if rollout is enabled). Additionally, you can specify a dictionary to update environment metadata in the training config under `config.experiment.env_meta_update_dict`. For example, if you wish to evaluate your model using absolute actions, you can update your training config as follows to override a specific controller setting:
```json
{
  ...
  "experiment": {
    ...
    "env_meta_update_dict": {
      "env_kwargs": {
          "controller_configs": {
              "control_delta": false
          },
      }
    },
    ...
  },
  ...
}
```

## Implement an Environment Wrapper

While we provide wrappers for [robosuite](https://robosuite.ai/) environments and  several standard [OpenAI-Gym](https://gym.openai.com/) environments, it is possible to implement your own wrapper for a new type of environment (for example, perhaps [PyBullet](https://pybullet.org/wordpress/)). This is useful if you have your own hdf5 dataset collected in this type of environment, and would like to conduct evaluation rollouts during the training process (note that no environment is needed if `config.experiment.rollout.enabled` is set to `False`). To do this, in addition to implementing a wrapper class by inheriting the `EnvBase` class, you will also need to (1) add a new environment type in the `robomimic.envs.env_base.EnvType` enum class and (2) modify `get_env_type()` and `get_env_class()` in `robomimic.utils.env_utils` to allow the environment class to be found automatically.

Below we outline important methods that each `EnvBase` subclass needs to implement or override. The implementation mostly follows the OpenAI-Gym convention.

- `__init__(self, ...)`
  - Create the wrapped environment instance and assign it to `self.env`. For example, in `EnvGym` it simply calls `self.env = gym.make(env_name, **kwargs)`. Refer to `EnvRobosuite` as an example of handling the constructor arguments.
- `step(self, action)`
  - Take a step in the environment with an input action, return `(observation, reward, done, info)`.
- `reset(self)`
  - Reset the environment, return `observation`
- `render(self, mode="human", height=None, width=None, camera_name=None, **kwargs)`
  - Render the environment if `mode=='human'`. Return an RGB array if `mode=='rgb_array'`
- `get_observation(self, obs=None)`
  - Return the current environment observation as a dictionary, unless `obs` is not None. This function should process the raw environment observation to align with the input expected by the policy model. For example, it should cast an image observation to float with value range `0-1` and shape format `[C, H, W]`. 
- `is_success(self)`
  - Check if the task condition(s) is reached. Should return a dictionary { str: bool } with at least a "task" key for the overall task success, and additional optional keys corresponding to other task criteria.
- `serialize(self)`
  - Aggregate and return all information needed to re-instantiate this environment in a dictionary. This is the same as @env_meta - environment metadata stored in hdf5 datasets and used in `robomimic/utils/env_utils.py`.
- `create_for_data_processing(cls, ...)`
  - (Optional) A class method that initialize an environment for data-postprocessing purposes, which includes extracting observations, labeling dense / sparse rewards, and annotating dones in transitions. This function should at least designate the list of observation modalities that are image / low-dimensional observations by calling `robomimic.utils.obs_utils.initialize_obs_utils_with_obs_specs()`. 
- `get_goal(self)`
  - (Optional) Get goal for a goal-conditional task
- `set_goal(self, goal)`
  - (optional) Set goal with external specification
- `get_state(self)`
  - (Optional) This function should return the underlying state of a simulated environment. Should be compatible with `reset_to`.
- `reset_to(self, state)`
  - (Optional) Reset to a specific simulator state. Useful for reproducing results.