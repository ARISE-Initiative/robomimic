"""
Wrapper environment class to enable using iGibson-based environments used in the MOMART paper
"""

from copy import deepcopy
import numpy as np
import json

import pybullet as p
import gibson2
from gibson2.envs.semantic_organize_and_fetch import SemanticOrganizeAndFetch
from gibson2.utils.custom_utils import ObjectConfig
import gibson2.external.pybullet_tools.utils as PBU
import tempfile
import os
import yaml
import cv2

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB


# TODO: Once iG 2.0 is more stable, automate available environments, similar to robosuite
ENV_MAPPING = {
    "SemanticOrganizeAndFetch": SemanticOrganizeAndFetch,
}


class EnvGibsonMOMART(EB.EnvBase):
    """
    Wrapper class for gibson environments (https://github.com/StanfordVL/iGibson) specifically compatible with
    MoMaRT datasets
    """
    def __init__(
            self,
            env_name,
            ig_config,
            postprocess_visual_obs=True,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
            use_depth_obs=False,
            image_height=None,
            image_width=None,
            physics_timestep=1./240.,
            action_timestep=1./20.,
            **kwargs,
    ):
        """
        Args:
            ig_config (dict): YAML configuration to use for iGibson, as a dict

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @use_image_obs is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            use_depth_obs (bool): if True, environment is expected to render depth image observations
                on every env.step call. Set this to False for efficiency reasons, if depth
                observations are not required.

            render_mode (str): How to run simulation rendering. Options are {"pbgui", "iggui", or "headless"}

            image_height (int): If specified, overrides internal iG image height when rendering

            image_width (int): If specified, overrides internal iG image width when rendering

            physics_timestep (float): Pybullet physics timestep to use

            action_timestep (float): Action timestep to use for robot in simulation

            kwargs (unrolled dict): Any args to substitute in the ig_configuration
        """
        raise Exception("EnvGibsonMOMART is no longer supported.")

        self._env_name = env_name
        self.ig_config = deepcopy(ig_config)
        self.postprocess_visual_obs = postprocess_visual_obs
        self._init_kwargs = kwargs

        # Determine rendering mode
        self.render_mode = "iggui" if render else "headless"
        self.render_onscreen = render

        # Make sure rgb is part of obs in ig config
        self.ig_config["output"] = list(set(self.ig_config["output"] + ["rgb"]))

        # Warn user that iG always uses a renderer
        if (not render) and (not render_offscreen):
            print("WARNING: iGibson always uses a renderer -- using headless by default.")

        # Update ig config
        for k, v in kwargs.items():
            assert k in self.ig_config, f"Got unknown ig configuration key {k}!"
            self.ig_config[k] = v

        # Set rendering values
        self.obs_img_height = image_height if image_height is not None else self.ig_config.get("obs_image_height", 120)
        self.obs_img_width = image_width if image_width is not None else self.ig_config.get("obs_image_width", 120)

        # Get class to create
        envClass = ENV_MAPPING.get(self._env_name, None)

        # Make sure we have a valid environment class
        assert envClass is not None, "No valid environment for the requested task was found!"

        # Set device idx for rendering
        # ensure that we select the correct GPU device for rendering by testing for EGL rendering
        # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
        import egl_probe
        device_idx = 0
        valid_gpu_devices = egl_probe.get_available_devices()
        if len(valid_gpu_devices) > 0:
            device_idx = valid_gpu_devices[0]

        # Create environment
        self.env = envClass(
            config_file=deepcopy(self.ig_config),
            mode=self.render_mode,
            physics_timestep=physics_timestep,
            action_timestep=action_timestep,
            device_idx=device_idx,
        )

        # If we have a viewer, make sure to remove all bodies belonging to the visual markers
        self.exclude_body_ids = []      # Bodies to exclude when saving state
        if self.env.simulator.viewer is not None:
            self.exclude_body_ids.append(self.env.simulator.viewer.constraint_marker.body_id)
            self.exclude_body_ids.append(self.env.simulator.viewer.constraint_marker2.body_id)

    def step(self, action):
        """
        Step in the environment with an action

        Args:
            action: action to take

        Returns:
            observation: new observation
            reward: step reward
            done: whether the task is done
            info: extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """Reset environment"""
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific state
        Args:
            state (dict): contains:
                - states (np.ndarray): initial state of the mujoco environment
                - goal (dict): goal components to reset
        Returns:
            new observation
        """
        if "states" in state:
            self.env.reset_to(state["states"], exclude=self.exclude_body_ids)

        if "goal" in state:
            self.set_goal(**state["goal"])

        # Return obs
        return self.get_observation()

    def render(self, mode="human", camera_name="rgb", height=None, width=None):
        """
        Render

        Args:
            mode (str): Mode(s) to render. Options are either 'human' (rendering onscreen) or 'rgb' (rendering to
                frames offscreen)
            camera_name (str): Name of the camera to use -- valid options are "rgb" or "rgb_wrist"
            height (int): If specified with width, resizes the rendered image to this height
            width (int): If specified with height, resizes the rendered image to this width

        Returns:
            array or None: If rendering to frame, returns the rendered frame. Otherwise, returns None
        """
        # Only robotview camera is currently supported
        assert camera_name in {"rgb", "rgb_wrist"}, \
            f"Only rgb, rgb_wrist cameras currently supported, got {camera_name}."

        if mode == "human":
            assert self.render_onscreen, "Rendering has not been enabled for onscreen!"
            self.env.simulator.sync()
        else:
            assert self.env.simulator.renderer is not None, "No renderer enabled for this env!"

            frame = self.env.sensors["vision"].get_obs(self.env)[camera_name]

            # Reshape all frames
            if height is not None and width is not None:
                frame = cv2.resize(frame, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                return frame

    def resize_obs_frame(self, frame):
        """
        Resizes frame to be internal height and width values
        """
        return cv2.resize(frame, dsize=(self.obs_img_width, self.obs_img_height), interpolation=cv2.INTER_CUBIC)

    def get_observation(self, di=None):
        """Get environment observation"""
        if di is None:
            di = self.env.get_state()
        ret = {}
        for k in di:
            # RGB Images
            if "rgb" in k:
                ret[k] = di[k]
                # ret[k] = np.transpose(di[k], (2, 0, 1))
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k]), obs_key=k)

            # Depth images
            elif "depth" in k:
                # ret[k] = np.transpose(di[k], (2, 0, 1))
                # Values can be corrupted (negative or > 1.0, so we clip values)
                ret[k] = np.clip(di[k], 0.0, 1.0)
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k])[..., None], obs_key=k)

            # Segmentation Images
            elif "seg" in k:
                ret[k] = di[k][..., None]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k]), obs_key=k)

            # Scans
            elif "scan" in k:
                ret[k] = np.transpose(np.array(di[k]), axes=(1, 0))

        # Compose proprio obs
        proprio_obs = di["proprio"]

        # Compute intermediate values
        lin_vel = np.linalg.norm(proprio_obs["base_lin_vel"][:2])
        ang_vel = proprio_obs["base_ang_vel"][2]

        ret["proprio"] = np.concatenate([
            proprio_obs["head_joint_pos"],
            proprio_obs["grasped"],
            proprio_obs["eef_pos"],
            proprio_obs["eef_quat"],
        ])

        # Proprio info that's only relevant for navigation
        ret["proprio_nav"] = np.concatenate([
            [lin_vel],
            [ang_vel],
        ])

        # Compose task obs
        ret["object"] = np.concatenate([
            np.array(di["task_obs"]["object-state"]),
        ])

        # Add ground truth navigational state
        ret["gt_nav"] = np.concatenate([
            proprio_obs["base_pos"][:2],
            [np.sin(proprio_obs["base_rpy"][2])],
            [np.cos(proprio_obs["base_rpy"][2])],
        ])

        return ret

    def sync_task(self):
        """
        Method to synchronize iG task, since we're not actually resetting the env but instead setting states directly.
        Should only be called after resetting the initial state of an episode
        """
        self.env.task.update_target_object_init_pos()
        self.env.task.update_location_info()

    def set_task_conditions(self, task_conditions):
        """
        Method to override task conditions (e.g.: target object), useful in cases such as playing back
            from demonstrations

        Args:
            task_conditions (dict): Keyword-mapped arguments to pass to task instance to set internally
        """
        self.env.set_task_conditions(task_conditions)

    def get_state(self):
        """Get iG flattened state"""
        return {"states": PBU.WorldSaver(exclude_body_ids=self.exclude_body_ids).serialize()}

    def get_reward(self):
        return self.env.task.get_reward(self.env)[0]
        # return float(self.is_success()["task"])

    def get_goal(self):
        """Get goal specification"""
        # No support yet in iG
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        # No support yet in iG
        raise NotImplementedError

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env.check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @classmethod
    def create_for_data_processing(
            cls,
            env_name,
            camera_names,
            camera_height,
            camera_width,
            reward_shaping,
            render=None, 
            render_offscreen=None, 
            use_image_obs=None, 
            use_depth_obs=None, 
            **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
            render (bool or None): optionally override rendering behavior
            render_offscreen (bool or None): optionally override rendering behavior
            use_image_obs (bool or None): optionally override rendering behavior
        """
        has_camera = (len(camera_names) > 0)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=(False if render is None else render), 
            render_offscreen=(has_camera if render_offscreen is None else render_offscreen), 
            use_image_obs=(has_camera if use_image_obs is None else use_image_obs), 
            postprocess_visual_obs=False,
            image_height=camera_height,
            image_width=camera_width,
            **kwargs,
        )

    @property
    def action_dimension(self):
        """Action dimension"""
        return self.env.robots[0].action_dim

    @property
    def name(self):
        """Environment name"""
        return self._env_name

    @property
    def type(self):
        """Environment type"""
        return EB.EnvType.IG_MOMART_TYPE

    def serialize(self):
        """Serialize to dictionary"""
        return dict(env_name=self.name, type=self.type,
                    ig_config=self.ig_config,
                    env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def deserialize(cls, info, postprocess_visual_obs=True):
        """Create environment with external info"""
        return cls(env_name=info["env_name"], ig_config=info["ig_config"], postprocess_visual_obs=postprocess_visual_obs, **info["env_kwargs"])

    @property
    def rollout_exceptions(self):
        """Return tuple of exceptions to except when doing rollouts"""
        return (RuntimeError)

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4) + \
               "\niGibson Config: \n" + json.dumps(self.ig_config, sort_keys=True, indent=4)
