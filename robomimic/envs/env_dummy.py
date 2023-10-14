import json
import numpy as np
from copy import deepcopy
import robomimic.envs.env_base as EB


class EnvDummy(EB.EnvBase):
    """Dummy env used for real-world cases when env doesn't exist"""
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
    self._env_name = env_name

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reset_to(self, state):
        raise NotImplementedError

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        raise NotImplementedError

    def get_observation(self, obs=None):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_goal(self):
        raise NotImplementedError

    def set_goal(self, **kwargs):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError

    @property
    def action_dimension(self):
        raise NotImplementedError

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.GYM_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type)

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        raise NotImplementedError

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return f'{self.name} Dummy Env'

