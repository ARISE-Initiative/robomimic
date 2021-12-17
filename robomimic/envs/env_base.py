"""
This file contains the base class for environment wrappers that are used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import abc


class EnvType:
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """
    ROBOSUITE_TYPE = 1
    GYM_TYPE = 2
    IG_MOMART_TYPE = 3


class EnvBase(abc.ABC):
    """A base class method for environments used by this repo."""
    @abc.abstractmethod
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        return

    @abc.abstractmethod
    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        return

    @abc.abstractmethod
    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        return

    @abc.abstractmethod
    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        return

    @abc.abstractmethod
    def render(self, mode="human", height=None, width=None, camera_name=None):
        """Render"""
        return

    @abc.abstractmethod
    def get_observation(self):
        """Get environment observation"""
        return

    @abc.abstractmethod
    def get_state(self):
        """Get environment simulator state, compatible with @reset_to"""
        return

    @abc.abstractmethod
    def get_reward(self):
        """
        Get current reward.
        """
        return

    @abc.abstractmethod
    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return

    @abc.abstractmethod
    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return

    @abc.abstractmethod
    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return

    @abc.abstractmethod
    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return

    @property
    @abc.abstractmethod
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return

    @property
    @abc.abstractmethod
    def name(self):
        """
        Returns name of environment name (str).
        """
        return

    @property
    @abc.abstractmethod
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return

    @abc.abstractmethod
    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return

    @classmethod
    @abc.abstractmethod
    def create_for_data_processing(cls, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            camera_names ([str]): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards

        Returns:
            env (EnvBase instance)
        """
        return

    @property
    @abc.abstractmethod
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return
    
