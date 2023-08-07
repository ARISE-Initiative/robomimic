"""
This file contains the base class for environment wrappers that are used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import time
import json
import sys
import numpy as np
from copy import deepcopy

import cv2

import RobotTeleop
import RobotTeleop.utils as U
from RobotTeleop.utils import Rate, RateMeasure, Timers

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

class EnvRealPanda(EB.EnvBase):
    """Wrapper class for real panda environment"""
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
        use_depth_obs=False,
        postprocess_visual_obs=True,
        control_freq=20.,
        action_scale=None,
        camera_names_to_sizes=None,
        init_ros_node=True,
        publish_target_pose=False,
        fake_controller=False,
        use_moveit=True,
    ):
        """
        Args:
            env_name (str): name of environment.

            render (bool): ignored - on-screen rendering is not supported

            render_offscreen (bool): ignored - image observations are supplied by default

            use_image_obs (bool): ignored - image observations are used by default.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            control_freq (int): real-world control frequency to try and enforce through rate-limiting

            action_scale (list): list of 7 numbers for what the -1 and 1 action in each dimension corresponds to
                for the physical robot action space

            camera_names_to_sizes (dict):  dictionary that maps camera names to tuple of image height and width
                to return
        """
        self._env_name = env_name
        self.postprocess_visual_obs = postprocess_visual_obs
        self.control_freq = control_freq

        # to enforce control rate
        self.rate = Rate(control_freq)
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * control_freq))
        self.timers = Timers(history=100, disable_on_creation=False)

        assert (action_scale is not None), "must provide action scaling bounds"
        assert len(action_scale) == 7, "must provide scaling for all dimensions"
        self.action_scale = np.array(action_scale).reshape(-1)

        camera_names_to_sizes = deepcopy(camera_names_to_sizes)
        if camera_names_to_sizes is None:
            self.camera_names_to_sizes = {}
        else:
            self.camera_names_to_sizes = camera_names_to_sizes

        # save kwargs for serialization
        kwargs = dict(
            camera_names_to_sizes=camera_names_to_sizes,
            action_scale=action_scale,
            init_ros_node=init_ros_node,
            publish_target_pose=publish_target_pose,
            fake_controller=fake_controller,
            use_moveit=use_moveit,
            control_freq=control_freq
        )
        self._init_kwargs = deepcopy(kwargs)

        # connect to robot
        # if (sys.version_info > (3, 0)):
        #     from RobotTeleop.robots.panda_redis_interface import PandaRedisInterface
        #     self.robot_interface = PandaRedisInterface(
        #         init_ros_node=init_ros_node,
        #         publish_target_pose=publish_target_pose,
        #         fake_controller=fake_controller,
        #         use_moveit=use_moveit,
        #         camera_names_to_sizes=camera_names_to_sizes,
        #         debug_times=True,
        #     )
        # else:
        from RobotTeleop.robots.panda_ros_interface import PandaRosInterface
        self.robot_interface = PandaRosInterface(
            init_ros_node=init_ros_node,
            publish_target_pose=publish_target_pose,
            fake_controller=fake_controller,
            use_moveit=use_moveit,
            camera_names_to_sizes=camera_names_to_sizes,
            #use_redis=True,
        )

        # IMPORTANT: initialize JIT functions that may need to compile
        self._compile_jit_functions()

        # last grasp action - initialize to false, since gripper should start open
        self.did_grasp = False

    def _compile_jit_functions(self):
        """
        Helper function to incur the cost of compiling jit functions used by this class upfront.

        NOTE: this function looks strange because we apparently need to make it look like the env.step function
              for it to compile properly, otherwise we will have a heavy delay on the first env.step call...

        TODO: figure out why this needs to look like the step function code below...
        """

        # current robot state to use as reference
        ee_pos, ee_quat = self.robot_interface.ee_pose
        ee_mat = U.quat2mat(ee_quat)
        ee_quat_hat = U.mat2quat(ee_mat)

        # convert delta axis-angle to delta rotation matrix, and from there, to absolute target rotation
        drot = np.array([0., 0., 0.05])
        angle = np.linalg.norm(drot)
        if U.isclose(angle, 0.):
            drot_quat = np.array([0., 0., 0., 1.])
        else:
            axis = drot / angle
            drot_quat = U.axisangle2quat(axis, angle)

        # get target rotation
        drot_mat = U.quat2mat(drot_quat)
        target_rot_mat = (drot_mat.T).dot(ee_mat)
        target_rot_quat = U.mat2quat(target_rot_mat)

    def step(self, action, need_obs=True):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take, should be in [-1, 1]
            need_obs (bool): if False, don't return the observation, because this
                can involve copying image data around. This allows for more
                flexibility on when observations are retrieved.

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        assert len(action.shape) == 1 and action.shape[0] == 7, "action has incorrect dimensions"
        assert np.min(action) >= -1. and np.max(action) <= 1., "incorrect action bounds"

        # rate-limiting
        self.rate.sleep()
        self.rate_measure.measure()

        self.timers.tic("real_panda_step")

        # unscale action
        action = self.action_scale * action

        # extract action components
        dpos = action[:3]
        drot = action[3:6]
        gripper_command = action[6:7]

        # current robot state to use as reference
        ee_pos, ee_quat = self.robot_interface.ee_pose
        ee_mat = U.quat2mat(ee_quat)

        # absolute target position
        target_pos = ee_pos + dpos

        # convert delta axis-angle to delta rotation matrix, and from there, to absolute target rotation
        angle = np.linalg.norm(drot)
        if U.isclose(angle, 0.):
            drot_quat = np.array([0., 0., 0., 1.])
        else:
            axis = drot / angle
            drot_quat = U.axisangle2quat(axis, angle)
        drot_mat = U.quat2mat(drot_quat)
        target_rot_mat = (drot_mat.T).dot(ee_mat)
        target_rot_quat = U.mat2quat(target_rot_mat)

        # play end effector action
        self.robot_interface.move_to_ee_pose(pos=target_pos, ori=target_rot_quat)

        # convert continuous control signal in [-1, 1] to boolean
        should_close = (float(gripper_command) < 0.)

        # only send command if trying to change gripper state.
        # this is due to hardware limitations - robot grippers suck.
        if should_close != self.did_grasp:
            if should_close:
                self.robot_interface.gripper_close()
            else:
                self.robot_interface.gripper_open()

        # remember last grasp command
        self.did_grasp = should_close

        # get observation
        obs = None
        if need_obs:
            obs = self.get_observation()
        r = self.get_reward()
        done = self.is_done()

        self.timers.toc("real_panda_step")

        return obs, r, done, {}

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self.robot_interface.gripper_open()
        self.robot_interface.reset_teleop()
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * self.control_freq))

        return self.get_observation()

    def reset_to(self, state):
        """
        Reset to a specific state. On real robot, we visualize the start image,
        and a human should manually reset the scene.

        Reset to a specific simulator state.

        Args:
            state (dict): initial state that contains:
                - image (np.ndarray): initial workspace image

        Returns:
            None
        """
        assert "front_image" in state
        ref_img = cv2.cvtColor(state["front_image"], cv2.COLOR_RGB2BGR)

        print("\n" + "*" * 50)
        print("Reset environment to image shown in left pane")
        print("Press 'c' when ready to continue.")
        print("*" * 50 + "\n")
        while(True):
            # read current image
            cur_img = self.robot_interface.get_camera_frame(camera_name="front_image")
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)

            # concatenate frames to display
            img = np.concatenate([ref_img, cur_img], axis=1)

            # display frame
            cv2.imshow('initial state alignment window', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                break

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            raise Exception("on-screen rendering not supported currently")
        if mode == "rgb_array":
            # assert (height is None) and (width is None), "cannot resize images"
            assert camera_name in self.camera_names_to_sizes, "invalid camera name"
            return self.robot_interface.get_camera_frame(camera_name=camera_name)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current observation dictionary.
        """
        self.timers.tic("get_observation")
        observation = {}
        observation["ee_pose"] = np.concatenate(self.robot_interface.ee_pose)
        observation["joint_positions"] = self.robot_interface.joint_position
        observation["joint_velocities"] = self.robot_interface.joint_velocity
        observation["gripper_position"] = self.robot_interface.gripper_position
        observation["gripper_velocity"] = self.robot_interface.gripper_velocity
        for cam_name in self.camera_names_to_sizes:
            im = self.robot_interface.get_camera_frame(camera_name=cam_name)
            if self.postprocess_visual_obs:
                im = ObsUtils.process_image(im)
            observation[cam_name] = im
        self.timers.toc("get_observation")
        return observation

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return dict(states=np.zeros(1))
        # raise Exception("Real robot has no simulation state.")

    def get_reward(self):
        """
        Get current reward.
        """
        return 0.

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """

        # real robot environments don't usually have a success check - this must be done manually
        return { "task" : False }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        # return self._env_name

        # for real robot. ensure class name is stored in env meta (as env name) for use with any external
        # class registries
        return self.__class__.__name__

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.REAL_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvRealPanda instance)
        """

        # initialize obs utils so it knows which modalities are image modalities
        assert "camera_names_to_sizes" in kwargs
        image_modalities = list(kwargs["camera_names_to_sizes"].keys())
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "image": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=False, 
            use_image_obs=True, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        # we don't wrap any env
        return self

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)