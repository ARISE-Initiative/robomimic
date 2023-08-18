"""
Real robot env wrapper for Yifeng's GPRS control stack.
"""
import os
import time
import json
import sys
import numpy as np
from copy import deepcopy
from easydict import EasyDict as edict

import cv2
from PIL import Image

import RobotTeleop
import RobotTeleop.utils as U
from RobotTeleop.utils import Rate, RateMeasure, Timers

try:
    # GPRS imports
    from gprs.franka_interface import FrankaInterface
    from gprs.camera_redis_interface import CameraRedisSubInterface
    from gprs.utils import YamlConfig
    from gprs import config_root

    from rpl_vision_utils.utils import img_utils as ImgUtils
except ImportError:
    print("WARNING: no GPRS...")

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning

try:
    import robosuite.utils.transform_utils as T
except ImportError:
    print("WARNING: could not import robosuite transform utils (needed for using absolute actions with GPRS")


def center_crop(im, t_h, t_w):
    assert(im.shape[-3] >= t_h and im.shape[-2] >= t_w)
    assert(im.shape[-1] in [1, 3])
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h:crop_h + t_h, crop_w:crop_w + t_w, :]


def get_depth_scale(camera_name):
    """
    Returns scaling factor that converts from uint16 depth to real-valued depth (in meters).
    """

    # TODO: fix duplication
    if camera_name == "front":
        return 0.0010000000474974513
    if camera_name == "wrist":
        return 0.0010000000474974513
    raise Exception("should not reach here")
    # from RobotTeleop.scripts.debug_april_tag import get_depth_scale_unified
    # return get_depth_scale_unified(camera_name=camera_name)


class EnvRealPandaGPRS(EB.EnvBase):
    """Wrapper class for real panda environment"""
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
        postprocess_visual_obs=True,
        control_freq=20.,
        camera_names_to_sizes=None,
        center_crop_images=True,
        general_cfg_file=None,
        controller_type=None,
        controller_cfg_file=None,
        controller_cfg_dict=None,
        use_depth_obs=False,
        absolute_actions=False, # use absolute pos and rot (axis-angle) in 7-dim action vector
        # additional GPRS-specific args
        state_freq=100.,
        control_timeout=1.0,
        has_gripper=True,
        use_visualizer=False,
        debug=False,
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

            camera_names_to_sizes (dict):  dictionary that maps camera names to tuple of image height and width
                to return
        """
        self._env_name = env_name
        self.postprocess_visual_obs = postprocess_visual_obs
        self.control_freq = control_freq
        self.absolute_actions = absolute_actions
        self.general_cfg_file = general_cfg_file
        self.controller_type = controller_type
        self.controller_cfg_file = controller_cfg_file
        self.controller_cfg_dict = deepcopy(controller_cfg_dict) if controller_cfg_dict is not None else None
        if self.controller_cfg_dict is not None:
            # control code expects easydict
            self.controller_cfg = edict(self.controller_cfg_dict)
        else:
            assert controller_cfg_file is not None
            self.controller_cfg = YamlConfig(os.path.join(config_root, controller_cfg_file)).as_easydict()
        self.use_depth_obs = use_depth_obs

        # to enforce control rate
        self.rate = Rate(control_freq)
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * control_freq))
        self.timers = Timers(history=100, disable_on_creation=False)

        camera_names_to_sizes = deepcopy(camera_names_to_sizes)
        if camera_names_to_sizes is None:
            self.camera_names_to_sizes = {}
        else:
            self.camera_names_to_sizes = camera_names_to_sizes
        self.center_crop_images = center_crop_images

        self._exclude_depth_from_obs = (not self.use_depth_obs)
        if self.use_depth_obs and self.postprocess_visual_obs:
            for cam_name in self.camera_names_to_sizes:
                depth_mod = "{}_depth".format(cam_name)
                if not ((depth_mod in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=depth_mod, obs_modality="depth")):
                    log_warning("depth observation {} will not be postprocessed since robomimic is not aware of it".format(depth_mod))
                    # # HACK: assume this means we don't actually need depth, but we might the camera interface to support it for TAMP / perception
                    # self.use_depth_obs = False
                    self._exclude_depth_from_obs = True 

        # save kwargs for serialization
        kwargs = dict(
            env_name=env_name,
            camera_names_to_sizes=camera_names_to_sizes,
            center_crop_images=center_crop_images,
            general_cfg_file=general_cfg_file,
            control_freq=control_freq,
            controller_type=controller_type,
            controller_cfg_file=controller_cfg_file,
            controller_cfg_dict=controller_cfg_dict,
            use_depth_obs=use_depth_obs,
            state_freq=state_freq,
            control_timeout=control_timeout,
            has_gripper=has_gripper,
            use_visualizer=use_visualizer,
            debug=debug,
        )
        self._init_kwargs = deepcopy(kwargs)

        # connect to robot
        self.robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, general_cfg_file),
            control_freq=control_freq,
            state_freq=state_freq,
            control_timeout=control_timeout,
            has_gripper=has_gripper,
            use_visualizer=use_visualizer,
            debug=debug,
        )

        # TODO: clean up camera ID definition later

        # start camera interfaces
        camera_ids = list(range(len(self.camera_names_to_sizes)))
        self.cr_interfaces = {}
        for c_id, c_name in enumerate(self.camera_names_to_sizes):
            cr_interface = CameraRedisSubInterface(camera_id=c_id, use_depth=self.use_depth_obs)
            cr_interface.start()
            self.cr_interfaces[c_name] = cr_interface

        # IMPORTANT: initialize JIT functions that may need to compile
        self._compile_jit_functions()

    def _compile_jit_functions(self):
        """
        Helper function to incur the cost of compiling jit functions used by this class upfront.

        NOTE: this function looks strange because we apparently need to make it look like the env.step function
              for it to compile properly, otherwise we will have a heavy delay on the first env.step call...

        TODO: figure out why this needs to look like the step function code below...
        """

        # current robot state to use as reference
        # ee_pos, ee_quat = self.robot_interface.ee_pose
        ee_mat = U.quat2mat(np.array([0., 0., 0., 1.]))
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

        if self.absolute_actions:
            test_mat = T.quat2mat(T.axisangle2quat(drot))

    def _get_unified_getter(self):
        """
        For HITL-TAMP teleoperation only - provides access to important information for perception.
        """
        from htamp.scripts.test_real_world import UnifiedGetter
        return UnifiedGetter(
            use_real_robot=True,
            robot_interface=self.robot_interface,
            camera_interface=self.cr_interfaces["front_image"],
        )

    def switch_controllers(self, controller_dict):
        """
        Switch the controller type and controller config being used. Useful
        for switching inbetween two different kinds of controllers during an
        episode - for example, OSC and Joint Impedance.

        Args:
            controller_dict (dict): dictionary that contains two keys
                type (str): type of controller
                cfg (easydict): controller config

        Returns:
            old_controller_dict (dict): the previous @controller_dict
        """
        old_controller_dict = dict(type=self.controller_type, cfg=deepcopy(self.controller_cfg))
        print("*" * 50)
        print("SWITCH TO CONTROLLER TYPE: {}".format(controller_dict["type"]))
        print("*" * 50)
        self.controller_type = controller_dict["type"]
        self.controller_cfg = controller_dict["cfg"]
        return old_controller_dict

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
        # print("step got action: {}".format(action))
        if self.controller_type == "OSC_POSE":
            assert len(action.shape) == 1 and action.shape[0] == 7, "action has incorrect dimensions"

            if self.absolute_actions:
                # convert action from absolute to relative for compatibility with rest of code
                action = np.array(action)

                # absolute pose target
                target_pos = action[:3]
                target_rot = T.quat2mat(T.axisangle2quat(action[3:6]))

                # current pose
                last_robot_state = self.robot_interface._state_buffer[-1]
                ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
                start_pos = ee_pose[:3, 3]
                start_rot = ee_pose[:3, :3]

                # TODO: remove hardcode
                max_dpos = np.array([0.08, 0.08, 0.08])
                max_drot = np.array([0.5, 0.5, 0.5])

                # copied from MG class (TODO: unify)
                delta_position = target_pos - start_pos
                delta_position = np.clip(delta_position / max_dpos, -1., 1.)

                delta_rot_mat = target_rot.dot(start_rot.T)
                delta_rot_quat = U.mat2quat(delta_rot_mat)
                delta_rot_aa = U.quat2axisangle(delta_rot_quat)
                delta_rotation = delta_rot_aa[0] * delta_rot_aa[1]
                delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

                # relative action
                action[:3] = delta_position
                action[3:6] = delta_rotation
                action[6:] = np.clip(action[6:], -1., 1.)

            assert np.min(action) >= -1. and np.max(action) <= 1., "incorrect action bounds"
        elif self.controller_type == "JOINT_IMPEDANCE":
            assert len(action.shape) == 1 and action.shape[0] == 8, "action has incorrect dimensions"
            assert not self.absolute_actions
            if not np.any(action[:7]):
                raise Exception("GOT ZERO ACTION WITH JOINT IMPEDANCE CONTROLLER - TERMINATING")
            
            # compare current joint position with issued action
            last_robot_state = self.robot_interface._state_buffer[-1]
            cur_q = np.array(last_robot_state.q)

            # print("joint action: {}".format(action[:7]))
            # print("current joints: {}".format(cur_q))
            # print("absolute error: {}".format(np.abs(action[:7] - cur_q)))
            # print("max absolute error: {}".format(np.max(np.abs(action[:7] - cur_q))))

            # if np.max(np.abs(action[:7] - cur_q)) > 0.2:
            #     raise Exception("max absolute error too high - stopping")

            # TODO: joint impedance controller takes in raw joint positions - we might need to change this later, if we want to learn from these actions
            # assert np.min(action) >= -1. and np.max(action) <= 1., "incorrect action bounds"

        # meaure rate-limiting
        # self.rate.sleep()
        self.rate_measure.measure()

        self.timers.tic("real_panda_step")

        self.robot_interface.control(
            control_type=self.controller_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )

        # remember the last gripper action taken in this variable
        gripper_command = action[-1:]
        self.did_grasp = (gripper_command[0] > 0.)

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

        # self.robot_interface.close()
        # del self.robot_interface
        # self.robot_interface = FrankaInterface(
        #     general_cfg_file=os.path.join(config_root, self._init_kwargs['general_cfg_file']),
        #     control_freq=self._init_kwargs['control_freq'],
        #     state_freq=self._init_kwargs['state_freq'],
        #     control_timeout=self._init_kwargs['control_timeout'],
        #     has_gripper=self._init_kwargs['has_gripper'],
        #     use_visualizer=self._init_kwargs['use_visualizer'],
        #     debug=self._init_kwargs['debug'],
        # )

        self.robot_interface.clear_buffer()

        print("restarting the robot interface")

        # Code below based on https://github.com/UT-Austin-RPL/robot_infra/blob/master/gprs/examples/reset_robot_joints.py

        # Golden resetting joints
        reset_joint_positions = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.30396583422025, 0.8480939705504309]

        # This is for varying initialization of joints a little bit to
        # increase data variation.
        # reset_joint_positions = [e + np.clip(np.random.randn() * 0.005, -0.005, 0.005) for e in reset_joint_positions]
        action = reset_joint_positions + [-1.]

        # temp robot interface to use for joint position control
        # tmp_robot_interface = FrankaInterface(os.path.join(config_root, self.general_cfg_file), use_visualizer=False)
        # tmp_controller_cfg = YamlConfig(os.path.join(config_root, self.controller_cfg_file)).as_easydict()
        tmp_controller_cfg = deepcopy(self.controller_cfg)

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                # print(self.robot_interface._state_buffer[-1].q)
                # print(reset_joint_positions)
                # print(np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))))
                # print("-----------------------")

                # if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-3:
                if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-2:
                    break

            self.robot_interface.control(
                control_type="JOINT_POSITION",
                action=action,
                controller_cfg=tmp_controller_cfg,
            )

        # tmp_robot_interface.close()

        # We added this sleep here to give the C++ controller time to reset from joint control mode to no control mode
        # to prevent some issues.
        time.sleep(1.0)
        print("RESET DONE")

        self.did_grasp = False

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
            cur_img = self._get_image(camera_name="front_image")
            if self.use_depth_obs:
                cur_img = cur_img[0]

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
            imgs = self.cr_interfaces[camera_name].get_img()
            return imgs["color"][..., ::-1]
            # return self._get_image(camera_name=camera_name)[..., ::-1]
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
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        ee_pos = ee_pose[:3, 3]
        ee_quat = U.mat2quat(ee_pose[:3, :3])
        observation["ee_pose"] = np.concatenate([ee_pos, ee_quat])
        observation["joint_positions"] = np.array(last_robot_state.q)
        observation["joint_velocities"] = np.array(last_robot_state.dq)
        observation["gripper_position"] = np.array(last_gripper_state.width)
        # observation["gripper_velocity"] = self.robot_interface.gripper_velocity
        for cam_name in self.camera_names_to_sizes:
            im = self._get_image(camera_name=cam_name)
            if self.use_depth_obs:
                im, depth_im = im
                # im, depth_im, depth_im_unaligned = im
                # observation[cam_name + "_depth"] = depth_im
                # observation[cam_name + "_unaligned_depth"] = depth_im_unaligned
                if (not self._exclude_depth_from_obs):
                    depth_im_mod = cam_name + "_depth"
                    if self.postprocess_visual_obs and (depth_im_mod in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=depth_im_mod, obs_modality="depth"):
                        depth_im = ObsUtils.process_obs(obs=depth_im, obs_key=depth_im_mod)
                    observation[depth_im_mod] = depth_im
            im = im[..., ::-1]
            if self.postprocess_visual_obs:
                # NOTE: commented out for now, since run-trained-agent was running into issues with unneeded agent modalities that were present in @self.camera_names_to_sizes
                # assert (cam_name in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=cam_name, obs_modality="rgb")
                im = ObsUtils.process_obs(obs=im, obs_key=cam_name)
            observation[cam_name] = im
        self.timers.toc("get_observation")
        return observation

    def _get_image(self, camera_name):
        """
        Get image from camera interface
        """

        # get image
        imgs = self.cr_interfaces[camera_name].get_img()
        im = imgs["color"]
        
        # resize image
        im_size = self.camera_names_to_sizes[camera_name]
        if im_size is not None:
            im = Image.fromarray(im).resize((im_size[1], im_size[0]), Image.BILINEAR)
        im = np.array(im).astype(np.uint8)

        if self.center_crop_images:
            # center crop image
            crop_size = min(im.shape[:2])
            im = center_crop(im, crop_size, crop_size)

        if self.use_depth_obs:
            depth_im = imgs["depth"]
            if im_size is not None:
                # depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]), Image.BILINEAR)
                depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]))
            # note: depth images are uint16, with default scale 0.001m
            depth_im = np.array(depth_im).astype(np.uint16)
            if len(depth_im.shape) < 3:
                depth_im = depth_im[..., None] # add channel dimension
            if self.center_crop_images:
                depth_im = center_crop(depth_im, crop_size, crop_size)
            return im, depth_im
            # depth_images = []
            # for k in ["depth", "unaligned_depth"]:
            #     depth_im = imgs[k]
            #     if im_size is not None:
            #         # depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]), Image.BILINEAR)
            #         depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]))
            #     # note: depth images are uint16, with default scale 0.001m
            #     depth_im = np.array(depth_im).astype(np.uint16)
            #     if len(depth_im.shape) < 3:
            #         depth_im = depth_im[..., None]  # add channel dimension
            #     if self.center_crop_images:
            #         depth_im = center_crop(depth_im, crop_size, crop_size)
            #     depth_images.append(depth_im)
            # return im, depth_images[0], depth_images[1]
        return im

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
        if self.controller_type == "OSC_POSE":
            return 7
        elif self.controller_type == "JOINT_IMPEDANCE":
            return 8
        assert False, "should never get here"
    
    @property
    def action_dim(self):
        """
        Returns dimension of actions (int).
        """
        return self.action_dimension

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
        return EB.EnvType.GPRS_REAL_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

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
            render_offscreen=True, 
            use_image_obs=True, 
            use_depth_obs=use_depth_obs if use_depth_obs is not None else False, 
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

    def close(self):
        """
        Clean up env
        """
        for c_name in self.cr_interfaces:
            self.cr_interfaces[c_name].stop()
        self.robot_interface.close()
