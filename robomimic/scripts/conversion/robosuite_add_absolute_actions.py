"""
This script converts robosuite dataset with delta actions to absolute actions.
It reads the robomimic dataset, processes each demo to convert delta actions (`actions`) to 
absolute actions, and saves the results back to the dataset under a new key `actions_abs`.

Arguments:
    dataset (str): path to the robomimic dataset
    num_workers (int): number of workers to use for parallel processing

Example usage:
    python scripts/conversion/robosuite_add_absolute_actions.py --dataset /path/to/your/demo.hdf5 --num_workers 10
"""

import multiprocessing
import os
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
import argparse
import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

import robosuite

from robomimic.config import config_factory

"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
"""
class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        """
        Class to convert robomimic dataset with delta actions to absolute actions.
        Args:
            dataset_path (str): path to the robomimic dataset
            algo_name (str): name of the algorithm to use for config
        """
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        if robosuite.__version__ < "1.5":
            abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        else:
            abs_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        if robosuite.__version__ < "1.5":
            assert not abs_env.env.robots[0].controller.use_delta
            self.controller_config = abs_env._init_kwargs['controller_configs']
        else:
            assert not abs_env._init_kwargs['controller_configs']['body_parts']['right']['control_delta']
            self.controller_config = abs_env._init_kwargs['controller_configs']['body_parts']['right']

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def get_demo_keys(self):
        return list(self.file['data'].keys())

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray,
            initial_state: dict) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        env = self.env
        d_a = len(env.env.robots[0].action_limits[0])

        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)

        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_remainder = stacked_actions[...,6:]
        for i in range(len(states)):
            if i == 0:
                _ = env.reset_to(initial_state)
            else:
                _ = env.reset_to({'states': states[i]})

            for idx, robot in enumerate(env.env.robots):
                if robosuite.__version__ < "1.5":
                    # run controller goal generator
                    robot.control(stacked_actions[i,idx], policy_step=True)
                
                    # read pos and ori from robots
                    controller = robot.controller
                    action_goal_pos[i,idx] = controller.goal_pos
                    action_goal_ori[i,idx] = Rotation.from_matrix(
                        controller.goal_ori).as_rotvec()
                
                else:
                    # run controller goal generator
                    robot.control(stacked_actions[i,idx], policy_step=True)

                    # read pos and ori from robots
                    controller = robot.part_controllers['right']
                    action_goal_pos[i,idx] = controller.goal_pos
                    action_goal_ori[i,idx] = Rotation.from_matrix(
                        controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_remainder
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_demo(self, demo_key):
        file = self.file
        demo = file["data/{}".format(demo_key)]
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

        # generate abs actions
        abs_actions = self.convert_actions(states, actions, initial_state=initial_state)
        return abs_actions


"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/scripts/robomimic_dataset_conversion.py
"""
def worker(x):
    path, demo_key = x
    converter = RobomimicAbsoluteActionConverter(path)
    abs_actions = converter.convert_demo(demo_key)
    info = dict()
    return abs_actions, info


def add_absolute_actions_to_dataset(dataset, num_workers):
    """
    Entry-point for adding absolute actions to robomimic dataset.
    Args:
        dataset (str): path to the robomimic dataset
        num_workers (int): number of workers to use for parallel processing
    """
    # process inputs
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    # initialize converter
    converter = RobomimicAbsoluteActionConverter(dataset)
    demo_keys = converter.get_demo_keys()
    del converter
    
    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(dataset, demo_key) for demo_key in demo_keys])

    # modify action
    with h5py.File(dataset, 'r+') as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file["data/{}".format(demo_keys[i])]
            if "actions_abs" not in demo:
                demo.create_dataset("actions_abs", data=np.array(abs_actions))
            else:
                demo['actions_abs'][:] = abs_actions    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    
    args = parser.parse_args()

    add_absolute_actions_to_dataset(
        dataset=args.dataset,
        num_workers=args.num_workers,
    )
