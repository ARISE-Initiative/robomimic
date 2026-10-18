"""
This script converts a robosuite dataset with absolute actions to delta actions.
It reads the robomimic dataset, processes each demo to convert absolute actions to
delta actions, and saves the results back to the dataset under a new key
`actions_delta`. This is the inverse operation of
`robosuite_add_absolute_actions.py`, which converts delta actions (`actions`) to
absolute actions (`actions_abs`).

The absolute actions are read from `actions_abs` when present (e.g. datasets
produced by `robosuite_add_absolute_actions.py`), and from `actions` otherwise
(e.g. datasets collected with a controller in absolute mode).

Arguments:
    dataset (str): path to the robomimic dataset
    num_workers (int): number of workers to use for parallel processing

Example usage:
    python scripts/conversion/robosuite_add_delta_actions.py --dataset /path/to/your/demo.hdf5 --num_workers 10
"""

import multiprocessing
import pathlib
import h5py
from tqdm import tqdm
import argparse
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

import robosuite

from robomimic.config import config_factory

class RobomimicDeltaActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        """
        Class to convert robomimic dataset with absolute actions to delta actions.
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

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        self.env = env
        self.file = h5py.File(dataset_path, 'r')

    def get_demo_keys(self):
        return list(self.file['data'].keys())

    def convert_actions(self,
            states: np.ndarray,
            actions: np.ndarray,
            initial_state: dict) -> np.ndarray:
        """
        Given a state and absolute action sequence, generate equivalent delta
        actions for each step. The absolute goal pose for each step is expressed
        relative to the robot pose achieved at that step, by inverting the
        controller's delta action scaling (see robosuite
        `Controller.scale_action`) and goal update (see robosuite
        `OSCController.compute_goal_pos` and `compute_goal_ori`). The original
        gripper actions are kept intact.
        """
        env = self.env
        d_a = len(env.env.robots[0].action_limits[0])

        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)

        # generate delta actions
        action_delta_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,),
            dtype=stacked_actions.dtype)
        action_delta_ori = np.zeros(
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
                    controller = robot.controller
                else:
                    controller = robot.part_controllers['right']

                # sync the controller reference (current eef pose) with the
                # state that this action was taken from, exactly like the
                # controller does on every control step. Passing a zero delta
                # action does not advance the simulation.
                robot.control(np.zeros(d_a), policy_step=True)

                # read the robot pose that this step's delta action starts
                # from, in the same reference frame the controller expresses
                # goals in
                ref_pos = np.array(controller.ref_pos)
                ref_ori_mat = np.array(controller.ref_ori_mat)
                if getattr(controller, "input_ref_frame", "world") == "base":
                    ref_pos = np.array(controller.world_to_origin_frame(ref_pos))
                    ref_ori_mat = np.array(controller.goal_origin_to_eef_pose())[:3, :3]

                # read the absolute goal pose from the absolute action
                goal_pos = stacked_actions[i, idx, 0:3]
                goal_ori = Rotation.from_rotvec(stacked_actions[i, idx, 3:6]).as_matrix()

                # invert the controller's affine input scaling
                # (see robosuite Controller.scale_action)
                action_scale = (controller.output_max - controller.output_min) / (controller.input_max - controller.input_min)
                action_output_transform = (controller.output_max + controller.output_min) / 2.0
                action_input_transform = (controller.input_max + controller.input_min) / 2.0

                # delta position: goal is achieved position plus a scaled delta
                delta_pos = goal_pos - ref_pos
                action_delta_pos[i, idx] = (delta_pos - action_output_transform[0:3]) / action_scale[0:3] + action_input_transform[0:3]

                # delta orientation: goal orientation is achieved orientation
                # pre-multiplied with the rotation corresponding to the scaled
                # delta (in the controller reference frame)
                delta_ori_mat = goal_ori @ ref_ori_mat.T
                delta_ori = Rotation.from_matrix(delta_ori_mat).as_rotvec()
                action_delta_ori[i, idx] = (delta_ori - action_output_transform[3:6]) / action_scale[3:6] + action_input_transform[3:6]

        stacked_delta_actions = np.concatenate([
            action_delta_pos,
            action_delta_ori,
            action_remainder
        ], axis=-1)
        delta_actions = stacked_delta_actions.reshape(actions.shape)
        return delta_actions

    def convert_demo(self, demo_key, input_key):
        file = self.file
        demo = file["data/{}".format(demo_key)]
        # input
        states = demo['states'][:]
        actions = demo[input_key][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

        # generate delta actions
        delta_actions = self.convert_actions(states, actions, initial_state=initial_state)
        return delta_actions


def worker(x):
    path, demo_key, input_key = x
    converter = RobomimicDeltaActionConverter(path)
    delta_actions = converter.convert_demo(demo_key, input_key)
    info = dict()
    return delta_actions, info


def add_delta_actions_to_dataset(dataset, num_workers):
    """
    Entry-point for adding delta actions to robomimic dataset.
    Args:
        dataset (str): path to the robomimic dataset
        num_workers (int): number of workers to use for parallel processing
    """
    # process inputs
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    # read absolute actions from `actions_abs` if present, and from
    # `actions` otherwise (e.g. collected with a controller in absolute mode)
    with h5py.File(dataset, 'r') as in_file:
        demo_keys = list(in_file['data'].keys())
        demo_key = demo_keys[0]
        if "actions_abs" in in_file["data/{}".format(demo_key)]:
            input_key = "actions_abs"
        else:
            input_key = "actions"
    print("====== Reading absolute actions from key: {} ======".format(input_key))

    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(dataset, demo_key, input_key) for demo_key in demo_keys])

    # modify action
    with h5py.File(dataset, 'r+') as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            delta_actions, info = results[i]
            demo = out_file["data/{}".format(demo_keys[i])]
            if "actions_delta" not in demo:
                demo.create_dataset("actions_delta", data=np.array(delta_actions))
            else:
                demo['actions_delta'][:] = delta_actions


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

    add_delta_actions_to_dataset(
        dataset=args.dataset,
        num_workers=args.num_workers,
    )
