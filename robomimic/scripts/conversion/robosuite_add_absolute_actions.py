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

from robomimic.config import config_factory

"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
"""
class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

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
        assert not abs_env.env.robots[0].controller.use_delta

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

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
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

    def convert_and_eval_demo(self, demo_key):
        raise NotImplementedError
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file["data/{}".format(demo_key)]
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info

"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/scripts/robomimic_dataset_conversion.py
"""
def worker(x):
    path, demo_key, do_eval = x
    converter = RobomimicAbsoluteActionConverter(path)
    if do_eval:
        abs_actions, info = converter.convert_and_eval_demo(demo_key)
    else:
        abs_actions = converter.convert_demo(demo_key)
        info = dict()
    return abs_actions, info


def add_absolute_actions_to_dataset(dataset, eval_dir, num_workers):
    # process inputs
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()
        do_eval = True
    
    converter = RobomimicAbsoluteActionConverter(dataset)
    demo_keys = converter.get_demo_keys()
    del converter
    
    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(dataset, demo_key, do_eval) for demo_key in demo_keys])

    # modify action
    with h5py.File(dataset, 'r+') as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file["data/{}".format(demo_keys[i])]
            if "actions_abs" not in demo:
                demo.create_dataset("actions_abs", data=np.array(abs_actions))
            else:
                demo['actions_abs'][:] = abs_actions
    
    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath('error_stats.pkl').open('wb'))

        print("Generating visualization")
        metrics = ['pos', 'rot']
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt
        plt.switch_backend('PDF')

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10,4)
        fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
        fig.savefig(str(eval_dir.joinpath('error_stats.png')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        help="directory to output evaluation metrics",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    
    args = parser.parse_args()
    
    
    add_absolute_actions_to_dataset(
        dataset=args.dataset,
        eval_dir=args.eval_dir,
        num_workers=args.num_workers,
    )