"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Use 10 vectorized envs
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --n_envs 10 --horizon 400 --seed 0 \
        --logdir /path/to/log/ \
"""
import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.log_utils import log_warning
from robomimic.utils.python_utils import DictionaryAction
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper, FrameStackWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS

def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    rollout_model, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if args.horizon is not None:
        rollout_horizon = args.horizon
    else:
        rollout_horizon = config.experiment.rollout.horizon
    eval_seed = config.train.seed if args.seed is None else args.seed
    num_episodes = args.n_rollouts

    # create envs
    envs = {}
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]
    env_name = env_meta["env_name"]
    def create_env_helper(seed_id):
        np.random.seed(seed=seed_id + eval_seed)
        # maybe incorporate any additional kwargs for this specific env
        env_meta_for_this_env = deepcopy(env_meta)

        env_name_for_this_env = env_name
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta_for_this_env,
            env_name=env_name_for_this_env, 
            render=config.experiment.render, 
            render_offscreen=True,
            use_image_obs=shape_meta["use_images"], 
            use_depth_obs=shape_meta["use_depths"], 
        )
        env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment wrapper, if applicable
        return env
    if args.n_envs > 1:
        from tianshou.env import SubprocVectorEnv
        print(f"Creating {args.n_envs} vector envs for evaluation")
        env_fns = [lambda seed_id=i: create_env_helper(seed_id) for i in range(args.n_envs)]
        env = SubprocVectorEnv(env_fns)
    else:
        env = create_env_helper(0)
    envs = {env_name: env}
    print(env_name)


    # run rollouts
    # create_logdir
    video_dir = f"{args.logdir}/videos"
    os.makedirs(video_dir, exist_ok=True)
    render_args = {
        "camera_name": args.camera_name,
        "width": args.resolution[0],
        "height": args.resolution[1] 
    }

    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon={env_name: rollout_horizon},
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        extra_args={"render_args": render_args}
    )

    # summarize results from rollouts to tensorboard and terminal
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        print('Env: {}'.format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))
        with open(f"{args.logdir}/{env_name}.json", "w") as f:
            json.dump(rollout_logs, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="number of rollouts",
    )

    # number of rollouts
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="directory to save logs",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_name",
        type=str,
        default="agentview",
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # camera names to render
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="(optional) render resolution",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)
