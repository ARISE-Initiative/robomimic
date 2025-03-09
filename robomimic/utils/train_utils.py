"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import math
import signal
import contextlib
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from multiprocessing import TimeoutError

import torch
try:
    from tianshou.env import SubprocVectorEnv
except ImportError:
    print("tianshou is not installed")

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.macros as Macros

from robomimic.utils.dataset import SequenceDataset, MetaDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False, resume=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

        resume (bool): if True, resume an existing training run instead of creating a 
            new experiment directory
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """

    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expandvars(os.path.expanduser(config.train.output_dir))
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)

    if resume:
        assert os.path.exists(base_output_dir), "Resuming training run, but output dir {} does not exist".format(base_output_dir)
        subdir_lst = os.listdir(base_output_dir)
        assert len(subdir_lst) == 1, "Found more than one subdir {} in output dir {}".format(subdir_lst, base_output_dir)
        time_str = subdir_lst[0]
        assert os.path.isdir(os.path.join(base_output_dir, time_str)), "Found item {} that is not a subdirectory in {}".format(time_str, base_output_dir)
    elif os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir, exist_ok=resume)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir, exist_ok=resume)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir, exist_ok=resume)

    time_dir = os.path.join(base_output_dir, time_str)

    return log_dir, output_dir, video_dir, time_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)
    first_n_demos = config.train.hdf5_first_n_demos

    # load the dataset into memory
    if config.experiment.validate:
        assert isinstance(config.train.data, str)
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute, first_n_demos=first_n_demos)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute, first_n_demos=None)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute, first_n_demos=first_n_demos)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, first_n_demos=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

        first_n_demos (int or None): if provided, restrict training to use the first N demos

    Returns:
        dataset (SequenceDataset instance): dataset object
    """

    action_keys, action_config = FileUtils.get_action_info_from_config(config)

    base_ds_kwargs = dict(
        obs_keys=obs_keys,
        action_keys=action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=action_config,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        first_n_demos=first_n_demos,
    )
    dataset = dict()

    # we might have more than one dataset path stored in config.train.data - this might require
    # constructing multiple dataset objects and using a MetaDataset object to handle them
    ds_configs = config.train.data
    if isinstance(config.train.data, str):
        ds_configs = [
            dict(
                path=config.train.data,
                filter_key=filter_by_attribute,
                weight=1.0,
            ),
        ]

    # construct a dataset object for each dataset path
    dataset_objects = []
    for ds_ind in range(len(ds_configs)):
        ds_cfg = ds_configs[ds_ind]
        kwargs_for_ds_ind = deepcopy(base_ds_kwargs)
        kwargs_for_ds_ind["hdf5_path"] = ds_cfg["path"]
        kwargs_for_ds_ind["filter_by_attribute"] = ds_cfg.get("filter_key", filter_by_attribute)
        dataset_objects.append(
            SequenceDataset(**kwargs_for_ds_ind)
        )

    # maybe construct MetaDataset to handle constructing batches with sampling across all datasets.
    if len(ds_configs) == 1:
        # single dataset
        final_dataset = dataset_objects[0]
    else:
        dataset_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in ds_configs]
        final_dataset = MetaDataset(
            datasets=dataset_objects,
            ds_weights=dataset_weights,
            normalize_weights_by_ds_size=True,
        )
    dataset["data"] = final_dataset

    return dataset


def batchify_obs(obs_list):
    """
    TODO: add comments
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs


def run_rollout(
        policy, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        obs_video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        render_args={"width": 512, "height": 512, "camera_name": "agentview"}
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper) or isinstance(env, SubprocVectorEnv)
    
    batched = isinstance(env, SubprocVectorEnv)

    results = {}
    video_count = 0  # video frame counter

    rews = []
    got_exception = False
    got_exception_retry = False
    num_steps = 0
    success = dict(task=False)


    # ensure env calls are inside try-block in case they trigger a rollout exception
    ob_dict = env.reset()
    policy.start_episode()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()
    success = None #{ k: False for k in env.is_success() } # success metrics
    
    if batched:
        end_step = [None for _ in range(len(env))]
    else:
        end_step = None

    video_frames = []
    obs_frames = []
    iterator = LogUtils.custom_tqdm(range(horizon), total=horizon)
    
    for step_i in iterator:
        start_time = time.time()

        # get action from policy
        if batched:
            policy_ob = batchify_obs(ob_dict)
            ac = policy(ob=policy_ob, goal=goal_dict, batched=True) #, return_ob=True)
        else:
            policy_ob = ob_dict
            ac = policy(ob=policy_ob, goal=goal_dict) #, return_ob=True)

        ob_dict, r, done, info = env.step(ac)

        num_steps += 1

        if render and not batched:
            env.render(mode="human")

        rews.append(r)

        if batched:
            cur_success_metrics = TensorUtils.list_of_flat_dict_to_dict_of_list([info[i]["is_success"] for i in range(len(info))])
            cur_success_metrics = {k: np.array(v) for (k, v) in cur_success_metrics.items()}
        else:
            cur_success_metrics = info["is_success"]

        if success is None:
            success = deepcopy(cur_success_metrics)
        else:
            for k in success:
                success[k] = success[k] | cur_success_metrics[k]

        if video_writer is not None or obs_video_writer is not None:
            if video_count % video_skip == 0:
                if obs_video_writer is not None:
                    policy_ob = deepcopy(policy_ob)
                    im_names = []
                    for obs_k in policy_ob.keys():
                        if (obs_k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=obs_k, obs_modality="rgb"):
                            im_names.append(obs_k)
                    im_names = sorted(im_names)
                    imgs = [policy_ob[im_name] for im_name in im_names]
                    if not batched:
                        imgs = [x[None] for x in imgs]
                    imgs = np.concatenate(imgs, axis=-1).transpose(0, 2, 3, 1)
                    imgs = (imgs * 255.0).astype(np.uint8)
                    obs_frames.append(imgs)
                if video_writer is not None:
                    video_imgs = env.render(mode="rgb_array", **render_args)
                    if batched:
                        video_imgs = [policy.modify_rollout_video_frame(video_img) for video_img in video_imgs]
                        video_imgs = np.stack(video_imgs, axis=0)
                    else:
                        video_imgs = policy.modify_rollout_video_frame(video_imgs)[None]
                    video_frames.append(video_imgs)

            video_count += 1

        # break if done
        
        if batched:
            for env_i in range(len(env)):
                if end_step[env_i] is not None:
                    continue
                
                if done[env_i] or (terminate_on_success and success["task"][env_i]):
                    end_step[env_i] = step_i
        else:
            if done or (terminate_on_success and success["task"]):
                end_step = step_i
                break

    if video_writer is not None:
        if batched:
            for env_i in range(len(env)):
                for frame in video_frames:
                    video_writer.append_data(frame[env_i])
        else:
            for frame in video_frames:
                video_writer.append_data(frame[0])
    if obs_video_writer is not None:
        if batched:
            for env_i in range(len(env)):
                for frame in obs_frames:
                    obs_video_writer.append_data(frame[env_i])
        else:
            for frame in obs_frames:
                obs_video_writer.append_data(frame[0])

    if batched:
        total_reward = np.zeros(len(env))
        rews = np.array(rews)
        for env_i in range(len(env)):
            end_step_env_i = end_step[env_i] or step_i
            total_reward[env_i] = np.sum(rews[:end_step_env_i+1, env_i])
            end_step[env_i] = end_step_env_i
        
        results["Return"] = total_reward.tolist()
        results["Horizon"] = (np.array(end_step) + 1).tolist()
        results["Success_Rate"] = success["task"].astype(float).tolist()
    else:
        end_step = end_step or step_i
        total_reward = np.sum(rews[:end_step + 1])
        
        results["Return"] = [total_reward]
        results["Horizon"] = [end_step + 1]
        results["Success_Rate"] = [float(success["task"])]
        
    # TODO dummy fill for now
    results["Policy_Horizon"] = results["Horizon"]
    results["Exception_Rate"] = [0.0] * len(results["Return"])
    results["Exception_Rate_Retry"] = [0.0] * len(results["Return"])

    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = [float(success[k])]

    return results


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        extra_args={}
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance or dict): policy to use for rollouts. Can be a dict to use a 
            different policy for each env - in this case it should map env name to policy

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int or dict): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy) or isinstance(policy, dict)
    if isinstance(policy, dict):
        for env_name in policy:
            assert isinstance(policy[env_name], RolloutPolicy)
    else:
        # easy way to unify code - turn policy into dict
        policy = { k : policy for k in envs }

    if not isinstance(horizon, dict):
        # easy way to unify code
        horizon = { k : horizon for k in horizon }

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }
        obs_video_writers = { k : imageio.get_writer(video_paths[k].replace(".mp4", "_obs.mp4"), fps=20) for k in envs }

            
    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]
            env_obs_video_writer = obs_video_writers[env_name]
        else:
            env_video_writer = None
            env_obs_video_writer = None

            
        batched = isinstance(env, SubprocVectorEnv)
        if batched:
            num_env = len(env)
        else:
            num_env = 1
            
        
        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}, num_env={}".format(
            env_name, horizon[env_name], use_goals, num_episodes, num_env
        ))
        rollout_logs = []
            
        num_chunk = num_episodes // num_env

        num_success = 0
        num_episodes_attempted = 0
        num_exception = 0
        num_exception_retry = 0
        for ep_i in range(num_chunk):
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy[env_name],
                env=env,
                horizon=horizon[env_name],
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                obs_video_writer=env_obs_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
                **extra_args
            )
            # amortize time across all envs
            rollout_info["time"] = [(time.time() - rollout_timestamp) / num_env] * num_env

            rollout_logs.append(rollout_info)
            num_success += sum(rollout_info["Success_Rate"])
            num_episodes_attempted += len(rollout_info["Success_Rate"])
            num_exception += sum(rollout_info["Exception_Rate"])
            
            print("Chunk {}/{}, horizon={}, success={}/{}, acc success={}/{}".format(ep_i + 1, num_chunk, horizon[env_name], sum(rollout_info["Success_Rate"]), num_env, num_success, num_episodes_attempted))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs_all = {}
        for k in rollout_logs[0]:
            # aggregate all lists into a single list
            rollout_logs_all[k] = sum([rollout_info[k] for rollout_info in rollout_logs], [])
        rollout_logs = rollout_logs_all
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        rollout_logs_mean["Num_Episode"] = num_episodes_attempted # number of episodes attempted
        rollout_logs_mean["Exception_Rate_Retry"] = (float(num_exception_retry) / num_episodes_attempted)

        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        epoch,
        all_rollout_logs,
        best_return,
        best_success_rate,
        best_success_rate_epoch,
        best_avg_task_success_rate,
        best_avg_task_success_rate_epoch,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        epoch (int): current epoch number

        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        best_success_rate_epoch (dict): dictionary that stores the epoch at which the best average success rate
            was achieved, for each environment

        best_avg_task_success_rate (float or None): stores best average task success rate across all tasks

        best_avg_task_success_rate_epoch (int): stores epoch at which best average task success rate was
            achieved across all environments

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    all_env_sr = []
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if (env_name not in best_return) or (rollout_logs["Return"] > best_return[env_name]):
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        all_env_sr.append(rollout_logs["Success_Rate"])
        if (env_name not in best_success_rate) or (rollout_logs["Success_Rate"] > best_success_rate[env_name]):
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            best_success_rate_epoch[env_name] = epoch
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    cur_avg_env_sr = np.mean(all_env_sr)
    if (best_avg_task_success_rate is None) or (cur_avg_env_sr > best_avg_task_success_rate):
        best_avg_task_success_rate = cur_avg_env_sr
        best_avg_task_success_rate_epoch = epoch
        # save checkpoint for best avg task success if evaluating on more than one task
        if save_on_best_rollout_success_rate and (len(all_env_sr) > 1):
            epoch_ckpt_name += "_avg_task_success_{}".format(best_avg_task_success_rate)
            should_save_ckpt = True
            ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        best_success_rate_epoch=best_success_rate_epoch,
        best_avg_task_success_rate=best_avg_task_success_rate,
        best_avg_task_success_rate_epoch=best_avg_task_success_rate_epoch,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(
        model,
        config,
        env_meta,
        shape_meta,
        ckpt_path,
        variable_state=None,
        obs_normalization_stats=None,
        action_normalization_stats=None,
    ):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        variable_state (dict): internal variable state in main train loop, used for restoring training process
            from ckpt

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
        variable_state=variable_state,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    if action_normalization_stats is not None:
        action_normalization_stats = deepcopy(action_normalization_stats)
        params["action_normalization_stats"] = TensorUtils.to_list(action_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        first_data_loader = list(data_loader.values())[0]
        num_steps = len(first_data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = { k: iter(data_loader[k]) for k in data_loader }
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        batch = dict()
        t = time.time()
        for k in data_loader:
            try:
                batch[k] = next(data_loader_iter[k])
            except StopIteration:
                # reset for next dataset pass
                data_loader_iter[k] = iter(data_loader[k])
                batch[k] = next(data_loader_iter[k])
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0


@contextlib.contextmanager
def timeout_context(timeout):
    """
    Note: from https://gitlab-master.nvidia.com/srl/simpler/-/blob/ce33f1a794e54bfe47c77c0bbe67892c4bcce71b/src/simpler/util.py#L81

    A context manager that enforces a timeout on its Python context.

    Args:
        timeout: The number of seconds before the timeout.

    Raises:
        TimeoutError: If `timeout` seconds pass before the context is exited.
    """
    if timeout == math.inf:
        yield
        return
    timeout = int(math.ceil(timeout))

    def raise_timeout(*args):
        raise TimeoutError(f"Timed out after {timeout:d} seconds.")

    if timeout <= 0:
        raise_timeout()

    # Register `raise_timeout` to raise a `TimeoutError` on the signal
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after `timeout`
    signal.alarm(timeout)
    try:
        yield
    finally:
        # Unregister the signal
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def maybe_timeout_context(timeout=None):
    """
    Args:
        timeout (float or None): if provided, sets up a context that raises TimeoutError after @timeout
            seconds have passed without leaving the context, otherwise sets up a dummy context
    """
    return timeout_context(timeout=timeout) if timeout is not None else TorchUtils.dummy_context_mgr()
