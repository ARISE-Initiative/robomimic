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
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.macros as Macros


from robomimic.utils.dataset import SequenceDataset, R2D2Dataset, MetaDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    assert not (Macros.USE_MAGLEV and Macros.USE_NGC)
    if Macros.USE_MAGLEV or Macros.USE_NGC:
        # remove existing experiment directory automatically if path exists so that we don't block on user input
        auto_remove_exp_dir = True

    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expandvars(os.path.expanduser(config.train.output_dir))
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
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
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)

    # establish sync path for syncing important training results back
    set_absolute_sync_path(
        output_dir=config.train.output_dir,
        exp_name=config.experiment.name,
        time_str=time_str,
    )

    return log_dir, output_dir, video_dir


def set_absolute_sync_path(output_dir, exp_name, time_str=None):
    """
    Establish sync path for syncing important training results back and puts the path
    into Macros.RESULTS_SYNC_PATH_ABS
    """
    need_sync_results = (Macros.USE_MAGLEV and (Macros.MAGLEV_SCRATCH_SYNC_PATH is not None)) or \
        (Macros.USE_NGC and (Macros.NGC_SCRATCH_SYNC_PATH is not None)) or \
        ((not Macros.USE_MAGLEV) and (not Macros.USE_NGC) and (Macros.RESULTS_SYNC_PATH is not None))
    if need_sync_results:
        # get path where we will sync results
        assert Macros.RESULTS_SYNC_PATH_ABS is None
        base_output_dir_name = os.path.basename(os.path.normpath(os.path.expandvars(os.path.expanduser(output_dir))))
        
        if Macros.USE_MAGLEV:
            # turn relative scratch space path into absolute scratch space path
            sync_prefix = os.path.join(
                os.getenv("WORKFLOW_SCRATCH"),
                "test_disk", # NOTE: most workflows mount scratch space under this prefix
                Macros.MAGLEV_SCRATCH_SYNC_PATH,
            )
        elif Macros.USE_NGC:
            sync_prefix = os.path.expandvars(os.path.expanduser(Macros.NGC_SCRATCH_SYNC_PATH))
        else:
            sync_prefix = os.path.expandvars(os.path.expanduser(Macros.RESULTS_SYNC_PATH))

        # store at results_sync_path/output_dir_name/experiment_name/time_str
        sync_path_without_time_dir = os.path.join(
            sync_prefix,
            base_output_dir_name,
            exp_name,
        )
        if os.path.exists(sync_path_without_time_dir):
            # only keep one time directory per exp name
            shutil.rmtree(sync_path_without_time_dir)
        Macros.RESULTS_SYNC_PATH_ABS = sync_path_without_time_dir
        if time_str is not None:
            Macros.RESULTS_SYNC_PATH_ABS = os.path.join(sync_path_without_time_dir, time_str)
        os.makedirs(Macros.RESULTS_SYNC_PATH_ABS)
    elif (Macros.USE_MAGLEV or Macros.USE_NGC):
        LogUtils.log_warning(
            "Using MagLev / NGC, but MAGLEV_SCRATCH_SYNC_PATH / NGC_SCRATCH_SYNC_PATH is unset in macros.py."
            "No results will be synced back to scratch space."
        )


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

    # load the dataset into memory
    if config.experiment.validate:
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
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
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

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
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
        filter_by_attribute=filter_by_attribute
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.train.data]
    ds_kwargs["filter_by_attribute"] = [filter_by_attribute for ds_cfg in config.train.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.train.data]
    ds_labels = [ds_cfg.get("label", "dummy") for ds_cfg in config.train.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=R2D2Dataset if config.train.data_format == "r2d2" else SequenceDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_labels=ds_labels,
        normalize_weights_by_ds_size=False,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_labels,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            ds_labels=ds_labels,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def run_rollout(
        policy, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
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
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics
    got_exception = False

    try:
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    results["Exception_Rate"] = float(got_exception)

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

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
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

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
    assert isinstance(policy, RolloutPolicy)

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

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
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
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

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
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None, action_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

        action_normalization_stats (dict): TODO
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
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
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
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


def get_model_from_output_folder(models_path, videos_path=None, epoch=None, best=False, last=False):
    """
    Gets path to model (and video) for a certain epoch number (or the best or last epoch).

    Args:
        models_path (str): path to models folder (in output directory)
        videos_path (str): path to videos folder (in output directory)
        epoch (int): if provided, get model ckpt and video for this epoch
        best (bool): if True, get the model and video for the best checkpoint (according to success rate)
        last (bool): if True, get the model and video for the last checkpoint (according to epoch number)

    Returns:
        model_path (str): path to model pth
        video_path (str): path to mp4
        epoch (int): epoch number for retrieved model and video paths
    """

    # make sure we either grab a specific epoch, best epoch, or last epoch
    assert sum([(epoch is not None), best, last]) == 1

    # run through models to find the epoch we want
    best_success_rate = -0.1
    need_particular_epoch = (epoch is not None)
    need_best_epoch = best
    need_max_epoch = last

    selected_epoch = -1
    selected_model_path = None
    for f in os.scandir(models_path):
        model_epoch = int(f.name.split("_")[2].strip(".pth"))

        if need_particular_epoch and (model_epoch == epoch):
            selected_epoch = epoch
            selected_model_path = os.path.join(models_path, f.name)

        elif need_best_epoch: 
            # this block assumes that the experiment run opted to save the model with the best checkpoint
            if "success" in f.name:
                # example name: model_epoch_250_NutAssemblySquareTarget_6_success_0.86.pth
                # take last piece - "0.86.pth" -> "0.86" -> convert to float
                success_rate = float(f.name.split("success_")[-1][:-4])
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    selected_epoch = model_epoch
                    selected_model_path = os.path.join(models_path, f.name)

        elif need_max_epoch:
            # find last epoch
            if model_epoch > selected_epoch:
                selected_epoch = model_epoch
                selected_model_path = os.path.join(models_path, f.name)

    assert selected_epoch != -1
    assert selected_model_path is not None

    selected_video_path = None
    if videos_path is not None:
        # get random video filename
        video_fname = None
        for f in os.scandir(videos_path):
            video_fname = f.name
            break
        # example video file name: NutAssemblySquareTarget_6_epoch_150.mp4
        # take name skeleton and use it to infer name of source videos we want, then copy them
        video_name_prefix = video_fname.split("epoch")[0]
        selected_video_path = os.path.join(videos_path, "{}epoch_{}.mp4".format(video_name_prefix, selected_epoch))
    return selected_model_path, selected_video_path, selected_epoch
