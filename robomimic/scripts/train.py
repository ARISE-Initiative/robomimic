"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import signal
import socket
import traceback
import shlex
import subprocess
import tempfile
from copy import deepcopy

from collections import OrderedDict
from datetime import datetime

import robomimic

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings, log_warning

def train(config, device, auto_remove_exp=False, resume=False):
    """
    Train a model using the algorithm.
    """

    # time this run
    start_time = time.time()
    time_elapsed = 0.

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=auto_remove_exp, resume=resume)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists

    dataset_paths_to_check = [config.train.data]
    if isinstance(config.train.data, str):
        dataset_paths_to_check = [os.path.expandvars(os.path.expanduser(config.train.data))]
    else:
        dataset_paths_to_check = [os.path.expandvars(os.path.expanduser(ds_cfg["path"])) for ds_cfg in config.train.data]
    for dataset_path in dataset_paths_to_check:
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # NOTE: for now, env_meta, action_keys, and shape_meta are all inferred from the first dataset if multiple datasets are used for training.
    if len(dataset_paths_to_check) > 1:
        log_warning("Env meta and shape meta will be inferred from first dataset at path {}".format(dataset_paths_to_check[0]))

    # assert len(dataset_paths_to_check) == 1
    dataset_path = dataset_paths_to_check[0]

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    # update env meta if applicable
    TensorUtils.deep_update(env_meta, config.experiment.env_meta_update_dict)

    action_keys, _ = FileUtils.get_action_info_from_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        action_keys=action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environments
    envs = OrderedDict()
    env_rollout_horizons = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]
        env_horizons = [config.experiment.rollout.horizon]

        def create_env_helper(seed_id):
            np.random.seed(seed=seed_id + config.train.seed)
            env_id = 0
            # maybe incorporate any additional kwargs for this specific env
            env_meta_for_this_env = deepcopy(env_meta)

            env_name_for_this_env = env_names[env_id]
            env_horizon_for_this_env = env_horizons[env_id]
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta_for_this_env,
                env_name=env_name_for_this_env, 
                render=config.experiment.render, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"], 
                use_depth_obs=shape_meta["use_depths"], 
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment wrapper, if applicable
            return env
        if config.experiment.rollout.batched:
            from tianshou.env import SubprocVectorEnv
            print(f"Creating {config.experiment.rollout.num_batch_envs} vector envs for evaluation")
            env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
            env = SubprocVectorEnv(env_fns)
        else:
            env = create_env_helper(0)
        env_name = env_names[0]
        envs[env_name] = env
        env_rollout_horizons[env_name] = env_horizons[0]
        print(env_name)

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)

        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"])
        print("*" * 50)
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = { k: trainset[k].get_dataset_sampler() for k in trainset }
    print("\n============= Training Datasets =============")
    for k in trainset:
        print("Dataset Key: {}".format(k))
        print(trainset[k])
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalization
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset["data"].get_obs_normalization_stats()
    action_normalization_stats = trainset["data"].get_action_normalization_stats()

    # initialize data loaders
    train_loader = {
        k: DataLoader(
            dataset=trainset[k],
            sampler=train_sampler[k],
            batch_size=config.train.batch_size,
            shuffle=(train_sampler[k] is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        ) for k in trainset
    }

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = {k: validset[k].get_dataset_sampler() for k in validset}
        valid_loader = {
            k: DataLoader(
                dataset=validset[k],
                sampler=valid_sampler[k],
                batch_size=config.train.batch_size,
                shuffle=(valid_sampler[k] is None),
                num_workers=num_workers,
                drop_last=True
            ) for k in validset
        }
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = None
    best_success_rate = None
    best_success_rate_epoch = None
    best_avg_task_success_rate = None
    best_avg_task_success_rate_epoch = None
    if config.experiment.rollout.enabled:
        best_return = dict()
        best_success_rate = dict()
        best_success_rate_epoch = dict()
    last_saved_epoch_with_rollouts = None
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        best_success_rate_epoch = variable_state["best_success_rate_epoch"]
        best_avg_task_success_rate = variable_state["best_avg_task_success_rate"]
        best_avg_task_success_rate_epoch = variable_state["best_avg_task_success_rate_epoch"]
        last_saved_epoch_with_rollouts = variable_state["last_saved_epoch_with_rollouts"]
        time_elapsed = variable_state["time_elapsed"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        did_rollouts = False
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            # run rollouts in current process
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=env_rollout_horizons,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=config.experiment.render,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                epoch=epoch,
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                best_success_rate_epoch=best_success_rate_epoch,
                best_avg_task_success_rate=best_avg_task_success_rate,
                best_avg_task_success_rate_epoch=best_avg_task_success_rate_epoch,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            best_success_rate_epoch = updated_stats["best_success_rate_epoch"]
            best_avg_task_success_rate = updated_stats["best_avg_task_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

            did_rollouts = True

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # # maybe upload rollout videos to wandb
        # if Macros.USE_WANDB and (video_paths is not None):
        #     for k in video_paths:
        #         if os.path.exists(video_paths[k]):
        #             wandb.log({"video": wandb.Video(video_paths[k], format="mp4")})

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt and did_rollouts:
            last_saved_epoch_with_rollouts = epoch

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
            best_success_rate_epoch=best_success_rate_epoch,
            best_avg_task_success_rate=best_avg_task_success_rate,
            best_avg_task_success_rate_epoch=best_avg_task_success_rate_epoch,
            last_saved_epoch_with_rollouts=last_saved_epoch_with_rollouts,
            time_elapsed=(time.time() - start_time + time_elapsed), # keep track of total time elapsed, including previous runs
        )

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()

    # collect important statistics
    important_stats = dict()
    prefix = "Rollout/Success_Rate/"
    exception_prefix = "Rollout/Exception_Rate/"
    exception_retry_prefix = "Rollout/Exception_Rate_Retry/"

    success_rates_by_env = dict()
    all_success_rates_by_epoch = dict() # all success rates by epoch (one per env)
    for k in data_logger._data:
        if k.startswith(prefix):
            env_name = k[len(prefix):]
            success_rates_by_env[env_name] = deepcopy(data_logger._data[k]) # dict mapping epoch to success rate
            for sr_epoch in success_rates_by_env[env_name]:
                if sr_epoch not in all_success_rates_by_epoch:
                    all_success_rates_by_epoch[sr_epoch] = []
                all_success_rates_by_epoch[sr_epoch].append(success_rates_by_env[env_name][sr_epoch])
            stats = data_logger.get_stats(k)
            important_stats["{}-max".format(env_name)] = stats["max"]
            important_stats["{}-mean".format(env_name)] = stats["mean"]
        elif k.startswith(exception_prefix):
            env_name = k[len(exception_prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-exception-rate-max".format(env_name)] = stats["max"]
            important_stats["{}-exception-rate-mean".format(env_name)] = stats["mean"]
        elif k.startswith(exception_retry_prefix):
            env_name = k[len(exception_retry_prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-exception-rate-retry-max".format(env_name)] = stats["max"]
            important_stats["{}-exception-rate-retry-mean".format(env_name)] = stats["mean"]

    if config.experiment.rollout.enabled:
        # get best average task success rate across all tasks and epochs
        avg_task_success_rates_by_epoch = dict() # average task success rate across all envs
        for sr_epoch in all_success_rates_by_epoch:
            if len(all_success_rates_by_epoch[sr_epoch]) == len(success_rates_by_env):
                avg_task_success_rates_by_epoch[sr_epoch] = np.mean(all_success_rates_by_epoch[sr_epoch])
        best_avg_task_epoch = max(avg_task_success_rates_by_epoch, key=lambda k: avg_task_success_rates_by_epoch[k])
        last_avg_task_epoch = max(list(avg_task_success_rates_by_epoch.keys()))
        important_stats["avg-task-max"] = avg_task_success_rates_by_epoch[best_avg_task_epoch]
        important_stats["avg-task-last"] = avg_task_success_rates_by_epoch[last_avg_task_epoch]
        important_stats["avg-task-max-at-epoch"] = best_avg_task_epoch
        important_stats["avg-task-max-all-envs"] = {
            env_name : success_rates_by_env[env_name][best_avg_task_epoch]
            for env_name in success_rates_by_env
        }

        # get per-env best SR and last SR as well
        all_env_names = list(success_rates_by_env.keys())
        max_epoch = max(list(success_rates_by_env[all_env_names[0]].keys()))
        best_epoch_by_env = dict()
        for env_name in all_env_names:
            best_epoch = max(success_rates_by_env[env_name], key=lambda k: success_rates_by_env[env_name][k])
            best_epoch_by_env[env_name] = best_epoch

            # record best epoch
            important_stats["{}-max-at-epoch".format(env_name)] = best_epoch

            # record success rate at last epoch
            important_stats["{}-last".format(env_name)] = success_rates_by_env[env_name][max_epoch]

            # record success rates for all other envs at this env's best epoch
            for env_name2 in all_env_names:
                if env_name == env_name2:
                    continue
                important_stats["{}-at-{}-max".format(env_name2, env_name)] = success_rates_by_env[env_name2][best_epoch]

    # add in time taken
    important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time + time_elapsed) / 3600.)

    # write stats to disk
    json_file_path = os.path.join(log_dir, "important_stats.json")
    with open(json_file_path, 'w') as f:
        # preserve original key ordering
        json.dump(important_stats, f, sort_keys=False, indent=4)

    return important_stats


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.output is not None:
        config.train.output_dir = args.output

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # # wandb project name
    # wandb_project_name = args.wandb_project_name

    # maybe modify config for debugging purposes
    if args.debug:
        Macros.DEBUG = True

        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        if config.experiment.rollout.get("batched", False):
            config.experiment.rollout.num_batch_envs = 2

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

        # # set wandb project name to "test" since it isn't an official run
        # wandb_project_name = "test"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # # maybe setup wandb
    # if Macros.USE_WANDB:
    #     # set api key as env variable
    #     os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

    #     wandb.init(
    #         project=wandb_project_name,
    #         entity=Macros.WANDB_ENTITY,
    #         sync_tensorboard=True,
    #         name=config.experiment.name,
    #         config=config,
    #     )

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    important_stats_str = None
    important_stats = train(config, device=device, auto_remove_exp=args.auto_remove_exp, resume=args.resume)
    print(res_str)
    if important_stats is not None:
        important_stats_str = json.dumps(important_stats, indent=4)
        print("\nRollout Success Rate Stats")
        print(important_stats_str)

    # # maybe cleanup wandb
    # if Macros.USE_WANDB:
    #     wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # force delete the experiment folder if it exists
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes",
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    # # wandb project name
    # parser.add_argument(
    #     "--wandb_project_name",
    #     type=str,
    #     default="test",
    # )

    args = parser.parse_args()
    main(args)

