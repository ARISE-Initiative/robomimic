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
import socket
import traceback
import tqdm

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import tensorflow as tf

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.action_utils as ActionUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.rlds_utils import droid_dataset_transform, robomimic_transform, R2D2_TO_RLDS_OBS_KEY_MAP, R2D2_TO_RLDS_LOW_DIM_OBS_KEY_MAP, TorchRLDSDataset

from octo.data.dataset import make_interleaved_dataset


def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    ds_format = config.train.data_format

    if ds_format == "r2d2_rlds":
        # # load basic metadata from training file
        # print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=None, ds_format=ds_format)
         # TODO (Ashwin): make sure setting to None here is OK, are observations pre-normalized the way we want?
        obs_normalization_stats = None

        # FOR RLDS
        tf.config.set_visible_devices([], "GPU")

        obs_modalities = config.observation.modalities.obs.rgb
        # NOTE(Ashwin): must be 2 cam for now, can clean this up later
        assert(len(obs_modalities) == 2)

        BASE_DATASET_KWARGS = {
                "data_dir": config.train.data_path,
                "image_obs_keys": {"primary": R2D2_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]], "secondary": R2D2_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]]},
                "state_obs_keys": [R2D2_TO_RLDS_LOW_DIM_OBS_KEY_MAP[a] for a in config.observation.modalities.obs.low_dim],
                "language_key": "language_instruction",
                "action_proprio_normalization_type": "bounds",
                "absolute_action_mask": [True] * 10,
                "action_normalization_mask": [True] * 10,
                "standardize_fn": droid_dataset_transform,
            }

        # you can add more datasets here & the sampling weights below if you want to mix
        dataset_names = config.train.dataset_names
        dataset_kwargs_list = [
            {"name": d_name,  **BASE_DATASET_KWARGS} for d_name in dataset_names
        ]

        # TODO(Ashwin): Wrap more of the parameters below in the robomimic configs
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            config.train.sample_weights,
            train=True,
            shuffle_buffer_size=config.train.shuffle_buffer_size,
            batch_size=None,  # batching will be handles in PyTorch Dataloader object
            balance_weights=True,
            traj_transform_kwargs=dict(
                # NOTE(Ashwin): window_size and future_action_window_size may break if 
                # not using diffusion policy
                window_size=config.algo.horizon.observation_horizon,
                future_action_window_size=config.algo.horizon.prediction_horizon-1,
                subsample_length=100,
                skip_unlabeled=True,    # skip all trajectories without language
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs=dict(
                ),
                resize_size=dict(
                    primary=config.observation.image_dim,
                    secondary=config.observation.image_dim,
                ),
                num_parallel_calls=200,
            ),
            traj_transform_threads=48,
            traj_read_threads=48,
        )
        # TODO(Ashwin): this assumes that you are doing co-training and that the co-training dataset
        # that corresponds to the eval setting is last
        rlds_dataset_stats = dataset.dataset_statistics[-1]["action"]
        action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats, config.train.action_keys, config.train.action_shapes)
        action_config = config.train.action_config
        action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
        dataset = dataset.map(robomimic_transform, num_parallel_calls=48)

        pytorch_dataset = TorchRLDSDataset(dataset)
        train_loader = DataLoader(
            pytorch_dataset,
            batch_size=config.train.batch_size,
            num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        )


        # For RLDS, get batch from train loader to compute shapes
        data_loader_iter = iter(train_loader)
        rlds_batch = next(data_loader_iter)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=None,
            batch=rlds_batch,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True,
            config = config
        )
    else:
        # make sure the dataset exists
        eval_dataset_cfg = config.train.data[0]
        dataset_path = os.path.expanduser(eval_dataset_cfg["path"])

        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)


        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            batch=None,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True,
            config = config
        )
        # load training data
        trainset, validset = TrainUtils.load_data_for_training(
            config, obs_keys=shape_meta["all_obs_keys"])
        train_sampler = trainset.get_dataset_sampler()
        print("\n============= Training Dataset =============")
        print(trainset)
        print("")
        if validset is not None:
            print("\n============= Validation Dataset =============")
            print(validset)
            print("")

        # # maybe retreve statistics for normalizing observations
        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()

        # maybe retreve statistics for normalizing actions
        action_normalization_stats = trainset.get_action_normalization_stats()

        # initialize data loaders
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name, 
                render=False, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"], 
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
            envs[env.name] = env
            print(envs[env.name])

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
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # if checkpoint is specified, load in model weights
    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None:
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    ##### ------------------------------------------------------------------------------------ ######

    # TODO(Ashwin): Support loading validation splits for RLDS
    if ds_format != "r2d2_rlds" and config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
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
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    data_loader_iter = iter(train_loader)
    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log, data_loader_iter = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
            data_loader_iter = data_loader_iter
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
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
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
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
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
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # check if we need to save model MSE
        #TODO(Ashwin): support MSE Logging with RLDS dataloading
        if ds_format != "r2d2_rlds":
            should_save_mse = False
            if config.experiment.mse.enabled:
                if config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0:
                    should_save_mse = True
                if config.experiment.mse.on_save_ckpt and should_save_ckpt:
                    should_save_mse = True
            if should_save_mse:
                print("Computing MSE ...")
                if config.experiment.mse.visualize:
                    save_vis_dir = os.path.join(vis_dir, epoch_ckpt_name)
                else:
                    save_vis_dir = None
                mse_log, vis_log = model.compute_mse_visualize(
                    trainset,
                    validset,
                    num_samples=config.experiment.mse.num_samples,
                    savedir=save_vis_dir,
                )
                for k, v in mse_log.items():
                    data_logger.record("{}".format(k), v, epoch)
                
                for k, v in vis_log.items():
                    data_logger.record("{}".format(k), v, epoch, data_type='image')


                print("MSE Log Epoch {}".format(epoch))
                print(json.dumps(mse_log, sort_keys=True, indent=4))
        else:
            should_save_batch_samples = False
            # TODO(Ashwin): eventually clean up to use different config parameters for
            # batch visualization vs. mse visualization
            if config.experiment.mse.enabled:
                if config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0:
                    should_save_batch_samples = True
                if config.experiment.mse.on_save_ckpt and should_save_ckpt:
                    should_save_batch_samples = True
            if should_save_batch_samples:
                print("Computing Batch Visualization ...")
                if config.experiment.mse.visualize:
                    save_vis_dir = os.path.join(vis_dir, epoch_ckpt_name)
                else:
                    save_vis_dir = None
                vis_log = model.compute_batch_visualize(
                    batch=rlds_batch,
                    num_samples=config.experiment.mse.num_samples,
                    savedir=save_vis_dir,
                )

                for k, v in vis_log.items():
                    data_logger.record("{}".format(k), v, epoch, data_type='image')

                print("Batch Log Epoch {}".format(epoch))
        
        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:    
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


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

    if config.train.data_format != "r2d2_rlds" and args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 200
        config.experiment.mse.every_n_epochs = 2
        config.experiment.save.every_n_epochs = 1

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


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

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)
