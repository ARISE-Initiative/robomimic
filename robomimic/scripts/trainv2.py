# -*- coding: utf-8 -*-
"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    resume_checkpoint (str): if provided, path to a checkpoint file to resume training from.

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

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings


def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)


    # --- Checkpoint Loading Initialization ---
    ckpt_path = config.train.resume_checkpoint_path # Get path from config
    start_epoch = 1
    ckpt_obs_stats = None
    loaded_optimizer_states = None
    loaded_model_state = None

    if ckpt_path is not None:
        print(f"\n=== Resuming training from checkpoint: {ckpt_path} ===")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device) # Load checkpoint dict
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at {ckpt_path}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            raise

        # Load training state
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("WARNING: Checkpoint does not contain epoch number. Starting from epoch 1.")
            # start_epoch remains 1

        # Load model state dict - will be applied after model creation
        if "model" in checkpoint:
            loaded_model_state = checkpoint["model"]
            print("Found model state_dict in checkpoint.")
        else:
            print("WARNING: Checkpoint does not contain model state_dict.")

        # Load optimizer state dict(s) - will be applied after model/optimizer creation
        if "optimizer" in checkpoint:
            loaded_optimizer_states = checkpoint["optimizer"]
            print("Found optimizer state_dict(s) in checkpoint.")
        else:
            print("WARNING: Checkpoint does not contain optimizer state_dict(s). Optimizer(s) will be reset.")

        # Load observation normalization stats if they exist
        if "obs_normalization_stats" in checkpoint:
             ckpt_obs_stats = checkpoint["obs_normalization_stats"]
             print("Found observation normalization stats in checkpoint.")
        # We don't load config from checkpoint to allow overriding, but could add checks for compatibility
        # if "config" in checkpoint: loaded_ckpt_config = checkpoint["config"]

    else: # Not resuming
        print("\n=== Starting new training run ===")
    # --- End Checkpoint Loading Initialization ---


    print("\n============= Config for this run =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    # Ensure checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # Load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # Create environment(s)
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
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
                use_depth_obs=shape_meta["use_depths"],
            )
            env = EnvUtils.wrap_env_from_config(env, config=config)
            envs[env.name] = env
            print(envs[env.name])
    print("")

    # Setup data logger
    # If resuming, logger might append to existing files if experiment name is the same
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )

    # Create model instance
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # --- Load Model and Optimizer States If Resuming ---
    if loaded_model_state is not None:
        try:
            model.load_state_dict(loaded_model_state)
            print("Successfully loaded model state_dict from checkpoint.")
        except Exception as e:
            print(f"WARNING: Failed to load model state_dict: {e}. Training with initialized model.")

    if loaded_optimizer_states is not None:
        try:
            model.load_optimizer_states(loaded_optimizer_states) # Assuming model has this method
            print("Successfully loaded optimizer state_dict(s) from checkpoint.")
        except AttributeError:
            print("WARNING: Model class does not have 'load_optimizer_states' method. Trying common structure.")
            # Attempt common structure (e.g., policy optimizer)
            try:
                if isinstance(loaded_optimizer_states, dict) and 'policy' in loaded_optimizer_states and 'policy' in model.optimizers:
                     model.optimizers['policy'].load_state_dict(loaded_optimizer_states['policy'])
                     print("Successfully loaded 'policy' optimizer state.")
                     # Load other optimizers if present
                elif isinstance(loaded_optimizer_states, dict) and not model.optimizers: # Single optimizer case
                    # Assume model.optimizer is the optimizer if model.optimizers is empty/None
                     model.optimizer.load_state_dict(loaded_optimizer_states)
                     print("Successfully loaded single optimizer state.")
                else:
                     print("WARNING: Could not determine how to load optimizer state. Optimizer(s) reset.")
            except Exception as e:
                 print(f"WARNING: Failed to load optimizer state_dict(s): {e}. Optimizer(s) reset.")
        except Exception as e:
            print(f"WARNING: Failed to load optimizer state_dict(s): {e}. Optimizer(s) reset.")
    # --- End Loading States ---

    # Save the config for this run (possibly resuming)
    config_save_path = os.path.join(log_dir, '..', 'config.json')
    try:
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved config for this run to {config_save_path}")
    except Exception as e:
        print(f"WARNING: Failed to save config.json: {e}")


    print("\n============= Model Summary =============")
    print(model)
    print("")

    # Load training data
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

    # Handle observation normalization stats
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        if ckpt_obs_stats is not None: # Use stats from checkpoint if available
             obs_normalization_stats = ckpt_obs_stats
             print("Using observation normalization stats from checkpoint.")
        else: # Calculate stats from trainset if not resuming or not in checkpoint
            if ckpt_path is not None: # Resuming but no stats in checkpoint
                print("WARNING: Resuming training but checkpoint did not contain observation stats. Calculating fresh stats from dataset.")
            obs_normalization_stats = trainset.get_obs_normalization_stats()
            print("Calculated observation normalization stats from training set.")

    # Initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    valid_loader = None
    if config.experiment.validate and validset is not None:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size, # Use train batch size for validation? Maybe smaller?
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )

    # Print warnings
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # Main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps


    # --- Adjust loop start ---
    print(f"\n=== Starting training loop from epoch {start_epoch} to {config.train.num_epochs} ===\n")
    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats, # Pass potentially loaded stats
        )
        model.on_epoch_end(epoch) # Call model's epoch end hook

        # Setup checkpoint path naming
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # Check checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
            # Save last epoch always?
            if epoch == config.train.num_epochs:
                 print("Saving checkpoint for last epoch.")
                 should_save_ckpt = True

        ckpt_reason = None
        if should_save_ckpt and not (time_check or epoch_check or epoch_list_check):
             ckpt_reason = "last epoch" # Identify reason if forced by last epoch
        elif should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time/epoch" # General reason for periodic saves

        # Log training step results
        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            data_key = f"Train/{k}"
            if k.startswith("Time_"):
                data_key = f"Timing_Stats/Train_{k[5:]}"
            data_logger.record(data_key, v, epoch)

        # Evaluate on validation set
        if config.experiment.validate and valid_loader is not None:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps, obs_normalization_stats=obs_normalization_stats)
            for k, v in step_log.items():
                data_key = f"Valid/{k}"
                if k.startswith("Time_"):
                    data_key = f"Timing_Stats/Valid_{k[5:]}"
                data_logger.record(data_key, v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # Save checkpoint based on validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    # Update epoch name included in filename
                    best_valid_ckpt_name = f"{epoch_ckpt_name}_best_validation_{best_valid_loss:.4f}" # Use current epoch name base
                    print(f"New best validation loss: {best_valid_loss}. Marking for save.")
                    should_save_ckpt = True
                    new_ckpt_reason = "valid"
                    # Overwrite reason if it's better than time/epoch reason
                    ckpt_reason = new_ckpt_reason if ckpt_reason is None else ckpt_reason

        # Evaluate with rollouts
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or \
                        (should_save_ckpt and ckpt_reason == "time/epoch") or \
                        (epoch == config.train.num_epochs) # Also run on last epoch

        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
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

            # Log rollout results
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    data_key = f"Rollout/{k}/{env_name}"
                    if k.startswith("Time_"):
                        data_key = f"Timing_Stats/Rollout_{env_name}_{k[5:]}"
                    # Ensure stats logging happens correctly
                    data_logger.record(data_key, v, epoch, log_stats=(not k.startswith("Time_")))


                print(f"\nEpoch {epoch} Rollouts took {rollout_logs.get('time', 'N/A')}s (avg) with results:")
                print(f'Env: {env_name}')
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # Checkpoint saving logic based on rollouts
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                # Use current base name, best indicator will be appended
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            # Name might be updated with best return/success suffix
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            if updated_stats["should_save_ckpt"] and config.experiment.save.enabled:
                should_save_ckpt = True
                if updated_stats["ckpt_reason"] is not None:
                     # Prioritize rollout reasons over time/epoch reasons
                     ckpt_reason = updated_stats["ckpt_reason"] if ckpt_reason != "valid" else ckpt_reason

        # Video saving logic
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                try:
                    os.remove(video_paths[env_name])
                except FileNotFoundError:
                    pass # Video might not have been created

        # --- Save Checkpoint ---
        if should_save_ckpt:
            # Decide final checkpoint name based on priority (best > time/epoch > last)
            final_ckpt_name = epoch_ckpt_name # Potentially includes _best_... suffix
            if ckpt_reason == "valid": # Use the name generated during validation check
                 final_ckpt_name = best_valid_ckpt_name # includes loss value

            ckpt_save_path = os.path.join(ckpt_dir, final_ckpt_name + ".pth")

            # Create comprehensive state dictionary
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                # Save optimizer states (handle potential variations)
                'optimizer': model.get_optimizer_states() if hasattr(model, 'get_optimizer_states') else getattr(model, 'optimizers', {}),
                'config': config.dump(), # Save config dictionary
                'env_meta': env_meta,
                'shape_meta': shape_meta,
                'obs_normalization_stats': obs_normalization_stats,
                # Optionally save current best metrics for exact state recovery
                # 'best_valid_loss': best_valid_loss,
                # 'best_return': best_return,
                # 'best_success_rate': best_success_rate,
            }
            try:
                 torch.save(save_dict, ckpt_save_path)
                 print(f"Saved checkpoint to: {ckpt_save_path} (Reason: {ckpt_reason})")

                 # Create a 'latest.pth' link/copy for easy access
                 latest_path = os.path.join(ckpt_dir, "latest.pth")
                 try:
                     # Use shutil.copy to make it work on Windows too
                     shutil.copy(ckpt_save_path, latest_path)
                     # os.symlink(os.path.basename(ckpt_save_path), latest_path) # Doesn't work reliably on Windows
                 except Exception as e:
                     print(f"WARNING: Could not create/update latest.pth link: {e}")

            except Exception as e:
                 print(f"ERROR: Failed to save checkpoint {ckpt_save_path}: {e}")

            # # --- Removed the old TrainUtils.save_model call ---
            # TrainUtils.save_model(...)
        # --- End Save Checkpoint ---

        # Log memory usage
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000) # MB
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # End of training loop
    print("="*50 + "\nTraining finished.\n" + "="*50)
    data_logger.close()


def main(args):

    # Load config from file or defaults
    if args.config is not None:
        try:
            with open(args.config, 'r') as f:
                ext_cfg = json.load(f)
            algo_name = ext_cfg.get("algo_name", args.algo) # Get algo name from config or arg
            if algo_name is None:
                raise ValueError("Algorithm name ('algo_name') not found in config or provided via --algo.")
            config = config_factory(algo_name)
        except FileNotFoundError:
             print(f"Error: Config file not found at {args.config}")
             sys.exit(1)
        except json.JSONDecodeError:
             print(f"Error: Could not decode JSON from config file {args.config}")
             sys.exit(1)
        except Exception as e:
             print(f"Error loading config: {e}")
             sys.exit(1)

        # Update config with external json
        try:
            with config.values_unlocked():
                config.update(ext_cfg)
        except Exception as e:
             print(f"Error updating config with external JSON values: {e}")
             # Maybe print key that caused issue?
             sys.exit(1)
    elif args.algo is not None:
        config = config_factory(args.algo)
    else:
        parser.error("Either --config or --algo must be specified.")

    # Override specific config values with args
    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # --- Add resume checkpoint path to config ---
    # Use a default None value if not provided
    config.train.resume_checkpoint_path = args.resume_checkpoint
    # ---

    # Get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # Debug mode modifications
    if args.debug:
        print("--- Running in DEBUG mode ---")
        config.unlock()
        config.lock_keys()

        config.experiment.epoch_every_n_steps = 5
        config.experiment.validation_epoch_every_n_steps = 5
        config.train.num_epochs = 2 # Run only 2 epochs total

        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # Optionally reduce dataset size if possible? Depends on dataset class
        # config.train.num_data_workers = 0 # Easier debugging

        # Ensure output dir is temporary or clearly marked
        if config.train.output_dir != "/tmp/tmp_trained_models":
             config.train.output_dir = os.path.join(config.train.output_dir, "DEBUG_RUN")
        print(f"Debug output directory: {config.train.output_dir}")


    # Lock config before training
    config.lock()

    # Run training
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except KeyboardInterrupt:
        res_str = "run terminated by user (KeyboardInterrupt)."
        print("\n" + res_str)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
        print("\n" + res_str) # Also print traceback to console if logger is active
    finally:
        # Ensure logs are flushed etc.
        if 'logger' in locals() and isinstance(logger, PrintLogger):
             logger.close() # Close log file

    print(res_str) # Print final status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Existing Arguments ---
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None, # Make it optional if config provides algo_name
        help="(optional) name of algorithm to run. Required if not specified in --config.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) override the experiment name defined in the config.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) override the dataset path defined in the config.",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes."
    )

    # --- New Argument for Resuming ---
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="(optional) path to a checkpoint file (.pth) to resume training from."
    )
    # ---

    args = parser.parse_args()
    main(args)