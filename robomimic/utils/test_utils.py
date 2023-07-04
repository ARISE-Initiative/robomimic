"""
Utilities for testing algorithm implementations - used mainly by scripts in tests directory.
"""
import os
import json
import shutil
import traceback
from termcolor import colored

import numpy as np
import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import Config, config_factory
from robomimic.scripts.train import train


def maybe_remove_dir(dir_to_remove):
    """
    Remove directory if it exists.

    Args:
        dir_to_remove (str): path to directory to remove
    """
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)


def maybe_remove_file(file_to_remove):
    """
    Remove file if it exists.

    Args:
        file_to_remove (str): path to file to remove
    """
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)


def example_dataset_path():
    """
    Path to dataset to use for testing and example purposes. It should
    exist under the tests/assets directory, and will be downloaded 
    from a server if it does not exist.
    """
    dataset_folder = os.path.join(robomimic.__path__[0], "../tests/assets/")
    dataset_path = os.path.join(dataset_folder, "test_v141.hdf5")
    if not os.path.exists(dataset_path):
        print("\nWARNING: test hdf5 does not exist! Downloading from server...")
        os.makedirs(dataset_folder, exist_ok=True)
        FileUtils.download_url(
            url="http://downloads.cs.stanford.edu/downloads/rt_benchmark/test_v141.hdf5", 
            download_dir=dataset_folder,
        )
    return dataset_path


def example_momart_dataset_path():
    """
    Path to momart dataset to use for testing and example purposes. It should
    exist under the tests/assets directory, and will be downloaded
    from a server if it does not exist.
    """
    dataset_folder = os.path.join(robomimic.__path__[0], "../tests/assets/")
    dataset_path = os.path.join(dataset_folder, "test_momart.hdf5")
    if not os.path.exists(dataset_path):
        user_response = input("\nWARNING: momart test hdf5 does not exist! We will download sample dataset. "
                              "This will take 0.6GB space. Proceed? y/n\n")
        assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

        print("\nDownloading from server...")

        os.makedirs(dataset_folder, exist_ok=True)
        FileUtils.download_url(
            url="http://downloads.cs.stanford.edu/downloads/rt_mm/sample/test_momart.hdf5",
            download_dir=dataset_folder,
        )
    return dataset_path


def temp_model_dir_path():
    """
    Path to a temporary model directory to write to for testing and example purposes.
    """
    return os.path.join(robomimic.__path__[0], "../tests/tmp_model_dir")


def temp_dataset_path():
    """
    Defines default dataset path to write to for testing.
    """
    return os.path.join(robomimic.__path__[0], "../tests/", "tmp.hdf5")


def temp_video_path():
    """
    Defines default video path to write to for testing.
    """
    return os.path.join(robomimic.__path__[0], "../tests/", "tmp.mp4")


def get_base_config(algo_name):
    """
    Base config for testing algorithms.

    Args:
        algo_name (str): name of algorithm - loads the corresponding json
            from the config templates directory
    """

    # we will load and override defaults from template config
    base_config_path = os.path.join(robomimic.__path__[0], "exps/templates/{}.json".format(algo_name))
    with open(base_config_path, 'r') as f:
        config = Config(json.load(f))

    # small dataset with a handful of trajectories
    config.train.data = example_dataset_path()

    # temporary model dir
    model_dir = temp_model_dir_path()
    maybe_remove_dir(model_dir)
    config.train.output_dir = model_dir

    # train and validate for 3 gradient steps
    config.experiment.name = "test"
    config.experiment.validate = True
    config.experiment.epoch_every_n_steps = 3
    config.experiment.validation_epoch_every_n_steps = 3
    config.train.num_epochs = 1

    # default train and validation filter keys
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"

    # ensure model saving, rollout, and offscreen video rendering are tested too
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 1
    config.experiment.rollout.enabled = True
    config.experiment.rollout.rate = 1
    config.experiment.rollout.n = 1
    config.experiment.rollout.horizon = 10
    config.experiment.render_video = True

    # turn off logging to stdout, since that can interfere with testing code outputs
    config.experiment.logging.terminal_output_to_txt = False

    # test cuda (if available)
    config.train.cuda = True

    return config


def config_from_modifier(base_config, config_modifier):
    """
    Helper function to load a base config, modify it using
    the passed @config modifier function, and finalize it
    for training.

    Args:
        base_config (BaseConfig instance): starting config object that is
            loaded (to change algorithm config defaults), and then modified
            with @config_modifier

        config_modifier (function): function that takes a config object as
            input, and modifies it
    """

    # algo name to default config for this algorithm
    algo_name = base_config["algo_name"]
    config = config_factory(algo_name)

    # update config with the settings specified in the base config
    with config.unlocked():
        config.update(base_config)

        # modify the config and finalize it for training (no more modifications allowed)
        config = config_modifier(config)

    return config


def checkpoint_path_from_test_run():
    """
    Helper function that gets the path of a model checkpoint after a test training run is finished.
    """
    exp_dir = os.path.join(temp_model_dir_path(), "test")
    time_dir_names = [f.name for f in os.scandir(exp_dir) if f.is_dir()]
    assert len(time_dir_names) == 1
    path_to_models = os.path.join(exp_dir, time_dir_names[0], "models")
    epoch_name = [f.name for f in os.scandir(path_to_models) if f.name.startswith("model")][0]
    return os.path.join(path_to_models, epoch_name)


def test_eval_agent_from_checkpoint(ckpt_path, device):
    """
    Test loading a model from checkpoint and running a rollout with the 
    trained agent for a small number of steps.

    Args:
        ckpt_path (str): path to a checkpoint pth file

        device (torch.Device): torch device
    """

    # get policy and env from checkpoint
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, verbose=True)

    # run a test rollout
    ob_dict = env.reset()
    policy.start_episode()
    for _ in range(15):
        ac = policy(ob=ob_dict)
        ob_dict, r, done, _ = env.step(ac)


def test_run(base_config, config_modifier):
    """
    Takes a base_config and config_modifier (function that modifies a passed Config object)
    and runs training as a test. It also takes the trained checkpoint, tries to load the
    policy and environment from the checkpoint, and run an evaluation rollout. Returns
    a string that is colored green if the run finished successfully without any issues,
    and colored red if an error occurred. If an error occurs, the traceback is included
    in the string.

    Args:
        base_config (BaseConfig instance): starting config object that is
            loaded (to change algorithm config defaults), and then modified
            with @config_modifier

        config_modifier (function): function that takes a config object as
            input, and modifies it

    Returns:
        ret (str): a green "passed!" string, or a red "failed with error" string that contains
            the traceback
    """
    try:
        # get config
        config = config_from_modifier(base_config=base_config, config_modifier=config_modifier)

        # set torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

        # run training
        train(config, device=device)

        # test evaluating a trained agent using saved checkpoint
        ckpt_path = checkpoint_path_from_test_run()
        test_eval_agent_from_checkpoint(ckpt_path, device=device)

        # indicate success
        ret = colored("passed!", "green")

    except Exception as e:
        # indicate failure by returning error string
        ret = colored("failed with error:\n{}\n\n{}".format(e, traceback.format_exc()), "red")

    # make sure model directory is cleaned up before returning from this function
    maybe_remove_dir(temp_model_dir_path())

    return ret
