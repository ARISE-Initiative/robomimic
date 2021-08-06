"""
Tests for a handful of scripts. Excludes stdout output by 
default (pass --verbose to see stdout output).
"""
import argparse
import traceback
import h5py
import numpy as np
import torch
from collections import OrderedDict
from termcolor import colored

import robomimic
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import Config
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr
from robomimic.scripts.train import train
from robomimic.scripts.playback_dataset import playback_dataset
from robomimic.scripts.run_trained_agent import run_trained_agent


def get_checkpoint_to_test():
    """
    Run a quick training run to get a checkpoint. This function runs a basic bc-image
    training run. Image modality is used for a harder test case for the run agent
    script, which will need to also try writing image observations to the rollout
    dataset.
    """

    # prepare image training run
    config = TestUtils.get_base_config(algo_name="bc")

    def image_modifier(conf):
        # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
        conf.train.hdf5_cache_mode = "low_dim"
        conf.train.num_data_workers = 0
        conf.train.batch_size = 16

        # replace object with image modality
        conf.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        conf.observation.modalities.obs.image = ["agentview_image"]

        # set up visual encoders
        conf.observation.encoder.visual_core = 'ResNet18Conv'
        conf.observation.encoder.visual_core_kwargs = Config()
        conf.observation.encoder.obs_randomizer_class = None
        conf.observation.encoder.visual_feature_dimension = 64
        conf.observation.encoder.use_spatial_softmax = True
        conf.observation.encoder.spatial_softmax_kwargs.num_kp = 32
        conf.observation.encoder.spatial_softmax_kwargs.learnable_temperature = False
        conf.observation.encoder.spatial_softmax_kwargs.temperature = 1.0
        conf.observation.encoder.spatial_softmax_kwargs.noise_std = 0.0
        return conf

    config = TestUtils.config_from_modifier(base_config=config, config_modifier=image_modifier)

    # run training
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    train(config, device=device)

    # return checkpoint
    ckpt_path = TestUtils.checkpoint_path_from_test_run()
    return ckpt_path


def test_playback_script(silence=True, use_actions=False, use_obs=False):
    context = silence_stdout() if silence else dummy_context_mgr()
    with context:

        try:
            # setup args and run script
            args = argparse.Namespace()
            args.dataset = TestUtils.example_dataset_path()
            args.filter_key = None
            args.n = 3 # playback 3 demonstrations
            args.use_actions = use_actions
            args.use_obs = use_obs
            args.render = False
            args.video_path = TestUtils.temp_video_path() # dump video
            args.video_skip = 5
            if use_obs:
                # camera observation names
                args.render_image_names = ["agentview_image", "robot0_eye_in_hand_image"]
            else:
                # camera names
                args.render_image_names = ["agentview", "robot0_eye_in_hand"]
            args.first = False
            playback_dataset(args)

            # indicate success
            ret = colored("passed!", "green")

        except Exception as e:
            # indicate failure by returning error string
            ret = colored("failed with error:\n{}\n\n{}".format(e, traceback.format_exc()), "red")

        # delete output video
        TestUtils.maybe_remove_file(TestUtils.temp_video_path())

    act_str = "-action_playback" if use_actions else ""
    obs_str = "-obs" if use_obs else ""
    test_name = "playback-script{}{}".format(act_str, obs_str)
    print("{}: {}".format(test_name, ret))


def test_run_agent_script(silence=True):
    context = silence_stdout() if silence else dummy_context_mgr()
    with context:

        try:
            # get a model checkpoint
            ckpt_path = get_checkpoint_to_test()

            # setup args and run script
            args = argparse.Namespace()
            args.agent = ckpt_path
            args.n_rollouts = 3 # 3 rollouts
            args.horizon = 10 # short rollouts - 10 steps
            args.env = None
            args.render = False
            args.video_path = TestUtils.temp_video_path() # dump video
            args.video_skip = 5
            args.camera_names = ["agentview", "robot0_eye_in_hand"]
            args.dataset_path = TestUtils.temp_dataset_path() # dump dataset
            args.dataset_obs = True
            args.seed = 0
            run_trained_agent(args)

            # simple sanity check for shape of image observations in rollout dataset
            f = h5py.File(TestUtils.temp_dataset_path(), "r")
            assert f["data/demo_1/obs/agentview_image"].shape == (10, 84, 84, 3)
            assert f["data/demo_1/obs/agentview_image"].dtype == np.uint8
            f.close()

            # indicate success
            ret = colored("passed!", "green")

        except Exception as e:
            # indicate failure by returning error string
            ret = colored("failed with error:\n{}\n\n{}".format(e, traceback.format_exc()), "red")

        # delete trained model directory, output video, and output dataset
        TestUtils.maybe_remove_dir(TestUtils.temp_model_dir_path())
        TestUtils.maybe_remove_file(TestUtils.temp_video_path())
        TestUtils.maybe_remove_file(TestUtils.temp_dataset_path())

    test_name = "run-agent-script"
    print("{}: {}".format(test_name, ret))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_playback_script(silence=(not args.verbose), use_actions=False, use_obs=False)
    test_playback_script(silence=(not args.verbose), use_actions=True, use_obs=False)
    test_playback_script(silence=(not args.verbose), use_actions=False, use_obs=True)
    test_run_agent_script(silence=(not args.verbose))
