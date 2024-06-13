"""
Test script for IQL algorithms. Each test trains a variant of IQL
for a handful of gradient steps and tries one rollout with
the model. Excludes stdout output by default (pass --verbose
to see stdout output).
"""

import argparse
from collections import OrderedDict

import robomimic
from robomimic.config import Config
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


def get_algo_base_config():
    """
    Base config for testing IQL algorithms.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="iql")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm,
    # for example HBC)
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    config.observation.modalities.obs.rgb = []

    return config


def convert_config_for_images(config):
    """
    Modify config to use image observations.
    """

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.train.batch_size = 16

    # replace object with rgb modality
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    config.observation.modalities.obs.rgb = ["agentview_image"]

    # set up visual encoders
    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"  # ResNet backbone for image observations (unused if no image observations)
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = (
        False  # kwargs for visual core
    )
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = (
        "SpatialSoftmax"  # Alternate options are "SpatialMeanPool" or None (no pooling)
    )
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = (
        32  # Default arguments for "SpatialSoftmax"
    )
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = (
        False  # Default arguments for "SpatialSoftmax"
    )
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = (
        1.0  # Default arguments for "SpatialSoftmax"
    )
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

    # observation randomizer class - set to None to use no randomization, or 'CropRandomizer' to use crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    return config


def make_image_modifier(config_modifier):
    """
    turn a config modifier into its image version. Note that
    this explicit function definition is needed for proper
    scoping of @config_modifier
    """
    return lambda x: config_modifier(convert_config_for_images(x))


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()


def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier

    return decorator


@register_mod("iql-gaussian")
def iql_default_modifier(config):
    config.algo.actor.net.type = "gaussian"
    return config


@register_mod("iql-gmm")
def iql_default_modifier(config):
    config.algo.actor.net.type = "gmm"
    return config


@register_mod("iql-clip-adv")
def iql_default_modifier(config):
    config.algo.adv.clip_adv_value = 1.0
    return config


# add image version of all tests
image_modifiers = OrderedDict()
for test_name in MODIFIERS:
    lst = test_name.split("-")
    name = "-".join(lst[:1] + ["rgb"] + lst[1:])
    image_modifiers[name] = make_image_modifier(MODIFIERS[test_name])
MODIFIERS.update(image_modifiers)


# test for image crop randomization
@register_mod("iql-image-crop")
def iql_image_crop_modifier(config):
    config = convert_config_for_images(config)

    # observation randomizer class - using Crop randomizer
    config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"

    # kwargs for observation randomizers (for the CropRandomizer, this is size and number of crops)
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
    return config


def test_iql(silence=True):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr()
        with context:
            base_config = get_algo_base_config()
            res_str = TestUtils.test_run(
                base_config=base_config, config_modifier=MODIFIERS[test_name]
            )
        print("{}: {}".format(test_name, res_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_iql(silence=(not args.verbose))
