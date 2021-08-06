"""
Test script for CQL algorithms. Each test trains a variant of CQL
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
    Base config for testing CQL algorithms.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="cql")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm, 
    # for example HBC)
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.modalities.obs.image = []

    # by default, vanilla CQL
    config.algo.actor.bc_start_steps = 40           # BC training initially
    config.algo.critic.target_q_gap = 5.0           # use automatic cql tuning
    config.algo.actor.target_entropy = "default"    # use automatic entropy tuning

    # lower batch size to 100 to accomodate small test dataset
    config.train.batch_size = 100

    return config


def convert_config_for_images(config):
    """
    Modify config to use image observations.
    """

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.train.batch_size = 16

    # replace object with image modality
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    config.observation.modalities.obs.image = ["agentview_image"]

    # set up visual encoders
    config.observation.encoder.visual_core = 'ResNet18Conv'
    config.observation.encoder.visual_core_kwargs = Config()
    config.observation.encoder.obs_randomizer_class = None
    config.observation.encoder.visual_feature_dimension = 64
    config.observation.encoder.use_spatial_softmax = True
    config.observation.encoder.spatial_softmax_kwargs.num_kp = 32
    config.observation.encoder.spatial_softmax_kwargs.learnable_temperature = False
    config.observation.encoder.spatial_softmax_kwargs.temperature = 1.0
    config.observation.encoder.spatial_softmax_kwargs.noise_std = 0.

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


@register_mod("cql-fixed-entropy")
def cql_entropy_modifier(config):
    config.algo.actor.target_entropy = None
    return config


@register_mod("cql-fixed-q-gap")
def cql_q_gap_modifier(config):
    config.algo.critic.target_q_gap = None
    config.algo.critic.cql_weight = 1.0
    return config


@register_mod("cql-fixed-gaussian")
def cql_gaussian_modifier(config):
    config.algo.actor.net.gaussian.fixed_std = True
    return config


# add image version of all tests
image_modifiers = OrderedDict()
for test_name in MODIFIERS:
    lst = test_name.split("-")
    name = "-".join(lst[:1] + ["image"] + lst[1:])
    image_modifiers[name] = make_image_modifier(MODIFIERS[test_name])
MODIFIERS.update(image_modifiers)


# test for image crop randomization
@register_mod("cql-image-crop")
def cql_image_crop_modifier(config):
    config = convert_config_for_images(config)
    config.observation.encoder.obs_randomizer_class = 'CropRandomizer'  # observation randomizer class
    config.observation.encoder.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.obs_randomizer_kwargs.pos_enc = False
    return config


def test_cql(silence=True):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr()
        with context:
            base_config = get_algo_base_config()
            res_str = TestUtils.test_run(base_config=base_config, config_modifier=MODIFIERS[test_name])
        print("{}: {}".format(test_name, res_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_cql(silence=(not args.verbose))
