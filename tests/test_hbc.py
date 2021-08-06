"""
Test script for HBC algorithm. Each test trains a variant of HBC
for a handful of gradient steps and tries one rollout with 
the model. Excludes stdout output by default (pass --verbose
to see stdout output).
"""
import argparse
from collections import OrderedDict

import robomimic
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


def get_algo_base_config():
    """
    Base config for testing BCQ algorithms.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="hbc")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm, 
    # for example HBC)
    config.observation.planner.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.planner.modalities.obs.image = []

    config.observation.planner.modalities.subgoal.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.planner.modalities.subgoal.image = []

    config.observation.actor.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.actor.modalities.obs.image = []

    # by default, planner is deterministic prediction
    config.algo.planner.vae.enabled = False

    return config


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("hbc")
def hbc_modifier(config):
    # no-op
    return config


@register_mod("hbc-vae, N(0, 1) prior")
def hbc_vae_modifier_1(config):
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = False
    config.algo.planner.vae.prior.is_conditioned = False
    return config


@register_mod("hbc-vae, Gaussian prior (obs-independent)")
def hbc_vae_modifier_2(config):
    # learn parameters of Gaussian prior (obs-independent)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = False
    config.algo.planner.vae.prior.use_gmm = False
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, Gaussian prior (obs-dependent)")
def hbc_vae_modifier_3(config):
    # learn parameters of Gaussian prior (obs-dependent)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = True
    config.algo.planner.vae.prior.use_gmm = False
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, GMM prior (obs-independent, weights-fixed)")
def hbc_vae_modifier_4(config):
    # learn parameters of GMM prior (obs-independent, weights-fixed)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = False
    config.algo.planner.vae.prior.use_gmm = True
    config.algo.planner.vae.prior.gmm_learn_weights = False
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, GMM prior (obs-independent, weights-learned)")
def hbc_vae_modifier_5(config):
    # learn parameters of GMM prior (obs-independent, weights-learned)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = False
    config.algo.planner.vae.prior.use_gmm = True
    config.algo.planner.vae.prior.gmm_learn_weights = True
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, GMM prior (obs-dependent, weights-fixed)")
def hbc_vae_modifier_6(config):
    # learn parameters of GMM prior (obs-dependent, weights-fixed)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = True
    config.algo.planner.vae.prior.use_gmm = True
    config.algo.planner.vae.prior.gmm_learn_weights = False
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, GMM prior (obs-dependent, weights-learned)")
def hbc_vae_modifier_7(config):
    # learn parameters of GMM prior (obs-dependent, weights-learned)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = True
    config.algo.planner.vae.prior.use_gmm = True
    config.algo.planner.vae.prior.gmm_learn_weights = True
    config.algo.planner.vae.prior.use_categorical = False
    return config


@register_mod("hbc-vae, uniform categorical prior")
def hbc_vae_modifier_8(config):
    # uniform categorical prior
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = False
    config.algo.planner.vae.prior.is_conditioned = False
    config.algo.planner.vae.prior.use_gmm = False
    config.algo.planner.vae.prior.use_categorical = True
    return config


@register_mod("hbc-vae, categorical prior (obs-independent)")
def hbc_vae_modifier_9(config):
    # learn parameters of categorical prior (obs-independent)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = False
    config.algo.planner.vae.prior.use_gmm = False
    config.algo.planner.vae.prior.use_categorical = True
    return config


@register_mod("hbc-vae, categorical prior (obs-dependent)")
def hbc_vae_modifier_10(config):
    # learn parameters of categorical prior (obs-dependent)
    config.algo.planner.vae.enabled = True
    config.algo.planner.vae.prior.learn = True
    config.algo.planner.vae.prior.is_conditioned = True
    config.algo.planner.vae.prior.use_gmm = False
    config.algo.planner.vae.prior.use_categorical = True
    return config


def test_hbc(silence=True):
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

    test_hbc(silence=(not args.verbose))
