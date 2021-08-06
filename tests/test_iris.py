"""
Test script for IRIS algorithms. Each test trains a variant of IRIS
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
    config = TestUtils.get_base_config(algo_name="iris")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm, 
    # for example iris)
    config.observation.value_planner.planner.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.value_planner.planner.modalities.obs.image = []

    config.observation.value_planner.planner.modalities.subgoal.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.value_planner.planner.modalities.subgoal.image = []

    config.observation.value_planner.value.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.value_planner.value.modalities.obs.image = []

    config.observation.actor.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.actor.modalities.obs.image = []

    # by default, basic N(0, 1) prior for both planner VAE and BCQ cVAE
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = False
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = False
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False

    return config


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("iris")
def iris_modifier_1(config):
    # no-op
    return config


@register_mod("iris, planner vae Gaussian prior (obs-independent)")
def iris_modifier_2(config):
    # learn parameters of Gaussian prior (obs-independent)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.planner.vae.prior.use_gmm = False
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae Gaussian prior (obs-dependent)")
def iris_modifier_3(config):
    # learn parameters of Gaussian prior (obs-dependent)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = True
    config.algo.value_planner.planner.vae.prior.use_gmm = False
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae GMM prior (obs-independent, weights-fixed)")
def iris_modifier_4(config):
    # learn parameters of GMM prior (obs-independent, weights-fixed)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.planner.vae.prior.use_gmm = True
    config.algo.value_planner.planner.vae.prior.gmm_learn_weights = False
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae GMM prior (obs-independent, weights-learned)")
def iris_modifier_5(config):
    # learn parameters of GMM prior (obs-independent, weights-learned)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.planner.vae.prior.use_gmm = True
    config.algo.value_planner.planner.vae.prior.gmm_learn_weights = True
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae GMM prior (obs-dependent, weights-fixed)")
def iris_modifier_6(config):
    # learn parameters of GMM prior (obs-dependent, weights-fixed)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = True
    config.algo.value_planner.planner.vae.prior.use_gmm = True
    config.algo.value_planner.planner.vae.prior.gmm_learn_weights = False
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae GMM prior (obs-dependent, weights-learned)")
def iris_modifier_7(config):
    # learn parameters of GMM prior (obs-dependent, weights-learned)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = True
    config.algo.value_planner.planner.vae.prior.use_gmm = True
    config.algo.value_planner.planner.vae.prior.gmm_learn_weights = True
    config.algo.value_planner.planner.vae.prior.use_categorical = False
    return config


@register_mod("iris, planner vae uniform categorical prior")
def iris_modifier_8(config):
    # uniform categorical prior
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = False
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.planner.vae.prior.use_gmm = False
    config.algo.value_planner.planner.vae.prior.use_categorical = True
    return config


@register_mod("iris, planner vae categorical prior (obs-independent)")
def iris_modifier_9(config):
    # learn parameters of categorical prior (obs-independent)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = False
    config.algo.value_planner.planner.vae.prior.use_gmm = False
    config.algo.value_planner.planner.vae.prior.use_categorical = True
    return config


@register_mod("iris, planner vae categorical prior (obs-dependent)")
def iris_modifier_10(config):
    # learn parameters of categorical prior (obs-dependent)
    config.algo.value_planner.planner.vae.enabled = True
    config.algo.value_planner.planner.vae.prior.learn = True
    config.algo.value_planner.planner.vae.prior.is_conditioned = True
    config.algo.value_planner.planner.vae.prior.use_gmm = False
    config.algo.value_planner.planner.vae.prior.use_categorical = True
    return config


@register_mod("iris, bcq gmm")
def iris_modifier_11(config):
    # bcq action sampler is GMM
    config.algo.value_planner.value.action_sampler.gmm.enabled = True
    config.algo.value_planner.value.action_sampler.vae.enabled = False
    return config


@register_mod("iris, bcq distributional")
def iris_modifier_12(config):
    # bcq value function is distributional
    config.algo.value_planner.value.critic.distributional.enabled = True
    config.algo.value_planner.value.critic.value_bounds = [-100., 100.]
    return config

@register_mod("iris, bcq cVAE Gaussian prior (obs-independent)")
def iris_modifier_13(config):
    # learn parameters of Gaussian prior (obs-independent)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE Gaussian prior (obs-dependent)")
def iris_modifier_14(config):
    # learn parameters of Gaussian prior (obs-dependent)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE GMM prior (obs-independent, weights-fixed)")
def iris_modifier_15(config):
    # learn parameters of GMM prior (obs-independent, weights-fixed)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = True
    config.algo.value_planner.value.action_sampler.vae.prior.gmm_learn_weights = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE GMM prior (obs-independent, weights-learned)")
def iris_modifier_16(config):
    # learn parameters of GMM prior (obs-independent, weights-learned)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = True
    config.algo.value_planner.value.action_sampler.vae.prior.gmm_learn_weights = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE GMM prior (obs-dependent, weights-fixed)")
def iris_modifier_17(config):
    # learn parameters of GMM prior (obs-dependent, weights-fixed)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = True
    config.algo.value_planner.value.action_sampler.vae.prior.gmm_learn_weights = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE GMM prior (obs-dependent, weights-learned)")
def iris_modifier_18(config):
    # learn parameters of GMM prior (obs-dependent, weights-learned)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = True
    config.algo.value_planner.value.action_sampler.vae.prior.gmm_learn_weights = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = False
    return config


@register_mod("iris, bcq cVAE uniform categorical prior")
def iris_modifier_19(config):
    # uniform categorical prior
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = False
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = True
    return config


@register_mod("iris, bcq cVAE categorical prior (obs-independent)")
def iris_modifier_20(config):
    # learn parameters of categorical prior (obs-independent)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = True
    return config


@register_mod("iris, bcq cVAE categorical prior (obs-dependent)")
def iris_modifier_21(config):
    # learn parameters of categorical prior (obs-dependent)
    config.algo.value_planner.value.action_sampler.vae.enabled = True
    config.algo.value_planner.value.action_sampler.vae.prior.learn = True
    config.algo.value_planner.value.action_sampler.vae.prior.is_conditioned = True
    config.algo.value_planner.value.action_sampler.vae.prior.use_gmm = False
    config.algo.value_planner.value.action_sampler.vae.prior.use_categorical = True
    return config


def test_iris(silence=True):
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

    test_iris(silence=(not args.verbose))
