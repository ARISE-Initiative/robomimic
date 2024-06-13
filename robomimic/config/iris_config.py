"""
Config for IRIS algorithm.
"""

from robomimic.config.bcq_config import BCQConfig
from robomimic.config.gl_config import GLConfig
from robomimic.config.bc_config import BCConfig
from robomimic.config.hbc_config import HBCConfig


class IRISConfig(HBCConfig):
    ALGO_NAME = "iris"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # One of ["separate", "cascade"]. In "separate" mode (default),
        # the planner and actor are trained independently and then the planner subgoal predictions are
        # used to condition the actor at test-time. In "cascade" mode, the actor is trained directly
        # on planner subgoal predictions. In "actor_only" mode, only the actor is trained, and in
        # "planner_only" mode, only the planner is trained.
        self.algo.mode = "separate"

        self.algo.actor_use_random_subgoals = (
            False  # whether to sample subgoal index from [1, subgoal_horizon]
        )
        self.algo.subgoal_update_interval = 10  # how frequently the subgoal should be updated at test-time (usually matches train.seq_length)

        # ================== Latent Subgoal Config ==================

        # NOTE: latent subgoals are not supported by IRIS, but superclass expects this config
        self.algo.latent_subgoal.enabled = False
        self.algo.latent_subgoal.prior_correction.enabled = False
        self.algo.latent_subgoal.prior_correction.num_samples = 100

        # ================== Planner Config ==================

        # The ValuePlanner planner component is a Goal Learning VAE model
        self.algo.value_planner.planner = GLConfig().algo  # config for goal learning
        # set subgoal horizon explicitly
        self.algo.value_planner.planner.subgoal_horizon = 10
        # ensure VAE is used
        self.algo.value_planner.planner.vae.enabled = True

        # The ValuePlanner value component is a BCQ model
        self.algo.value_planner.value = BCQConfig().algo
        self.algo.value_planner.value.actor.enabled = False  # ensure no BCQ actor
        # number of subgoal samples to use for value planner
        self.algo.value_planner.num_samples = 100

        # ================== Actor Config ===================
        self.algo.actor = BCConfig().algo
        # use RNN
        self.algo.actor.rnn.enabled = True
        self.algo.actor.rnn.horizon = 10
        # remove unused parts of BCConfig algo config
        del self.algo.actor.gaussian
        del self.algo.actor.gmm
        del self.algo.actor.vae

    def observation_config(self):
        """
        Update from superclass so that value planner and actor each get their own obs config.
        """
        self.observation.value_planner.planner = GLConfig().observation
        self.observation.value_planner.value = BCQConfig().observation
        self.observation.actor = BCConfig().observation

    @property
    def use_goals(self):
        """
        Update from superclass - value planner goal modalities determine goal-conditioning.
        """
        return (
            len(
                self.observation.value_planner.planner.modalities.goal.low_dim
                + self.observation.value_planner.planner.modalities.goal.rgb
            )
            > 0
        )

    @property
    def all_obs_keys(self):
        """
        Update from superclass to include modalities from value planner and actor.
        """
        # pool all modalities
        return sorted(
            tuple(
                set(
                    [
                        obs_key
                        for group in [
                            self.observation.value_planner.planner.modalities.obs.values(),
                            self.observation.value_planner.planner.modalities.goal.values(),
                            self.observation.value_planner.planner.modalities.subgoal.values(),
                            self.observation.value_planner.value.modalities.obs.values(),
                            self.observation.value_planner.value.modalities.goal.values(),
                            self.observation.actor.modalities.obs.values(),
                            self.observation.actor.modalities.goal.values(),
                        ]
                        for modality in group
                        for obs_key in modality
                    ]
                )
            )
        )
