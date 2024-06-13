"""
Config for HBC algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.gl_config import GLConfig
from robomimic.config.bc_config import BCConfig


class HBCConfig(BaseConfig):
    ALGO_NAME = "hbc"

    def train_config(self):
        """
        Update from superclass to change default sequence length to load from dataset.
        """
        super(HBCConfig, self).train_config()
        self.train.seq_length = (
            10  # length of experience sequence to fetch from the buffer
        )

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
        self.algo.subgoal_update_interval = (
            10  # how frequently the subgoal should be updated at test-time
        )

        # ================== Latent Subgoal Config ==================
        self.algo.latent_subgoal.enabled = False  # if True, use VAE latent space as subgoals for actor, instead of reconstructions

        # prior correction trick for actor and value training: instead of using encoder for
        # transforming subgoals to latent subgoals, generate prior samples and choose
        # the closest one to the encoder output
        self.algo.latent_subgoal.prior_correction.enabled = False
        self.algo.latent_subgoal.prior_correction.num_samples = 100

        # ================== Planner Config ==================
        self.algo.planner = GLConfig().algo  # config for goal learning
        # set subgoal horizon explicitly
        self.algo.planner.subgoal_horizon = 10
        # ensure VAE is used
        self.algo.planner.vae.enabled = True

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
        Update from superclass so that planner and actor each get their own observation config.
        """
        self.observation.planner = GLConfig().observation
        self.observation.actor = BCConfig().observation

    @property
    def use_goals(self):
        """
        Update from superclass - planner goal modalities determine goal-conditioning
        """
        return (
            len(
                self.observation.planner.modalities.goal.low_dim
                + self.observation.planner.modalities.goal.rgb
            )
            > 0
        )

    @property
    def all_obs_keys(self):
        """
        Update from superclass to include modalities from planner and actor.
        """
        # pool all modalities
        return sorted(
            tuple(
                set(
                    [
                        obs_key
                        for group in [
                            self.observation.planner.modalities.obs.values(),
                            self.observation.planner.modalities.goal.values(),
                            self.observation.planner.modalities.subgoal.values(),
                            self.observation.actor.modalities.obs.values(),
                            self.observation.actor.modalities.goal.values(),
                        ]
                        for modality in group
                        for obs_key in modality
                    ]
                )
            )
        )
