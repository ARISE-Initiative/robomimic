"""
Config for Goal Learning (sub-algorithm used by hierarchical models like HBC and IRIS).
This class of model predicts (or samples) subgoal observations given a current observation.
"""

from robomimic.config.base_config import BaseConfig


class GLConfig(BaseConfig):
    ALGO_NAME = "gl"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.goal_network.learning_rate.initial = (
            1e-4  # goal network learning rate
        )
        self.algo.optim_params.goal_network.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.goal_network.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.goal_network.regularization.L2 = 0.00

        # subgoal definition: observation that is @subgoal_horizon number of timesteps in future from current observation
        self.algo.subgoal_horizon = 10

        # MLP size for deterministic goal network (unused if VAE is enabled)
        self.algo.ae.planner_layer_dims = (300, 400)

        # ================== VAE config ==================
        self.algo.vae.enabled = True  # set to true to use VAE network
        self.algo.vae.latent_dim = 16  # VAE latent dimension
        self.algo.vae.latent_clip = (
            None  # clip latent space when decoding (set to None to disable)
        )
        self.algo.vae.kl_weight = 1.0  # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = (
            True  # whether decoder should condition on observation
        )
        self.algo.vae.decoder.reconstruction_sum_across_elements = (
            False  # sum instead of mean for reconstruction loss
        )

        # VAE prior settings
        self.algo.vae.prior.learn = (
            False  # learn Gaussian / GMM prior instead of N(0, 1)
        )
        self.algo.vae.prior.is_conditioned = (
            False  # whether to condition prior on observations
        )
        self.algo.vae.prior.use_gmm = False  # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10  # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False  # whether to learn GMM weights
        self.algo.vae.prior.use_categorical = False  # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = (
            10  # the number of categorical classes for each latent dimension
        )
        self.algo.vae.prior.categorical_gumbel_softmax_hard = (
            False  # use hard selection in forward pass
        )
        self.algo.vae.prior.categorical_init_temp = 1.0  # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = (
            0.001  # linear temp annealing rate
        )
        self.algo.vae.prior.categorical_min_temp = 0.3  # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)  # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)  # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (
            300,
            400,
        )  # prior MLP layer dimensions (if learning conditioned prior)

    def observation_config(self):
        """
        Update from superclass to specify subgoal modalities.
        """
        super(GLConfig, self).observation_config()
        self.observation.modalities.subgoal.low_dim = (
            [  # specify low-dim subgoal observations for agent to predict
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ]
        )
        self.observation.modalities.subgoal.rgb = (
            []
        )  # specify rgb image subgoal observations for agent to predict
        self.observation.modalities.subgoal.depth = []
        self.observation.modalities.subgoal.scan = []
        self.observation.modalities.subgoal.do_not_lock_keys()

    @property
    def all_obs_keys(self):
        """
        Update from superclass to include subgoals.
        """
        # pool all modalities
        return sorted(
            tuple(
                set(
                    [
                        obs_key
                        for group in [
                            self.observation.modalities.obs.values(),
                            self.observation.modalities.goal.values(),
                            self.observation.modalities.subgoal.values(),
                        ]
                        for modality in group
                        for obs_key in modality
                    ]
                )
            )
        )
