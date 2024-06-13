"""
Config for CQL algorithm.
"""

from robomimic.config.base_config import BaseConfig


class CQLConfig(BaseConfig):
    ALGO_NAME = "cql"

    def train_config(self):
        """
        Update from superclass to change default batch size.
        """
        super(CQLConfig, self).train_config()

        # increase batch size to 1024 (found to work better for most manipulation experiments)
        self.train.batch_size = 1024

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.critic.learning_rate.initial = (
            1e-3  # critic learning rate
        )
        self.algo.optim_params.critic.learning_rate.decay_factor = (
            0.0  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.critic.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = (
            0.00  # L2 regularization strength
        )

        self.algo.optim_params.actor.learning_rate.initial = 3e-4  # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = (
            0.0  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.actor.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = (
            0.00  # L2 regularization strength
        )

        # target network related parameters
        self.algo.discount = 0.99  # discount factor to use
        self.algo.n_step = 1  # for using n-step returns in TD-updates
        self.algo.target_tau = 0.005  # update rate for target networks

        # ================== Actor Network Config ===================
        self.algo.actor.bc_start_steps = (
            0  # uses BC policy loss for first n-training steps
        )
        self.algo.actor.target_entropy = "default"  # None is fixed entropy, otherwise is automatically tuned to match target. Can specify "default" as well for default tuning target
        self.algo.actor.max_gradient_norm = None  # L2 gradient clipping for actor

        # Actor network settings
        self.algo.actor.net.type = (
            "gaussian"  # Options are currently only "gaussian" (no support for GMM yet)
        )

        # Actor network settings - shared
        self.algo.actor.net.common.std_activation = (
            "exp"  # Activation to use for std output from policy net
        )
        self.algo.actor.net.common.use_tanh = (
            True  # Whether to use tanh at output of actor network
        )
        self.algo.actor.net.common.low_noise_eval = (
            True  # Whether to use deterministic action sampling at eval stage
        )

        # Actor network settings - gaussian
        self.algo.actor.net.gaussian.init_last_fc_weight = 0.001  # If set, will override the initialization of the final fc layer to be uniformly sampled limited by this value
        self.algo.actor.net.gaussian.init_std = (
            0.3  # Relative scaling factor for std from policy net
        )
        self.algo.actor.net.gaussian.fixed_std = (
            False  # Whether to learn std dev or not
        )

        self.algo.actor.layer_dims = (300, 400)  # actor MLP layer dimensions

        # ================== Critic Network Config ===================
        self.algo.critic.use_huber = False  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = (
            None  # L2 gradient clipping for critic (None to use no clipping)
        )

        self.algo.critic.value_bounds = (
            None  # optional 2-tuple to ensure lower and upper bound on value estimates
        )

        self.algo.critic.num_action_samples = 1  # number of actions to sample per training batch to get target critic value; use maximum Q value from n random sampled actions when doing TD error backup

        # cql settings for critic
        self.algo.critic.cql_weight = 1.0  # weighting for cql component of critic loss (only used if target_q_gap is < 0 or None)
        self.algo.critic.deterministic_backup = (
            True  # if not set, subtract weighted logprob of action when doing backup
        )
        self.algo.critic.min_q_weight = 1.0  # min q weight (scaling factor) to apply
        self.algo.critic.target_q_gap = 5.0  # if set, sets the diff threshold at which Q-values will be penalized more (note: this overrides cql weight above!) Use None or a negative value if not set
        self.algo.critic.num_random_actions = (
            10  # Number of random actions to sample when calculating CQL loss
        )

        # critic ensemble parameters (TD3 trick)
        self.algo.critic.ensemble.n = 2  # number of Q networks in the ensemble

        self.algo.critic.layer_dims = (300, 400)  # critic MLP layer dimensions
