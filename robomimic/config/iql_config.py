"""
Config for IQL algorithm.
"""

from robomimic.config.base_config import BaseConfig


class IQLConfig(BaseConfig):
    ALGO_NAME = "iql"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(IQLConfig, self).algo_config()

        # optimization parameters        
        self.algo.optim_params.critic.learning_rate.initial = 1e-4          # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00              # L2 regularization strength

        self.algo.optim_params.vf.learning_rate.initial = 1e-4              # vf learning rate
        self.algo.optim_params.vf.learning_rate.decay_factor = 0.0          # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.vf.learning_rate.epoch_schedule = []         # epochs where LR decay occurs
        self.algo.optim_params.vf.regularization.L2 = 0.00                  # L2 regularization strength

        self.algo.optim_params.actor.learning_rate.initial = 1e-4           # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.0       # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []      # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = 0.00               # L2 regularization strength

        # target network related parameters
        self.algo.discount = 0.99                                           # discount factor to use
        self.algo.target_tau = 0.01                                         # update rate for target networks

        # ================== Actor Network Config ===================
        # Actor network settings
        self.algo.actor.net.type = "gaussian"                               # Options are currently ["gaussian", "gmm"]

        # Actor network settings - shared
        self.algo.actor.net.common.std_activation = "softplus"              # Activation to use for std output from policy net
        self.algo.actor.net.common.low_noise_eval = True                    # Whether to use deterministic action sampling at eval stage
        self.algo.actor.net.common.use_tanh = False                         # Whether to use tanh at output of actor network

        # Actor network settings - gaussian
        self.algo.actor.net.gaussian.init_last_fc_weight = 0.001            # If set, will override the initialization of the final fc layer to be uniformly sampled limited by this value
        self.algo.actor.net.gaussian.init_std = 0.3                         # Relative scaling factor for std from policy net
        self.algo.actor.net.gaussian.fixed_std = False                      # Whether to learn std dev or not

        self.algo.actor.net.gmm.num_modes = 5                               # number of GMM modes
        self.algo.actor.net.gmm.min_std = 0.0001                            # minimum std output from network

        self.algo.actor.layer_dims = (300, 400)                             # actor MLP layer dimensions

        self.algo.actor.max_gradient_norm = None                            # L2 gradient clipping for actor

        # ================== Critic Network Config ===================
        # critic ensemble parameters
        self.algo.critic.ensemble.n = 2                                     # number of Q networks in the ensemble
        self.algo.critic.layer_dims = (300, 400)                            # critic MLP layer dimensions
        self.algo.critic.use_huber = False                                  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = None                           # L2 gradient clipping for actor

        # ================== Adv Config ==============================
        self.algo.adv.clip_adv_value = None                                 # whether to clip raw advantage estimates
        self.algo.adv.beta = 1.0                                            # temperature for operator
        self.algo.adv.use_final_clip = True                                 # whether to clip final weight calculations

        self.algo.vf_quantile = 0.9                                         # quantile factor in quantile regression
