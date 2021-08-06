"""
Config for BCQ algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.bc_config import BCConfig


class BCQConfig(BaseConfig):
    ALGO_NAME = "bcq"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.critic.learning_rate.initial = 1e-3              # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.1          # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []         # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00                  # L2 regularization strength
        self.algo.optim_params.critic.start_epoch = -1                          # number of epochs before starting critic training (-1 means start right away)
        self.algo.optim_params.critic.end_epoch = -1                            # number of epochs before ending critic training (-1 means start right away)

        self.algo.optim_params.action_sampler.learning_rate.initial = 1e-3      # action sampler learning rate
        self.algo.optim_params.action_sampler.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.action_sampler.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.action_sampler.regularization.L2 = 0.00          # L2 regularization strength
        self.algo.optim_params.action_sampler.start_epoch = -1                  # number of epochs before starting action sampler training (-1 means start right away)
        self.algo.optim_params.action_sampler.end_epoch = -1                    # number of epochs before ending action sampler training (-1 means start right away)

        self.algo.optim_params.actor.learning_rate.initial = 1e-3               # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.1           # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []          # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = 0.00                   # L2 regularization strength
        self.algo.optim_params.actor.start_epoch = -1                           # number of epochs before starting actor training (-1 means start right away)
        self.algo.optim_params.actor.end_epoch = -1                             # number of epochs before ending actor training (-1 means start right away)

        # target network related parameters
        self.algo.discount = 0.99                           # discount factor to use
        self.algo.n_step = 1                                # for using n-step returns in TD-updates
        self.algo.target_tau = 0.005                        # update rate for target networks
        self.algo.infinite_horizon = False                  # if True, scale terminal rewards by 1 / (1 - discount) to treat as infinite horizon

        # ================== Critic Network Config ===================
        self.algo.critic.use_huber = False                  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = None           # L2 gradient clipping for critic (None to use no clipping)
        self.algo.critic.value_bounds = None                # optional 2-tuple to ensure lower and upper bound on value estimates 
        self.algo.critic.num_action_samples = 10            # number of actions to sample per training batch to get target critic value
        self.algo.critic.num_action_samples_rollout = 100   # number of actions to sample per environment step

        # critic ensemble parameters (TD3 trick)
        self.algo.critic.ensemble.n = 2                     # number of Q networks in the ensemble
        self.algo.critic.ensemble.weight = 0.75             # weighting for mixing min and max for target Q value

        # distributional critic
        self.algo.critic.distributional.enabled = False     # train distributional critic (C51)
        self.algo.critic.distributional.num_atoms = 51      # number of values in categorical distribution

        self.algo.critic.layer_dims = (300, 400)            # size of critic MLP

        # ================== Action Sampler Config ===================
        self.algo.action_sampler = BCConfig().algo
        # use VAE by default
        self.algo.action_sampler.vae.enabled = True
        # remove unused parts of BCConfig algo config
        del self.algo.action_sampler.optim_params           # since action sampler optim params specified at top-level
        del self.algo.action_sampler.loss
        del self.algo.action_sampler.gaussian
        del self.algo.action_sampler.rnn

        # Number of epochs before freezing encoder (-1 for no freezing). Only applies to cVAE-based action samplers.
        with self.algo.action_sampler.unlocked():
            self.algo.action_sampler.freeze_encoder_epoch = -1

        # ================== Actor Network Config ===================
        self.algo.actor.enabled = False                     # whether to use the actor perturbation network
        self.algo.actor.perturbation_scale = 0.05           # size of learned action perturbations
        self.algo.actor.layer_dims = (300, 400)             # size of actor MLP
