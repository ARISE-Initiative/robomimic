"""
Config for TD3_BC.
"""

from robomimic.config.base_config import BaseConfig


class TD3_BCConfig(BaseConfig):
    ALGO_NAME = "td3_bc"

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(TD3_BCConfig, self).experiment_config()

        # no validation and no video rendering
        self.experiment.validate = False
        self.experiment.render_video = False

        # save 10 checkpoints throughout training
        self.experiment.save.every_n_epochs = 20

        # save models that achieve best rollout return instead of best success rate
        self.experiment.save.on_best_rollout_return = True
        self.experiment.save.on_best_rollout_success_rate = False

        # epoch definition - 5000 gradient steps per epoch, with 200 epochs = 1M gradient steps, and eval every 1 epochs
        self.experiment.epoch_every_n_steps = 5000

        # evaluate with normal environment rollouts
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 50  # paper uses 10, but we can afford to do 50
        self.experiment.rollout.horizon = 1000
        self.experiment.rollout.rate = 1  # rollout every epoch to match paper

    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(TD3_BCConfig, self).train_config()

        # update to normalize observations
        self.train.hdf5_normalize_obs = True

        # increase batch size to 256
        self.train.batch_size = 256

        # 200 epochs, with each epoch lasting 5000 gradient steps, for 1M total steps
        self.train.num_epochs = 200

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.critic.learning_rate.initial = (
            3e-4  # critic learning rate
        )
        self.algo.optim_params.critic.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.critic.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = (
            0.00  # L2 regularization strength
        )
        self.algo.optim_params.critic.start_epoch = (
            -1
        )  # number of epochs before starting critic training (-1 means start right away)
        self.algo.optim_params.critic.end_epoch = (
            -1
        )  # number of epochs before ending critic training (-1 means start right away)

        self.algo.optim_params.actor.learning_rate.initial = 3e-4  # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.actor.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = (
            0.00  # L2 regularization strength
        )
        self.algo.optim_params.actor.start_epoch = (
            -1
        )  # number of epochs before starting actor training (-1 means start right away)
        self.algo.optim_params.actor.end_epoch = (
            -1
        )  # number of epochs before ending actor training (-1 means start right away)

        # alpha value - for weighting critic loss vs. BC loss
        self.algo.alpha = 2.5

        # target network related parameters
        self.algo.discount = 0.99  # discount factor to use
        self.algo.n_step = 1  # for using n-step returns in TD-updates
        self.algo.target_tau = 0.005  # update rate for target networks
        self.algo.infinite_horizon = False  # if True, scale terminal rewards by 1 / (1 - discount) to treat as infinite horizon

        # ================== Critic Network Config ===================
        self.algo.critic.use_huber = False  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = (
            None  # L2 gradient clipping for critic (None to use no clipping)
        )
        self.algo.critic.value_bounds = (
            None  # optional 2-tuple to ensure lower and upper bound on value estimates
        )

        # critic ensemble parameters (TD3 trick)
        self.algo.critic.ensemble.n = 2  # number of Q networks in the ensemble
        self.algo.critic.ensemble.weight = (
            1.0  # weighting for mixing min and max for target Q value
        )

        self.algo.critic.layer_dims = (256, 256)  # size of critic MLP

        # ================== Actor Network Config ===================

        # update actor and target networks every n gradients steps for each critic gradient step
        self.algo.actor.update_freq = 2

        # exploration noise used to form target action for Q-update - clipped Gaussian noise
        self.algo.actor.noise_std = (
            0.2  # zero-mean gaussian noise with this std is applied to actions
        )
        self.algo.actor.noise_clip = (
            0.5  # noise is clipped in each dimension to (-noise_clip, noise_clip)
        )

        self.algo.actor.layer_dims = (256, 256)  # size of actor MLP

    def observation_config(self):
        """
        Update from superclass to use flat observations from gym envs.
        """
        super(TD3_BCConfig, self).observation_config()
        self.observation.modalities.obs.low_dim = ["flat"]
