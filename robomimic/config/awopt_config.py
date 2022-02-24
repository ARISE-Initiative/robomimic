"""
Config for AW-Opt algorithm.
"""

from robomimic.config.base_config import BaseConfig


class AW_Opt_Config(BaseConfig):
    ALGO_NAME = "awopt"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []  # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0  # L2 loss weight
        self.algo.loss.l1_weight = 0.0  # L1 loss weight
        self.algo.loss.cos_weight = 0.0  # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)
