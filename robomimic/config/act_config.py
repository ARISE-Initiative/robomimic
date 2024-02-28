"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class ACTConfig(BaseConfig):
    ALGO_NAME = "act"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(ACTConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 5e-5      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "linear" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.0001          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 0.0      # L2 loss weight
        self.algo.loss.l1_weight = 1.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # ACT policy settings
        self.algo.act.hidden_dim = 512                              # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.act.dim_feedforward = 3200                        # dimension for embeddings used by transformer
        self.algo.act.backbone = "resnet18"                         # number of transformer blocks to stack
        self.algo.act.enc_layers = 4                                # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.act.dec_layers = 7                                # dropout probability for embedding inputs in transformer
        self.algo.act.nheads = 8                                    # dropout probability for attention outputs for each transformer block
        self.algo.act.latent_dim = 32                               # latent dim of VAE
        self.algo.act.kl_weight = 20                                # KL weight of VAE
