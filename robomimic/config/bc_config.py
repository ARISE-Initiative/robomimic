"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class BCConfig(BaseConfig):
    ALGO_NAME = "bc"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(BCConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False              # whether to train a Gaussian policy
        self.algo.gaussian.fixed_std = False            # whether to train std output or keep it constant
        self.algo.gaussian.init_std = 0.1               # initial standard deviation (or constant)
        self.algo.gaussian.min_std = 0.01               # minimum std output from network
        self.algo.gaussian.std_activation = "softplus"  # activation to use for std output from policy net
        self.algo.gaussian.low_noise_eval = True        # low-std at test-time 

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.gmm.num_modes = 5                     # number of GMM modes
        self.algo.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.gmm.low_noise_eval = True             # low-std at test-time 

        # stochastic VAE policy settings
        self.algo.vae.enabled = False                   # whether to train a VAE policy
        self.algo.vae.latent_dim = 14                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 1.                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights 
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)

        # RNN policy settings
        self.algo.rnn.enabled = False                               # whether to train RNN policy
        self.algo.rnn.horizon = 10                                  # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400                              # hidden dimension size    
        self.algo.rnn.rnn_type = "LSTM"                             # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2                                # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False                             # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False                  # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Transformer policy settings
        self.algo.transformer.enabled = False                       # whether to train transformer policy
        self.algo.transformer.context_length = 10                   # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
        self.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
        self.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.transformer.emb_dropout = 0.1                     # dropout probability for embedding inputs in transformer
        self.algo.transformer.attn_dropout = 0.1                    # dropout probability for attention outputs for each transformer block
        self.algo.transformer.block_output_dropout = 0.1            # dropout probability for final outputs for each transformer block
        self.algo.transformer.sinusoidal_embedding = False          # if True, use standard positional encodings (sin/cos)
        self.algo.transformer.activation = "gelu"                   # activation function for MLP in Transformer Block
        self.algo.transformer.supervise_all_steps = False           # if true, supervise all intermediate actions, otherwise only final one
        self.algo.transformer.nn_parameter_for_timesteps = True     # if true, use nn.Parameter otherwise use nn.Embedding
        self.algo.transformer.pred_future_acs = False               # shift action prediction forward to predict future actions instead of past actions
        self.algo.transformer.causal = True                         # whether the transformer is causal

        self.algo.language_conditioned = False                      # whether policy is language conditioned
