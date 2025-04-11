"""
Configuration Class for the Diffusion Graph Attention Network (DiffGAT) Algorithm.
"""

from robomimic.config.base_config import BaseConfig


class DiffGATConfig(BaseConfig):
    """
    Configuration settings for the DiffGAT algorithm.

    Organizes parameters related to:
    - Experiment setup (name, logging)
    - Training parameters (batch size, sequence length, frame stack, optimizer)
    - Diffusion process (number of timesteps)
    - Network architecture (GNN layers/heads/dims, Transformer layers/heads/dims, dropout)
    - Loss function weights (currently only MSE implied by diffusion framework)
    - Observation modalities
    """
    ALGO_NAME = 'diff_gat'

    def experiment_config(self):
        """Configure experiment settings."""
        super().experiment_config()
        # Unique name for this experiment run
        self.experiment.name = "diff_gat_panda_example"
        # Configure rollout settings for evaluation
        self.experiment.rollout.n = 10       # Number of rollout episodes
        self.experiment.rollout.horizon = 400 # Max steps per rollout
        self.experiment.rollout.rate = 50    # Frequency of rollouts (e.g., every 50 epochs)
        self.experiment.rollout.warmstart = 200 # Steps before starting rollouts
        self.experiment.rollout.terminate_on_success = True # End rollout if task succeeds

    def train_config(self):
        """Configure training loop settings."""
        super().train_config()
        # Diffusion models typically predict action sequences based on past observations.
        # Loading "next_obs" from HDF5 is usually not required, saving memory and I/O.
        self.train.hdf5_load_next_obs = False
        self.train.data = "datasets/can/ph/low_dim_v141.hdf5"

        # Core training parameters
        self.train.seq_length = 3     # Length of action sequences predicted by the policy
        self.train.frame_stack = 3    # Number of observation frames provided as input context
        self.train.batch_size = 256   # Number of sequences per training batch (adjust based on GPU memory)
        self.train.num_epochs = 2000  # Total number of training epochs
        self.train.num_data_workers = 0 # Number of parallel data loading workers

    def algo_config(self):
        """Configure algorithm-specific hyperparameters."""
        super().algo_config() # Ensure base algo config is initialized

        # --- Action Space ---
        self.algo.action_dim = 7 # Dimension of the action vector at each step (pos and quat for gripper)
        self.algo.num_joints = 7 # Number of joints in the robot arm (e.g., 7 for Panda + 1 for gripper)
        # Full action dimension over the sequence (useful for reference, not direct config)
        # self.algo.full_action_dim = self.train.seq_length * self.algo.action_dim

        # --- Optimization ---
        optim_params = self.algo.optim_params.policy
        optim_params.optimizer_type = "adam" # Adam optimizer is standard
        optim_params.learning_rate.initial = 1e-4     # Initial learning rate
        optim_params.learning_rate.decay_factor = 0.1 # Multiplicative factor for LR decay
        optim_params.learning_rate.epoch_schedule = [] # Epochs at which to decay LR (e.g., [1000, 1500]) - empty means no decay
        optim_params.learning_rate.scheduler_type = "multistep" # 'multistep' or 'cosine'
        optim_params.regularization.L2 = 1e-6       # L2 weight decay (0 means none)
        self.algo.grad_clip = 1.0                     # Max norm for gradient clipping (helps stability)

        # --- Diffusion Process ---
        self.algo.diffusion.num_timesteps = 100 # Number of noise levels in the diffusion process (T)

        # --- Network Architecture ---
        # GNN (GATv2 Backbone) parameters
        gnn = self.algo.gnn
        gnn.num_layers = 4         # Number of message-passing layers
        gnn.hidden_dim = 512       # Hidden dimension within GNN layers and output embedding size
        gnn.num_heads = 4          # Number of attention heads in GATv2 layers (hidden_dim must be divisible by num_heads)
        gnn.attention_dropout = 0.3 # Dropout rate specifically on attention weights

        # Transformer Decoder Head parameters
        transformer = self.algo.transformer
        transformer.num_layers = 4        # Number of decoder layers
        transformer.num_heads = 4         # Number of attention heads in decoder layers
        # Feedforward dimension is often a multiple of the model dimension (hidden_dim)
        transformer.ff_dim_multiplier = 2

        # General Network parameters
        network = self.algo.network
        network.dropout = 0.2 # General dropout rate applied in MLPs, positional encoding, etc.

        # --- Loss Configuration ---
        # For standard diffusion models, the primary loss is MSE between predicted and actual noise.
        # Weights below are placeholders if other loss components were added (e.g., L1).
        loss = self.algo.loss
        loss.l2_weight = 1.0 # Weight for the primary MSE noise prediction loss
        loss.l1_weight = 0.0 # Weight for potential L1 loss component
        loss.cos_weight = 0.0 # Weight for potential cosine similarity loss component
        loss.tradeoff = 0.9 # Weight for tradeoff between q_pos loss and noise loss

    def observation_config(self):
        """Configure which observation modalities are used."""
        super().observation_config() # Ensure base observation config is initialized
        obs_modalities = self.observation.modalities.obs

        # Define low-dimensional observations expected by the NodeFeatureProcessor
        obs_modalities.low_dim = [
            "robot0_joint_pos",     # Robot joint angles
            "robot0_eef_pos",       # End-effector position
            "robot0_eef_quat",      # End-effector orientation (quaternion)
            "robot0_gripper_qpos",  # Gripper joint positions
            "object",               # Object-related features (e.g., pose, relative pose)
        ]
        # Disable image observations if not used
        obs_modalities.rgb = []
        obs_modalities.depth = []
        obs_modalities.scan = []

        # Configure observation processing (e.g., normalization)
        # By default, Robomimic normalizes low_dim and image observations.
        self.observation.encoder.low_dim.core_kwargs.normalization = True
        self.observation.encoder.low_dim.obs_randomizer_kwargs.gaussian_noise_std = 0.0 # Disable obs noise injection