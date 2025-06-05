
"""
Configuration Class for the Flow Graph Attention Network (FlowGAT) Algorithm.
"""

from robomimic.config.base_config import BaseConfig


class FlowGATConfig(BaseConfig):
    """
    Configuration settings for the FlowGAT algorithm.

    Organizes parameters related to:
    - Experiment setup (name, logging, rollouts)
    - Training parameters (dataset, batch size, sequence length, frame stack, optimizer)
    - Network architecture details (handled within model definition, e.g., GNN layers/heads/dims)
    - Observation modalities (implicitly defined by dataset and model)
    """
    ALGO_NAME = 'flow_gat'

    def experiment_config(self):
        """Configure experiment settings."""
        super().experiment_config()
        # Unique name for this experiment run
        self.experiment.name = "graph_structure_experiment"
        # Configure rollout settings for evaluation
        self.experiment.rollout.n = 25       # Number of rollout episodes
        self.experiment.rollout.horizon = 400 # Max steps per rollout
        self.experiment.rollout.rate = 50    # Frequency of rollouts (e.g., every 50 epochs)
        self.experiment.rollout.warmstart = 100 # Steps before starting rollouts
        self.experiment.rollout.terminate_on_success = True # End rollout if task succeeds

        self.experiment.logging.log_wandb = True # Enable logging to Weights & Biases
        self.experiment.logging.wandb_proj_name = "thesis_evaluation_graph_structure"
        self.experiment.render_video = True # Disable video rendering during rollouts
        self.experiment.save.enabled = True

    def train_config(self):
        """Configure training loop settings."""
        super().train_config()
        # Diffusion models typically predict action sequences based on past observations.
        # Loading "next_obs" from HDF5 is usually not required, saving memory and I/O.
        self.train.hdf5_load_next_obs = True

        self.train.data = "datasets/can/ph/low_dim_v15.hdf5" # Path to the dataset (HDF5 file)s
        self.train.graph_config = "robomimic/algo/flow_gat_files/pickplace.json"

        # Core training parameters
        self.train.seq_length = 2     # Length of action sequences predicted by the policy
        self.train.frame_stack = 2    # Number of observation frames provided as input context
        self.train.batch_size = 256   # Number of sequences per training batch (adjust based on GPU memory)
        self.train.num_epochs = 2000  # Total number of training epochs
        self.train.num_data_workers = 0 # Number of parallel data loading workers
        self.train.seed = 0
    def algo_config(self):
        """Configure algorithm-specific hyperparameters."""
        super().algo_config() # Ensure base algo config is initialized
        self.algo.name = "flow_gat" # Algorithm name
        self.algo.graph_name = "skip_graph" # Graph structure name (e.g., "skip_graph")


        # --- Action Space ---
        self.algo.action_dim = 7 # Dimension of the action vector at each step (pos and quat for gripper)
        self.algo.num_joints = 7 # Number of joints in the robot arm (e.g., 7 for Panda + 1 for gripper)
        self.algo.grad_clip = 1.0   
        self.algo.t_a = 2 # Action execution horizon
        self.algo.graph_frame_stack = 2    # ≤ frame_stack; number of obs frames the GNN actually sees
        self.algo.inference_euler_steps = 5


        # --- Optimization ---
        optim_params = self.algo.optim_params.policy
        optim_params.optimizer_type = "adam" # Adam optimizer is standard
        optim_params.learning_rate.initial = 1e-4     # Initial learning rate
        optim_params.learning_rate.decay_factor = 0.01 # Multiplicative factor for LR decay
        optim_params.learning_rate.epoch_schedule = [1000, 1500] # Epochs at which to decay LR (e.g., [1000, 1500]) - empty means no decay
        optim_params.learning_rate.scheduler_type = "cosine_warmup" # 'multistep' or 'cosine'
        optim_params.learning_rate.cosine_max = 2000
        optim_params.learning_rate.warmup_steps = 5 # Number of warmup steps for learning rate
        optim_params.regularization.L2 = 1e-5       # L2 weight decay (0 means none)                  # Max norm for gradient clipping (helps stability)

        # --- Network Architecture ---
        # GNN (GATv2 Backbone) parameters
        gnn = self.algo.gnn
        gnn.num_layers = 4         # Number of message-passing layers
        gnn.node_dim = 64          # Dimension of node features (e.g., 64 for each joint)
        gnn.hidden_dim = 128       # Hidden dimension within GNN layers and output embedding size
        gnn.num_heads = 4          # Number of attention heads in GATv2 layers (hidden_dim must be divisible by num_heads)
        gnn.attention_dropout = 0.1 # Dropout rate specifically on attention weights
        gnn.node_input_dim = 22
        # --- Cross Attention ---
        # Transformer parameters
        transformer = self.algo.transformer
        transformer.num_layers = 4
        transformer.num_heads = 4          # Number of attention heads in Transformer layers
        transformer.hidden_dim = 128       # Hidden dimension within Transformer layers
        transformer.attention_dropout = 0.1 # Dropout rate specifically on attention weights

        # General Network parameters
        network = self.algo.network
        network.dropout = 0.1 # General dropout rate applied in MLPs, positional encoding, etc.

        # EMA
        ema = self.algo.ema
        ema.enabled = True
        ema.power = 0.9999 # Exponential moving average decay rate

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
            "object",
        ]
        # obs_modalities.rgb = [
        #     "robot0_eye_in_hand_image", # RGB image from the robot's camera
        # ]

        # Configure observation processing (e.g., normalization)
        # By default, Robomimic normalizes low_dim and image observations.
        self.observation.encoder.low_dim.core_kwargs.normalization = True
        self.observation.encoder.low_dim.obs_randomizer_kwargs.gaussian_noise_std = 0.0 # Disable obs noise injection

        self.observation.encoder.rgb.core_kwargs.feature_dimension = 64
        self.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = True
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
        self.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0
        self.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
        self.observation.encoder.rgb.obs_randomizer_kwargs = {
            "crop_height": 76,
            "crop_width": 76,
            "num_crops": 1,
            "pos_enc":  False,
        }