"""
Configuration Class for the Flow Graph Attention Network (FlowGAT) Algorithm.
"""

from robomimic.config.base_config import BaseConfig


class FlowGATConfig(BaseConfig):
    """
    Configuration settings for the FlowGAT algorithm.
    Organizes parameters related to experiment setup, training, and model architecture.
    """
    ALGO_NAME = 'flow_gat'

    def experiment_config(self):
        super().experiment_config()
        self.experiment.name = "GNN_Backbone"
        self.experiment.rollout.n = 25
        self.experiment.rollout.horizon = 400
        self.experiment.rollout.rate = 100
        self.experiment.rollout.warmstart = 0
        self.experiment.rollout.terminate_on_success = True
        self.experiment.logging.log_wandb = True
        self.experiment.logging.wandb_proj_name = "dev"
        self.experiment.render_video = True
        self.experiment.save.enabled = True
        self.experiment.save.every_n_epochs = 100
        self.experiment.validate = True
        self.experiment.epoch_every_n_steps = None

    def train_config(self):
        super().train_config()
        self.train.hdf5_load_next_obs = True
        self.train.data = "datasets/square/ph/low_dim_v15.hdf5"
        self.train.graph_config = "robomimic/algo/flow_gat_files/nut_assembly_fc_robot_sparse_env.json"
        self.train.seq_length = 10
        self.train.frame_stack = 2
        self.train.batch_size = 512
        self.train.num_epochs = 4000
        self.train.num_data_workers = 0
        self.train.seed = 0
        self.train.hdf5_filter_key = "train"
        self.train.hdf5_validation_filter_key = "valid"
        self.train.hdf5_normalize_obs = False
        self.train.global_feature_size = 14

    def algo_config(self):
        super().algo_config()
        self.algo.name = "flow_gcn"
        self.algo.graph_name = "skip_graph"
        self.algo.action_dim = 7
        self.algo.num_joints = 7
        self.algo.grad_clip = 1.0
        self.algo.t_a = 5
        self.algo.t_p = self.train.seq_length
        self.algo.graph_frame_stack = 2
        self.algo.inference_euler_steps = 5
        self.algo.temp_edges = False
        self.algo.has_edge_attr = True
        self.algo.num_edge_attr = 5
        # Edge feature configuration for ablation studies
        # Available features: 'relative_position', 'distance', 'edge_type'
        self.algo.edge_features = ['relative_position', 'edge_type']
        # Optimization
        optim_params = self.algo.optim_params.policy
        optim_params.optimizer_type = "adamw"
        optim_params.learning_rate.initial = 2e-4 
        optim_params.learning_rate.decay_factor = 0.001
        optim_params.learning_rate.epoch_schedule = [1000, 1500]
        optim_params.learning_rate.scheduler_type = "cosine_warmup"
        optim_params.learning_rate.cosine_max = 4000
        optim_params.learning_rate.warmup_steps = 50
        optim_params.regularization.L2 = 1e-3 
        # GNN Backbone
        gnn = self.algo.gnn
        gnn.num_layers = 3
        gnn.node_dim = 64
        gnn.hidden_dim = 64
        gnn.num_heads = 2
        gnn.attention_dropout = 0.2
        gnn.node_input_dim = 22
        gnn.noise_std_dev = 0.02
        # Transformer
        transformer = self.algo.transformer
        transformer.num_layers = 2
        transformer.num_heads = 1
        transformer.hidden_dim = 256
        transformer.attention_dropout = gnn.attention_dropout
        # General Network
        network = self.algo.network
        network.emb_dropout = 0
        # EMA
        ema = self.algo.ema
        ema.enabled = True
        ema.power = 0.75

    def observation_config(self):
        super().observation_config()
        obs_modalities = self.observation.modalities.obs
        obs_modalities.low_dim = [
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]
        self.observation.encoder.low_dim.core_kwargs.normalization = True
        self.observation.encoder.low_dim.obs_randomizer_kwargs.gaussian_noise_std = 0.0
