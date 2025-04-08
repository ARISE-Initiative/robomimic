from robomimic.config.base_config import BaseConfig

class GATConfig(BaseConfig):
    ALGO_NAME='gat_bc'

    def experiment_config(self):
        super().experiment_config()
        self.experiment.name = "4_layer" # name of experiment

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(GATConfig, self).train_config()
        super(GATConfig,self).observation_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        self.train.seq_length = 5
        self.experiment.rollout.n = 10
        self.experiment.rollout.horizon = 500
        self.experiment.save.every_n_epochs = 100
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4    # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.num_layers = 3
        self.algo.hidden_dim = 256
        self.algo.heads = 6
        self.algo.dropout = 0.1
        self.algo.action_dim = 35

        # Replace individual dims with a list specifying the node encoder layers
        self.algo.node_encoder_dims = [64]

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        self.algo.grad_clip = 1.0           # gradient clipping threshold


    def observation_config(self):
        self.observation.modalities.obs.low_dim = [             # specify low-dim observations for agent
            "robot0_joint_pos",
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ]
