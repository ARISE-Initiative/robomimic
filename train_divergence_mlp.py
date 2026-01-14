import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train

# Path to your dataset with divergence info
# Update this path if your file is located elsewhere
dataset_path = os.path.expanduser("/app/robomimic/datasets/lift/ph/low_dim_v15_w_cdm.hdf5")

# Create default BC configuration
config = config_factory(algo_name="bc")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Set output directory for results
    config.train.output_dir = os.path.expanduser("./bc_divergence_results")
    config.experiment.name = "bc_mlp_divergence_test"

    # Configure observation keys
    # CRITICAL: 'robot0_eef_pos' and 'robot0_eef_quat' are required for 
    # the divergence computation (div_v_t) in the loss function.
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    
    # Disable RNN/Transformer to ensure we are training an MLP
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = False
    
    # MLP architecture settings (standard BC-MLP)
    config.algo.actor_layer_dims = [1024, 1024]
    
    # NEW: Set divergence loss weight
    config.algo.loss.cdm_weight = 0.1 

    # Training settings
    config.train.batch_size = 256
    config.train.num_epochs = 200
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 50
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
