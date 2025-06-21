import h5py
import numpy as np
import sys
import os
import json
from tqdm import tqdm

# IMPORTANT: Define the order of observation keys.
# This order MUST EXACTLY MATCH the order of concatenation used in your policy
# (e.g., in `process_batch_for_training`).
OBS_KEYS_IN_ORDER = [
    'object', 
    'robot0_eef_pos', 
    'robot0_eef_quat', 
    'robot0_gripper_qpos', 
    'robot0_joint_pos',
]

def compute_action_stats(h5_file_path):
    """
    Computes mean, std, min, and max for the 'actions' dataset across all demonstrations.
    """
    print("Computing action statistics...")
    with h5py.File(h5_file_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        all_actions = []
        for demo_key in tqdm(demo_keys, desc="Processing demos for actions"):
            actions = f[f'data/{demo_key}/actions'][:]
            all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        mean = np.mean(all_actions, axis=0)
        std = np.std(all_actions, axis=0)
        min_vals = np.min(all_actions, axis=0)
        max_vals = np.max(all_actions, axis=0)
        
        return mean, std, min_vals, max_vals

def compute_obs_stats(h5_file_path, obs_keys):
    """
    Computes mean and std for the concatenated observation vector across all demonstrations.
    """
    print(f"\nComputing observation statistics using keys: {obs_keys}")
    with h5py.File(h5_file_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        all_obs_vectors = []

        for demo_key in tqdm(demo_keys, desc="Processing demos for obs"):
            # Check if all specified obs_keys exist in this demo
            obs_group = f[f'data/{demo_key}/obs']
            if not all(key in obs_group for key in obs_keys):
                print(f"Warning: Skipping demo {demo_key} because it's missing one of the required observation keys.")
                continue

            # Get the number of timesteps from the first key
            num_timesteps = obs_group[obs_keys[0]].shape[0]

            # Concatenate observations for each timestep
            for t in range(num_timesteps):
                # For each timestep t, create a list of the numpy arrays for each obs key
                timestep_obs_parts = [obs_group[key][t] for key in obs_keys]
                
                # Concatenate the parts into a single flat vector
                concatenated_obs = np.concatenate(timestep_obs_parts, axis=0)
                all_obs_vectors.append(concatenated_obs)

        # Compute stats over the list of all concatenated vectors
        all_obs_vectors = np.array(all_obs_vectors)
        mean = np.mean(all_obs_vectors, axis=0)
        std = np.std(all_obs_vectors, axis=0)
        
        # Add a small epsilon to std to prevent division by zero during normalization
        std[std < 1e-6] = 1.0 # Set near-zero std to 1.0 to avoid scaling issues
        
        return mean, std


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_dataset_stats.py <path_to_h5_file>")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    
    # --- Compute and Save Action Stats ---
    action_mean, action_std, action_min, action_max = compute_action_stats(h5_file_path)
    
    print("\n--- Action Stats ---")
    print(f"action_mean={action_mean.tolist()}")
    print(f"action_std={action_std.tolist()}")
    print(f"action_min={action_min.tolist()}")
    print(f"action_max={action_max.tolist()}")

    action_stats = {
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist()
    }
    dir_path = os.path.dirname(h5_file_path)
    action_out_path = os.path.join(dir_path, "action_stats.json")
    with open(action_out_path, "w") as f:
        json.dump(action_stats, f, indent=4)
    print(f"\nSaved action stats to {action_out_path}")

    # --- Compute and Save Observation Stats ---
    obs_mean, obs_std = compute_obs_stats(h5_file_path, OBS_KEYS_IN_ORDER)

    print("\n--- Observation Stats ---")
    print(f"obs_mean={obs_mean.tolist()}")
    print(f"obs_std={obs_std.tolist()}")

    obs_stats = {
        "obs_mean": obs_mean.tolist(),
        "obs_std": obs_std.tolist(),
    }
    obs_out_path = os.path.join(dir_path, "obs_stats.json")
    with open(obs_out_path, "w") as f:
        json.dump(obs_stats, f, indent=4)
    print(f"\nSaved observation stats to {obs_out_path}")