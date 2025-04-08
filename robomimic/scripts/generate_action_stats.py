import h5py
import numpy as np
import sys

def compute_action_stats(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        all_actions = []
        for demo_key in demo_keys:
            actions = f[f'data/{demo_key}/actions'][:]
            all_actions.append(actions)
        all_actions = np.concatenate(all_actions, axis=0)
        mean = np.mean(all_actions, axis=0)
        std = np.std(all_actions, axis=0)
        min_vals = np.min(all_actions, axis=0)
        max_vals = np.max(all_actions, axis=0)
        return mean, std, min_vals, max_vals

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_action_stats.py <path_to_h5_file>")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    mean, std, min_vals, max_vals = compute_action_stats(h5_file_path)
    print(f"action_mean={mean.tolist()}")
    print(f"action_std={std.tolist()}")
    print(f"action_min={min_vals.tolist()}")
    print(f"action_max={max_vals.tolist()}")
