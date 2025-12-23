import h5py
import numpy as np
import torch_utils
from robomimic.utils.dataset import SequenceDataset


def load_training_data(hdf5_path, demo_keys=None, verbose=True):
    """
    Load training data from an HDF5 dataset with flexible access to different parts.
    
    Args:
        hdf5_path (str): Path to the HDF5 dataset file
        demo_keys (list): List of demonstration keys to load. If None, loads all demos.
        verbose (bool): Whether to print dataset information
    
    Returns:
        dict: Dictionary containing:
            - 'data': h5py File object for raw access
            - 'demos': List of demonstration keys
            - 'attrs': Dataset attributes (environment info, etc.)
            - 'obs_keys': Available observation keys
            - 'get_demo': Function to get a specific demo's data
            - 'get_obs': Function to get observations from a demo
            - 'get_actions': Function to get actions from a demo
            - 'get_rewards': Function to get rewards from a demo
    """
    f = h5py.File(hdf5_path, 'r')
    
    # Get available demos
    all_demos = sorted([k for k in f['data'].keys()])
    demos = demo_keys if demo_keys is not None else all_demos
    
    # Get dataset attributes
    attrs = dict(f['data'].attrs)
    
    # Get observation keys from first demo
    first_demo = f['data'][demos[0]]
    obs_keys = list(first_demo['obs'].keys()) if 'obs' in first_demo else []
    
    if verbose:
        print(f"Dataset: {hdf5_path}")
        print(f"Total demonstrations: {len(all_demos)}")
        print(f"Loading demonstrations: {len(demos)}")
        print(f"Environment: {attrs.get('env', 'Unknown')}")
        print(f"Observation keys: {obs_keys}")
        print(f"Attributes: {list(attrs.keys())}")
    
    # Helper functions for data access
    def get_demo(demo_key):
        """Get all data for a specific demonstration."""
        if demo_key not in demos:
            raise ValueError(f"Demo {demo_key} not in loaded demos: {demos}")
        demo = f['data'][demo_key]
        return {
            'obs': {k: demo['obs'][k][()] for k in demo['obs'].keys()},
            'actions': demo['actions'][()],
            'rewards': demo['rewards'][()] if 'rewards' in demo else None,
            'dones': demo['dones'][()] if 'dones' in demo else None,
            'states': demo['states'][()] if 'states' in demo else None,
        }
    
    def get_obs(demo_key, obs_key=None):
        """Get observations from a demo. If obs_key is None, returns all observations."""
        demo = f['data'][demo_key]
        if obs_key is None:
            return {k: demo['obs'][k][()] for k in demo['obs'].keys()}
        return demo['obs'][obs_key][()]
    
    def get_actions(demo_key, indices=None):
        """Get actions from a demo. Optionally slice by indices."""
        actions = f['data'][demo_key]['actions'][()]
        return actions[indices] if indices is not None else actions
    
    def get_rewards(demo_key):
        """Get rewards from a demo."""
        demo = f['data'][demo_key]
        return demo['rewards'][()] if 'rewards' in demo else None
    
    return {
        'data': f,
        'demos': demos,
        'attrs': attrs,
        'obs_keys': obs_keys,
        'get_demo': get_demo,
        'get_obs': get_obs,
        'get_actions': get_actions,
        'get_rewards': get_rewards,
    }


if __name__ == "__main__":
    """
    Demo code showing how to use load_training_data() to explore a dataset.
    
    Usage:
        python robomimic/utils/divergence_utils.py --dataset /path/to/dataset.hdf5
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo key to inspect")
    args = parser.parse_args()
    
    print("="*60)
    print("DEMO: Loading and exploring training data")
    print("="*60)
    
    # Load the dataset
    print("\n1. Loading dataset...")
    data = load_training_data(args.dataset)
    
    # Show available demos
    print(f"\n2. Available demos: {data['demos'][:]}...")  # Show all
    
    # Get full demo data
    print(f"\n3. Loading demo '{args.demo}'...")
    demo_data = data['get_demo'](args.demo)
    
    print(f"\n   Actions shape: {demo_data['actions'].shape}")
    print(f"   Actions range: [{demo_data['actions'].min():.3f}, {demo_data['actions'].max():.3f}]")
    
    if demo_data['rewards'] is not None:
        print(f"   Total reward: {demo_data['rewards'].sum():.3f}")
        print(f"   Episode length: {len(demo_data['rewards'])}")
    
    # Explore observations
    print(f"\n4. Observation keys in demo:")
    for obs_key in demo_data['obs'].keys():
        obs_data = demo_data['obs'][obs_key]
        print(f"   - {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
    
    # Get specific observation
    if data['obs_keys']:
        first_obs_key = data['obs_keys'][0]
        print(f"\n5. Accessing specific observation: '{first_obs_key}'")
        obs = data['get_obs'](args.demo, first_obs_key)
        print(f"   Shape: {obs.shape}")
        print(f"   First timestep: {obs[0]}")
    
    # Get action slices
    print(f"\n6. Accessing action slices...")
    first_10_actions = data['get_actions'](args.demo, slice(0, 10))
    print(f"   First 10 actions shape: {first_10_actions.shape}")
    print(f"   First action: {first_10_actions[0]}")
    
    # Compare multiple demos
    print(f"\n7. Comparing trajectory lengths across demos...")
    for demo_key in data['demos'][:3]:  # First 3 demos
        actions = data['get_actions'](demo_key)
        print(f"   {demo_key}: {len(actions)} timesteps")
    
    # calculate trajectory length stats
    n_actions = []
    for demo_key in data['demos']:
        actions = data['get_actions'](demo_key)
        n_actions.append(len(actions))
    mean_n_actions = np.mean(n_actions)
    std_n_actions = np.std(n_actions)
    min_n_actions = np.min(n_actions)
    max_n_actions = np.max(n_actions)
    
    print(f"\n8. Trajectory length statistics across all demos:")
    print(f"   Mean: {mean_n_actions:.1f} timesteps")
    print(f"   Std:  {std_n_actions:.1f} timesteps")
    print(f"   Min:  {min_n_actions} timesteps")
    print(f"   Max:  {max_n_actions} timesteps")
    
    print("\n" + "="*60)
    print("Done! Remember to close the file when finished:")
    print("  data['data'].close()")
    print("="*60)

