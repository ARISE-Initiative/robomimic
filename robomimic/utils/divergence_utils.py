import h5py
import numpy as np
import torch_utils
import torch
from robomimic.utils.dataset import SequenceDataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# GMM UTILS
def _fit_gmm_single_timestep(args):
    """
    Helper function to fit GMM for a single time step (for parallel processing).
    
    Args:
        args: tuple of (time_step_data, num_components, max_components)
    
    Returns:
        tuple: (means, sigmas, weights) for this timestep
    """
    time_step_data, num_components, max_components = args
    
    # Find best number of components using BIC
    best_n_components = 1
    best_bic = np.inf
    best_gmm = None
    
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='diag',
                random_state=42,
                max_iter=200
            )
            gmm.fit(time_step_data)
            bic = gmm.bic(time_step_data)
            
            if bic < best_bic:
                best_bic = bic
                best_n_components = n
                best_gmm = gmm
        except:
            # Skip if fitting fails (e.g., singular covariance)
            continue
    
    # Initialize output arrays
    means_t = np.zeros((num_components, time_step_data.shape[1]))
    sigmas_t = np.zeros((num_components, time_step_data.shape[1]))
    weights_t = np.zeros(num_components)
    counts_t = np.zeros(num_components, dtype=np.int64)
    
    # Extract parameters from best GMM
    if best_gmm is not None:
        # Get cluster assignments
        assignments = best_gmm.predict(time_step_data)
        
        # Fill in the components that were fit
        for i in range(best_n_components):
            means_t[i] = best_gmm.means_[i]
            sigmas_t[i] = np.sqrt(best_gmm.covariances_[i])
            weights_t[i] = best_gmm.weights_[i]
            counts_t[i] = np.sum(assignments == i)
        
        # Normalize weights to sum to 1
        if weights_t.sum() > 0:
            weights_t = weights_t / weights_t.sum()
    else:
        # Fallback: single component with mean and std of data
        means_t[0] = time_step_data.mean(axis=0)
        sigmas_t[0] = time_step_data.std(axis=0) + 1e-6
        weights_t[0] = 1.0
        counts_t[0] = time_step_data.shape[0]
    
    return means_t, sigmas_t, weights_t, counts_t


def fit_gmm(data, num_components=5, n_jobs=-1):
    """
    Method for fitting a gmm to data. Fits the gmm with the best number of components to explain the data 
    at each time step up to the max number of components. Uses parallel processing for speed.
    
    Args:
        data: torch.tensor [batch, steps, dims], the data
        num_components: int, maximum number of gaussians to allow
        n_jobs: int, number of parallel jobs (-1 for all CPUs, 1 for sequential)
    
    Returns:
        means: torch.tensor [steps, num_components, dims], the means of each gaussian
        sigmas: torch.tensor [steps, num_components, dims], the std dev of each gaussian
        weights: torch.tensor [steps, num_components], the mixture weights
        counts: torch.tensor [steps, num_components], number of samples assigned to each cluster
    """
    from multiprocessing import Pool, cpu_count
    
    batch_size, n_steps, n_dims = data.shape
    
    # Convert to numpy for sklearn
    data_np = data.cpu().numpy()
    
    # Prepare arguments for parallel processing
    max_components = min(num_components, batch_size)
    args_list = [
        (data_np[:, t, :], num_components, max_components) 
        for t in range(n_steps)
    ]
    
    # Fit GMMs in parallel
    if n_jobs == 1:
        # Sequential processing
        results = [_fit_gmm_single_timestep(args) for args in args_list]
    else:
        # Parallel processing
        n_processes = cpu_count() if n_jobs == -1 else n_jobs
        with Pool(processes=n_processes) as pool:
            results = pool.map(_fit_gmm_single_timestep, args_list)
    # Stack results into tensors
    means = torch.zeros(n_steps, num_components, n_dims)
    sigmas = torch.zeros(n_steps, num_components, n_dims)
    weights = torch.zeros(n_steps, num_components)
    counts = torch.zeros(n_steps, num_components, dtype=torch.int64)
    
    for t, (means_t, sigmas_t, weights_t, counts_t) in enumerate(results):
        means[t] = torch.from_numpy(means_t)
        sigmas[t] = torch.from_numpy(sigmas_t)
        weights[t] = torch.from_numpy(weights_t)
        counts[t] = torch.from_numpy(counts_t)
    
    return means, sigmas, weights, counts


# DIVERGENCE UTILS
def _construct_state_graph(data):
    """
    Construct a separate graph for each time step, showing how all batch samples
    are connected at that specific time.
    
    Args:
        data: Interpolated samples [batch, n_steps, dim]
    
    Returns:
        graphs: List of dictionaries, one per time step, each containing:
            - 'time_step': int
            - 'nodes': tensor [batch, dim] - states at this time step
            - 'distances': tensor [batch, batch] - pairwise distances
            - 'adjacency_list': dict mapping node_idx -> [(neighbor_idx, distance), ...]
    """
    batch_size, n_steps, dim = data.shape
    
    graphs = []
    
    for step in range(n_steps):
        # Extract all states at this time step [batch, dim]
        nodes_at_step = data[:, step, :]  # [batch, dim]
        
        # Compute pairwise distances between all states at this time step
        # Using broadcasting: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        nodes_norm_sq = (nodes_at_step ** 2).sum(dim=1, keepdim=True)  # [batch, 1]
        distances = torch.sqrt(
            nodes_norm_sq + nodes_norm_sq.T - 2 * nodes_at_step @ nodes_at_step.T
        )  # [batch, batch]
        
        # Create adjacency list for this time step
        adjacency_list = {}
        for i in range(batch_size):
            neighbors = []
            for j in range(batch_size):
                if i != j:  # Don't include self-loops
                    dist = distances[i, j].item()
                    neighbors.append((j, dist))
            adjacency_list[i] = neighbors
        
        graphs.append({
            'time_step': step,
            'nodes': nodes_at_step,
            'distances': distances,
            'adjacency_list': adjacency_list
        })
    
    return graphs


def _compute_divergence_via_neighbors(graphs, dt, k=4):
    """
    Estimates divergence by tracking the distance to the k-th neighbor.
    
    Args:
        graphs: List of state graphs (one per time step)
        dt: tensor [n_steps], Time between each step in graph
        k: Number of nearest neighbors
    
    Returns:
        divergence: tensor [batch, n_steps] - divergence at each state
    """
    n_steps = len(graphs)
    dim = graphs[0]['nodes'].shape[1]
    batch_size = graphs[0]['nodes'].shape[0]
    log_radii = []
    
    for graph in graphs:
        distances = graph['distances'] # [batch, batch]
        # Get distance to k-th nearest neighbor for every point
        # We use topk (smallest)
        # values, indices = torch.topk(distances, k+1, largest=False)
        # The 0-th neighbor is the point itself (dist=0), so take index k
        sorted_dists, _ = torch.sort(distances, dim=1)
        k_dist = sorted_dists[:, k] # [batch]
        
        # Keep the batch dimension
        log_radii.append(torch.log(k_dist + 1e-8))
        
    log_radii = torch.stack(log_radii, dim=1) # [batch, n_steps]
    
    # Divergence ≈ dim * d/dt(log radius)
    div_est = torch.zeros(batch_size, n_steps)
    div_est[:, 1:-1] = (log_radii[:, 2:] - log_radii[:, :-2]) / (dt[1:-1].unsqueeze(0) + dt[:-2].unsqueeze(0))  # Central difference for interior points: d/dt ≈ (f(t+dt) - f(t-dt)) / (dt[t+1] + dt[t])
    div_est[:, 0] = (log_radii[:, 1] - log_radii[:, 0]) / dt[0].unsqueeze(0)            # Forward difference for first point
    div_est[:, -1] = (log_radii[:, -1] - log_radii[:, -2]) / dt[-1].unsqueeze(0)        # Backward difference for last point

    return div_est * dim


def compute_divergence(data, dphase, k=4):
    """
    Method for computing approx. divergence at each time step using nearest neighbors
    Args:
        data: tensor [batch, n_steps, dim], the data we are calculating the divergence of
        dphase: tensor [n_steps], the change in the percent of task completion between points in time in the graph
        k: int, number of neighbors to consider when calculating the divergence
    Returns:
        divergence: tensor [batch_size, n_steps], the divergence of the flow at each datapoint
    """
    graphs = _construct_state_graph(data)

    return _compute_divergence_via_neighbors(graphs, dphase, k)


# MISC. UTILS
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


def visualize_observation_trajectories():
    """
    Method for visualizing trajectories to help understand what number of gmms should be fit to the traj's
    """
    pass


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

