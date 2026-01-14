import h5py
import numpy as np
import torch
import warnings
from robomimic.utils.dataset import SequenceDataset
import argparse
from robomimic.utils.tf_utils import compute_twist_between_poses, add_twist_to_pose
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torch.func import jvp

warnings.filterwarnings('ignore', message='logm result may be inaccurate')

############################
# Private methods
############################
# model divergence enstimation
def _estimate_divergence_jvp(model, batch, obs_key_pos='robot0_eef_pos', obs_key_quat='robot0_eef_quat', 
                            goal_dict=None, n_samples=1):
    """
    Estimates the exact local divergence of the policy's action field with respect to 
    end-effector pose perturbations using Forward-Mode Automatic Differentiation (JVP) 
    on the SE(3) manifold.
    
    This method works with all BC model architectures (MLP, RNN, Transformer, Gaussian, GMM, VAE)
    by:
    1. Extracting end-effector pose from obs_dict
    2. Perturbing the pose using SE(3) exponential map
    3. Computing action sensitivity via JVP
    4. Estimating divergence using Hutchinson's trace estimator
    
    **Handles Sequence Outputs:** For RNN/Transformer models that output action sequences
    [batch, time, action_dim], automatically extracts the last timestep action (the one
    that would be executed at the current state).
    
    **Handles Action Dimension Mismatch:** If action_dim ≠ 6, computes divergence on the
    overlapping subspace and scales appropriately.
    
    Algorithm Steps:
    ----------------
    1.  **Tangent Space Sampling**:
        Sample random probe vectors `epsilon` ~ N(0, I) with shape [batch, 6].
        These vectors define directions in the Lie Algebra se(3) along which
        we measure the model's sensitivity.

    2.  **Define Differentiable Manifold Wrapper**:
        Create a function that perturbs the end-effector pose in obs_dict:
        
        def f(twist):
            # Apply twist to EE pose in obs_dict
            perturbed_obs_dict = obs_dict with EE pose perturbed by twist
            # Run BC model (any architecture)
            actions = model(perturbed_obs_dict, goal_dict)
            # Extract last timestep if sequence output
            if actions.dim() == 3:
                actions = actions[:, -1, :]
            return actions

    3.  **Jacobian-Vector Product (JVP)**:
        Compute the directional derivative of model output w.r.t. twist perturbation
        at twist=0 along direction epsilon. This computes (J @ epsilon) efficiently.

    4.  **Trace Estimation (Hutchinson's Estimator)**:
        Divergence ≈ E[epsilon^T @ (J @ epsilon)]
        Average over multiple samples for better approximation.

    Args:
        model: BC policy network (ActorNetwork, GaussianActorNetwork, RNN_MIMO_MLP, 
               MIMO_Transformer, VAE, etc.). Must have forward(obs_dict, goal_dict) method.
        batch: Dictionary with 'obs' key containing observation dict with at least:
            - obs_key_pos: [batch, 3] end-effector position
            - obs_key_quat: [batch, 4] end-effector quaternion (wxyz or xyzw format)
        obs_key_pos: str, observation key for end-effector position (default: 'robot0_eef_pos')
        obs_key_quat: str, observation key for end-effector quaternion (default: 'robot0_eef_quat')
        goal_dict: Optional goal observations for goal-conditioned policies
        n_samples: int, number of random samples for Hutchinson estimator (default: 1)

    Returns:
        divergence: Tensor [batch] representing the estimated divergence scalar at each state.
        
    Note:
        - For sequence models (RNN/Transformer): Uses the last timestep action
        - For action_dim < 6: Scales divergence to account for subspace
        - For action_dim > 6: Uses first 6 dimensions (typically EE control)
    """
    
    # Extract observations and end-effector pose
    obs_dict = batch['obs']
    
    # Check if required keys exist
    if obs_key_pos not in obs_dict or obs_key_quat not in obs_dict:
        raise ValueError(f"obs_dict must contain '{obs_key_pos}' and '{obs_key_quat}' keys. "
                        f"Available keys: {list(obs_dict.keys())}")
    
    ee_pos = obs_dict[obs_key_pos]  # [batch, 3]
    ee_quat = obs_dict[obs_key_quat]  # [batch, 4]
    
    batch_size = ee_pos.shape[0]
    device = ee_pos.device
    dtype = ee_pos.dtype
    
    # Construct base end-effector pose [batch, 7] = [pos(3), quat(4)]
    base_ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)
    
    # Accumulate divergence estimates across samples
    divergence_samples = []
    
    for _ in range(n_samples):
        # Step 1: Sample random probe vector in tangent space (se(3) - 6D twist)
        # epsilon represents a random direction in the Lie algebra
        epsilon = torch.randn(batch_size, 6, device=device, dtype=dtype)
        
        # Step 2: Define differentiable manifold wrapper
        # This function applies twist perturbations to the EE pose in obs_dict
        def f(twist):
            """
            Apply twist perturbation to end-effector pose in observation dict.
            
            Args:
                twist: [batch, 6] - perturbation in tangent space se(3)
            
            Returns:
                [batch, action_dim] - model's action prediction
            """
            # Perturb the base end-effector pose using twist
            perturbed_ee_pose = add_twist_to_pose(base_ee_pose, twist, dt=1.0, w_first=False)
            
            # Split back into position and quaternion
            perturbed_pos = perturbed_ee_pose[:, :3]
            perturbed_quat = perturbed_ee_pose[:, 3:]
            
            # Create perturbed observation dictionary
            perturbed_obs_dict = {k: v.clone() for k, v in obs_dict.items()}
            perturbed_obs_dict[obs_key_pos] = perturbed_pos
            perturbed_obs_dict[obs_key_quat] = perturbed_quat
            
            # Run model with perturbed observations
            # This works for all BC architectures: MLP, RNN, Transformer, Gaussian, GMM, VAE
            output = model(obs_dict=perturbed_obs_dict, goal_dict=goal_dict)
            
            # Handle different output types:
            # - Deterministic policies (ActorNetwork): return actions directly [batch, action_dim]
            # - RNN/Transformer: return action sequences [batch, time, action_dim]
            # - Stochastic policies in eval mode (GaussianActorNetwork, GMMActorNetwork): return actions
            # - VAE: returns actions
            # For stochastic policies using forward_train(), they return distributions,
            # but here we use forward() which returns deterministic actions (means)
            
            # Handle sequence outputs (RNN/Transformer): take last timestep action
            # This is the action that would actually be executed at the current state
            if output.dim() == 3:
                # Output is [batch, time, action_dim], take last timestep
                output = output[:, -1, :]
            
            # Ensure output is 2D [batch, action_dim]
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            return output
        
        # Step 3: Compute Jacobian-Vector Product at zero perturbation
        # This efficiently computes J @ epsilon where J is the Jacobian d(actions)/d(twist)
        zero_twist = torch.zeros(batch_size, 6, device=device, dtype=dtype)
        
        # jvp returns (f(zero_twist), J @ epsilon)
        # Note: J is [action_dim, 6], so J @ epsilon is [action_dim]
        output, jvp_val = jvp(f, (zero_twist,), (epsilon,))
        
        # Step 4: Trace estimation using Hutchinson's estimator
        # Divergence = Trace(J) ≈ E[epsilon^T @ (J @ epsilon)]
        # 
        # If action_dim != 6, we need to handle the mismatch:
        # - If action_dim > 6: Take first 6 dimensions (typically EE pose control)
        # - If action_dim < 6: Pad or only use available dimensions
        # 
        # For robustness, we compute divergence on the minimum of the two dimensions
        action_dim = jvp_val.shape[-1]
        effective_dim = min(action_dim, 6)
        
        # Slice both vectors to effective dimension
        epsilon_slice = epsilon[:, :effective_dim]
        jvp_slice = jvp_val[:, :effective_dim]
        
        # Trace estimate: epsilon^T @ (J @ epsilon)
        # For the full 6x6 block of J, we use effective_dim
        divergence_sample = torch.sum(epsilon_slice * jvp_slice, dim=-1)  # [batch]
        
        # Scale by dimension to get proper divergence estimate
        # (since we're only measuring divergence in a subspace)
        if effective_dim < 6:
            divergence_sample = divergence_sample * (6.0 / effective_dim)
        
        divergence_samples.append(divergence_sample)
    
    # Average over samples for better estimate
    divergence = torch.stack(divergence_samples).mean(dim=0)  # [batch]
    
    return divergence

# training data divergence and score estimation
def _load_training_data(hdf5_path, demo_keys=None, verbose=True):
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
            'phase': None,  # Placeholder, will be filled by _add_phase
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

def _construct_state_graph(x):
    """
    Constructs distance matrix for each time step.
    Removed the adjacency_list loop (O(N^2) in Python) as it wasn't used.
    """
    batch_size, n_steps, dim = x.shape
    graphs = []
    
    for step in range(n_steps):
        nodes_at_step = x[:, step, :]  # [batch, dim]
        
        if dim == 7:  # Poses
            # Expansion for pairwise twist computation
            # [batch, 1, 7] and [1, batch, 7] implicit broadcasting is usually 
            # faster/cleaner than explicit expand() + reshape()
            start_poses = nodes_at_step.unsqueeze(1) # [batch, 1, 7]
            end_poses = nodes_at_step.unsqueeze(0)   # [1, batch, 7]
            
            # Assuming compute_twist_between_poses handles broadcasting 
            # or flattened inputs. Keeping your structure for safety:
            start_flat = start_poses.expand(batch_size, batch_size, 7).reshape(-1, 7)
            end_flat = end_poses.expand(batch_size, batch_size, 7).reshape(-1, 7)
            
            twists = compute_twist_between_poses(start_flat, end_flat, w_first=False) 
            twists = twists.reshape(batch_size, batch_size, 6)
            distances = torch.norm(twists, dim=2)
            
        else:
            # Efficient Euclidean distance
            nodes_norm_sq = (nodes_at_step ** 2).sum(dim=1, keepdim=True)
            distances = torch.sqrt(
                (nodes_norm_sq + nodes_norm_sq.T - 2 * nodes_at_step @ nodes_at_step.T).clamp(min=1e-8)
            )
        
        graphs.append({
            'time_step': step,
            'nodes': nodes_at_step,
            'distances': distances
        })
    
    return graphs

def _compute_divergence_via_neighbors(graphs, dt, k=4, window_length=9, polyorder=2):
    """
    Estimates divergence using Savitzky-Golay filter for smooth differentiation.
    
    Args:
        dt: float or tensor. If tensor, we assume roughly constant steps for SavGol.
            If dt varies significantly, SavGol is not strictly valid.
        window_length: Length of SavGol filter window (must be odd).
        polyorder: Order of polynomial to fit.
    """
    n_steps = len(graphs)
    dim = graphs[0]['nodes'].shape[1]
    batch_size = graphs[0]['nodes'].shape[0]
    
    manifold_dim = 6 if dim == 7 else dim
    log_radii = []
    
    for graph in graphs:
        distances = graph['distances']
        # Sort to find k-th neighbor
        sorted_dists, _ = torch.sort(distances, dim=1)
        k_dist = sorted_dists[:, k]
        log_radii.append(torch.log(k_dist + 1e-8))
        
    log_radii = torch.stack(log_radii, dim=1) # [batch, n_steps]
    
    # --- SavGol Filtering ---
    # Move to CPU/Numpy for Scipy
    log_radii_np = log_radii.cpu().numpy()
    
    # Handle dt: SavGol assumes unit spacing, so we divide by dt afterwards.
    # If dt is a tensor, we assume the mean dt for the scale factor.
    if isinstance(dt, torch.Tensor):
        delta_t = dt.mean().item()
    else:
        delta_t = dt
        
    # Apply filter with deriv=1 to get d(log_R)/dt directly
    # axis=1 operates along the time dimension
    d_log_radii_np = savgol_filter(
        log_radii_np, 
        window_length=window_length, 
        polyorder=polyorder, 
        deriv=1, 
        delta=delta_t, 
        axis=1,
        mode='interp' # Handles boundaries better than raw Finite Diff
    )
    
    div_est = torch.from_numpy(d_log_radii_np).to(log_radii.device)
    
    return div_est * manifold_dim

def _compute_divergence(data, dphase, k=4):
    """
    Method for computing approx. divergence at each time step using nearest neighbors
    Args:
        data: tensor [batch, n_steps, dim], the data we are calculating the divergence of (can contain NaN values)
        phase: tensor [batch, n_steps, 1] or [batch, n_steps], the phase values at each time step (can contain NaN values)
        k: int, number of neighbors to consider when calculating the divergence
        time_window: int, number of timesteps before/after to include when finding neighbors (default 2)
    Returns:
        divergence: tensor [batch_size, n_steps], the divergence of the flow at each datapoint (NaN where input is NaN)
    """
    
    graphs = _construct_state_graph(data)

    return _compute_divergence_via_neighbors(graphs, dphase, k)

def _add_phase(data):
    """
    Add phase information to each demonstration trajectory.
    Phase goes from 0 to 1 linearly over the course of each trajectory.
    
    Args:
        data: Dictionary returned by _load_training_data() containing HDF5 file handle
    
    Returns:
        data: Updated data dictionary where each demo now has 'phase' accessible via data[demo_key]['phase']
    """
    demos = data['demos']
    
    # Store phase data in the data dictionary indexed by demo key
    for demo_key in demos:
        # get the number of trajectory steps
        actions = data['get_actions'](demo_key)
        n_steps = len(actions)
        
        # make a linspace vector that goes from 0 to 1 (inclusive) with the number of trajectory steps
        phase = torch.linspace(0, 1, n_steps)
        
        # Store the phase directly in data[demo_key]
        if demo_key not in data:
            data[demo_key] = {}
        data[demo_key]['phase'] = phase
    
    return data

def _bin_data(data, n_bins=100):
    """
    Create a phase tensor with fixed number of bins for all demonstrations. Each demo's phase values are placed in their nearest bins, with linear interpolation
    in twist space used to fill in the missing states.
    
    Args:
        data: Dictionary returned by _load_training_data() with phase added via _add_phase()
        n_bins: int, number of phase bins (default 100)
    
    Returns:
        phase_tensor: torch.tensor [n_demos, n_bins, 1]
        ee_state_tensor: torch.tensor [n_demos, n_bins, 7], ee_pose (pos + quat) with missing steps filled in using twist interpolation between poses
        demo_keys: list of demo keys corresponding to each row
        nan_mask: torch.tensor [n_demos, n_bins], boolean mask indicating which bins were originally NaN before interpolation
    """
    demos = data['demos']
    n_demos = len(demos)
    
    # Initialize tensors with NaN
    phase_tensor = torch.full((n_demos, n_bins, 1), float('nan'))
    ee_state_tensor = torch.full((n_demos, n_bins, 7), float('nan'))
    demo_keys = []
    
    # compute the dphase step that we are using for binning
    dphase = 1.0 / (n_bins - 1)

    # Process each demo
    for i, demo_key in enumerate(demos):
        demo_keys.append(demo_key)
        
        # Get phase for this demo
        phase = data[demo_key]['phase']  # [n_steps]
        
        # Get end-effector state (position + quaternion)
        ee_pos = data['get_obs'](demo_key, 'robot0_eef_pos')  # [n_steps, 3]
        ee_quat = data['get_obs'](demo_key, 'robot0_eef_quat')  # [n_steps, 4]
        
        # Convert to tensors if needed
        if not isinstance(ee_pos, torch.Tensor):
            ee_pos = torch.from_numpy(ee_pos) if isinstance(ee_pos, np.ndarray) else torch.tensor(ee_pos)
        if not isinstance(ee_quat, torch.Tensor):
            ee_quat = torch.from_numpy(ee_quat) if isinstance(ee_quat, np.ndarray) else torch.tensor(ee_quat)
        
        # Concatenate position and quaternion to form 7D state
        ee_state = torch.cat([ee_pos, ee_quat], dim=1)  # [n_steps, 7]
        
        n_steps = len(phase)
        
        # Map each phase value to its nearest bin
        # Phase goes from 0 to 1, bins go from 0 to n_bins-1
        bin_indices = torch.round(phase * (n_bins - 1)).long()
        
        # Place phase and ee_state in their corresponding bins
        for step_idx, bin_idx in enumerate(bin_indices):
            phase_tensor[i, bin_idx, 0] = phase[step_idx]
            ee_state_tensor[i, bin_idx] = ee_state[step_idx]
        
    nan_mask = torch.isnan(ee_state_tensor).any(dim=2)  # [n_demos, n_bins]

    # go through the bins and do linear interpolation for any missing ee_states
    for i in range(n_demos):
        # Get mask for this demo
        demo_nan_mask = nan_mask[i]  # [n_bins]
        
        # Find indices of valid (non-NaN) bins
        valid_indices = torch.where(~demo_nan_mask)[0]
        
        if len(valid_indices) < 2:
            # Can't interpolate with fewer than 2 valid points
            continue
        
        # Interpolate between each pair of consecutive valid bins
        for j in range(len(valid_indices) - 1):
            start_idx = valid_indices[j].item()
            end_idx = valid_indices[j + 1].item()
            
            # Only interpolate if there's a gap
            if end_idx - start_idx > 1:
                # Get start and end poses [7] = [pos(3), quat(4)]
                start_pose = ee_state_tensor[i, start_idx]  # [7]
                end_pose = ee_state_tensor[i, end_idx]  # [7]
                
                # Compute twist from start to end (note: poses are [pos, quat])
                # compute_twist_between_poses expects [batch, 7] or [7]
                twist = compute_twist_between_poses(start_pose.unsqueeze(0), end_pose.unsqueeze(0), w_first=False)  # [1, 6]
                twist = twist.squeeze(0)  # [6]
                
                # Number of interpolation steps
                n_interp = end_idx - start_idx
                
                # Interpolate using twist
                for k in range(1, n_interp):
                    # Linear interpolation factor
                    alpha = k / n_interp
                    
                    # Apply scaled twist to start pose
                    # print(f"Interpolating demo {i}, bin {start_idx} to {end_idx}, step {k}/{n_interp}, alpha={alpha:.3f}")
                    start_pose = start_pose.unsqueeze(0)  # [1, 7]
                    twist = twist.unsqueeze(0)  # [1, 6]
                    # print(f"  Start pose: {start_pose.shape}, End pose: {end_pose.shape}, Twist: {twist.shape}")
                    dt=torch.tensor(alpha).unsqueeze(0).unsqueeze(0)
                    # print(f"  Applying twist for dt {dt} with shape {dt.shape}")
                    interp_pose = add_twist_to_pose(start_pose, twist, dt=dt, w_first=False)  # [1, 7]
                    interp_pose = interp_pose.squeeze(0)  # [7]
                    
                    # Fill in the interpolated bin
                    bin_idx = start_idx + k
                    ee_state_tensor[i, bin_idx] = interp_pose
                    phase_tensor[i, bin_idx, 0] = phase_tensor[i, start_idx, 0] + alpha * (phase_tensor[i, end_idx, 0] - phase_tensor[i, start_idx, 0])
    
    return phase_tensor, ee_state_tensor, demo_keys, nan_mask, dphase

def _unbin_data(data, demo_keys, divergence, score, nan_mask):
    """
    Take the twist divergence data that has been binned and map it back to the original
    time steps of each demonstration in data.
    Args:
        data: Dictionary returned by _load_training_data() with phase added via _add_phase()
        demo_keys: list of demo keys corresponding to each row in div_twists
        divergence: tensor [n_demos, n_bins], divergence values in binned format
        score: tensor [n_demos, n_bins, 6], score values in binned format
        nan_mask: tensor [n_demos, n_bins], boolean mask indicating which bins were originally NaN before interpolation
    Returns:
        data: the data dictionary updated so that each demo now has 'divergence' accessible via data[demo_key]['divergence']
    """
        
    for i, demo_key in enumerate(demo_keys):
        # get diveregence for this demo where there were no NaNs originally
        demo_div = divergence[i][~nan_mask[i]]  # [n_valid_bins]
        demo_score = score[i][~nan_mask[i]]  # [n_valid_bins, 6]
        # print(f"Unbinned divergence for {demo_key}: shape={demo_div.shape}")
        data[demo_key]['divergence'] = demo_div
        data[demo_key]['score'] = demo_score
    
    return data

def _save_data_structure(data, save_path):
    """
    Save the data structure to a HDF5 file for later use.
    
    Args:
        data: Dictionary returned by _load_training_data().
        save_path: str, path to save the data file.
    """
    import shutil
    
    # Get the original file path
    original_file = data['data'].filename
    
    # Close the original file handle to allow copying
    data['data'].close()
    
    # Copy the entire HDF5 file to the new location
    shutil.copy2(original_file, save_path)
    
    # Open the new file in read-write mode
    with h5py.File(save_path, 'r+') as f:
        # Add divergence and phase data to each demo
        for demo_key in data['demos']:
            demo_group = f['data'][demo_key]
            
            # Add phase if it exists
            if demo_key in data and 'phase' in data[demo_key]:
                phase = data[demo_key]['phase']
                if isinstance(phase, torch.Tensor):
                    phase = phase.cpu().numpy()
                
                # Delete if already exists (for overwriting)
                if 'phase' in demo_group:
                    del demo_group['phase']
                demo_group.create_dataset('phase', data=phase)
            
            # Add divergence if it exists
            if demo_key in data and 'divergence' in data[demo_key]:
                divergence = data[demo_key]['divergence']
                if isinstance(divergence, torch.Tensor):
                    divergence = divergence.cpu().numpy()
                
                # Delete if already exists (for overwriting)
                if 'divergence' in demo_group:
                    del demo_group['divergence']
                demo_group.create_dataset('divergence', data=divergence)
    
    # Reopen the original file for continued use
    data['data'] = h5py.File(original_file, 'r')

    print(f"Data structure saved to {save_path}")

def _compute_score_via_neighbors(graphs, bandwidth=0.1):
    """
    Estimates the Stein Score (grad log p) of the state distribution
    using a Riemannian Gaussian Kernel (Mean Shift).
    
    Args:
        graphs: List of state graphs (from _construct_state_graph)
        bandwidth (float): The 'sigma' of the kernel. Controls smoothness.
                           Too small = noisy score pointing to nearest single neighbor.
                           Too large = score points to global mean (oversmoothed).
    
    Returns:
        scores: Tensor [batch, n_steps, 6] - The score vector (twist) at each state.
    """
    batch_size = graphs[0]['nodes'].shape[0]
    n_steps = len(graphs)
    scores = []

    for graph in graphs:
        # nodes: [batch, 7] (Poses)
        current_poses = graph['nodes']
        
        # 1. Compute pairwise twists between ALL particles at this step
        # (Reusing your efficient batch expansion logic)
        # start_poses (The reference points x_i): [batch, 1, 7]
        start_poses = current_poses.unsqueeze(1)
        # end_poses (The neighbors x_j): [1, batch, 7]
        end_poses = current_poses.unsqueeze(0)
        
        start_flat = start_poses.expand(batch_size, batch_size, 7).reshape(-1, 7)
        end_flat = end_poses.expand(batch_size, batch_size, 7).reshape(-1, 7)
        
        # twists_flat: Vectors pointing FROM row_i TO col_j
        twists_flat = compute_twist_between_poses(start_flat, end_flat) 
        twists = twists_flat.reshape(batch_size, batch_size, 6) # [batch, batch, 6]
        
        # 2. Compute Distances and Weights (Gaussian Kernel)
        # dist_sq: [batch, batch]
        dist_sq = (twists ** 2).sum(dim=2) 
        weights = torch.exp(-dist_sq / (2 * bandwidth**2))
        
        # Mask self-loops (optional, but usually good to avoid bias from self-weight 1.0)
        # However, for KDE, including self is mathematically valid (it adds 0 vector).
        
        # 3. Compute Mean Shift Vector
        # weighted_sum: [batch, 6] -> sum over dimension 1 (neighbors)
        # weights must be reshaped to [batch, batch, 1] for broadcasting
        weighted_sum = (weights.unsqueeze(-1) * twists).sum(dim=1)
        sum_of_weights = weights.sum(dim=1, keepdim=True)
        
        # The Score Estimate
        mean_shift = weighted_sum / (sum_of_weights + 1e-8)
        score_t = mean_shift / (bandwidth**2)
        
        scores.append(score_t)

    scores = torch.stack(scores, dim=1) # [batch, n_steps, 6]
    return scores

def _compute_score(data, bandwidth=0.1):
    """
    Method for computing approx. score at each time step using nearest neighbors
    Args:
        data: tensor [batch, n_steps, dim], the data we are calculating the score of (can contain NaN values)
        k: int, number of neighbors to consider when calculating the score
        bandwidth: float, bandwidth parameter for the Gaussian kernel
    Returns:
        score: tensor [batch_size, n_steps, 6], the score of the flow at each datapoint (NaN where input is NaN)
    """
    
    graphs = _construct_state_graph(data)

    return _compute_score_via_neighbors(graphs, bandwidth)

############################
# Public methods
############################
# model divergence enstimation
def compute_policy_divergence_during_training(model, batch, n_samples=3):
    """
    Convenience wrapper to compute policy divergence during BC training.
    This function can be called directly from train_on_batch() methods.
    
    Works with all BC model architectures:
    - BC (MLP): ActorNetwork
    - BC_RNN: RNN_MIMO_MLP  
    - BC_Transformer: MIMO_Transformer
    - BC_Gaussian: GaussianActorNetwork
    - BC_GMM: GMMActorNetwork
    - BC_VAE: VAE
    
    Args:
        model: The BC policy network (self.nets["policy"] from BC algo)
        batch: Training batch from process_batch_for_training() containing:
            - batch['obs']: observation dictionary with EE pose keys
            - batch['goal_obs']: optional goal observations
        n_samples: Number of samples for Hutchinson estimator (default: 3, increase for accuracy)
    
    Returns:
        divergence: Tensor [batch_size] with divergence values, or None if EE pose not available
        
    Example usage in BC.train_on_batch():
        ```python
        # After forward pass
        predictions = self._forward_training(batch)
        
        # Compute divergence (optional)
        try:
            divergence = compute_policy_divergence_during_training(
                self.nets["policy"], 
                batch, 
                n_samples=1
            )
            info["divergence"] = divergence.mean().item()  # Log average divergence
        except Exception as e:
            # Divergence computation failed (e.g., no EE pose in obs)
            pass
        ```
    """
    try:
        # Get goal observations if present
        goal_dict = batch.get('goal_obs', None)
        
        # Compute divergence using JVP method
        divergence = _estimate_divergence_jvp(
            model=model,
            batch=batch,
            goal_dict=goal_dict,
            n_samples=n_samples
        )
        return divergence
        
    except (KeyError, ValueError) as e:
        # EE pose keys not in observations, cannot compute divergence
        return None

# training data divergence and score estimation
def add_div_and_score_to_training_data(load_path, save_path=None, verbose=True, k=4, bandwidth=0.1):
    """
    Compute and add divergence information to each demonstration in the HDF5 dataset.
    
    Args:
        hdf5_path: str, path to the HDF5 dataset file.
        save_path: str, path to save the updated HDF5 dataset (if None, overwrites with load_path).
        verbose: bool, whether to print progress information.
        nn_k: int, number of nearest neighbors to use for divergence computation.
    """
    if save_path is None:
        save_path = load_path
        # add "_w_div" suffix to filename before the extension
        if '.' in load_path:
            save_path = load_path.rsplit('.', 1)[0] + "_w_cdm." + load_path.rsplit('.', 1)[1]
        else:
            save_path = load_path + "_w_cdm"

    if verbose:
        print(f"Loading training data from: {load_path}")
    data = _load_training_data(load_path)
    
    if verbose:
        print(f"Adding phase information to data...")
    data = _add_phase(data)
    
    # calculate trajectory length stats
    n_actions = []
    for demo_key in data['demos']:
        actions = data['get_actions'](demo_key)
        n_actions.append(len(actions))
    max_n_actions = np.max(n_actions)
    
    if verbose:
        print(f"Training data contains {len(data['demos'])} demos")
        print(f"Trajectory length stats (actions): min={np.min(n_actions)}, mean={np.mean(n_actions):.1f}, max={np.max(n_actions)}")
        print(f"\nComputing divergence of ee_states binned into {max_n_actions} bins:")
    phase_tensor, ee_state_tensor, demo_tensor_keys, nan_mask, dphase = _bin_data(data, n_bins=max_n_actions)
    
    # Count NaNs per bin
    nan_counts = torch.isnan(ee_state_tensor).any(dim=2).sum(dim=0)
    if verbose:
        print(f"NaNs per bin (min/mean/max): {nan_counts.min()}/{nan_counts.float().mean():.1f}/{nan_counts.max()}")
        print(f"\nComputing divergence and score of ee_states:")
    state_graphs = _construct_state_graph(ee_state_tensor)
    div_ee = _compute_divergence_via_neighbors(state_graphs, dphase,  k=k)
    score_ee = _compute_score_via_neighbors(state_graphs, bandwidth=bandwidth)

    if verbose:
        print(f"    Divergence tensor shape: {div_ee.shape}")
        valid_div = div_ee[~torch.isnan(div_ee)]
        print(f"    Divergence valid count: {valid_div.shape[0]}/{div_ee.numel()}")
        print(f"    Divergence stats (min/mean/max): {valid_div.min():.4f}/{valid_div.mean():.4f}/{valid_div.max():.4f}")
        print(f"    Score tensor shape: {score_ee.shape}")
        valid_score = score_ee[~torch.isnan(score_ee).any(dim=2)]
        print(f"    Score valid count: {valid_score.shape[0]}/{score_ee.shape[0] * score_ee.shape[1]}")

    if verbose:
        print(f"\nMapping divergence and score back to original demo time steps:")
    data = _unbin_data(data, demo_tensor_keys, div_ee, score_ee, nan_mask)
    
    # Verify unbinning by checking a few demos
    for demo_key in data['demos'][:3]:
        print("\n" + "-"*40)
        print(f"\n    Divergence stats for demo '{demo_key}':")
        if 'divergence' in data[demo_key]:
            div_twist_demo = data[demo_key]['divergence']
            valid_div_demo = div_twist_demo[~torch.isnan(div_twist_demo)]
            print(f"    {demo_key}: shape={div_twist_demo.shape}, valid={valid_div_demo.shape[0]}/{div_twist_demo.numel()}")
            if valid_div_demo.numel() > 0:
                print(f"      stats (min/mean/max): {valid_div_demo.min():.4f}/{valid_div_demo.mean():.4f}/{valid_div_demo.max():.4f}")
        
        print(f"\n    Score stats per component:")
        if 'score' in data[demo_key]:
            score_twist_demo = data[demo_key]['score']
            valid_score_demo = score_twist_demo[~torch.isnan(score_twist_demo).any(dim=1)]
            print(f"    {demo_key}: shape={score_twist_demo.shape}, valid={valid_score_demo.shape[0]}/{score_twist_demo.shape[0]}")
            if valid_score_demo.numel() > 0:
                print(f"      stats (min/mean/max) per component:")
                for i in range(score_twist_demo.shape[1]):
                    comp = valid_score_demo[:, i]
                    print(f"        Component {i}: {comp.min():.4f}/{comp.mean():.4f}/{comp.max():.4f}")

    if verbose:
        print(f"\nFinished computing divergence and score for all demos.")
        # print new dictionary structure keys
        print(f"Data dictionary keys: {list(data.keys())}")

    if verbose:
        print(f"\nSaving updated data structure to: {save_path}")
    _save_data_structure(data, save_path)
    
# visualization
def visualize_observation_trajectories(ee_state_tensor, demo_keys, nan_mask=None, 
                                       divergence=None, phase_tensor=None,
                                       frame_skip=10, demo_indices=None, 
                                       figsize=(20, 12), title="Trajectory Visualization"):
    """
    Visualize trajectories in 3D space with coordinate frames and separate subplots for 
    translation and rotation components.
    
    Args:
        ee_state_tensor: torch.tensor [n_demos, n_bins, 7], end-effector poses (pos + quat)
        demo_keys: list of demo keys corresponding to each trajectory
        nan_mask: torch.tensor [n_demos, n_bins], boolean mask for NaN values (optional)
        divergence: torch.tensor [n_demos, n_bins], divergence values (optional)
        phase_tensor: torch.tensor [n_demos, n_bins, 1], phase values (optional)
        frame_skip: int, plot coordinate frames every N steps (default 10)
        demo_indices: list of indices to visualize (if None, visualizes all demos)
        figsize: tuple, figure size (default (20, 12))
        title: str, overall figure title
    
    Returns:
        fig: matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from tf_utils import _quat_to_rot_mat
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            
            return np.min(zs)
    
    # Select demos to visualize
    n_demos, n_bins, _ = ee_state_tensor.shape
    if demo_indices is None:
        demo_indices = list(range(n_demos))
    
    # Convert to numpy for plotting
    ee_state_np = ee_state_tensor.cpu().numpy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.4, height_ratios=[1, 1, 1, 0.8])
    
    # 3D trajectory plot (spans left 3 columns, all 3 top rows)
    ax_3d = fig.add_subplot(gs[:3, :3], projection='3d')
    
    # Translation subplots (column 3)
    ax_x = fig.add_subplot(gs[0, 3])
    ax_y = fig.add_subplot(gs[1, 3])
    ax_z = fig.add_subplot(gs[2, 3])
    
    # Rotation subplots (columns 4-5, sin and cos for each angle)
    ax_roll_sin = fig.add_subplot(gs[0, 4])
    ax_roll_cos = fig.add_subplot(gs[0, 5])
    ax_pitch_sin = fig.add_subplot(gs[1, 4])
    ax_pitch_cos = fig.add_subplot(gs[1, 5])
    ax_yaw_sin = fig.add_subplot(gs[2, 4])
    ax_yaw_cos = fig.add_subplot(gs[2, 5])
    
    # Divergence subplot (bottom row, spans all columns)
    ax_div = fig.add_subplot(gs[3, :])
    
    # Color map for different demos
    colors = plt.cm.tab20(np.linspace(0, 1, len(demo_indices)))
    
    # Determine which demos to show in 3D (limit to 10 for clarity)
    demo_indices_3d = demo_indices if len(demo_indices) <= 10 else demo_indices[:10]
    
    # Adjust transparency based on number of trajectories
    line_alpha = 0.6 if len(demo_indices) <= 10 else 0.3
    
    for idx, demo_idx in enumerate(demo_indices):
        demo_key = demo_keys[demo_idx]
        color = colors[idx]
        
        # Get trajectory for this demo
        traj = ee_state_np[demo_idx]  # [n_bins, 7]
        
        # Get valid indices (non-NaN)
        if nan_mask is not None:
            valid_mask = ~nan_mask[demo_idx].cpu().numpy()
        else:
            valid_mask = ~np.isnan(traj).any(axis=1)
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue
        
        # Extract position and quaternion
        positions = traj[valid_indices, :3]  # [n_valid, 3]
        quaternions = traj[valid_indices, 3:]  # [n_valid, 4]
        
        # Plot 3D trajectory (only for subset if many demos)
        if demo_idx in demo_indices_3d:
            ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       color=color, alpha=0.6, linewidth=1.5, 
                       label=f"{demo_key}" if len(demo_indices_3d) <= 10 else None)
        
        # Plot coordinate frames at regular intervals (only for 3D subset)
        if demo_idx in demo_indices_3d:
            frame_indices = valid_indices[::frame_skip]
            for frame_idx in frame_indices:
                pos = traj[frame_idx, :3]
                quat = torch.from_numpy(traj[frame_idx, 3:]).unsqueeze(0)
                
                # Convert quaternion to rotation matrix
                rot_mat = _quat_to_rot_mat(quat, w_first=False).squeeze(0).numpy()
            
                # Draw coordinate frame axes - extract and normalize each axis
                axis_length = 0.02  # Length of coordinate frame axes
                
                # Extract each axis from rotation matrix columns
                x_axis = rot_mat[:, 0]  # First column
                y_axis = rot_mat[:, 1]  # Second column
                z_axis = rot_mat[:, 2]  # Third column
                
                # Normalize to ensure unit length (should already be normalized, but just in case)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
                
                # X-axis (red)
                ax_3d.add_artist(Arrow3D(
                    pos[0], pos[1], pos[2],
                    x_axis[0] * axis_length,
                    x_axis[1] * axis_length,
                    x_axis[2] * axis_length,
                    mutation_scale=10, lw=1.5, arrowstyle='-|>', color='red', alpha=0.5
                ))
                
                # Y-axis (green)
                ax_3d.add_artist(Arrow3D(
                    pos[0], pos[1], pos[2],
                    y_axis[0] * axis_length,
                    y_axis[1] * axis_length,
                    y_axis[2] * axis_length,
                    mutation_scale=10, lw=1.5, arrowstyle='-|>', color='green', alpha=0.5
                ))
                
                # Z-axis (blue)
                ax_3d.add_artist(Arrow3D(
                    pos[0], pos[1], pos[2],
                    z_axis[0] * axis_length,
                    z_axis[1] * axis_length,
                    z_axis[2] * axis_length,
                    mutation_scale=10, lw=1.5, arrowstyle='-|>', color='blue', alpha=0.5
                ))
        
        # Convert quaternions to Euler angles (roll, pitch, yaw) for rotation plots
        # Using rotation matrices to extract Euler angles
        rot_mats = _quat_to_rot_mat(torch.from_numpy(quaternions), w_first=False).numpy()
        
        # Extract roll, pitch, yaw from rotation matrices
        # Using ZYX Euler convention
        roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
        pitch = np.arctan2(-rot_mats[:, 2, 0], 
                          np.sqrt(rot_mats[:, 2, 1]**2 + rot_mats[:, 2, 2]**2))
        yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
        
        # Compute sine and cosine of angles to avoid discontinuities at ±π
        roll_sin = np.sin(roll)
        roll_cos = np.cos(roll)
        pitch_sin = np.sin(pitch)
        pitch_cos = np.cos(pitch)
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)
        
        # Extract phase values for plotting
        if phase_tensor is not None:
            phase_values = phase_tensor[demo_idx, valid_indices, 0].cpu().numpy()
        else:
            phase_values = valid_indices  # Fall back to time steps if no phase
        
        # Plot translation components
        ax_x.plot(phase_values, positions[:, 0], color=color, alpha=line_alpha, linewidth=1.5)
        ax_y.plot(phase_values, positions[:, 1], color=color, alpha=line_alpha, linewidth=1.5)
        ax_z.plot(phase_values, positions[:, 2], color=color, alpha=line_alpha, linewidth=1.5)
        
        # Plot rotation components (sine and cosine to avoid discontinuities)
        ax_roll_sin.plot(phase_values, roll_sin, color=color, alpha=line_alpha, linewidth=1.5)
        ax_roll_cos.plot(phase_values, roll_cos, color=color, alpha=line_alpha, linewidth=1.5)
        ax_pitch_sin.plot(phase_values, pitch_sin, color=color, alpha=line_alpha, linewidth=1.5)
        ax_pitch_cos.plot(phase_values, pitch_cos, color=color, alpha=line_alpha, linewidth=1.5)
        ax_yaw_sin.plot(phase_values, yaw_sin, color=color, alpha=line_alpha, linewidth=1.5)
        ax_yaw_cos.plot(phase_values, yaw_cos, color=color, alpha=line_alpha, linewidth=1.5)
        
        # Plot divergence if provided
        if divergence is not None:
            div_values = divergence[demo_idx].cpu().numpy()
            # Get valid divergence values (non-NaN)
            valid_div_mask = ~np.isnan(div_values)
            valid_div_indices = np.where(valid_div_mask)[0]
            
            if len(valid_div_indices) > 0:
                # Use phase if available, otherwise use bin indices
                if phase_tensor is not None:
                    phase_values = phase_tensor[demo_idx, valid_div_indices, 0].cpu().numpy()
                    ax_div.plot(phase_values, div_values[valid_div_indices], 
                               color=color, alpha=line_alpha, linewidth=1.5)
                else:
                    ax_div.plot(valid_div_indices, div_values[valid_div_indices], 
                               color=color, alpha=line_alpha, linewidth=1.5)
    
    # Configure 3D plot
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Z Position (m)')
    title_3d = '3D Trajectories with Coordinate Frames'
    if len(demo_indices) > 10:
        title_3d += f' (showing {len(demo_indices_3d)}/{len(demo_indices)})'
    ax_3d.set_title(title_3d, fontsize=12, fontweight='bold')
    if len(demo_indices_3d) <= 10:
        ax_3d.legend(loc='upper right', fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    
    # Configure translation subplots
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('Translation - X', fontsize=10, fontweight='bold')
    ax_x.grid(True, alpha=0.3)
    
    ax_y.set_ylabel('Y Position (m)')
    ax_y.set_title('Translation - Y', fontsize=10, fontweight='bold')
    ax_y.grid(True, alpha=0.3)
    
    ax_z.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Translation - Z', fontsize=10, fontweight='bold')
    ax_z.grid(True, alpha=0.3)
    
    # Configure rotation subplots
    ax_roll_sin.set_ylabel('sin(Roll)')
    ax_roll_sin.set_title('Roll - Sine', fontsize=10, fontweight='bold')
    ax_roll_sin.grid(True, alpha=0.3)
    ax_roll_sin.set_ylim([-1.1, 1.1])
    
    ax_roll_cos.set_ylabel('cos(Roll)')
    ax_roll_cos.set_title('Roll - Cosine', fontsize=10, fontweight='bold')
    ax_roll_cos.grid(True, alpha=0.3)
    ax_roll_cos.set_ylim([-1.1, 1.1])
    
    ax_pitch_sin.set_ylabel('sin(Pitch)')
    ax_pitch_sin.set_title('Pitch - Sine', fontsize=10, fontweight='bold')
    ax_pitch_sin.grid(True, alpha=0.3)
    ax_pitch_sin.set_ylim([-1.1, 1.1])
    
    ax_pitch_cos.set_ylabel('cos(Pitch)')
    ax_pitch_cos.set_title('Pitch - Cosine', fontsize=10, fontweight='bold')
    ax_pitch_cos.grid(True, alpha=0.3)
    ax_pitch_cos.set_ylim([-1.1, 1.1])
    
    ax_yaw_sin.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_yaw_sin.set_ylabel('sin(Yaw)')
    ax_yaw_sin.set_title('Yaw - Sine', fontsize=10, fontweight='bold')
    ax_yaw_sin.grid(True, alpha=0.3)
    ax_yaw_sin.set_ylim([-1.1, 1.1])
    
    ax_yaw_cos.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_yaw_cos.set_ylabel('cos(Yaw)')
    ax_yaw_cos.set_title('Yaw - Cosine', fontsize=10, fontweight='bold')
    ax_yaw_cos.grid(True, alpha=0.3)
    ax_yaw_cos.set_ylim([-1.1, 1.1])
    
    # Configure divergence subplot
    if divergence is not None:
        if phase_tensor is not None:
            ax_div.set_xlabel('Phase')
        else:
            ax_div.set_xlabel('Time Step')
        ax_div.set_ylabel('Divergence')
        ax_div.set_title('Divergence vs Phase', fontsize=10, fontweight='bold')
        ax_div.grid(True, alpha=0.3)
        ax_div.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    else:
        ax_div.text(0.5, 0.5, 'No divergence data', 
                   ha='center', va='center', transform=ax_div.transAxes, fontsize=12)
        ax_div.set_xticks([])
        ax_div.set_yticks([])
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Use tight_layout with rect to avoid overlap with suptitle
    # Skip tight_layout for 3D axes compatibility
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except:
        pass  # Ignore tight_layout issues with 3D plots
    
    return fig

def visualize_score_for_trajectories(
    ee_state_tensor,
    score_tensor,
    selected_demo_idx,
    demo_keys,
    nan_mask=None,
    phase_tensor=None,
    frame_skip=10,
    demo_indices=None,
    figsize=(24, 14),
    title="Trajectory Score Visualization"
):
    """
    Visualize a single trajectory with its score overlaid on all other trajectories.
    Shows the selected trajectory as a solid black line on top of all others to understand
    how it sits within the data distribution.
    
    Args:
        ee_state_tensor: torch.tensor [n_demos, n_bins, 7], end-effector poses (pos + quat)
        score_tensor: torch.tensor [n_bins, 6], score values (twist: linear xyz, angular xyz) for selected demo
        selected_demo_idx: int, index of the demo to highlight and show score for
        demo_keys: list of demo keys corresponding to each trajectory
        nan_mask: torch.tensor [n_demos, n_bins], boolean mask for NaN values (optional)
        phase_tensor: torch.tensor [n_demos, n_bins, 1], phase values (optional)
        frame_skip: int, plot coordinate frames every N steps (default 10)
        demo_indices: list of indices to visualize (if None, visualizes all demos)
        figsize: tuple, figure size (default (24, 14))
        title: str, overall figure title
    
    Returns:
        fig: matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from tf_utils import _quat_to_rot_mat
    import torch
    import numpy as np
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            
            return np.min(zs)
    
    # Select demos to visualize
    n_demos, n_bins, _ = ee_state_tensor.shape
    if demo_indices is None:
        demo_indices = list(range(n_demos))
    
    # Convert to numpy for plotting
    ee_state_np = ee_state_tensor.cpu().numpy()
    score_np = score_tensor.cpu().numpy()  # [n_bins, 6]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 6, hspace=0.4, wspace=0.4, height_ratios=[1, 1, 1, 0.8, 0.8])
    
    # 3D trajectory plot (spans left 3 columns, all 3 top rows)
    ax_3d = fig.add_subplot(gs[:3, :3], projection='3d')
    
    # Translation subplots (column 3)
    ax_x = fig.add_subplot(gs[0, 3])
    ax_y = fig.add_subplot(gs[1, 3])
    ax_z = fig.add_subplot(gs[2, 3])
    
    # Rotation subplots (columns 4-5, sin and cos for each angle)
    ax_roll_sin = fig.add_subplot(gs[0, 4])
    ax_roll_cos = fig.add_subplot(gs[0, 5])
    ax_pitch_sin = fig.add_subplot(gs[1, 4])
    ax_pitch_cos = fig.add_subplot(gs[1, 5])
    ax_yaw_sin = fig.add_subplot(gs[2, 4])
    ax_yaw_cos = fig.add_subplot(gs[2, 5])
    
    # Score subplots (4th row: linear components)
    ax_score_lin_x = fig.add_subplot(gs[3, 0:2])
    ax_score_lin_y = fig.add_subplot(gs[3, 2:4])
    ax_score_lin_z = fig.add_subplot(gs[3, 4:6])
    
    # Score subplots (5th row: angular components)
    ax_score_ang_x = fig.add_subplot(gs[4, 0:2])
    ax_score_ang_y = fig.add_subplot(gs[4, 2:4])
    ax_score_ang_z = fig.add_subplot(gs[4, 4:6])
    
    # Color map for different demos
    colors = plt.cm.tab20(np.linspace(0, 1, len(demo_indices)))
    
    # All demos shown with low transparency for context
    line_alpha = 0.15 if len(demo_indices) > 20 else 0.3
    
    # First pass: Plot all trajectories except the selected one
    for idx, demo_idx in enumerate(demo_indices):
        # Skip the selected demo - we'll plot it last on top
        if demo_idx == selected_demo_idx:
            continue
            
        demo_key = demo_keys[demo_idx]
        color = colors[idx]
        
        # Get trajectory for this demo
        traj = ee_state_np[demo_idx]  # [n_bins, 7]
        
        # Get valid indices (non-NaN)
        if nan_mask is not None:
            valid_mask = ~nan_mask[demo_idx].cpu().numpy()
        else:
            valid_mask = ~np.isnan(traj).any(axis=1)
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue
        
        # Extract position and quaternion
        positions = traj[valid_indices, :3]  # [n_valid, 3]
        quaternions = traj[valid_indices, 3:]  # [n_valid, 4]
        
        # Plot 3D trajectory with low alpha
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, alpha=line_alpha, linewidth=1.0)
        
        # Convert quaternions to Euler angles (roll, pitch, yaw) for rotation plots
        rot_mats = _quat_to_rot_mat(torch.from_numpy(quaternions), w_first=False).numpy()
        
        # Extract roll, pitch, yaw from rotation matrices (ZYX Euler convention)
        roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
        pitch = np.arctan2(-rot_mats[:, 2, 0], 
                          np.sqrt(rot_mats[:, 2, 1]**2 + rot_mats[:, 2, 2]**2))
        yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
        
        # Compute sine and cosine of angles to avoid discontinuities at ±π
        roll_sin = np.sin(roll)
        roll_cos = np.cos(roll)
        pitch_sin = np.sin(pitch)
        pitch_cos = np.cos(pitch)
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)
        
        # Extract phase values for plotting
        if phase_tensor is not None:
            phase_values = phase_tensor[demo_idx, valid_indices, 0].cpu().numpy()
        else:
            phase_values = valid_indices  # Fall back to time steps if no phase
        
        # Plot translation components
        ax_x.plot(phase_values, positions[:, 0], color=color, alpha=line_alpha, linewidth=1.0)
        ax_y.plot(phase_values, positions[:, 1], color=color, alpha=line_alpha, linewidth=1.0)
        ax_z.plot(phase_values, positions[:, 2], color=color, alpha=line_alpha, linewidth=1.0)
        
        # Plot rotation components (sine and cosine to avoid discontinuities)
        ax_roll_sin.plot(phase_values, roll_sin, color=color, alpha=line_alpha, linewidth=1.0)
        ax_roll_cos.plot(phase_values, roll_cos, color=color, alpha=line_alpha, linewidth=1.0)
        ax_pitch_sin.plot(phase_values, pitch_sin, color=color, alpha=line_alpha, linewidth=1.0)
        ax_pitch_cos.plot(phase_values, pitch_cos, color=color, alpha=line_alpha, linewidth=1.0)
        ax_yaw_sin.plot(phase_values, yaw_sin, color=color, alpha=line_alpha, linewidth=1.0)
        ax_yaw_cos.plot(phase_values, yaw_cos, color=color, alpha=line_alpha, linewidth=1.0)
    
    # Second pass: Plot the selected trajectory on top with solid black line
    selected_traj = ee_state_np[selected_demo_idx]  # [n_bins, 7]
    selected_demo_key = demo_keys[selected_demo_idx]
    
    # Get valid indices for selected demo
    if nan_mask is not None:
        selected_valid_mask = ~nan_mask[selected_demo_idx].cpu().numpy()
    else:
        selected_valid_mask = ~np.isnan(selected_traj).any(axis=1)
    
    selected_valid_indices = np.where(selected_valid_mask)[0]
    
    if len(selected_valid_indices) > 0:
        # Extract position and quaternion for selected demo
        selected_positions = selected_traj[selected_valid_indices, :3]  # [n_valid, 3]
        selected_quaternions = selected_traj[selected_valid_indices, 3:]  # [n_valid, 4]
        
        # Plot selected trajectory in 3D with solid black line
        ax_3d.plot(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2], 
                   color='black', alpha=1.0, linewidth=2.5, 
                   label=f"Selected: {selected_demo_key}", zorder=100)
        
        # Plot coordinate frames for selected trajectory
        frame_indices = selected_valid_indices[::frame_skip]
        for frame_idx in frame_indices:
            pos = selected_traj[frame_idx, :3]
            quat = torch.from_numpy(selected_traj[frame_idx, 3:]).unsqueeze(0)
            
            # Convert quaternion to rotation matrix
            rot_mat = _quat_to_rot_mat(quat, w_first=False).squeeze(0).numpy()
        
            # Draw coordinate frame axes
            axis_length = 0.025  # Slightly larger for visibility
            
            # Extract each axis from rotation matrix columns
            x_axis = rot_mat[:, 0]
            y_axis = rot_mat[:, 1]
            z_axis = rot_mat[:, 2]
            
            # Normalize to ensure unit length
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
            # X-axis (red)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                x_axis[0] * axis_length,
                x_axis[1] * axis_length,
                x_axis[2] * axis_length,
                mutation_scale=12, lw=2, arrowstyle='-|>', color='red', alpha=0.8
            ))
            
            # Y-axis (green)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                y_axis[0] * axis_length,
                y_axis[1] * axis_length,
                y_axis[2] * axis_length,
                mutation_scale=12, lw=2, arrowstyle='-|>', color='green', alpha=0.8
            ))
            
            # Z-axis (blue)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                z_axis[0] * axis_length,
                z_axis[1] * axis_length,
                z_axis[2] * axis_length,
                mutation_scale=12, lw=2, arrowstyle='-|>', color='blue', alpha=0.8
            ))
        
        # Convert quaternions to Euler angles for selected trajectory
        selected_rot_mats = _quat_to_rot_mat(torch.from_numpy(selected_quaternions), w_first=False).numpy()
        
        # Extract roll, pitch, yaw (ZYX Euler convention)
        selected_roll = np.arctan2(selected_rot_mats[:, 2, 1], selected_rot_mats[:, 2, 2])
        selected_pitch = np.arctan2(-selected_rot_mats[:, 2, 0], 
                                    np.sqrt(selected_rot_mats[:, 2, 1]**2 + selected_rot_mats[:, 2, 2]**2))
        selected_yaw = np.arctan2(selected_rot_mats[:, 1, 0], selected_rot_mats[:, 0, 0])
        
        # Compute sine and cosine
        selected_roll_sin = np.sin(selected_roll)
        selected_roll_cos = np.cos(selected_roll)
        selected_pitch_sin = np.sin(selected_pitch)
        selected_pitch_cos = np.cos(selected_pitch)
        selected_yaw_sin = np.sin(selected_yaw)
        selected_yaw_cos = np.cos(selected_yaw)
        
        # Extract phase values for selected trajectory
        if phase_tensor is not None:
            selected_phase_values = phase_tensor[selected_demo_idx, selected_valid_indices, 0].cpu().numpy()
        else:
            selected_phase_values = selected_valid_indices
        
        # Plot selected trajectory translation components with solid black line
        ax_x.plot(selected_phase_values, selected_positions[:, 0], color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_y.plot(selected_phase_values, selected_positions[:, 1], color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_z.plot(selected_phase_values, selected_positions[:, 2], color='black', alpha=1.0, linewidth=2.5, zorder=100)
        
        # Plot selected trajectory rotation components with solid black line
        ax_roll_sin.plot(selected_phase_values, selected_roll_sin, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_roll_cos.plot(selected_phase_values, selected_roll_cos, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_pitch_sin.plot(selected_phase_values, selected_pitch_sin, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_pitch_cos.plot(selected_phase_values, selected_pitch_cos, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_yaw_sin.plot(selected_phase_values, selected_yaw_sin, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        ax_yaw_cos.plot(selected_phase_values, selected_yaw_cos, color='black', alpha=1.0, linewidth=2.5, zorder=100)
        
        # Plot score components
        # Get valid score indices
        valid_score_mask = ~np.isnan(score_np).any(axis=1)
        valid_score_indices = np.where(valid_score_mask)[0]
        
        if len(valid_score_indices) > 0:
            if phase_tensor is not None:
                score_phase_values = phase_tensor[selected_demo_idx, valid_score_indices, 0].cpu().numpy()
            else:
                score_phase_values = valid_score_indices
            
            # Linear (translation) score components
            ax_score_lin_x.plot(score_phase_values, score_np[valid_score_indices, 0], 
                               color='darkred', linewidth=2)
            ax_score_lin_x.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_lin_x.grid(True, alpha=0.3)
            ax_score_lin_x.set_ylabel('Score (m/s)', fontsize=9)
            ax_score_lin_x.set_title('Score: Linear X', fontsize=10, fontweight='bold')
            
            ax_score_lin_y.plot(score_phase_values, score_np[valid_score_indices, 1], 
                               color='darkgreen', linewidth=2)
            ax_score_lin_y.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_lin_y.grid(True, alpha=0.3)
            ax_score_lin_y.set_ylabel('Score (m/s)', fontsize=9)
            ax_score_lin_y.set_title('Score: Linear Y', fontsize=10, fontweight='bold')
            
            ax_score_lin_z.plot(score_phase_values, score_np[valid_score_indices, 2], 
                               color='darkblue', linewidth=2)
            ax_score_lin_z.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_lin_z.grid(True, alpha=0.3)
            ax_score_lin_z.set_ylabel('Score (m/s)', fontsize=9)
            ax_score_lin_z.set_title('Score: Linear Z', fontsize=10, fontweight='bold')
            
            # Angular (rotation) score components
            ax_score_ang_x.plot(score_phase_values, score_np[valid_score_indices, 3], 
                               color='darkred', linewidth=2)
            ax_score_ang_x.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_ang_x.grid(True, alpha=0.3)
            ax_score_ang_x.set_xlabel('Phase' if phase_tensor is not None else 'Time Step', fontsize=9)
            ax_score_ang_x.set_ylabel('Score (rad/s)', fontsize=9)
            ax_score_ang_x.set_title('Score: Angular X', fontsize=10, fontweight='bold')
            
            ax_score_ang_y.plot(score_phase_values, score_np[valid_score_indices, 4], 
                               color='darkgreen', linewidth=2)
            ax_score_ang_y.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_ang_y.grid(True, alpha=0.3)
            ax_score_ang_y.set_xlabel('Phase' if phase_tensor is not None else 'Time Step', fontsize=9)
            ax_score_ang_y.set_ylabel('Score (rad/s)', fontsize=9)
            ax_score_ang_y.set_title('Score: Angular Y', fontsize=10, fontweight='bold')
            
            ax_score_ang_z.plot(score_phase_values, score_np[valid_score_indices, 5], 
                               color='darkblue', linewidth=2)
            ax_score_ang_z.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_score_ang_z.grid(True, alpha=0.3)
            ax_score_ang_z.set_xlabel('Phase' if phase_tensor is not None else 'Time Step', fontsize=9)
            ax_score_ang_z.set_ylabel('Score (rad/s)', fontsize=9)
            ax_score_ang_z.set_title('Score: Angular Z', fontsize=10, fontweight='bold')
    
    # Configure 3D plot
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectories (Selected in Black)', fontsize=12, fontweight='bold')
    ax_3d.legend(loc='upper right', fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    
    # Configure translation subplots
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('Translation - X', fontsize=10, fontweight='bold')
    ax_x.grid(True, alpha=0.3)
    
    ax_y.set_ylabel('Y Position (m)')
    ax_y.set_title('Translation - Y', fontsize=10, fontweight='bold')
    ax_y.grid(True, alpha=0.3)
    
    ax_z.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Translation - Z', fontsize=10, fontweight='bold')
    ax_z.grid(True, alpha=0.3)
    
    # Configure rotation subplots
    ax_roll_sin.set_ylabel('sin(Roll)')
    ax_roll_sin.set_title('Roll - Sine', fontsize=10, fontweight='bold')
    ax_roll_sin.grid(True, alpha=0.3)
    ax_roll_sin.set_ylim([-1.1, 1.1])
    
    ax_roll_cos.set_ylabel('cos(Roll)')
    ax_roll_cos.set_title('Roll - Cosine', fontsize=10, fontweight='bold')
    ax_roll_cos.grid(True, alpha=0.3)
    ax_roll_cos.set_ylim([-1.1, 1.1])
    
    ax_pitch_sin.set_ylabel('sin(Pitch)')
    ax_pitch_sin.set_title('Pitch - Sine', fontsize=10, fontweight='bold')
    ax_pitch_sin.grid(True, alpha=0.3)
    ax_pitch_sin.set_ylim([-1.1, 1.1])
    
    ax_pitch_cos.set_ylabel('cos(Pitch)')
    ax_pitch_cos.set_title('Pitch - Cosine', fontsize=10, fontweight='bold')
    ax_pitch_cos.grid(True, alpha=0.3)
    ax_pitch_cos.set_ylim([-1.1, 1.1])
    
    ax_yaw_sin.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_yaw_sin.set_ylabel('sin(Yaw)')
    ax_yaw_sin.set_title('Yaw - Sine', fontsize=10, fontweight='bold')
    ax_yaw_sin.grid(True, alpha=0.3)
    ax_yaw_sin.set_ylim([-1.1, 1.1])
    
    ax_yaw_cos.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_yaw_cos.set_ylabel('cos(Yaw)')
    ax_yaw_cos.set_title('Yaw - Cosine', fontsize=10, fontweight='bold')
    ax_yaw_cos.grid(True, alpha=0.3)
    ax_yaw_cos.set_ylim([-1.1, 1.1])
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Use tight_layout with rect to avoid overlap with suptitle
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except:
        pass  # Ignore tight_layout issues with 3D plots
    
    return fig

def test_and_visualize(args):
    ################################################################
    # Loading and exploring training data demo
    ################################################################
    print("="*60)
    print("0. Loading and exploring training data")
    print("="*60)
    
    # Load the dataset
    print("\n0.1. Loading dataset...")
    data = _load_training_data(args.dataset)
    
    # Show available demos
    print(f"\n0.2. Available demos: {data['demos'][:]}...")  # Show all

    print(f"\n0.2.1 Adding phases...")
    data = _add_phase(data)
    
    # Get full demo data
    print(f"\n0.3. Loading demo '{args.demo}'...")
    demo_data = data['get_demo'](args.demo)
    
    print(f"\n   Actions shape: {demo_data['actions'].shape}")
    print(f"   Actions range: [{demo_data['actions'].min():.3f}, {demo_data['actions'].max():.3f}]")

    # Access phase from the data dictionary
    phase = data[args.demo]['phase']
    print(f"\n   Phase shape: {phase.shape}")
    print(f"   Phase range: [{phase.min():.3f}, {phase.max():.3f}]")
    
    if demo_data['rewards'] is not None:
        print(f"   Total reward: {demo_data['rewards'].sum():.3f}")
        print(f"   Episode length: {len(demo_data['rewards'])}")
    
    # Explore observations
    print(f"\n0.4. Observation keys in demo:")
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
    print(f"\n0.6. Accessing action slices...")
    first_10_actions = data['get_actions'](args.demo, slice(0, 10))
    print(f"   First 10 actions shape: {first_10_actions.shape}")
    print(f"   First action: {first_10_actions[0]}")
    
    # Compare multiple demos
    print(f"\n0.7. Comparing trajectory lengths across demos...")
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
    
    print(f"\n0.8. Trajectory length statistics across all demos:")
    print(f"   Mean: {mean_n_actions:.1f} timesteps")
    print(f"   Std:  {std_n_actions:.1f} timesteps")
    print(f"   Min:  {min_n_actions} timesteps")
    print(f"   Max:  {max_n_actions} timesteps")

    ################################################################
    # True flow field divergence computation demo
    ################################################################
    print(f"\n" + "="*60)
    print("1. Computing divergence for ee poses across all demos")
    print(f"\n1.1. Binning data:")
    phase_tensor, ee_state_tensor, demo_tensor_keys, nan_mask, dphase = _bin_data(data, n_bins=max_n_actions)
    print(f"    Phase tensor shape: {phase_tensor.shape}")
    print(f"    EE state tensor shape: {ee_state_tensor.shape}")
    print(f"    Demo keys: {demo_tensor_keys[:3]}...")
    
    # Count NaNs per bin
    nan_counts = torch.isnan(ee_state_tensor).any(dim=2).sum(dim=0)
    print(f"    NaNs per bin (min/mean/max): {nan_counts.min()}/{nan_counts.float().mean():.1f}/{nan_counts.max()}")

    print(f"\n1.2. Computing divergence of ee_states:")
    
    # compute the state graphs
    state_graphs = _construct_state_graph(ee_state_tensor)

    # compute divergence via neighbors
    div_ee = _compute_divergence_via_neighbors(state_graphs, dphase,  k=4)
    print(f"    Divergence tensor shape: {div_ee.shape}")
    valid_div = div_ee[~torch.isnan(div_ee)]
    print(f"    Divergence valid count: {valid_div.shape[0]}/{div_ee.numel()}")
    print(f"    Divergence stats (min/mean/max): {valid_div.min():.4f}/{valid_div.mean():.4f}/{valid_div.max():.4f}")

    # Visualize just a few demos
    fig = visualize_observation_trajectories(
        ee_state_tensor, demo_tensor_keys, nan_mask,
        divergence=div_ee, phase_tensor=phase_tensor,
        # demo_indices=range(10),  
        frame_skip=5
    )
    plt.show()

    #################################################################
    # True training distribution score estimation demo
    #################################################################
    print(f"\n" + "="*60)
    print("2. Estimating score function for ee poses across all demos")
    print(f"\n2.1. Estimating score via nearest neighbors:")
    score_ee = _compute_score_via_neighbors(state_graphs, bandwidth=0.1)
    print(f"    Score tensor shape: {score_ee.shape}")
    valid_score = score_ee[~torch.isnan(score_ee).any(dim=2)]
    print(f"    Score valid count: {valid_score.shape[0]}/{score_ee.shape[0] * score_ee.shape[1]}")

    # Visualize score for a few demos
    for demo_idx in range(3):
        fig = visualize_score_for_trajectories(
            ee_state_tensor, score_ee[demo_idx], demo_idx, demo_tensor_keys,
            nan_mask=nan_mask, phase_tensor=phase_tensor,
            frame_skip=5
        )
        plt.show()

    print(f"\n2.2. Unbinning divergence and score data:")
    data = _unbin_data(data, demo_tensor_keys, div_ee, score_ee, nan_mask)
    
    # Verify unbinning by checking a few demos
    for demo_key in data['demos'][:3]:
        print("\n" + "-"*40)
        print(f"\n    Divergence stats for demo '{demo_key}':")
        if 'divergence' in data[demo_key]:
            div_twist_demo = data[demo_key]['divergence']
            valid_div_demo = div_twist_demo[~torch.isnan(div_twist_demo)]
            print(f"    {demo_key}: shape={div_twist_demo.shape}, valid={valid_div_demo.shape[0]}/{div_twist_demo.numel()}")
            if valid_div_demo.numel() > 0:
                print(f"      stats (min/mean/max): {valid_div_demo.min():.4f}/{valid_div_demo.mean():.4f}/{valid_div_demo.max():.4f}")
        
        print(f"\n    Score stats per component:")
        if 'score' in data[demo_key]:
            score_twist_demo = data[demo_key]['score']
            valid_score_demo = score_twist_demo[~torch.isnan(score_twist_demo).any(dim=1)]
            print(f"    {demo_key}: shape={score_twist_demo.shape}, valid={valid_score_demo.shape[0]}/{score_twist_demo.shape[0]}")
            if valid_score_demo.numel() > 0:
                print(f"      stats (min/mean/max) per component:")
                for i in range(score_twist_demo.shape[1]):
                    comp = valid_score_demo[:, i]
                    print(f"        Component {i}: {comp.min():.4f}/{comp.mean():.4f}/{comp.max():.4f}")

if __name__ == "__main__":
    """
    Demo code showing how to use _load_training_data() to explore a dataset.
    
    Usage:
        python robomimic/utils/divergence_utils.py --dataset /path/to/dataset.hdf5
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo key to inspect")
    args = parser.parse_args()
    
    # test_and_visualize(args)

    add_div_and_score_to_training_data(args.dataset)

