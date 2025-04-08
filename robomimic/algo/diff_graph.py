"""
Diffusion Graph Attention Network (DiffGAT) Algorithm for Robomimic.

This file defines the DiffGAT algorithm class, integrating the DiffusionPolicy
model with GATv2 backbone and Transformer head into the Robomimic framework.
It handles data processing (including graph construction), training loops,
and action sampling using DDIM.
"""

import math
import os
from collections import OrderedDict, deque
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.models.diffusion_policy import DiffusionPolicy


@register_algo_factory_func("diff_gat")
def algo_config_to_class(algo_config):
    """Factory function for DiffGAT algorithm."""
    return DIFF_GAT, {}


class NodeFeatureProcessor:
    """
    Processes raw observation dictionaries into structured node features and
    constructs graph representations (spatial and temporal) for the GNN.
    """
    def __init__(self, num_joints: int = 8 , chain: Optional[pk.SerialChain] = None):
        """
        Initializes the processor.

        Args:
            num_joints (int): Number of robot joints to include as nodes.
            chain (Optional[pk.SerialChain]): Pre-loaded PyTorch Kinematics chain
                                               for forward kinematics calculations.
        """
        self.chain = chain
        self.num_joints = num_joints
        # Offset from world origin to robot base (adjust if necessary)
        self.robot_base_offset = torch.tensor([-0.5, -0.1, 0.912])

        # Define the spatial adjacency matrix for a single frame
        # Rows/Cols: joint_0, ..., joint_6, eef, object
        # Represents direct physical connections (kinematic chain + eef-object interaction)
        self.adjacency_matrix = torch.tensor(
            [ # J0 J1 J2 J3 J4 J5 J6 EEF OBJ
                [0, 1, 0, 0, 0, 0, 0, 0, 0],  # joint_0 -> joint_1
                [1, 0, 1, 0, 0, 0, 0, 0, 0],  # joint_1 -> joint_0, joint_2
                [0, 1, 0, 1, 0, 0, 0, 0, 0],  # joint_2 -> joint_1, joint_3
                [0, 0, 1, 0, 1, 0, 0, 0, 0],  # joint_3 -> joint_2, joint_4
                [0, 0, 0, 1, 0, 1, 0, 0, 0],  # joint_4 -> joint_3, joint_5
                [0, 0, 0, 0, 1, 0, 1, 0, 0],  # joint_5 -> joint_4, joint_6
                [0, 0, 0, 0, 0, 1, 0, 1, 0],  # joint_6 -> joint_5, eef
                [0, 0, 0, 0, 0, 0, 1, 0, 1],  # eef -> joint_6, object 
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  # object -> eef 
            ],
            dtype=torch.bool,
        )
        self.num_nodes = self.adjacency_matrix.size(0) # Number of nodes per frame

        # Pre-compute static edge index from the adjacency matrix
        # Shape: (2, num_edges), where each column is [source_node, target_node]
        self.static_edge_index = (
            self.adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
        )

        # Will be set after processing first batch
        self.node_feature_dim: Optional[int] = None
        self.node_keys: Optional[list] = None
        self.batch_size: Optional[int] = None
        self.frame_stack: Optional[int] = None


    def process_features(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extracts and formats node features from the observation dictionary.

        Args:
            obs_dict (Dict[str, torch.Tensor]): Dictionary of observations, where each
                tensor has shape (batch_size, frame_stack, feature_dim).

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node names (e.g., "joint_0",
                "eef", "object") to their feature tensors of shape
                (batch_size, frame_stack, node_feature_dim). Features are padded
                to ensure consistent dimensionality across node types.
        """
        # Infer batch_size and frame_stack from an arbitrary observation tensor
        any_obs_tensor = next(iter(obs_dict.values()))
        self.batch_size = any_obs_tensor.shape[0]
        self.frame_stack = any_obs_tensor.shape[1]
        device = any_obs_tensor.device

        # Ensure robot base offset is on the correct device
        self.robot_base_offset = self.robot_base_offset.to(device=device)

        # Extract relevant observations
        joint_pos = obs_dict["robot0_joint_pos"].float()
        eef_pos = obs_dict["robot0_eef_pos"].float()
        eef_quat = obs_dict["robot0_eef_quat"].float()
        gripper_qpos = obs_dict["robot0_gripper_qpos"].float()
        object_features = obs_dict["object"].float() # Includes object state, potentially relative pos

        node_dict = {}

        # --- Joint Features ---
        if self.chain:
            self.chain = self.chain.to(device=device)
            # Flatten batch and frame dimensions for FK calculation
            flat_joint_pos = joint_pos.reshape(-1, self.num_joints)
            # Calculate FK for all links up to the specified number of joints
            fk_results = self.chain.forward_kinematics(flat_joint_pos, end_only=False)

            for i in range(self.num_joints):
                link_name = f"link{i}" # Assuming standard link naming convention
                if link_name in fk_results:
                    # Get transformation matrix (batch*frame, 4, 4)
                    transform_matrix = fk_results[link_name].get_matrix()
                    # Extract position (translation part)
                    link_pos = transform_matrix[:, :3, 3]
                    # Apply robot base offset
                    link_pos += self.robot_base_offset
                    # Reshape back to (batch, frame, 3)
                    link_pos = link_pos.view(self.batch_size, self.frame_stack, 3)
                    # Concatenate link position and corresponding joint angle
                    # Joint angle needs unsqueezing the last dim: (batch, frame) -> (batch, frame, 1)
                    node_dict[f"joint_{i}"] = torch.cat(
                        [link_pos, joint_pos[..., i].unsqueeze(-1)], dim=-1
                    ) # Feature: [x, y, z, angle]
                else:
                     # Fallback if FK fails or link name mismatch (e.g., zeros)
                    node_dict[f"joint_{i}"] = torch.zeros(
                        self.batch_size, self.frame_stack, 4, device=device
                    )
        else:
            # If no kinematic chain, use zeros for joint positions (only angle available)
            print("Warning: Kinematic chain not loaded. Using only joint angles for joint features.")
            for i in range(self.num_joints):
                 node_dict[f"joint_{i}"] = torch.cat(
                    [
                        torch.zeros(self.batch_size, self.frame_stack, 3, device=device),
                        joint_pos[..., i].unsqueeze(-1)
                     ], dim=-1
                 ) # Feature: [0, 0, 0, angle]


        # --- End Effector Features ---
        node_dict["eef"] = torch.cat([eef_pos, eef_quat, gripper_qpos], dim=-1)

        # --- Object Features ---
        node_dict["object"] = object_features

        # --- Padding for Consistent Feature Dimension ---
        # Find the maximum feature dimension across all node types
        max_len = max(v.shape[-1] for v in node_dict.values())

        # Pad features of nodes with smaller dimensions using zeros
        for key, tensor in node_dict.items():
            if tensor.shape[-1] < max_len:
                pad_size = max_len - tensor.shape[-1]
                # Pad only the last dimension (features)
                node_dict[key] = F.pad(tensor, (0, pad_size), mode='constant', value=0.0)

        # Store node keys in the order they were processed
        self.node_keys = list(node_dict.keys())
        # Store the final feature dimension after padding
        self.node_feature_dim = max_len

        return node_dict

    def build_graph(self, obs_dict: Dict[str, torch.Tensor]) -> Batch:
        """
        Constructs a PyG Batch object representing the spatio-temporal graph.

        Args:
            obs_dict (Dict[str, torch.Tensor]): Observation dictionary.

        Returns:
            torch_geometric.data.Batch: A batch containing graphs for each sample
                in the input batch. Each graph combines multiple time frames with
                spatial edges within frames and temporal edges between frames.
        """
        # Process raw observations into node features
        node_dict = self.process_features(obs_dict)
        device = next(iter(node_dict.values())).device

        # Ensure static tensors are on the correct device
        static_edge_index = self.static_edge_index.to(device)

        # --- Construct Spatio-Temporal Edge Index ---
        # 1. Spatial Edges: Repeat static edges for each frame
        # Shape: (2, num_static_edges * frame_stack)
        spatial_edge_index = static_edge_index.repeat(1, self.frame_stack)

        # Add offsets to node indices based on their frame
        # Example: frame 0 nodes are 0..N-1, frame 1 nodes are N..2N-1, etc.
        frame_offsets = torch.arange(self.frame_stack, device=device).repeat_interleave(
            static_edge_index.size(1) # Repeat offset for each edge in a frame
        ) * self.num_nodes # Offset by number of nodes per frame
        # Apply offsets to both source and target nodes
        spatial_edge_index = spatial_edge_index + frame_offsets

        # 2. Temporal Edges: Connect the same node across consecutive frames
        # Connect node k in frame t to node k in frame t+1
        num_temporal_nodes_to_connect = self.num_nodes * (self.frame_stack - 1)
        temporal_edge_sources = torch.arange(num_temporal_nodes_to_connect, device=device)
        temporal_edge_targets = temporal_edge_sources + self.num_nodes # Target is the same node in the next frame
        # Shape: (2, num_nodes * (frame_stack - 1))
        temporal_edge_index = torch.stack([temporal_edge_sources, temporal_edge_targets], dim=0)

        # Combine spatial and temporal edges
        # Shape: (2, num_spatial_edges + num_temporal_edges)
        combined_edge_index = torch.cat([spatial_edge_index, temporal_edge_index], dim=1)

        # --- Prepare Node Features ---
        # Stack features from the dictionary: (batch, num_nodes, frame_stack, feature_dim)
        node_features_stacked = torch.stack([node_dict[k] for k in self.node_keys], dim=1)

        # Optional: Add pairwise distances as node features (can help GNN learn spatial relationships)
        # Extract positions (first 3 dims): (batch, num_nodes, frame_stack, 3)
        positions = node_features_stacked[..., :3]
        # Reshape for pairwise distance calculation per frame: (batch, frame_stack, num_nodes, 3)
        positions_per_frame = positions.permute(0, 2, 1, 3)
        # Calculate pairwise distances within each frame
        distance_matrices = torch.cdist(positions_per_frame, positions_per_frame) # (batch, frame_stack, num_nodes, num_nodes)
        # Reshape distances to match node feature stacking: (batch, num_nodes, frame_stack, num_nodes)
        distance_matrices_reshaped = distance_matrices.permute(0, 2, 1, 3)
        # Concatenate distances to the original node features
        node_features_with_dist = torch.cat([node_features_stacked, distance_matrices_reshaped], dim=-1)

        # Reshape features for PyG: Flatten frame and node dimensions
        # Input: (batch, num_nodes, frame_stack, final_feature_dim)
        # Permute to: (batch, frame_stack, num_nodes, final_feature_dim)
        # Reshape to: (batch, frame_stack * num_nodes, final_feature_dim)
        # This creates a single long sequence of nodes per batch sample.
        batch_size, num_nodes, frame_stack, final_feature_dim = node_features_with_dist.shape
        permuted_features = node_features_with_dist.permute(0, 2, 1, 3)
        flat_node_features = permuted_features.reshape(
            batch_size, frame_stack * num_nodes, final_feature_dim
        )

        # --- Create PyG Data Objects and Batch ---
        graphs = []
        for i in range(self.batch_size):
            # Create a Data object for each sample in the batch
            # Note: edge_index is shared across all graphs in the batch,
            # PyG handles this correctly when creating the Batch object.
            graph_data = Data(x=flat_node_features[i], edge_index=combined_edge_index)
            graphs.append(graph_data)

            # --- Optional Debug Visualization (Visualize first graph of first batch) ---
            # if i == 0 and not hasattr(self, '_graph_visualized'):
            #     try:
            #         print("Attempting to visualize the first graph...")
            #         debug_dir = os.path.join(os.path.expanduser("~"), "robomimic_debug_diffgat")
            #         save_path = os.path.join(debug_dir, f"graph_visualization_fs{self.frame_stack}.png")
            #         self.visualize_graph(graph_data)
            #         # self._graph_visualized = True # Prevent repeated visualization
            #     except Exception as e:
            #         print(f"Warning: Failed to visualize graph: {e}")
            # --- End Debug Visualization ---

        # Combine individual Data objects into a single Batch object
        return Batch.from_data_list(graphs)

    def visualize_graph(self, graph_data: Data, save_path: str = None):
        """
        Visualize the graph structure in 3D space using Matplotlib and NetworkX.
        Nodes are colored by frame, spatial edges are solid black, temporal edges are dashed red.
        Node distances are visualized through edge color intensity.

        Args:
            graph_data (Data): A PyG Data object for a single sample (multiple frames).
            save_path (str, optional): Path to save the visualization image. If None, displays the plot.
        """
        if self.node_keys is None or self.frame_stack is None:
             print("Warning: Cannot visualize graph before processing features (node_keys/frame_stack unknown).")
             return

        # Convert PyG graph to NetworkX graph for easier plotting
        G = to_networkx(graph_data, node_attrs=['x'], to_undirected=True)

        # Extract node features (including positions) and edge indices
        node_features = graph_data.x.detach().cpu().numpy()
        positions_3d = node_features[:, :3] # Assuming first 3 features are x, y, z
        
        # Create two separate figures - one for 3D graph, one for distance heatmap
        fig = plt.figure(figsize=(20, 10))
        
        # --- 3D Graph Plot ---
        ax1 = fig.add_subplot(121, projection='3d')

        # Determine node attributes (frame, type) for plotting
        node_attributes = {}
        for i in range(len(positions_3d)):
            frame_idx = i // self.num_nodes
            node_in_frame_idx = i % self.num_nodes
            node_type = self.node_keys[node_in_frame_idx]
            node_attributes[i] = {"pos": positions_3d[i], "frame": frame_idx, "type": node_type}

        nx.set_node_attributes(G, node_attributes)

        # --- Plot Nodes ---
        cmap = plt.cm.viridis # Color map for frames
        unique_frames = sorted(list(set(nx.get_node_attributes(G, "frame").values())))

        for frame in unique_frames:
            nodes_in_frame = [n for n, attr in G.nodes(data=True) if attr["frame"] == frame]
            node_positions = np.array([G.nodes[n]["pos"] for n in nodes_in_frame])
            color = cmap(frame / max(1, len(unique_frames) - 1))

            if len(node_positions) > 0:
                ax1.scatter(
                    node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                    color=color, s=120, alpha=0.8, label=f"Frame {frame}"
                )
                # Annotate nodes with their type
                for i, node_idx in enumerate(nodes_in_frame):
                    ax1.text(
                        node_positions[i, 0], node_positions[i, 1], node_positions[i, 2],
                        f"{G.nodes[node_idx]['type']}", fontsize=7, ha='center', va='bottom'
                    )

        # --- Extract Distance Information ---
        # Extract distance matrices from node features
        # The distance matrix for each node is appended after the original features
        # In node_feature_with_dist, distances start after the original feature dimension
        num_nodes_per_frame = self.num_nodes
        num_frames = self.frame_stack
        
        # Compute a normalization factor for edge coloring based on distances
        dist_norm = plt.Normalize(0, 2.0)  # Adjust max value based on your data
        dist_cmap = plt.cm.YlOrRd  # Color map for distances (yellow to red)
        
        # --- Plot Edges with Distance Information ---
        for u, v, _ in G.edges(data=True):
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            frame_u = G.nodes[u]["frame"]
            frame_v = G.nodes[v]["frame"]
            
            # Compute actual Euclidean distance for edge color
            actual_dist = np.linalg.norm(pos_u - pos_v)
            edge_color = dist_cmap(dist_norm(actual_dist))
            
            if frame_u == frame_v:
                # Spatial edge (within the same frame) - colored by distance
                ax1.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                        color=edge_color, alpha=0.6, linewidth=1.5)
            else:
                # Temporal edge (between different frames) - dashed red
                ax1.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                        "r--", alpha=0.3, linewidth=0.8)

        # --- Setup 3D Plot Appearance ---
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f"Graph Visualization (Frames: {self.frame_stack}, Nodes/Frame: {num_nodes_per_frame})")
        ax1.legend(title="Frames")
        
        # Add colorbar for distance edges
        sm = plt.cm.ScalarMappable(cmap=dist_cmap, norm=dist_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, label='Edge Distance (m)')
        
        # --- Distance Heatmap Plot ---
        ax2 = fig.add_subplot(122)
        
        # For visualization simplicity, let's show distances for the first frame only
        first_frame_indices = [i for i, attr in G.nodes(data=True) if attr["frame"] == 0]
        first_frame_node_types = [G.nodes[i]["type"] for i in first_frame_indices]
        
        # Extract all pairwise distances between nodes in the first frame
        dist_matrix = np.zeros((len(first_frame_indices), len(first_frame_indices)))
        for i, node_i in enumerate(first_frame_indices):
            for j, node_j in enumerate(first_frame_indices):
                pos_i = G.nodes[node_i]["pos"]
                pos_j = G.nodes[node_j]["pos"]
                dist_matrix[i, j] = np.linalg.norm(pos_i - pos_j)
        
        # Create heatmap of distances
        im = ax2.imshow(dist_matrix, cmap='viridis')
        ax2.set_title("Pairwise Node Distances (First Frame)")
        
        # Set ticks with node type labels
        ax2.set_xticks(np.arange(len(first_frame_node_types)))
        ax2.set_yticks(np.arange(len(first_frame_node_types)))
        ax2.set_xticklabels(first_frame_node_types, rotation=45, ha="right")
        ax2.set_yticklabels(first_frame_node_types)
        
        # Add colorbar and distance values in cells
        fig.colorbar(im, ax=ax2, label='Distance (m)')
        
        # Annotate distance values in the heatmap
        for i in range(len(first_frame_node_types)):
            for j in range(len(first_frame_node_types)):
                text_color = "white" if dist_matrix[i, j] > dist_matrix.max() * 0.7 else "black"
                ax2.text(j, i, f"{dist_matrix[i, j]:.2f}", 
                         ha="center", va="center", color=text_color, fontsize=7)

        # Attempt to set equal aspect ratio for 3D plot
        try:
            extents = np.array([ax1.get_xlim(), ax1.get_ylim(), ax1.get_zlim()])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize / 2
            ax1.set_xlim(centers[0] - r, centers[0] + r)
            ax1.set_ylim(centers[1] - r, centers[1] + r)
            ax1.set_zlim(centers[2] - r, centers[2] + r)
        except Exception as e:
            print(f"Could not set equal aspect ratio: {e}")

        plt.tight_layout()
        
        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Graph visualization saved to {save_path}")
        else:
            plt.show()


class DIFF_GAT(PolicyAlgo):
    def __init__(
        self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
    ):
        # Call parent __init__ first - it sets up device, config, etc. and calls _create_networks
        super().__init__(
            algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
        )

        # Setup CUDNN, matmul precision
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.batch_size = self.global_config.train.batch_size
        self.seq_length = self.global_config.train.seq_length

        # --- Diffusion Schedule ---
        self.T = algo_config.diffusion.num_timesteps
        betas = self.cosine_beta_schedule(self.T).to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-9)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # --- Kinematics Loading ---
        self.chain = None
        try:
            mjcf_path = "robomimic/algo/panda/robot.xml"
            if os.path.exists(mjcf_path):
                 self.chain = pk.build_serial_chain_from_mjcf(
                    open(mjcf_path).read(), "link7"
                 ).to(dtype=torch.float32, device=self.device) 
                 print("Successfully loaded kinematic chain from robot.xml.")
            else:
                 print(f"Warning: Kinematic definition file not found at {mjcf_path}")
        except Exception as e:
            print(f"Warning: Failed to load robot.xml for kinematics: {e}")

        # Note: self.nets and self.node_feature_processor are created within _create_networks,
        # which was called by super().__init__() above.

        # --- Optimizer (needs self.nets) ---
        self.optimizers["policy"] = torch.optim.Adam(
            self.nets["policy"].parameters(),
            lr=self.algo_config.optim_params.policy.learning_rate.initial,
            weight_decay=self.algo_config.optim_params.policy.regularization.L2
        )

        # --- Instantiate NodeFeatureProcessor AFTER Policy ---
        node_processor_instance = NodeFeatureProcessor(
             num_joints=self.algo_config.num_joints, # Or get from config
             chain=self.chain
        )
        # Assign to self so other methods can use it
        self.node_feature_processor = node_processor_instance
        print("NodeFeatureProcessor initialized.")

        # --- Action Buffer ---
        self.action_buffer = deque()


    def _create_networks(self):
        """
        Creates the Diffusion Policy model (using lazy init) and the NodeFeatureProcessor.
        Called by the parent class's __init__.
        """
        self.nets = nn.ModuleDict()
        print("Creating networks using lazy initialization for GNN input dimension...")

        # --- Instantiate the Diffusion Policy Model with Lazy Input Dimension ---
        model = DiffusionPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            graph_input_feature_dim=-1, # Use -1 for lazy initialization
            timestep_emb_dim=128,       
            device=self.device,
        )
        # model = torch.compile(model)
        self.nets["policy"] = model
        print("Policy Network initialized (GNN input dim will be inferred on first forward pass).")
        
        return self.nets # Robomimic expects the nets dictionary returned

    def process_batch_for_training(self, batch: Dict) -> Dict:
        """
        Prepares a raw batch from the dataset for training.

        - Selects appropriate time slices for observations and actions.
        - Moves tensors to the designated device.
        - Constructs the spatio-temporal graph representation using NodeFeatureProcessor.

        Args:
            batch (Dict): A dictionary containing observation and action tensors.
                          Expected keys: "obs", "actions". "goal_obs" is optional.

        Returns:
            Dict: The processed batch containing device-mapped tensors and the
                  constructed graph ("graph" key with PyG Batch object).
        """
        frame_stack = self.global_config.train.frame_stack
        seq_len = self.global_config.train.seq_length

        # Ensure tensors have a batch dimension if loaded individually
        if len(batch["obs"]["robot0_joint_pos"].shape) == 2: # Assuming (seq, dim)
            batch = TensorUtils.unsqueeze_expand_batch(batch, 1) # Add batch dim

        # Select observation window frame_stack - 1 previous frames plus the current frame
        obs_data = {k: batch["obs"][k][:, :frame_stack, ...] for k in self.obs_key_shapes}

        # Extract the next observation for the current time step from next_obs key
        next_q_pos_obs = batch["obs"]["robot0_joint_pos"][:, frame_stack + 1, ...]

        # Select action sequence (corresponding to the observation window)
        # Actions start from the end of the frame stack context
        action_data = batch["actions"][:, frame_stack - 1 : frame_stack - 1 + seq_len, :]

        # Ensure action sequence has the expected length
        if action_data.shape[1] != seq_len:
             # Pad or truncate if necessary (e.g., end of trajectory)
             # This example pads with zeros, adjust if needed
             action_data = TensorUtils.pad_sequence(action_data, seq_len, batch_first=True, padding_value=0.0)

        processed_batch = {
            "obs": obs_data,
            "next_q_pos_obs": next_q_pos_obs, # Pass along next observation for training
            "goal_obs": batch.get("goal_obs", None), # Pass along goals if present
            "actions": action_data,
        }

        # Move tensors to device and ensure float32
        processed_batch = TensorUtils.to_float(
            TensorUtils.to_device(processed_batch, self.device)
        )

        # Build graph representation from observations
        # This adds a 'graph' key with a PyG Batch object
        processed_batch["graph"] = self.node_feature_processor.build_graph(
            processed_batch["obs"]
        ).to(self.device) # Ensure graph is on the correct device

        return processed_batch

    def process_batch_for_inference(self, obs_dict: Dict) -> Dict:
        """
        Prepares observations for inference/rollout (no actions needed).
        
        Args:
            obs_dict (Dict): Observation dictionary from the environment.
            
        Returns:
            Dict: Processed batch with observations and graph representation.
        """
        # Ensure tensors have a batch dimension
        if len(next(iter(obs_dict.values())).shape) < 3:  # Check if batch dim exists
            # Add batch dimension (assume we're processing a single observation)
            obs_dict = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
            
        # Ensure we only process keys that are in self.obs_key_shapes
        obs_data = {k: obs_dict[k] for k in self.obs_key_shapes if k in obs_dict}
        
        # Move tensors to device and ensure float32
        processed_batch = {
            "obs": TensorUtils.to_float(TensorUtils.to_device(obs_data, self.device)),
        }
        
        # Build graph representation from observations
        processed_batch["graph"] = self.node_feature_processor.build_graph(
            processed_batch["obs"]
        ).to(self.device)
        
        return processed_batch

    def _compute_losses(self, predictions, targets):
        """
        Compute different loss components for the diffusion model.
        
        Args:
            predictions (torch.Tensor): The predicted noise from the model.
            targets (torch.Tensor): The actual noise targets.
            
        Returns:
            OrderedDict: Dictionary containing individual loss components and the final weighted loss.
        """
        losses = OrderedDict()
        
        # Calculate primary loss components
        losses["mse_loss"] = F.mse_loss(predictions, targets)
        
        # Calculate additional loss components that might help with stability/performance
        losses["smooth_l1_loss"] = F.smooth_l1_loss(predictions, targets)
        
        # Optional: Add cosine similarity loss for directional consistency
        # This can be particularly helpful for action prediction tasks
        # Assuming the first 3 dimensions of the action are positional/directional
        if predictions.shape[-1] >= 3:
            losses["cos_loss"] = 1.0 - F.cosine_similarity(
                predictions[..., :3], targets[..., :3], dim=-1
            ).mean()
        else:
            losses["cos_loss"] = torch.tensor(0.0, device=self.device)
        
        # Weighted combination of all losses (using weights from config)
        losses["total_loss"] = (
            self.algo_config.loss.l2_weight * losses["mse_loss"] +
            self.algo_config.loss.l1_weight * losses["smooth_l1_loss"] +
            self.algo_config.loss.cos_weight * losses["cos_loss"]
        )
        
        return losses

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Performs a single training or validation step on a processed batch.

        Args:
            batch (Dict): Processed batch containing observations (graph), actions.
            epoch (int): Current training epoch number.
            validate (bool): If True, run in validation mode (no gradients/updates).

        Returns:
            Dict: Dictionary containing loss values and potentially other metrics.
                  Keys: "losses", "predictions".
        """
        # Use validation context manager if needed (disables gradients)
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Extract data from the processed batch
            clean_actions = batch["actions"] # Shape: (batch_size, seq_len, action_dim)
            graph = batch["graph"]           # PyG Batch object
            next_q_pos_obs = batch["next_q_pos_obs"] # Shape: (batch_size, action_dim)
            current_batch_size = clean_actions.shape[0] # Get actual batch size

            # 1. Sample Diffusion Timesteps
            timesteps = self.sample_timesteps(current_batch_size) # Shape: (batch_size,)

            # 2. Add Noise to Actions (Forward Diffusion Process)
            noisy_actions, noise = self.add_noise(clean_actions, timesteps)
            # noisy_actions: (batch, seq_len, action_dim), noise: (batch, seq_len, action_dim)

            # 3. Predict Noise using the Policy Network
            # The model conditions on the graph state and timesteps
            q_pos, predicted_noise = self.nets["policy"](
                noisy_action=noisy_actions,
                state=batch["obs"], # Pass obs dict (might be unused by policy if graph is primary)
                timestep=timesteps,
                graph=graph
            ) # Output Shape: (batch, seq_len, action_dim)

            # 4. Compute all loss components
            noise_losses = self._compute_losses(predicted_noise, noise)
            q_pos_losses = self._compute_losses(q_pos, next_q_pos_obs)
            
            weight = self.algo_config.loss.tradeoff
            # Combine losses (weighted sum)
            losses = OrderedDict()
            losses["mse_loss"] = weight * noise_losses["mse_loss"] + (1-weight) * q_pos_losses["mse_loss"]
            losses["smooth_l1_loss"] = weight * noise_losses["smooth_l1_loss"] + (1-weight) * q_pos_losses["smooth_l1_loss"]
            losses["cos_loss"] = weight * noise_losses["cos_loss"] + (1-weight) * q_pos_losses["cos_loss"]
            losses["total_loss"] = weight * noise_losses["total_loss"] + (1-weight) * q_pos_losses["total_loss"]
            losses["q_pos_loss"] = q_pos_losses["total_loss"] # Store q_pos loss separately
            losses["predicted_noise"] = noise_losses["total_loss"]

            # Store information for logging
            info = {
                "predictions": TensorUtils.detach(predicted_noise),
                "losses": TensorUtils.detach(losses),
            }

            # 5. Backpropagation and Optimization Step (if training)
            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info # Return loss info and other metrics

    def _train_step(self, losses):
        """
        Internal helper function for DIFF_GAT algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
            
        Returns:
            info (OrderedDict): dictionary with gradient norms and other training statistics
        """
        # Gradient step
        info = OrderedDict()
        
        # If gradient clipping is enabled, we need to handle it inside backprop_for_loss
        # by providing the clip parameter
        grad_clip_val = self.algo_config.grad_clip if hasattr(self.algo_config, "grad_clip") else None
        
        policy_grad_norm = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["total_loss"],
            max_grad_norm=grad_clip_val,
        )
        
        info["policy_grad_norms"] = policy_grad_norm
        return info

    def log_info(self, info: Dict) -> Dict:
        """Logs training/validation information."""
        log = super().log_info(info) # Get base log info
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Q_Pos_Loss"] = info["losses"]["q_pos_loss"].item()
        log["Predicted_Noise_Loss"] = info["losses"]["predicted_noise"].item()
        log["MSE_Loss"] = info["losses"]["mse_loss"].item()
        log["Smooth_L1_Loss"] = info["losses"]["smooth_l1_loss"].item()
        log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        # Add parameter count (useful for debugging model size)
        if "NumParams" not in log: # Log only once
             log["NumParams"] = sum(p.numel() for p in self.nets.parameters() if p.requires_grad)
        if "policy_grad_norms" in info:
            # Check if policy_grad_norms is a tensor or a float
            if isinstance(info["policy_grad_norms"], torch.Tensor):
                log["Policy_Grad_Norms"] = info["policy_grad_norms"].item()
            else:
                # It's already a float/scalar value
                log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict: Dict, goal_dict: Optional[Dict] = None) -> torch.Tensor:
        """
        Gets the next action(s) for the environment step.

        Uses DDIM sampling to generate an action sequence when the buffer is empty,
        then returns actions one by one from the buffer.

        Args:
            obs_dict (Dict): Current observation dictionary from the environment.
            goal_dict (Optional[Dict]): Goal observation dictionary (if applicable).

        Returns:
            torch.Tensor: The next action tensor of shape (1, action_dim).
        """
        # If the action buffer is empty, generate a new sequence
        if len(self.action_buffer) == 0:
            # Process the current observation using the inference-specific method
            processed_obs = self.process_batch_for_inference(obs_dict)
            graph = processed_obs["graph"]

            # Sample an action sequence using DDIM
            sampled_actions = self.sample(
                obs_dict=processed_obs["obs"],
                graph=graph,
                num_steps=50,
                eta=0.0,
            )

            # Convert to numpy and store in buffer
            action_sequence_np = sampled_actions.squeeze(0).cpu().numpy() # (seq_len, action_dim)
            self.action_buffer.clear()
            self.action_buffer.extend(list(action_sequence_np)) # Add actions as individual steps

        # Retrieve the next action from the buffer
        next_action_np = self.action_buffer.popleft() # Get first action
        # Convert back to tensor, add batch dimension, and move to device
        action_tensor = torch.tensor(
            next_action_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0) # Shape: (1, action_dim)

        return action_tensor

    # --- Diffusion Helper Methods ---

    def add_noise(
        self, x_start: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to clean data according to the forward diffusion process q(x_t | x_0).

        Args:
            x_start (torch.Tensor): The original clean data (e.g., actions),
                                    shape (batch_size, ...).
            timesteps (torch.Tensor): The diffusion timestep for each sample in the batch,
                                      shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - noisy_data (torch.Tensor): Data with noise added, same shape as x_start.
                - noise (torch.Tensor): The Gaussian noise added, same shape as x_start.
        """
        # Sample standard Gaussian noise
        noise = torch.randn_like(x_start)

        # Get sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape parameters to allow broadcasting: (batch_size,) -> (batch_size, 1, 1, ...)
        # Adds dimensions to match the shape of x_start beyond the batch dimension.
        view_shape = [x_start.shape[0]] + [1] * (x_start.dim() - 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(view_shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(view_shape)

        # Apply the forward process formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_data = (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return noisy_data, noise

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Generates a cosine variance schedule for betas.

        Proposed in: https://arxiv.org/abs/2102.09672 (Improved DDPMs)

        Args:
            timesteps (int): Total number of diffusion steps (T).
            s (float): Small offset to prevent betas from being too small near t=0.

        Returns:
            torch.Tensor: Tensor of beta values for each timestep, shape (timesteps,).
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        # Calculate f(t) = cos^2(((t/T) + s) / (1 + s) * pi/2)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        # Calculate alpha_bar_t = f(t) / f(0)
        alphas_cumprod = f_t / f_t[0]
        # Calculate beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clamp betas to prevent numerical instability
        return torch.clamp(betas, 0.0001, 0.999) # Min/Max values from DDPM paper

    def linear_beta_schedule(self, timesteps: int, start=0.0001, end=0.02) -> torch.Tensor:
        """Generates a linear variance schedule for betas (alternative)."""
        return torch.linspace(start, end, timesteps)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Samples random timesteps uniformly from [0, T-1]."""
        return torch.randint(0, self.T, (batch_size,), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def sample(
        self,
        obs_dict: Dict,
        graph: Batch,
        num_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0 # Placeholder for optional CFG
    ) -> torch.Tensor:
        """
        Generates an action sequence using DDIM sampling (reverse diffusion).

        Args:
            obs_dict (Dict): Observation dictionary for conditioning.
            graph (Batch): Graph representation of the state for conditioning.
            num_steps (int): Number of DDIM sampling steps (<= T).
            eta (float): DDIM parameter controlling stochasticity (0.0 = deterministic DDIM).
            guidance_scale (float): Scale for classifier-free guidance.

        Returns:
            torch.Tensor: The generated clean action sequence, shape (1, seq_len, action_dim).
        """
        batch_size = 1 # Inference is typically done one sample at a time

        # 1. Initialize Starting Noise
        # Start from pure Gaussian noise at timestep T
        x_t = torch.randn(
            (batch_size, self.seq_length, self.ac_dim), device=self.device
        )

        # 2. Prepare Timestep Schedule for Sampling
        # Sample 'num_steps' timesteps from T-1 down to 0
        times = torch.linspace(self.T - 1, 0, num_steps + 1, dtype=torch.long, device=self.device) # T, T-k, T-2k, ..., 0
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-1-k), (T-1-k, T-1-2k), ...]

        # 3. Iterative Denoising Loop (DDIM Steps)
        for t, t_next in time_pairs:
            # Create timestep tensor for the policy network (batch_size,)
            timestep_vec = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise using the policy network
            _, pred_noise = self.nets["policy"](
                 noisy_action=x_t,
                 state=obs_dict,
                 timestep=timestep_vec,
                 graph=graph
             )

            # --- Optional: Classifier-Free Guidance (CFG) ---
            # if guidance_scale > 1.0:
            #     # Requires running the model unconditionally (e.g., with null context)
            #     uncond_noise = self.nets["policy"](x_t, None, timestep_vec, graph=null_graph?) # How to get null graph?
            #     pred_noise = uncond_noise + guidance_scale * (pred_noise - uncond_noise)
            # --- End CFG ---

            # --- DDIM Update Rule ---
            # Get alpha_bar values for current (t) and next (t_next) timesteps
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device) # alpha_bar_{-1} = 1

            # Predict x_0 (estimated clean action) using the current state x_t and predicted noise
            # x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            # Optional: Clamp predicted x_0 to valid range (e.g., [-1, 1] for normalized actions)
            # x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Calculate coefficients for the DDIM update step
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next)
            )
            term1 = torch.sqrt(alpha_cumprod_t_next) * x_0_pred
            term2 = torch.sqrt(1 - alpha_cumprod_t_next - sigma_t**2) * pred_noise

            # Sample random noise for the stochastic part (if eta > 0)
            noise = torch.randn_like(x_t) if eta > 0 else 0.0

            # Update x_t -> x_{t_next}
            x_t = term1 + term2 + sigma_t * noise

            x_t = nn.Tanh()(x_t) # Apply tanh inside the loop because it fails otherwise ?#!&%$ 
            # --- End DDIM Update ---

        # Final result x_0 should be the clean action sequence
        # x_0_final = torch.tanh(x_t) # Tanh is common for normalized actions
        # x_0_final = torch.clamp(x_t, -1.0, 1.0) # Alternative: clamping
        x_0_final = x_t

        return x_0_final.contiguous() # Ensure tensor is contiguous