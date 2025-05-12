"""
Diffusion Graph Attention Network (DiffGAT) Algorithm for Robomimic.

This file defines the DiffGAT algorithm class, integrating the DiffusionPolicy
model with GATv2 backbone and Transformer head into the Robomimic framework.
It handles data processing (including graph construction), training loops,
and action sampling using DDIM.
"""

import math
import os
from typing import Callable, Union

from collections import OrderedDict, deque
from typing import Dict, Optional, Tuple
from packaging.version import parse as parse_version

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx, unbatch

# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.models.diffusion_policy import DiffusionPolicy, GNNPolicy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_nets as ObsNets


@register_algo_factory_func("diff_gat")
def algo_config_to_class(algo_config):
    """Factory function for DiffGAT algorithm."""
    return DIFF_GAT, {}


class NodeFeatureProcessor(nn.Module):
    """
    Processes raw observation dictionaries into structured node features and
    constructs graph representations (spatial and temporal) for the GNN.
    """

    def __init__(self, num_joints: int = 8, chain: Optional[pk.SerialChain] = None):
        super().__init__()
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
            [  # J0 J1 J2 J3 J4 J5 J6 EEF OBJ
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
        self.num_nodes = self.adjacency_matrix.size(0)  # Number of nodes per frame

        # Pre-compute static edge index from the adjacency matrix
        self.static_edge_index = (
            self.adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
        )

        # Define node types and mapping to IDs (for one-hot encoding)
        self.node_type_list = [f"joint_{i}" for i in range(num_joints)] + ["eef", "object"]
        self.node_type_to_id = {name: i for i, name in enumerate(self.node_type_list)}

        # Will be set after processing first batch
        self.node_feature_dim: Optional[int] = None
        self.node_keys: Optional[list] = None
        self.batch_size: Optional[int] = None
        self.frame_stack: Optional[int] = None

    def process_features(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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
        object_features = obs_dict["object"].float()

        # Compute se3_pose if not present (inference)
        if "robot0_joint_se3" not in obs_dict:
            # joint_pos: (batch, frame_stack, num_joints)
            B, T, J = joint_pos.shape
            qpos_flat = joint_pos.reshape(B * T, J)
            fk = self.chain.forward_kinematics(qpos_flat, end_only=False)
            se3s = []
            for i in range(self.num_joints):
                link_name = f"link{i}"
                mat = fk[link_name].get_matrix()  # (B*T, 4, 4)
                pos = mat[:, :3, 3]
                rot = mat[:, :3, :3]
                rot6d = rot[:, :, :2].reshape(B * T, 6)
                se3 = torch.cat([pos, rot6d], dim=-1)  # (B*T, 9)
                se3s.append(se3)
            se3_pose = torch.stack(se3s, dim=1)  # (B*T, num_joints, 9)
            se3_pose = se3_pose.permute(0, 1, 2).reshape(B, T, self.num_joints, 9)
            obs_dict["robot0_joint_se3"] = se3_pose.reshape(B, T, self.num_joints * 9)
        else:
            se3_pose = obs_dict["robot0_joint_se3"].float().view(
                self.batch_size, self.frame_stack, self.num_joints, 9
            )

        node_dict = {}

        # --- Joint Features ---
        for i in range(self.num_joints):
            node_name = f"joint_{i}"
            base_features = se3_pose[..., i, :]
            # Get one-hot encoding
            node_id = torch.tensor(self.node_type_to_id[node_name], device=device)
            one_hot_encoding = F.one_hot(node_id, num_classes=self.num_nodes).float() # Shape: (num_nodes,)
            # Expand encoding to match (B, T, num_nodes)
            one_hot_expanded = one_hot_encoding.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.frame_stack, -1)
            # Concatenate base features and one-hot encoding
            node_dict[node_name] = torch.cat([base_features, one_hot_expanded], dim=-1)

        # --- End Effector Features ---
        node_name = "eef"
        base_features = torch.cat([eef_pos, eef_quat, gripper_qpos], dim=-1)
        node_id = torch.tensor(self.node_type_to_id[node_name], device=device)
        one_hot_encoding = F.one_hot(node_id, num_classes=self.num_nodes).float()
        one_hot_expanded = one_hot_encoding.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.frame_stack, -1)
        node_dict[node_name] = torch.cat([base_features, one_hot_expanded], dim=-1)

        # --- Object Features ---
        node_name = "object"
        base_features = object_features
        node_id = torch.tensor(self.node_type_to_id[node_name], device=device)
        one_hot_encoding = F.one_hot(node_id, num_classes=self.num_nodes).float()
        one_hot_expanded = one_hot_encoding.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.frame_stack, -1)
        node_dict[node_name] = torch.cat([base_features, one_hot_expanded], dim=-1)

        # --- Padding for Consistent Feature Dimension ---
        # Find the maximum feature dimension across all node types (NOW includes one-hot dim)
        max_len = max(v.shape[-1] for v in node_dict.values())

        # Pad features of nodes with smaller dimensions using zeros
        for key, tensor in node_dict.items():
            if tensor.shape[-1] < max_len:
                pad_size = max_len - tensor.shape[-1]
                node_dict[key] = F.pad(
                    tensor, (0, pad_size), mode="constant", value=0.0
                )

        # Store node keys in the order they were processed (ensure consistent order)
        self.node_keys = self.node_type_list # Use the predefined order
        # Store the final feature dimension after padding
        self.node_feature_dim = max_len

        return node_dict

    def build_graph(self, obs_dict: Dict[str, torch.Tensor]) -> Batch:
        """
        Fast spatio-temporal graph builder with full pairwise distances:
         - Caches combined spatio-temporal edge_index per frame_stack
         - Vectorizes node-feature stacking and flattening
         - Keeps the full torch.cdist distance computation
        """
        # 1) Process your per-node features
        node_dict = self.process_features(obs_dict)
        device = next(iter(node_dict.values())).device
        B, T, N = self.batch_size, self.frame_stack, self.num_nodes
        F_node = self.node_feature_dim

        # 2) Stack node features: (B, N, T, F_node)
        node_feats = torch.stack([node_dict[k] for k in self.node_keys], dim=1)

        # 3) Compute full pairwise distances per frame
        #    positions: (B, N, T, 3)
        # positions = node_feats[..., :3]
        # #    reshape to (B, T, N, 3) for cdist
        # pos_per_frame = positions.permute(0, 2, 1, 3)
        # #    distance: (B, T, N, N)
        # dist = torch.cdist(pos_per_frame, pos_per_frame)
        # #    reshape back to (B, N, T, N)
        # dist = dist.permute(0, 2, 1, 3)

        # # 4) Concatenate distances onto features: (B, N, T, F_node + N)
        # feats_with_dist = torch.cat([node_feats, dist], dim=-1)
        final_dim = F_node #+ N

        # 5) Flatten into one big x: (B*T*N, final_dim)
        x = node_feats.permute(0, 2, 1, 3).reshape(B * T * N, final_dim)

        # 6) Build & cache the single-sample combined edge_index if needed
        if not hasattr(self, "_cached_edge_index") or self._cached_T != T:
            static = self.static_edge_index
            static = static.to(device)  # (2, E_static)
            # spatial edges repeated per frame
            spatial = static.repeat(1, T) + (
                torch.arange(T, device=device)
                .repeat_interleave(static.size(1))
                * N
            ).unsqueeze(0)
            # temporal edges between frames
            src = torch.arange(N * (T - 1), device=device)
            tgt = src + N
            temporal = torch.stack([src, tgt], dim=0)
            # cache
            self._cached_edge_index = torch.cat([spatial, temporal], dim=1)
            self._cached_T = T

        edge_single = self._cached_edge_index.to(device)  # (2, E_single)
        E = edge_single.size(1)

        # 7) Replicate for all B graphs, offsetting node indices
        batch_offsets = (
            torch.arange(B, device=device)
            .repeat_interleave(E)
            * (T * N)
        )
        edge_index = edge_single.repeat(1, B) + batch_offsets.unsqueeze(0)  # (2, B*E)

        # 8) Build the batch vector
        batch_idx = torch.arange(B, device=device).repeat_interleave(T * N)  # (B*T*N,)
        batch = Batch(x=x, edge_index=edge_index, batch=batch_idx)

        # --- Optional Debug Visualization (first graph only) ---
        # if not hasattr(self, "_graph_visualized"):
        #     try:
        #         # Number of nodes in one graph:
        #         num_nodes_one = T * N  # frame_stack * num_nodes_per_frame

        #         # 1) x slice for graph 0
        #         x0 = x[:num_nodes_one]

        #         # 2) edge mask for edges entirely within [0, num_nodes_one)
        #         row, col = edge_index
        #         mask = (row < num_nodes_one) & (col < num_nodes_one)
        #         edge0 = torch.stack([row[mask], col[mask]], dim=0)

        #         # 3) build Data and visualize
        #         first_graph = Data(x=x0, edge_index=edge0)
        #         debug_dir = os.path.join(os.path.expanduser("~"), "robomimic_debug_diffgat")
        #         os.makedirs(debug_dir, exist_ok=True)
        #         save_path = os.path.join(debug_dir, f"graph_vis_fs{T}.png")
        #         self.visualize_graph(first_graph, save_path=save_path)

        #         self._graph_visualized = True
        #     except Exception as e:
        #         print(f"Warning: Failed to visualize graph: {e}")
        # --- End Debug Visualization ---

        return batch 
    
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
            print(
                "Warning: Cannot visualize graph before processing features (node_keys/frame_stack unknown)."
            )
            return

        # Convert PyG graph to NetworkX graph for easier plotting
        G = to_networkx(graph_data, node_attrs=["x"], to_undirected=True)

        # Extract node features (including positions) and edge indices
        node_features = graph_data.x.detach().cpu().numpy()
        positions_3d = node_features[:, :3]  # Assuming first 3 features are x, y, z

        # Create two separate figures - one for 3D graph, one for distance heatmap
        fig = plt.figure(figsize=(20, 10))

        # --- 3D Graph Plot ---
        ax1 = fig.add_subplot(121, projection="3d")

        # Determine node attributes (frame, type) for plotting
        node_attributes = {}
        for i in range(len(positions_3d)):
            frame_idx = i // self.num_nodes
            node_in_frame_idx = i % self.num_nodes
            node_type = self.node_keys[node_in_frame_idx]
            node_attributes[i] = {
                "pos": positions_3d[i],
                "frame": frame_idx,
                "type": node_type,
            }

        nx.set_node_attributes(G, node_attributes)

        # --- Plot Nodes ---
        cmap = plt.cm.viridis  # Color map for frames
        unique_frames = sorted(list(set(nx.get_node_attributes(G, "frame").values())))

        for frame in unique_frames:
            nodes_in_frame = [
                n for n, attr in G.nodes(data=True) if attr["frame"] == frame
            ]
            node_positions = np.array([G.nodes[n]["pos"] for n in nodes_in_frame])
            color = cmap(frame / max(1, len(unique_frames) - 1))

            if len(node_positions) > 0:
                ax1.scatter(
                    node_positions[:, 0],
                    node_positions[:, 1],
                    node_positions[:, 2],
                    color=color,
                    s=120,
                    alpha=0.8,
                    label=f"Frame {frame}",
                )
                # Annotate nodes with their type
                for i, node_idx in enumerate(nodes_in_frame):
                    ax1.text(
                        node_positions[i, 0],
                        node_positions[i, 1],
                        node_positions[i, 2],
                        f"{G.nodes[node_idx]['type']}",
                        fontsize=7,
                        ha="center",
                        va="bottom",
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
                ax1.plot(
                    [pos_u[0], pos_v[0]],
                    [pos_u[1], pos_v[1]],
                    [pos_u[2], pos_v[2]],
                    color=edge_color,
                    alpha=0.6,
                    linewidth=1.5,
                )
            else:
                # Temporal edge (between different frames) - dashed red
                ax1.plot(
                    [pos_u[0], pos_v[0]],
                    [pos_u[1], pos_v[1]],
                    [pos_u[2], pos_v[2]],
                    "r--",
                    alpha=0.3,
                    linewidth=0.8,
                )

        # --- Setup 3D Plot Appearance ---
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(
            f"Graph Visualization (Frames: {self.frame_stack}, Nodes/Frame: {num_nodes_per_frame})"
        )
        ax1.legend(title="Frames")

        # Add colorbar for distance edges
        sm = plt.cm.ScalarMappable(cmap=dist_cmap, norm=dist_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, label="Edge Distance (m)")

        # --- Distance Heatmap Plot ---
        ax2 = fig.add_subplot(122)

        # For visualization simplicity, let's show distances for the first frame only
        first_frame_indices = [
            i for i, attr in G.nodes(data=True) if attr["frame"] == 0
        ]
        first_frame_node_types = [G.nodes[i]["type"] for i in first_frame_indices]

        # Extract all pairwise distances between nodes in the first frame
        dist_matrix = np.zeros((len(first_frame_indices), len(first_frame_indices)))
        for i, node_i in enumerate(first_frame_indices):
            for j, node_j in enumerate(first_frame_indices):
                pos_i = G.nodes[node_i]["pos"]
                pos_j = G.nodes[node_j]["pos"]
                dist_matrix[i, j] = np.linalg.norm(pos_i - pos_j)

        # Create heatmap of distances
        im = ax2.imshow(dist_matrix, cmap="viridis")
        ax2.set_title("Pairwise Node Distances (First Frame)")

        # Set ticks with node type labels
        ax2.set_xticks(np.arange(len(first_frame_node_types)))
        ax2.set_yticks(np.arange(len(first_frame_node_types)))
        ax2.set_xticklabels(first_frame_node_types, rotation=45, ha="right")
        ax2.set_yticklabels(first_frame_node_types)

        # Add colorbar and distance values in cells
        fig.colorbar(im, ax=ax2, label="Distance (m)")

        # Annotate distance values in the heatmap
        for i in range(len(first_frame_node_types)):
            for j in range(len(first_frame_node_types)):
                text_color = (
                    "white" if dist_matrix[i, j] > dist_matrix.max() * 0.7 else "black"
                )
                ax2.text(
                    j,
                    i,
                    f"{dist_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

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
        self.action_check_done = False

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

        # --- Instantiate NodeFeatureProcessor AFTER Policy ---
        node_processor_instance = NodeFeatureProcessor(
            num_joints=self.algo_config.num_joints,  # Or get from config
            chain=self.chain,
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

        # observation_group_shapes = OrderedDict()
        # observation_group_shapes["obs"] = OrderedDict({
        #         "robot0_eye_in_hand_image": self.obs_shapes["robot0_eye_in_hand_image"]
        #     })
        # encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
        #     self.obs_config.encoder
        # )

        # obs_encoder = ObsNets.ObservationGroupEncoder(
        #     observation_group_shapes=observation_group_shapes,
        #     encoder_kwargs=encoder_kwargs,
        # )

        # obs_encoder = replace_bn_with_gn(obs_encoder)

        # obs_dim = obs_encoder.output_shape()[0]

        # --- Instantiate the Diffusion Policy Model with Lazy Input Dimension ---
        diffusion_model = DiffusionPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            graph_input_feature_dim=23,  # Use -1 for lazy initialization
            timestep_emb_dim=256,
            device=self.device,
        )
        gnn_model = GNNPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            graph_input_feature_dim=23,  # Use -1 for lazy initialization
            timestep_emb_dim=256,
            device=self.device,
        )
        # model = torch.compile(model)
        # the final arch has 2 parts
        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {"obs_encoder": None, 
                     "diffusion_model": diffusion_model,}
                    #  "gnn_model": gnn_model}
                )
            }
        )
        nets = nets.float().to(self.device)

        print(
            "Policy Network initialized (GNN input dim will be inferred on first forward pass)."
        )

        noise_scheduler = None

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
            beta_schedule=self.algo_config.ddim.beta_schedule,
            clip_sample=self.algo_config.ddim.clip_sample,
            set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
            steps_offset=self.algo_config.ddim.steps_offset,
            prediction_type=self.algo_config.ddim.prediction_type,
        )

        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(
                parameters=nets.parameters(), decay=self.algo_config.ema.power
            )

        self.noise_scheduler = noise_scheduler
        self.nets = nets
        self.ema = ema

        return self.nets  # Robomimic expects the nets dictionary returned

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
        if len(batch["obs"]["robot0_joint_pos"].shape) == 2:  # Assuming (seq, dim)
            batch = TensorUtils.unsqueeze_expand_batch(batch, 1)  # Add batch dim

        # Select observation window
        obs_data = {k: batch["obs"][k][:, :frame_stack, ...] for k in batch["obs"]}

        # Extract the next observation for the current time step from next_obs key
        next_q_pos_obs = batch["obs"]["robot0_joint_pos"][:, frame_stack, ...]

        action_data = batch["actions"][:, frame_stack - 1 :, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = action_data
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError(
                    '"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.'
                )
            self.action_check_done = True

        processed_batch = {
            "obs": obs_data,
            "next_q_pos_obs": next_q_pos_obs,  # Pass along next observation for training
            "goal_obs": batch.get("goal_obs", None),  # Pass along goals if present
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
        ).to(
            self.device
        )  # Ensure graph is on the correct device


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
            actions = batch["actions"]  # Shape: (batch_size, seq_len, action_dim)
            next_q_pos_obs = batch["next_q_pos_obs"]  # Shape: (batch_size, action_dim)
            graph = batch["graph"]  # PyG Batch object
            B = actions.shape[0]  # Get actual batch size

            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert batch["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # obs = batch["obs"]
            # obs_features = TensorUtils.time_distributed({'obs':obs}, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            # obs_cond = obs_features.flatten(start_dim=1)

            # 1. Sample Diffusion Timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=self.device,
            ).long()

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)

            # 2. Add Noise to Actions (Forward Diffusion Process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps
            )  # noisy_actions: (batch, seq_len, action_dim), noise: (batch, seq_len, action_dim)

            # 3. Predict Noise using the Policy Network
            # The model conditions on the graph state and timesteps
            q_pos, predicted_noise = self.nets["policy"]["diffusion_model"](
                noisy_action=noisy_actions,
                obs_cond=None,
                timestep=timesteps,
                graph=graph,
            )  # Output Shape: (batch, seq_len, action_dim)

            # 4. Compute all loss components
            noise_losses = F.mse_loss(predicted_noise, noise)
            q_pos_losses = F.mse_loss(q_pos, next_q_pos_obs)

            # p_actions = self.nets["policy"]["gnn_model"](
            #     graph=graph)
            
            # loss = F.mse_loss(p_actions, actions)

            weight = self.algo_config.loss.tradeoff
            # Combine losses (weighted sum)
            losses = OrderedDict()
            losses["predicted_noise"] = noise_losses
            losses["q_pos_loss"] = q_pos_losses
            losses["l2_loss"] = (1 - weight) * q_pos_losses + weight * noise_losses

            # losses = OrderedDict()
            # losses["l2_loss"] = loss
            # Store information for logging
            info = {
                # "predictions": TensorUtils.detach(predicted_noise),
                "losses": TensorUtils.detach(losses),
            }

            # 5. Backpropagation and Optimization Step (if training)
            if not validate:
                policy_grad_norm = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=losses["l2_loss"],
                    max_grad_norm=self.algo_config.grad_clip,
                )

                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets.parameters())

                step_info = {"policy_grad_norms": policy_grad_norm}
                info.update(step_info)

        return info  # Return loss info and other metrics

    def log_info(self, info: Dict) -> Dict:
        """Logs training/validation information."""
        log = super().log_info(info)  # Get base log info
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Q_Pos_Loss"] = info["losses"]["q_pos_loss"].item()
        log["Predicted_Noise_Loss"] = info["losses"]["predicted_noise"].item()
        # Add parameter count (useful for debugging model size)
        if "NumParams" not in log:  # Log only once
            log["NumParams"] = sum(
                p.numel() for p in self.nets.parameters() if p.requires_grad
            )
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(
        self, obs_dict: Dict, goal_dict: Optional[Dict] = None
    ) -> torch.Tensor:
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
            # obs = processed_obs["obs"]
            # obs_features = TensorUtils.time_distributed(
            #     {'obs':obs}, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True
            # )
            # obs_cond = obs_features.flatten(start_dim=1)

            # Sample an action sequence using DDIM
            sampled_actions = self.sample(
                obs_cond=None,
                graph=graph,
                num_steps=self.algo_config.ddim.num_inference_timesteps,
            )
            # sampled_actions = self.nets["policy"]["gnn_model"](
            #     graph=graph
            # )

            # Convert to numpy and store in buffer
            self.action_buffer.clear()
            self.action_buffer.extend(
                sampled_actions[0]
            )  # Add actions as individual steps

        # Retrieve the next action from the buffer
        next_action_np = self.action_buffer.popleft()  # Get first action
        # Convert back to tensor, add batch dimension, and move to device
        action_tensor = next_action_np.unsqueeze(0)  # Shape: (1, action_dim)

        return action_tensor

    @torch.no_grad()
    def sample(
        self,
        obs_cond: torch.Tensor,
        graph: Batch,
        num_steps: int = 50,
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
        batch_size = 1  # Inference is typically done one sample at a time

        if self.ema is not None:
            self.ema.store(self.nets.parameters())
            self.ema.copy_to(self.nets.parameters())

        # 1. Initialize Starting Noise
        # Start from pure Gaussian noise at timestep T
        noisy_action = torch.randn(
            (batch_size, self.seq_length, self.ac_dim), device=self.device
        )
        naction = noisy_action.clone()

        # 2. Prepare Timestep Schedule for Sampling

        self.noise_scheduler.set_timesteps(num_steps)

        # 3. Iterative Denoising Loop (DDIM Steps)
        for k in self.noise_scheduler.timesteps:

            # Predict noise using the policy network
            _, pred_noise = self.nets["policy"]["diffusion_model"](
                noisy_action=naction, obs_cond=obs_cond, timestep=k, graph=graph
            )

            naction = self.noise_scheduler.step(
                model_output=pred_noise,
                timestep=k,
                sample=naction,
            ).prev_sample

        # Restore parameters from EMA
        if self.ema is not None:
            self.ema.restore(self.nets.parameters())
        # 4. Final Output

        # action = nn.Tanh()(naction)
        return naction.contiguous()  # Ensure tensor is contiguous

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.load_state_dict(model_dict["ema"])


# =================== Vision Encoder Utils =====================
def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module
