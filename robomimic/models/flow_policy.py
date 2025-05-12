"""
Implementation of a Diffusion Policy using Graph Attention Network v2 (GATv2)
as a backbone and a Transformer decoder head.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
import pytorch_kinematics as pk
import os
from robomimic.models.obs_nets import ObservationEncoder, ObservationDecoder
from robomimic.models.obs_core import CropRandomizer
import robomimic.utils.obs_utils as ObsUtils
import networkx as nx


# Note: robomimic imports are kept for potential future use or integration,
# but the core model here relies primarily on torch and torch_geometric.
# from robomimic.models.obs_nets import ObservationGroupEncoder
# from robomimic.models.base_nets import MLP
# import robomimic.utils.obs_utils as ObsUtils
def rot6d_to_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrices via Gram-Schmidt.
    """
    a1, a2 = ortho6d[:, :3], ortho6d[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # [V,3,3]


class RBF(torch.nn.Module):
    def __init__(self, K=16, cutoff=2.0, device="cpu"):
        super().__init__()
        self.means = torch.linspace(0, cutoff, K, device=device)
        self.width = (self.means[1] - self.means[0]).item()

    def forward(self, dist):
        # dist: [E,]
        # returns [E, K]
        return torch.exp(-((dist.unsqueeze(1) - self.means) ** 2) / self.width**2)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal position embeddings for diffusion timesteps.

    Copied from Phil Wang's implementation:
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    """

    def __init__(self, dim: int):
        """
        Initializes the module.

        Args:
            dim (int): The dimension of the embeddings to generate.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension 'dim' must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generates embeddings for the input timesteps.

        Args:
            t (torch.Tensor): Tensor of shape (batch_size,) containing diffusion timesteps.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, dim) containing embeddings.
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = (
            t.float()[:, None] * embeddings[None, :]
        )  # Ensure t is float for multiplication
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GATv2Backbone(nn.Module):
    """
    Graph Attention Network v2 (GATv2) backbone using PyTorch Geometric.

    Processes graph structures (nodes, edges) and outputs a graph-level embedding
    using GATv2 convolutions, Layer Normalization, SiLU activation, global pooling,
    and skip connections.
    """

    def __init__(
        self,
        input_feature_dim: dict,
        num_layers: int,
        node_encode_dim: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        activation: str = "silu",
    ):
        """
        Initializes the GATv2 backbone.

        Args:
            input_feature_dim (int): Dimension of input node features.
            num_layers (int): Number of GATv2 layers.
            hidden_dim (int): Hidden dimension size for GATv2 layers. Must be
                              divisible by num_heads.
            num_heads (int): Number of attention heads in each GATv2 layer.
            attention_dropout (float): Dropout rate for attention weights.
            activation (str): Activation function ('silu' or 'relu'). Defaults to 'silu'.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = num_heads
        self.attention_dropout = attention_dropout
        self.input_feature_dim = input_feature_dim
        self.node_encode_dim = node_encode_dim  # Store node_encode_dim

        self.node_encoder = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # For skip connection dimension matching

        out_channels_per_head = self.hidden_dim // self.heads

        for i, (key, dim) in enumerate(input_feature_dim.items()):
            # Create a linear layer for each node type
            self.node_encoder[key] = nn.Linear(
                dim + 4, node_encode_dim
            )  # +4 for static features
            # Initialize the weights of the linear layer
            nn.init.xavier_uniform_(self.node_encoder[key].weight)
            nn.init.zeros_(self.node_encoder[key].bias)

        current_dim = node_encode_dim
        for i in range(self.num_layers):
            self.convs.append(
            GATv2Conv(

                in_channels=current_dim,
                out_channels=out_channels_per_head,
                heads=self.heads,
                concat=True,
                dropout=self.attention_dropout,
                add_self_loops=True,  # Recommended for GAT variants
                residual=True,
                edge_dim=3,  # Include relative distance edge features
            )
            )
            
            current_dim = self.hidden_dim  # Input for the next layer
            self.norms.append(nn.LayerNorm(current_dim))


        if activation.lower() == "silu":
            self.activation = nn.SiLU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.pool = global_mean_pool

    def forward(self, graph_features: dict) -> torch.Tensor:
        """
        Processes the input graph batch.

        Args:
            graph (torch_geometric.data.Batch): A PyG Batch object containing graph data
                (x: node features, edge_index: connectivity, batch: graph indices).

        Returns:
            torch.Tensor: Graph-level embeddings of shape (batch_size, hidden_dim).
        """
        graph = self.build_graph(graph_features)
        visualize = False
        if visualize:
            self.visualize_graph(graph, graph_idx=0, show_edge_weights=True, plot_3d=True)

        x, edge_index, batch_indices = graph.x, graph.edge_index, graph.batch
        edge_attr = graph.edge_attr  # Relative distance edge features
        if x.dtype != torch.float32:
            x = x.float()  # Ensure float32 for GNN layers

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = self.activation(x)

        # Pool node features to get a single embedding per graph
        graph_embedding = self.pool(x, batch_indices)  # Shape: (batch_size, hidden_dim)
        return graph_embedding

    def build_graph(self, node_dict: dict) -> Batch:
        """
        Fast spatio-temporal graph builder with full pairwise distances:
         - Caches combined spatio-temporal edge_index per frame_stack
         - Vectorizes node-feature stacking and flattening
         - Keeps the full torch.cdist distance computation
        """
        device = node_dict[next(iter(node_dict))].device
        # adjacency_matrix = torch.tensor(
        #     [  # J0 J1 J2 J3 J4 J5 J6 EEF OBJ
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # joint_0 -> joint_1
        #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # joint_1 -> joint_0, joint_2
        #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # joint_2 -> joint_1, joint_3
        #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # joint_3 -> joint_2, joint_4
        #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # joint_4 -> joint_3, joint_5
        #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # joint_5 -> joint_4, joint_6
        #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # joint_6 -> joint_5, eef
        #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # eef -> joint_6, insertion_hook, wrench
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # base_frame -> insertion_hook, wrench
        #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # insertion_hook -> eef, base_frame
        #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # ratcheting_wrench -> eef, base_frame
        #     ],
        #     dtype=torch.bool,
        # )
        # Adjacency matrix for the can task.
        adjacency_matrix = torch.tensor(
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
        # Add self loops
        adjacency_matrix = adjacency_matrix | torch.eye(adjacency_matrix.size(0), dtype=torch.bool)

        B,T,_ = node_dict[next(iter(node_dict))].shape # (Batch, T, Features)
        N = adjacency_matrix.size(0)  # Number of nodes
        # Pre-compute static edge index from the adjacency matrix
        static_edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()

        # === 1. NODE TYPE ENCODING (0=joint, 1=eef, 2=object, 3=base_frame)
        node_types = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2], device=device)
        node_type_onehot = F.one_hot(node_types, num_classes=3).float()  # (N, 4)

        # === 2. SHORTEST PATH DISTANCE TO EEF
        import networkx as nx
        G = nx.from_numpy_array(adjacency_matrix.cpu().numpy(), create_using=nx.Graph)
        dist_to_eef = torch.tensor(
            [nx.shortest_path_length(G, source=i, target=7) for i in range(N)],
            dtype=torch.float32,
            device=device
        ).unsqueeze(1)  # (N, 1)

        # === 3. COMBINE STATIC FEATURES
        static_features = torch.cat([node_type_onehot, dist_to_eef], dim=1)  # (N, 5)

        node_order = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
            'eef', 'object',#'base_frame', 'insertion_hook', 'wrench'
        ]
        if len(node_order) != N:
            raise ValueError(f"Mismatch between node_order length ({len(node_order)}) and N ({N})")

        processed_features = {}
        expected_node_encode_dim = -1 # To store and check consistency

        for i, key in enumerate(node_order):
            if key not in node_dict:
                raise ValueError(f"Missing node feature in node_dict: {key}")
            if key not in self.node_encoder:
                raise ValueError(f"Missing encoder for node type: {key}")

            dynamic_feats = node_dict[key] # Shape: (B, T, F_k)
            node_static_feats = static_features[i] # Shape: (5,)

            # Expand static features to match batch and time dimensions
            B, T, F_k = dynamic_feats.shape
            # Ensure static features are on the same device
            expanded_static = node_static_feats.to(dynamic_feats.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1) # Shape: (B, T, 5)

            # Concatenate RAW dynamic features with the expanded static features
            combined_raw_features = torch.cat([dynamic_feats, expanded_static], dim=-1) # Shape: (B, T, F_k + 5)
            # Encode the combined features
            encoded_combined = self.node_encoder[key](combined_raw_features) # Shape: (B, T, node_encode_dim)

            processed_features[key] = encoded_combined

        # Stack the processed (encoded) features for all nodes in the defined order
        # Ensure all encoded features have the same final dimension (self.node_encode_dim)
        node_feats = torch.stack([processed_features[k] for k in node_order], dim=1) # Shape: (B, N, T, node_encode_dim)

        # 5) Flatten into one big x: (B*T*N, node_encode_dim)
        # The dimension here should be the output dimension of the node encoders
        x = node_feats.permute(0, 2, 1, 3).reshape(B * T * N, self.node_encode_dim)

        # 6) Build & cache the single-sample combined edge_index if needed
        static = static_edge_index
        static = static.to(device)  # (2, E_static)
        # spatial edges repeated per frame
        spatial = static.repeat(1, T) + (
            torch.arange(T, device=device).repeat_interleave(static.size(1)) * N
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
        batch_offsets = torch.arange(B, device=device).repeat_interleave(E) * (T * N)
        edge_index = edge_single.repeat(1, B) + batch_offsets.unsqueeze(0)  # (2, B*E)

        # 8) Build the batch vector
        batch_idx = torch.arange(B, device=device).repeat_interleave(T * N)  # (B*T*N,)

        # Compute raw node positions for edge features
        node_pos = torch.stack([node_dict[k][..., :3] for k in node_order], dim=1)  # (B, N, T, 3)
        node_pos = node_pos.permute(0, 2, 1, 3).reshape(B * T * N, 3)  # (B*T*N, 3)
        src, dst = edge_index  # (E,), (E,)
        edge_attr = node_pos[dst] - node_pos[src]  # (E, 3)

        batch = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, pos = node_pos)
        return batch

    def visualize_graph(self, graph: Batch, graph_idx: int = 0, show_edge_weights: bool = False, plot_3d: bool = False):
        """Lightweight graph visualization for debugging (2D or 3D)."""
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt

        edge_index = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy() if hasattr(graph, 'edge_attr') else None
        batch_arr = graph.batch.cpu().numpy()
        # select nodes belonging to one subgraph
        idx = np.where(batch_arr == graph_idx)[0]
        idx_map = {orig: i for i, orig in enumerate(idx)}
        # build networkx graph
        G = nx.Graph()
        for orig in idx:
            G.add_node(idx_map[orig])
        for i, (u, v) in enumerate(edge_index.T):
            if batch_arr[u] == graph_idx and batch_arr[v] == graph_idx:
                u2, v2 = idx_map[u], idx_map[v]
                if show_edge_weights and edge_attr is not None:
                    weight = float(np.linalg.norm(edge_attr[i]))
                    G.add_edge(u2, v2, weight=weight)
                else:
                    G.add_edge(u2, v2)
        # Retrieve node positions
        pos_attr = graph.pos.cpu().numpy() if hasattr(graph, 'pos') else None
        # 3D plot if requested
        if plot_3d and pos_attr is not None:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # scatter and label nodes
            for orig in idx:
                p = pos_attr[orig]
                ax.scatter(p[0], p[1], p[2], s=50)
                ax.text(p[0], p[1], p[2], str(idx_map[orig]), size=8)
            # draw edges in 3D
            for i, (u, v) in enumerate(edge_index.T):
                if batch_arr[u] == graph_idx and batch_arr[v] == graph_idx:
                    p1, p2 = pos_attr[u], pos_attr[v]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=1)
                    if show_edge_weights and edge_attr is not None:
                        weight = float(np.linalg.norm(edge_attr[i]))
                        mid = (p1 + p2) / 2
                        ax.text(mid[0], mid[1], mid[2], f"{weight:.2f}", size=8)
            ax.set_title(f"3D Graph visualization (index={graph_idx})")
            plt.show()
            return
        # 2D fallback: use first two dims for layout
        if pos_attr is not None:
            pos = {idx_map[orig]: (pos_attr[orig, 0], pos_attr[orig, 1]) for orig in idx}
        else:
            pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=50)
        if show_edge_weights:
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
        plt.title(f"Graph visualization (index={graph_idx})")
        plt.show()

class PositionalEncoding(nn.Module):
    """
    Standard fixed sinusoidal positional encoding for sequences.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        """
        Initializes the positional encoding module.

        Args:
            d_model (int): The embedding dimension (must match the input tensor's feature dim).
            dropout (float): Dropout rate applied after adding positional encodings.
            max_len (int): Maximum sequence length for pre-computed encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)  # Shape: (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register 'pe' as a buffer, not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
                              Assumes batch_first=True.

        Returns:
            torch.Tensor: Output tensor with added positional encoding, same shape as input.
        """
        # self.pe is (max_len, 1, d_model). Select up to seq_len and remove the middle dim.
        # x is (batch_size, seq_len, d_model). Broadcasting adds pe to each batch element.
        x = x + self.pe[: x.size(1)].squeeze(1)
        return self.dropout(x)


class JointDecoder(nn.Module):
    """
    Joint decoder module for infering joint agnles from graph embedding.

    """

    def __init__(self, algo_config, hidden_dim: int, q_dim: int):
        """
        Initializes the JointDecoder.

        Args:
            algo_config: Configuration object containing parameters for the decoder.
            d_model (int): The main embedding dimension used throughout the decoder.
            q_dim (int): Dimension of the joint angles.
        """
        super().__init__()
        # Linear layer to project GNN output to action dimension
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(
                hidden_dim * 2, q_dim
            ),  # Output dimension is q_dim (joint angles)
        )
        # Linear layer to project GNN output to action dimension

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the JointDecoder.

        Args:
            graph_embedding (torch.Tensor): Graph-level embedding, shape (batch_size, d_model).

        Returns:
            torch.Tensor: Predicted joint angles, shape (batch_size, action_dim).
        """
        # Project graph embedding to action dimension
        joint_angles = self.mlp(graph_embedding)  # Shape: (batch_size, q_dim)
        return joint_angles


class DiffFwdKinEncoder(nn.Module):
    """
    Compute the forward kinematics from joint angles to end effector positions.
    Create embedding for the end effector positions.
    """

    def __init__(self, algo_config, q_dim: int, hidden_dim: int, device: str = "cpu"):
        """
        Initializes the DiffFwdKin.

        Args:
            algo_config: Configuration object containing parameters for the decoder.
            q_dim (int): Dimension of the joint angles.
            hidden_dim (int): The main embedding dimension used throughout the decoder.
        """
        super().__init__()

        # --- Kinematics Loading ---
        self.q_dim = q_dim
        self.hidden_dim = hidden_dim
        self.chain = None
        self.device = torch.device(device)
        try:
            mjcf_path = "robomimic/algo/panda/robot.xml"
            if os.path.exists(mjcf_path):
                self.chain = pk.build_serial_chain_from_mjcf(
                    open(mjcf_path).read(), "right_hand"
                ).to(dtype=torch.float32, device=self.device)
                print("Successfully loaded kinematic chain from robot.xml.")
            else:
                print(f"Warning: Kinematic definition file not found at {mjcf_path}")
        except Exception as e:
            print(f"Warning: Failed to load robot.xml for kinematics: {e}")

        # Linear layer to project fwd kinematics to hidden dimension
        input_dim = 3 + 6  # 3 for position, 6 for 6D rotation matrix
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.device)

    def forward(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the JointDecoder.

        Args:
            graph_embedding (torch.Tensor): Graph-level embedding, shape (batch_size, d_model).

        Returns:
            torch.Tensor: Predicted joint angles, shape (batch_size, action_dim).
        """
        # Calculate forward kinematics
        ret = self.chain.forward_kinematics(joint_pos)
        # Convert to tensor and move to the appropriate device
        transf = ret.get_matrix()

        eef_pos = transf[:, :3, 3]  # Extract end effector position (x, y, z)
        eef_rot = transf[
            :, :3, :2
        ]  # Extract end effector rotation using 6D rotation representation
        eef_rot_flat = eef_rot.reshape(eef_rot.shape[0], -1)  # Shape: (batch_size, 6)
        eef_pose = torch.cat(
            (eef_pos, eef_rot_flat), dim=-1
        )  # Concatenate position and flattened rotation

        eef_pose_embedding = self.mlp(eef_pose)  # Shape: (batch_size, hidden_dim)
        return eef_pose_embedding


class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        in_dim = hidden_dim + action_dim
        out_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for action prediction.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        """
        Initializes the TransformerDecoder.

        Args:
            hidden_dim (int): Dimension of the hidden embeddings.
            action_dim (int): Dimension of the output actions.
            num_layers (int): Number of Transformer decoder layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Transformer decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Final projection layer to map hidden states to action dimensions
        self.output_projection = nn.Linear(hidden_dim, action_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TransformerDecoder.

        Args:
            tgt (torch.Tensor): Target sequence embeddings [B, T, H].
            memory (torch.Tensor): Memory embeddings from the encoder [B, S, H].

        Returns:
            torch.Tensor: Decoded actions [B, T, A].
        """
        # Pass through the Transformer decoder
        decoded = self.transformer_decoder(tgt, memory)  # [B, T, H]

        # Project to action dimensions
        actions = self.output_projection(decoded)  # [B, T, A]
        return actions


class FlowPolicy(nn.Module):
    """
    Flow-based policy that uses graph embeddings and attention to predict actions.
    Combines graph representation with time embeddings and previous actions.
    """

    def __init__(
        self,
        algo_config,
        global_config,
        device: str = "cpu",
    ):
        """
        Initialize the Flow Policy.

        Args:
            algo_config: Configuration for model architecture
            global_config: Global configuration settings
            timestep_emb_dim: Dimension for timestep embeddings
            device: Computing device
        """
        super().__init__()

        # Configuration parameters
        seq_len = global_config.train.seq_length
        action_dim = algo_config.action_dim
        hidden = algo_config.transformer.hidden_dim
        self.seq_length = seq_len
        self.hidden = hidden
        self.t_p = seq_len
        self.t_a = algo_config.t_a  # Action execution horizon
        self.num_feedback_actions = self.t_p - self.t_a

        assert (
            self.t_a <= self.t_p
        ), "Action execution horizon cannot exceed prediction horizon"

        # Time encoding components
        self.time_encoder = SinusoidalPositionEmbeddings(hidden)

        # Action encoding components
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden))

        # Graph encoding components
        self.graph_encoder = GATv2Backbone(
            input_feature_dim=algo_config.gnn.node_input_dim,
            num_layers=algo_config.gnn.num_layers,
            node_encode_dim=algo_config.gnn.node_dim,
            hidden_dim=algo_config.gnn.hidden_dim,
            num_heads=algo_config.gnn.num_heads,
            attention_dropout=algo_config.gnn.attention_dropout,
        )

        # Joint decoder for inferring joint angles
        self.joint_decoder = JointDecoder(
            algo_config=algo_config,
            hidden_dim=hidden,
            q_dim=algo_config.action_dim,
        )

        # Forward kinematics encoder for end effector positions
        self.fwd_kin_encoder = DiffFwdKinEncoder(
            algo_config=algo_config,
            q_dim=algo_config.action_dim,
            hidden_dim=hidden,
            device=device,
        )

        # Dynamics model for predicting next state
        self.dynamics_model = DynamicsModel(hidden, action_dim)

        # Transformer-based action decoder
        self.decoder = TransformerDecoder(
            hidden_dim=hidden,
            action_dim=action_dim,
            num_layers=algo_config.transformer.num_layers,
            num_heads=algo_config.transformer.num_heads,
            dropout=algo_config.transformer.attention_dropout,
        )

    def forward(
        self,
        action,
        timestep,
        graph_data: dict,
        previous_unexecuted_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the policy.

        Args:
            action: Input actions [B, T, A]
            timestep: Input timesteps
            graph_data: Dictionary of graph features
            previous_unexecuted_actions: Previously planned but unexecuted actions

        Returns:
            tuple: (output_actions, current_graph_embedding, next_graph_embedding)
        """
        # Encode the current state as a graph embedding
        graph_emb = self.graph_encoder(graph_data).unsqueeze(1)  # [B, 1, H]

        # Early return during initialization
        if action is None:
            return None, None, graph_emb, None

        # Extract dimensions
        B, T, A = action.shape
        H = self.hidden

        # 1. Encode timesteps
        t_flat = timestep.view(-1)  # [B*T]
        te_flat = self.time_encoder(t_flat)  # [B*T, H]
        time_emb = te_flat.view(B, T, H)  # [B, T, H]

        # pred_q = self.joint_decoder(graph_emb).squeeze()  # [B, 1, H]
        # fk_emb = self.fwd_kin_encoder(pred_q)
        # fk_emb = fk_emb.unsqueeze(1)
        # 2. Encode actions
        action_emb = self.action_encoder(action)  # [B, T, H]
        context = torch.cat([time_emb, graph_emb], dim=1)  # [B, T, H]

        # 5. Decode to output actions using TransformerDecoder
        out = self.decoder(tgt=action_emb, memory=context)  # [B, T, H]

        # 3. Predict next state embedding using dynamics model
        next_graph_emb = self.dynamics_model(
            torch.cat([graph_emb, out[:, 0, :].unsqueeze(1)], dim=2)
        )

        return out, None, graph_emb, next_graph_emb
