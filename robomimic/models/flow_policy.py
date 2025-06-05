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
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
import pytorch_kinematics as pk
import os
from robomimic.models.obs_nets import ObservationEncoder, ObservationDecoder
from robomimic.models.obs_core import CropRandomizer
import robomimic.utils.obs_utils as ObsUtils
import networkx as nx

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
        input_feature_dim: int,
        num_layers: int,
        node_encode_dim: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
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

        # Node feature encoder - works regardless of number of nodes
        self.node_encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_encode_dim)
        )

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
                    edge_dim=6,  # Include relative distance edge features
                )
            )
            
            current_dim = self.hidden_dim  # Input for the next layer 
            self.norms.append(nn.LayerNorm(current_dim))
        
        self.pool = self._masked_pooling
        # Cross-attention layer to combine pooled node-type embeddings
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.heads, dropout=self.attention_dropout)

    def _masked_pooling(self, x: torch.Tensor, batch: torch.Tensor, T: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Masked pooling per node‐type.  
        x.flatten to [B*T*N,F], mask is [B,N] → expand to [B,T,N] → flat [B*T*N].
        """
        if mask is None:
            return global_mean_pool(x, batch)

        B = int(batch.max().item()) + 1
        # infer N = total nodes per graph
        N = x.size(0) // (B * T)
        assert mask.shape == (B, N), f"mask should be [B, N], got {mask.shape}"

        # [B, N] → [B, T, N]
        mask_exp = mask.unsqueeze(1).expand(-1, T, -1)
        # → [B*T*N]
        mask_flat = mask_exp.reshape(-1)

        x_masked = x[mask_flat]
        batch_masked = batch[mask_flat]

        return global_mean_pool(x_masked, batch_masked)  # [B, F]


    def forward(self, graph) -> torch.Tensor:
        """
        Processes the input graph batch.
        """
        x, edge_index, edge_attr, batch_indices = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        T = graph.t
        if x.dtype != torch.float32:
            x = x.float()

        # 1. GNN Message Passing
        x = self.node_encoder(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = nn.SiLU()(x)
        

        # 2. Vectorized Per-Type, Per-Timestep Pooling
        B = int(batch_indices.max().item()) + 1
        T_val = T if isinstance(T, int) else T.item()

        node_type_expanded = graph.node_type.unsqueeze(1).expand(-1, T_val, -1)
        
        # Now, flatten all tensors for the pooling index calculation.
        node_type_flat = node_type_expanded.reshape(-1)
        temporal_mask_flat = graph.node_temporal_mask.reshape(-1)
        
        assert batch_indices.shape == node_type_flat.shape == temporal_mask_flat.shape, "Shape mismatch before pooling!"

        max_node_type = int(node_type_flat.max().item()) + 1

        # Create a unique ID for each (batch, time, type) group.
        pool_idx = (
            batch_indices * T_val * max_node_type + 
            temporal_mask_flat * max_node_type + 
            node_type_flat
        )

        pooled_x = global_mean_pool(x, pool_idx)

        dense_pooled_embeddings = torch.zeros(
            B * T_val * max_node_type, self.hidden_dim, device=x.device
        )
        unique_ids = torch.unique(pool_idx, return_inverse=False)
        dense_pooled_embeddings[unique_ids] = pooled_x
        
        per_type_temp_emb = dense_pooled_embeddings.view(
            B, T_val, max_node_type, self.hidden_dim
        )

        # Vectorized Cross-Attention
        reshaped_emb = per_type_temp_emb.view(
            B * T_val, max_node_type, self.hidden_dim
        )
        query = reshaped_emb[:, :1, :].transpose(0, 1)
        key_value = reshaped_emb.transpose(0, 1)
        attn_output, _ = self.cross_attn(query, key_value, key_value)
        final_embeddings = attn_output.squeeze(0).view(B, T_val, self.hidden_dim)

        return final_embeddings

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

class ObservationEncoder(nn.Module):
    def __init__(self, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm for better stability
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm for the second layer
            nn.SiLU(),
            nn.Dropout(p=0.1)  # Add Dropout for regularization
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, T, F = obs.shape
        obs = obs.view(B * T, F)
        obs = self.mlp(obs)
        obs = obs.view(B, T, -1)
        return obs

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
        obs_dim: int = 0,
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

        # Positional encoding for actions
        self.positional_encoding = PositionalEncoding(
            d_model=hidden,
            dropout=algo_config.transformer.attention_dropout,
            max_len=seq_len,
        )

        # Graph encoding components
        self.graph_encoder = GATv2Backbone(
            input_feature_dim=algo_config.gnn.node_input_dim,
            num_layers=algo_config.gnn.num_layers,
            node_encode_dim=algo_config.gnn.node_dim,
            hidden_dim=algo_config.gnn.hidden_dim,
            num_heads=algo_config.gnn.num_heads,
            attention_dropout=algo_config.gnn.attention_dropout,
        )

        # # Alternative MLP-based observation encoder
        # self.obs_encoder = ObservationEncoder(
        #     hidden_dim=hidden,
        #     obs_dim=obs_dim
        # )

        # Joint decoder for inferring joint angles
        # self.joint_decoder = JointDecoder(
        #     algo_config=algo_config,
        #     hidden_dim=hidden,
        #     q_dim=algo_config.action_dim,
        # )

        # Forward kinematics encoder for end effector positions
        # self.fwd_kin_encoder = DiffFwdKinEncoder(
        #     algo_config=algo_config,
        #     q_dim=algo_config.action_dim,
        #     hidden_dim=hidden,
        #     device=device,
        # )

        # Dynamics model for predicting next state
        # self.dynamics_model = DynamicsModel(hidden, action_dim)

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
        obs: torch.Tensor = None,
        graph = None
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
        if obs is None:
            graph_emb = self.graph_encoder(graph) # [B, T, H]
            obs_emb = graph_emb

        else:
            obs_emb = self.obs_encoder(obs)  # [B, T, H]

        # Early return during initialization
        if action is None and obs is None:
            return None, None, obs_emb, None
        
        # Extract dimensions
        B, T, A = action.shape
        H = self.hidden

        # 1. Encode timesteps
        t_flat = timestep.view(-1)  # [B*T]
        te_flat = self.time_encoder(t_flat)  # [B*T, H]
        time_emb = te_flat.view(B, T, H)  # [B, T, H]


        # 2. Encode actions
        action_emb = self.action_encoder(action)  # [B, T, H]
        action_emb = self.positional_encoding(action_emb)
        context = time_emb + graph_emb 

        # 5. Decode to output actions using TransformerDecoder
        out = self.decoder(tgt=action_emb, memory=context)  # [B, T, H]


        return out, None, obs_emb, None
