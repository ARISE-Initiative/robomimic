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
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv, global_add_pool
import pytorch_kinematics as pk
import os
from torch_geometric.utils import softmax

from robomimic.models.obs_nets import ObservationEncoder, ObservationDecoder
from robomimic.models.obs_core import CropRandomizer
import robomimic.utils.obs_utils as ObsUtils

# import networkx as nx
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing


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


# +++ 1. ADD THE ATTENTION POOLING MODULE +++
class GlobalAttentionPool(nn.Module):
    """
    A PyG-compatible global attention pooling layer.

    This layer learns a weighted average of node features per graph (or group),
    where the weights are determined by an attention mechanism.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # A small network to compute attention scores for each node.
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # placeholders for logging
        self.last_attn_weights = None     # shape [num_nodes, 1]
        self.last_attn_batch_idx = None   # shape [num_nodes]

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The node features for the entire batch.
                              Shape: (num_total_nodes, hidden_dim)
            batch_idx (torch.Tensor): A tensor defining the group/graph assignment
                                      for each node. Shape: (num_total_nodes)

        Returns:
            torch.Tensor: The pooled embeddings for each group.
                          Shape: (num_groups, hidden_dim)
        """
        # 1. Compute raw attention scores for every node in the batch.
        attn_scores_raw = self.attention_net(x)

        # 2. Normalize scores per group using PyG's specialized softmax.
        # This is crucial for correct batch processing.
        attn_weights = softmax(attn_scores_raw, index=batch_idx)

        # save for logging
        self.last_attn_weights    = attn_weights.detach().cpu()
        self.last_attn_batch_idx  = batch_idx.detach().cpu()

        # 3. Compute the weighted sum of features within each group.
        # Element-wise multiplication broadcasts weights to all features.
        weighted_x = x * attn_weights

        # 4. Sum the weighted features for each group.
        pooled_x = global_add_pool(weighted_x, batch_idx)

        return pooled_x


# +++ 1. GATED FUSION MODULE +++
class GatedFusion(nn.Module):
    """
    Fuse per-type embeddings via learnable scalar gates.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # project each embedding to a scalar gate logit
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.last_gates = None

    def forward(self, per_type_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_type_emb: Tensor of shape [B, T, M, H]
        Returns:
            fused embedding of shape [B, T, H]
        """
        B, T, M, H = per_type_emb.size()
        # flatten to [B*T*M, H]
        flat = per_type_emb.view(B * T * M, H)

        # compute one logit per type embedding
        logits = self.gate_proj(flat)  # [B*T*M, 1]
        logits = logits.view(B * T, M)  # [B*T, M]

        # softmax over types
        gates = F.softmax(logits, dim=-1)  # [B*T, M]
        gates = gates.unsqueeze(-1)  # [B*T, M, 1]

        # reshape flat back to [B*T, M, H]
        flat_mt = flat.view(B * T, M, H)
        # weighted sum across types
        fused = (flat_mt * gates).sum(dim=1)  # [B*T, H]

        self.last_gates = gates.view(B,T,M).detach().cpu()

        # return [B, T, H]
        return fused.view(B, T, H)


# +++ 3. UPDATED GATv2Backbone WITH DYNAMIC GATED FUSION +++
class GATv2Backbone(nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        num_layers: int,
        node_encode_dim: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = num_heads
        self.attention_dropout = attention_dropout
        self.node_noise_std = 0.02

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_encode_dim),
        )

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        out_dim = hidden_dim // num_heads
        current_dim = node_encode_dim
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=attention_dropout,
                    add_self_loops=True,
                    edge_dim=6,
                )
            )
            current_dim = hidden_dim
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Pooling and fusion
        self.pooling = GlobalAttentionPool(hidden_dim)
        self.fusion = GatedFusion(hidden_dim)

    def forward(self, graph) -> torch.Tensor:
        x, edge_index, edge_attr, batch_idx = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
        )
        T = graph.t
        # encode and GNN
        x = x.float()
        x = self.node_encoder(x)

        if self.training and self.node_noise_std > 0:
            noise = torch.randn_like(x) * self.node_noise_std
            x = x + noise

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = nn.SiLU()(x)

        # global pooling
        # x = global_mean_pool(x, batch_idx)

        # per-type, per-time pooling
        B = int(batch_idx.max().item()) + 1
        T_val = T if isinstance(T, int) else T.item()
        # compute flat indices
        node_type_expanded = graph.node_type.unsqueeze(1).expand(-1, T_val, -1)
        type_flat = node_type_expanded.reshape(-1)
        # --------------------------------------------------------------
        # type_flat = torch.zeros_like(type_flat) # Pool all types together
        # --------------------------------------------------------------
        temp_flat = graph.node_temporal_mask.reshape(-1)
        pool_idx = (
            batch_idx * T_val * (type_flat.max().item() + 1)
            + temp_flat * (type_flat.max().item() + 1)
            + type_flat
        )
        pooled = self.pooling(x, pool_idx)

        # build dense [B, T, M, H]
        M = int(type_flat.max().item()) + 1
        total = B * T_val * M
        dense = x.new_zeros(total, self.hidden_dim)
        ids = torch.unique(pool_idx)
        dense[ids] = pooled
        per_type = dense.view(B, T_val, M, self.hidden_dim)

        # fuse per-type embeddings dynamically
        final_emb = self.fusion(per_type)  # [B, T, H]
        # x = x.view(-1, 1, self.hidden_dim)  # Reshape to [B*T, 1, H]
        return final_emb


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
            nn.Dropout(p=0.1),  # Add Dropout for regularization
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, T, F = obs.shape
        obs = obs.view(B * T, F)
        obs = self.mlp(obs)
        obs = obs.view(B, T, -1)
        return obs


class FlowPolicy(nn.Module):
    """
    Flow-based policy using an Encoder-Decoder Transformer architecture.
    The Encoder processes observation and time context.
    The Decoder processes the noisy action, conditioned on the context.
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
        """
        super().__init__()

        # Configuration parameters
        seq_len = global_config.train.seq_length
        frame_stack = global_config.train.frame_stack
        action_dim = algo_config.action_dim
        hidden = algo_config.transformer.hidden_dim
        num_heads = algo_config.transformer.num_heads
        num_encoder_layers = (
            algo_config.transformer.num_layers
        )  # You may need to add this to your config
        num_decoder_layers = (
            algo_config.transformer.num_layers
        )  # Renamed from num_layers

        self.hidden = hidden

        # --- ENCODING COMPONENTS ---

        # 1. Time Encoder
        self.time_encoder = SinusoidalPositionEmbeddings(hidden)

        # 2. Action Encoder (for decoder input)
        self.action_encoder = nn.Linear(action_dim, hidden)
        self.action_positional_encoding = PositionalEncoding(
            d_model=hidden,
            dropout=algo_config.transformer.attention_dropout,
            max_len=seq_len,
        )

        # 3. Observation Encoder (Graph or MLP)
        self.graph_encoder = GATv2Backbone(
            input_feature_dim=algo_config.gnn.node_input_dim,
            num_layers=algo_config.gnn.num_layers,
            node_encode_dim=algo_config.gnn.node_dim,
            hidden_dim=algo_config.gnn.hidden_dim,
            num_heads=algo_config.gnn.num_heads,
            attention_dropout=algo_config.gnn.attention_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=algo_config.transformer.attention_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=algo_config.transformer.attention_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        # --- OUTPUT HEAD ---
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, action_dim)
        )

        # Debug Decoder MLP
        self.debug_decoder_mlp = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim * seq_len),
        )

    def forward(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor = None,
        graph=None,
    ):
        """
        Forward pass updated for a SHARED, GLOBAL timestep.
        """
        if obs is None:
            obs_emb = self.graph_encoder(graph)
        else:
            obs_emb = self.obs_encoder(obs)

        if action is None:
            return None, None, obs_emb, None

        B, T_action, A = action.shape

        action_emb = self.action_encoder(action)
        tgt = self.action_positional_encoding(action_emb)

        time_emb_single = self.time_encoder(timestep.squeeze(-1))

        context_sequence = torch.cat([obs_emb, time_emb_single], dim=1)

        memory = self.context_encoder(context_sequence)

        output_embedding = self.action_decoder(tgt=tgt, memory=memory)

        predicted_vector_field = self.output_head(output_embedding)

        # Combine action, obs and time for decoder mlp
        # x = torch.cat(
        #     [obs_emb, time_emb_single, tgt],
        #     dim=1,
        # )
        # x = x.view(B, -1)
        # # Pass through the debug decoder MLP
        # predicted_vector_field = self.debug_decoder_mlp(x)
        # # Reshape to match the expected output shape
        # predicted_vector_field = predicted_vector_field.view(B, T_action, A)

        return predicted_vector_field, None, obs_emb, None
