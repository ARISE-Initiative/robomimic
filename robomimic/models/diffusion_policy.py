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
# Note: robomimic imports are kept for potential future use or integration,
# but the core model here relies primarily on torch and torch_geometric.
# from robomimic.models.obs_nets import ObservationGroupEncoder
# from robomimic.models.base_nets import MLP
# import robomimic.utils.obs_utils as ObsUtils


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
        embeddings = t.float()[:, None] * embeddings[None, :] # Ensure t is float for multiplication
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GATv2Backbone(nn.Module):
    """
    Graph Attention Network v2 (GATv2) backbone using PyTorch Geometric.

    Processes graph structures (nodes, edges) and outputs a graph-level embedding
    using GATv2 convolutions, Layer Normalization, SiLU activation, and global pooling.
    """
    def __init__(
        self,
        input_feature_dim: int,
        num_layers: int,
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = input_feature_dim
        out_channels_per_head = self.hidden_dim // self.heads

        for _ in range(self.num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=current_dim,
                    out_channels=out_channels_per_head,
                    heads=self.heads,
                    concat=True,  # Concatenate head outputs
                    dropout=self.attention_dropout,
                    add_self_loops=True, # Recommended for GAT variants
                )
            )
            # LayerNorm is applied on the concatenated output (hidden_dim)
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            current_dim = self.hidden_dim  # Input for the next layer

        if activation.lower() == "silu":
            self.activation = nn.SiLU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.pool = global_mean_pool

    def forward(self, graph: Batch) -> torch.Tensor:
        """
        Processes the input graph batch.

        Args:
            graph (torch_geometric.data.Batch): A PyG Batch object containing graph data
                (x: node features, edge_index: connectivity, batch: graph indices).

        Returns:
            torch.Tensor: Graph-level embeddings of shape (batch_size, hidden_dim).
        """
        x, edge_index, batch_indices = graph.x, graph.edge_index, graph.batch
        if x.dtype != torch.float32:
             x = x.float() # Ensure float32 for GNN layers

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # Apply Layer Normalization *before* activation for stability
            x = self.norms[i](x)
            x = self.activation(x)
            # Optional: Add dropout after activation if needed
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Pool node features to get a single embedding per graph
        graph_embedding = self.pool(x, batch_indices)  # Shape: (batch_size, hidden_dim)
        return graph_embedding


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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # Shape: (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register 'pe' as a buffer, not a parameter
        self.register_buffer('pe', pe)

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
        x = x + self.pe[:x.size(1)].squeeze(1)
        return self.dropout(x)


class TransformerDiffusionHead(nn.Module):
    """
    Transformer decoder head for the diffusion policy.

    Takes noisy actions, timestep embeddings, and context vectors (from GNN)
    as input, and predicts the noise added to the actions using a Transformer decoder.
    """
    def __init__(self,
                 action_dim: int,
                 d_model: int,
                 context_dim: int,
                 nhead: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 timestep_emb_dim: int,
                 max_seq_len: int = 20, # Max length for positional encoding
                 activation: str = "silu",
                 ):
        """
        Initializes the Transformer decoder head.

        Args:
            action_dim (int): Dimension of the action space (per step).
            d_model (int): The main embedding dimension used throughout the Transformer.
                           Should typically match the GNN output dimension (hidden_dim).
            context_dim (int): Dimension of the context vector input (GNN output).
            nhead (int): Number of attention heads in the Transformer decoder layers.
            num_decoder_layers (int): Number of layers in the Transformer decoder stack.
            dim_feedforward (int): Dimension of the feedforward network in each decoder layer.
            dropout (float): Dropout rate used in projections, positional encoding, and decoder layers.
            timestep_emb_dim (int): Dimension of the input timestep embedding.
            max_seq_len (int): Maximum expected sequence length for positional encoding.
            activation (str): Activation function ('silu' or 'relu'). Defaults to 'silu'.
        """
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # Project concatenated (action + timestep embedding) to d_model
        self.action_time_proj = nn.Linear(action_dim + timestep_emb_dim, d_model)

        # Project context vector (e.g., GNN output) to d_model for decoder memory
        self.context_proj = nn.Linear(context_dim, d_model)

        # Positional encoding for the action sequence (decoder target)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        if activation.lower() == "silu":
            transformer_activation = F.silu
        elif activation.lower() == "relu":
            transformer_activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Standard Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True, # Expect input/output as (batch, seq, feature)
            norm_first=True   # Apply LayerNorm before attention/FFN (more stable)
        )

        # Stack of decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model) # Final LayerNorm after all decoder layers
        )

        # Output layer mapping decoder output back to action dimension (noise prediction)
        self.output_mlp = nn.Linear(d_model, action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer decoder head.

        Args:
            noisy_action (torch.Tensor): Noisy action sequence, shape (batch_size, seq_len, action_dim).
            timestep_embedding (torch.Tensor): Timestep embeddings, shape (batch_size, timestep_emb_dim).
            context_vector (torch.Tensor): Context vector (e.g., GNN output), shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Predicted noise, shape (batch_size, seq_len, action_dim).
        """
        batch_size, seq_len, _ = noisy_action.shape

        # 1. Prepare Target Sequence (tgt) for Decoder
        # Expand timestep embedding to match sequence length: (batch, 1, emb_dim) -> (batch, seq_len, emb_dim)
        timestep_embedding_expanded = timestep_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate noisy action and expanded timestep embedding along the feature dimension
        action_time = torch.cat([noisy_action, timestep_embedding_expanded], dim=-1) # Shape: (batch, seq_len, action_dim + emb_dim)

        # Project concatenated features to d_model: (batch, seq_len, d_model)
        tgt = self.action_time_proj(action_time)

        # Add positional encoding to the target sequence
        tgt = self.pos_encoder(tgt) # Shape: (batch, seq_len, d_model)

        # 2. Prepare Memory Sequence (memory) for Decoder
        # Project context vector (GNN output) to d_model: (batch, d_model)
        context_proj = self.context_proj(context_vector) # Shape: (batch, d_model)

        # Add a sequence dimension (length 1) for the decoder's memory input: (batch, 1, d_model)
        memory = context_proj.unsqueeze(1)

        # 3. Pass through Transformer Decoder
        # tgt: (batch, seq_len, d_model), memory: (batch, 1, d_model)
        # Output shape: (batch, seq_len, d_model)
        # Note: LayerNorms are handled internally by the decoder (norm_first=True and final norm)
        decoder_output = self.transformer_decoder(tgt=tgt, memory=memory)

        # 4. Project Decoder Output to Noise Prediction
        # Input: (batch, seq_len, d_model) -> Output: (batch, seq_len, action_dim)
        noise_pred = self.output_mlp(decoder_output)

        return noise_pred

class JointDecoder(nn.Module):
    """
    Joint decoder module for infering joint agnles from graph embedding.

    """
    def __init__(self, algo_config, d_model: int, q_dim: int):
        """
        Initializes the JointDecoder.

        Args:
            algo_config: Configuration object containing parameters for the decoder.
            d_model (int): The main embedding dimension used throughout the decoder.
            q_dim (int): Dimension of the joint angles.
        """
        super().__init__()
        self.q_dim = q_dim
        self.d_model = d_model

        # Linear layer to project GNN output to action dimension
        self.linear = nn.Linear(d_model, q_dim)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the JointDecoder.

        Args:
            graph_embedding (torch.Tensor): Graph-level embedding, shape (batch_size, d_model).

        Returns:
            torch.Tensor: Predicted joint angles, shape (batch_size, action_dim).
        """
        # Project graph embedding to action dimension
        joint_angles = self.linear(graph_embedding)
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
                    open(mjcf_path).read(), "link7"
                 ).to(dtype=torch.float32, device=self.device) 
                 print("Successfully loaded kinematic chain from robot.xml.")
            else:
                 print(f"Warning: Kinematic definition file not found at {mjcf_path}")
        except Exception as e:
            print(f"Warning: Failed to load robot.xml for kinematics: {e}")


        # Linear layer to project fwd kinematics to hidden dimension
        input_dim = 3 + 9 # 3 for position, 9 for rotation matrix
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
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
        eef_rot = transf[:, :3, :3]  # Extract end effector rotation (3x3 matrix)
        # Flatten the rotation matrix from (batch, 3, 3) to (batch, 9)
        eef_rot_flat = eef_rot.flatten(start_dim=1)  # Flatten all dims except batch dim
        eef_pose = torch.cat((eef_pos, eef_rot_flat), dim=-1)  # Concatenate position and flattened rotation
        
        eef_pose_embedding = self.mlp(eef_pose)  # Shape: (batch_size, hidden_dim)
        return eef_pose_embedding


class DiffusionPolicy(nn.Module):
    """
    Main Diffusion Policy model combining GATv2 backbone and Transformer head.

    Processes graph-structured state observations using GATv2, incorporates diffusion
    timestep information, and uses a Transformer decoder to predict noise for
    denoising action sequences.
    """
    def __init__(
        self,
        algo_config, # Configuration object 
        global_config,
        graph_input_feature_dim: int,
        timestep_emb_dim: int = 128,
        device: str = "cpu",
    ):
        """
        Initializes the Diffusion Policy model.

        Args:
            algo_config: Configuration object containing parameters for GNN,
                         Transformer, action dim, dropout, etc.
            graph_input_feature_dim (int): The dimension of features for each node
                                           in the input graph.
            timestep_emb_dim (int): Dimension for the sinusoidal timestep embeddings.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        self.algo_config = algo_config

        # --- Configuration Parameters ---
        gnn_config = algo_config.gnn
        transformer_config = algo_config.transformer
        network_config = algo_config.network

        hidden_dim = gnn_config.hidden_dim # GNN output dim, also used as Transformer d_model
        transformer_nhead = transformer_config.num_heads
        transformer_layers = transformer_config.num_layers
        transformer_ff_dim = hidden_dim * transformer_config.ff_dim_multiplier

        # Context dimension is the output of the GNN backbone
        context_dim = hidden_dim * 2 # GNN output dim + EEF embedding dim

        # --- Timestep Encoding ---
        # Projects timestep index to an embedding, then processes it through MLPs
        self.timestep_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, hidden_dim * 2), # Intermediate projection
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, timestep_emb_dim), # Final embedding dim for Transformer head
        ).to(device)

        # --- GATv2 Graph Encoder Backbone ---
        self.gnn = GATv2Backbone(
            input_feature_dim=graph_input_feature_dim,
            num_layers=gnn_config.num_layers,
            hidden_dim=hidden_dim,
            num_heads=gnn_config.num_heads,
            attention_dropout=gnn_config.attention_dropout,
            activation="silu", # Hardcoded or could be from config
        ).to(device)

        # --- Transformer Decoder Head ---
        self.transformer_head = TransformerDiffusionHead(
            action_dim=algo_config.action_dim,
            d_model=hidden_dim,                # Matches GNN output dim
            context_dim=context_dim,           # GNN output dimension provides context
            nhead=transformer_nhead,
            num_decoder_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=network_config.dropout,    # General network dropout
            timestep_emb_dim=timestep_emb_dim, # Dimension of the processed timestep embedding
            max_seq_len=global_config.train.seq_length + 5, # Provide margin over sequence length
            activation="silu", # Hardcoded or could be from config
        ).to(device)


        # --- Joint Decoder ---
        self.joint_decoder = JointDecoder(
            algo_config=algo_config,
            d_model=hidden_dim,
            q_dim=algo_config.num_joints
        ).to(device)


        # --- Differential Forward Kinematics Encoder ---
        self.fwd_kin_encoder = DiffFwdKinEncoder(
            algo_config=algo_config,
            q_dim=algo_config.num_joints,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        # Initialize model weights
        self.apply(self._init_weights)
        print(f"DiffusionPolicy initialized on {self.device}")

    def _init_weights(self, module):
        """Applies standard weight initialization."""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # Zero initialization for biases
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for LayerNorm
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        # GATv2Conv layers use default PyG initialization, which is usually fine.

    def forward(
        self,
        noisy_action: torch.Tensor,
        state: dict, # Original observation dict (unused if only graph is needed)
        timestep: torch.Tensor,
        graph: Optional[Batch] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Diffusion Policy.

        Args:
            noisy_action (torch.Tensor): Noisy action sequence, shape (batch, seq_len, action_dim).
            state (dict): Dictionary of raw observations (potentially unused if graph provides all state).
            timestep (torch.Tensor): Diffusion timesteps, shape (batch,).
            graph (Optional[torch_geometric.data.Batch]): Batched PyG graph representing the state.
                                                           Required if using the GNN backbone.

        Returns:
            torch.Tensor: Predicted noise, shape (batch_size, seq_len, action_dim).

        Raises:
            ValueError: If graph input is required but not provided.
        """
        if graph is None:
            # If only raw state was to be used, an alternative encoder would be needed here.
            raise ValueError("Graph data (PyG Batch) must be provided for GAT backbone.")

        # 1. Encode Diffusion Timestep
        # Input: (batch,) -> Output: (batch, timestep_emb_dim)
        t_emb = self.timestep_encoder(timestep)

        # 2. Encode State using GNN
        # Input: PyG Batch -> Output: (batch_size, hidden_dim)
        s_emb = self.gnn(graph) # Graph embedding serves as context

        q_pos = self.joint_decoder(s_emb)
        eef_pos_embedding = self.fwd_kin_encoder(q_pos)

        # 3. Define Context Vector
        # Currently uses only the GNN embedding. Could be extended to include other features.
        context_vector = torch.cat((s_emb, eef_pos_embedding), dim=-1) # Shape: (batch_size, hidden_dim + 3 + 9)

        # 4. Predict Noise using Transformer Head
        # Inputs: noisy_action, timestep embedding, context vector
        # Output: (batch_size, seq_len, action_dim)
        noise_pred = self.transformer_head(
            noisy_action=noisy_action,
            timestep_embedding=t_emb,
            context_vector=context_vector
        )

        return q_pos, noise_pred