"""
Diffusion Policy implementation using a time-conditional Graph Convolutional 
Network (GCN) as a backbone and a small Transformer Encoder-Decoder head.
This architecture is designed to provide maximum expressive power while remaining
small enough to avoid overfitting on the given dataset.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax

# =====================
# Embedding Modules
# =====================


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal position embeddings for diffusion timesteps.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.embedding_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# =====================
# Attention Pooling & Fusion
# =====================


class GlobalAttentionPool(nn.Module):
    """
    Global attention pooling layer for node features per graph/group.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.last_attn_weights = None
        self.last_attn_batch_idx = None

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        attn_scores_raw = self.attention_net(x)
        attn_weights = softmax(attn_scores_raw, index=batch_idx)
        self.last_attn_weights = attn_weights.detach().cpu()
        self.last_attn_batch_idx = batch_idx.detach().cpu()
        weighted_x = x * attn_weights
        pooled_x = global_add_pool(weighted_x, batch_idx)
        return pooled_x


class GatedFusion(nn.Module):
    """
    Fuse per-type embeddings via learnable scalar gates.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.last_gates = None

    def forward(self, per_type_emb: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_types, hidden_dim = per_type_emb.size()
        flat = per_type_emb.view(batch_size * seq_len * num_types, hidden_dim)
        logits = self.gate_proj(flat).view(batch_size * seq_len, num_types)
        gates = F.softmax(logits, dim=-1).unsqueeze(-1)
        flat_mt = flat.view(batch_size * seq_len, num_types, hidden_dim)
        fused = (flat_mt * gates).sum(dim=1)
        self.last_gates = gates.view(batch_size, seq_len, num_types).detach().cpu()
        return fused.view(batch_size, seq_len, hidden_dim)


# =====================
# GNN Backbone (Time-Conditional GCN)
# =====================

class EdgeGCNConv(MessagePassing):
    """
    A simple Graph Convolutional Network layer that incorporates edge features.
    """
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__(aggr='add')
        self.message_proj = nn.Linear(node_in_dim + edge_in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = torch.cat([x_j, edge_attr], dim=1)
        return self.message_proj(msg)


class GCNBackbone(nn.Module):
    """
    A time-conditional GCN-based graph encoder. It injects the diffusion time
    embedding into the node features before message passing.
    """
    def __init__(
        self,
        input_feature_dim: int,
        edge_feature_dim: int,
        time_emb_dim: int,
        num_layers: int,
        node_encode_dim: int,
        hidden_dim: int,
        noise_std_dev: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise_std_dev = noise_std_dev
        self.node_encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_encode_dim),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        first_layer_dim = node_encode_dim + time_emb_dim
        self.convs.append(
            EdgeGCNConv(node_in_dim=first_layer_dim, edge_in_dim=edge_feature_dim, out_dim=hidden_dim)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(
                EdgeGCNConv(node_in_dim=hidden_dim, edge_in_dim=edge_feature_dim, out_dim=hidden_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.pooling = GlobalAttentionPool(hidden_dim)
        self.fusion = GatedFusion(hidden_dim)

    def forward(self, graph, time_emb) -> torch.Tensor:
        x, edge_index, edge_attr, batch_idx = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        num_timesteps = graph.t
        x = x.float()

        x = self.node_encoder(x)
        time_emb_expanded = time_emb[batch_idx]
        x = torch.cat([x, time_emb_expanded], dim=-1)

        if self.training and self.noise_std_dev > 0:
            x = x + torch.randn_like(x) * self.noise_std_dev

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i > 0:
                x_res = x
                x = conv(x, edge_index, edge_attr.float())
                x = norm(x)
                x = nn.SiLU()(x)
                x = x + x_res
            else:
                x = conv(x, edge_index, edge_attr.float())
                x = norm(x)
                x = nn.SiLU()(x)
            
        batch_size = int(batch_idx.max().item()) + 1
        t_val = num_timesteps if isinstance(num_timesteps, int) else num_timesteps.item()
        node_type_expanded = graph.node_type.unsqueeze(1).expand(-1, t_val, -1)
        type_flat = node_type_expanded.reshape(-1)
        temp_flat = graph.node_temporal_mask.reshape(-1)
        pool_idx = (
            batch_idx * t_val * (type_flat.max().item() + 1)
            + temp_flat * (type_flat.max().item() + 1)
            + type_flat
        )
        pooled = self.pooling(x, pool_idx)
        num_types = int(type_flat.max().item()) + 1
        total = batch_size * t_val * num_types
        dense = x.new_zeros(total, self.hidden_dim)
        ids = torch.unique(pool_idx)
        dense[ids] = pooled
        per_type = dense.view(batch_size, t_val, num_types, self.hidden_dim)
        final_emb = self.fusion(per_type)
        return final_emb

# =====================
# Main Policy
# =====================

class FlowPolicy(nn.Module):
    """
    Diffusion policy with a Time-Conditional GCN backbone and a Transformer
    Encoder-Decoder head.
    """

    def __init__(
        self,
        algo_config,
        global_config,
        device: str = "cpu",
        obs_dim: int = 30,
    ):
        super().__init__()
        seq_len = global_config.train.seq_length
        obs_len = global_config.train.frame_stack
        action_dim = algo_config.action_dim
        hidden_dim = algo_config.transformer.hidden_dim
        emb_dropout = algo_config.network.emb_dropout
        attn_dropout = algo_config.transformer.attention_dropout
        num_heads = algo_config.transformer.num_heads
        num_layers = algo_config.transformer.num_layers

        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.time_encoder = SinusoidalPositionEmbeddings(hidden_dim)
        # Positional embedding is now only for the observation sequence
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, obs_len, hidden_dim)) 
        nn.init.normal_(self.obs_pos_emb, mean=0.0, std=0.02)

        self.graph_encoder = GCNBackbone(
            input_feature_dim=algo_config.gnn.node_input_dim,
            edge_feature_dim=6,
            time_emb_dim=hidden_dim,
            num_layers=algo_config.gnn.num_layers,
            node_encode_dim=algo_config.gnn.node_dim,
            hidden_dim=algo_config.gnn.hidden_dim,
            noise_std_dev=algo_config.gnn.get("noise_std_dev", 0.01)
        )
        self.graph_emb_projection = nn.Linear(algo_config.gnn.hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=emb_dropout)

        self.action_pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.normal_(self.action_pos_emb, mean=0.0, std=0.02)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Context Encoder to process observation sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=attn_dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Causal mask for the decoder
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, float(0.0))
        self.register_buffer("tgt_mask", mask)

        # Action Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=attn_dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor = None,
        graph=None,
    ):
        # Time embedding is now created once and used by the GNN
        time_emb = self.time_encoder(timestep.squeeze(-1)).squeeze(1)  # Shape: (batch_size, 1, hidden_dim)

        # 1. Encode observations
        if obs is None:
            # GNN is now time-conditional
            obs_emb = self.graph_encoder(graph, time_emb)
            obs_emb = self.graph_emb_projection(obs_emb)
        else:
            obs_emb = self.obs_encoder(obs)
            if self.training:
                 obs_emb = obs_emb + torch.randn_like(obs_emb) * 0.01

        # 2. Apply positional embedding to the time-aware observation sequence
        context_sequence = self.dropout(obs_emb + self.obs_pos_emb.to(obs_emb.device))

        # 3. Process context sequence with the encoder to create memory
        # The separate time token has been removed.
        memory = self.context_encoder(context_sequence)
        
        # 4. Prepare action sequence for the decoder
        action_emb = self.action_encoder(action)
        tgt = self.dropout(action_emb + self.action_pos_emb.to(action_emb.device))

        # 5. Decode actions using the Transformer Decoder
        output_embedding = self.action_decoder(
            tgt=tgt, memory=memory, tgt_mask=self.tgt_mask
        )
        
        # 6. Project to action dimension
        predicted_vector_field = self.output_head(output_embedding)
        return predicted_vector_field
