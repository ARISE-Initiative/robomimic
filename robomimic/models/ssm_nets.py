"""
Selective State-Space Model (SSM) building blocks for sequential policy learning.

Implements a Mamba-style selective scan mechanism as an alternative to Transformer
self-attention for processing observation sequences. The selective scan operates
in O(n) time complexity with respect to sequence length, compared to O(n^2) for
self-attention, making it suitable for long-horizon policy rollouts.

References:
    Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.
    https://arxiv.org/abs/2312.00752

    Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces," 2022.
    https://arxiv.org/abs/2111.00396
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from robomimic.models.base_nets import Module


class SelectiveSSMBlock(Module):
    """
    A single selective state-space block with input-dependent dynamics.

    The block follows the Mamba architecture: the input is projected and split
    into two branches. One branch passes through a 1D convolution and the SSM
    scan. The other branch provides a gating signal via SiLU activation. The
    gated output is projected back to the embedding dimension.

    Block diagram:

        x ─┬─── Linear(d, 2*expand*d) ──┬── Conv1d + SiLU ── SSM Scan ── (*)──
           │                             │                                  │
           └─────────────────────────────┴── SiLU (gate) ──────────────────┘
                                                                           │
                                                                     Linear ── y

    Args:
        embed_dim (int): input and output embedding dimension.
        state_dim (int): hidden state dimension for the SSM recurrence.
        conv_dim (int): kernel size for the local convolution.
        expand_factor (int): expansion factor for the inner dimension.
        dropout (float): dropout probability applied after the output projection.
    """

    def __init__(
        self,
        embed_dim,
        state_dim=16,
        conv_dim=4,
        expand_factor=2,
        dropout=0.1,
    ):
        super(SelectiveSSMBlock, self).__init__()

        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor
        self.inner_dim = embed_dim * expand_factor

        # input projection: produces both the main path and gate path
        self.in_proj = nn.Linear(embed_dim, 2 * self.inner_dim, bias=False)

        # local convolution on the main path
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_dim,
            padding=conv_dim - 1,
            groups=self.inner_dim,
            bias=True,
        )

        # input-dependent SSM parameters
        # B and C are produced from the convolved input
        self.x_proj = nn.Linear(self.inner_dim, state_dim * 2 + 1, bias=False)

        # log of the diagonal state transition matrix (learned per channel)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(self.inner_dim, -1)
            .clone()
        )

        # scaling parameter for the discretization
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # output projection
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=False)

        # layer norm before the block
        self.norm = nn.LayerNorm(embed_dim)

        # dropout
        self.drop = nn.Dropout(dropout)

    def _selective_scan(self, x, delta, B, C):
        """
        Run the selective SSM scan over a sequence.

        Computes the recurrence:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C_t * h_t

        where A_bar and B_bar are discretized using the zero-order hold:
            A_bar = exp(delta * A)
            B_bar = delta * B

        Args:
            x (torch.Tensor): input of shape (B, T, inner_dim).
            delta (torch.Tensor): discretization step of shape (B, T, inner_dim).
            B (torch.Tensor): input matrix of shape (B, T, state_dim).
            C (torch.Tensor): output matrix of shape (B, T, state_dim).

        Returns:
            y (torch.Tensor): output of shape (B, T, inner_dim).
        """
        batch_size, seq_len, _ = x.shape

        # A is diagonal, stored as log for numerical stability
        # A shape: (inner_dim, state_dim)
        A = -torch.exp(self.A_log)

        # discretize: A_bar = exp(delta * A), B_bar = delta * B
        # delta: (B, T, inner_dim) -> (B, T, inner_dim, 1)
        delta_unsqueezed = delta.unsqueeze(-1)

        # A_bar: (B, T, inner_dim, state_dim)
        A_bar = torch.exp(delta_unsqueezed * A.unsqueeze(0).unsqueeze(0))

        # B_bar: (B, T, inner_dim, state_dim)
        B_bar = delta_unsqueezed * B.unsqueeze(2)

        # sequential scan
        h = torch.zeros(
            batch_size, self.inner_dim, self.state_dim,
            device=x.device, dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :].unsqueeze(-1)
            # y = C * h (summed over state_dim)
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # skip connection scaled by D
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x):
        """
        Forward pass through the SelectiveSSMBlock.

        Args:
            x (torch.Tensor): input of shape (B, T, embed_dim).

        Returns:
            output (torch.Tensor): output of shape (B, T, embed_dim).
        """
        residual = x
        x = self.norm(x)

        # project to inner dimension, split into main path and gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # 1D convolution on the main path
        # (B, T, inner_dim) -> (B, inner_dim, T) for Conv1d
        x_main = x_main.transpose(1, 2)
        x_main = self.conv1d(x_main)[:, :, :x.shape[1]]
        x_main = x_main.transpose(1, 2)
        x_main = F.silu(x_main)

        # compute input-dependent SSM parameters
        ssm_params = self.x_proj(x_main)
        B_param = ssm_params[:, :, :self.state_dim]
        C_param = ssm_params[:, :, self.state_dim:2 * self.state_dim]
        delta = F.softplus(ssm_params[:, :, -1]).unsqueeze(-1).expand(
            -1, -1, self.inner_dim
        )

        # selective scan
        y = self._selective_scan(x_main, delta, B_param, C_param)

        # gate and project back
        y = y * F.silu(z)
        output = self.out_proj(y)
        output = self.drop(output)

        return output + residual

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape.
        """
        return list(input_shape)


class SSM_Backbone(Module):
    """
    Stacked selective state-space model backbone for sequential policy learning.

    Analogous to GPT_Backbone, this module chains multiple SelectiveSSMBlock layers
    with a final layer normalization. Input and output tensors have shape (B, T, D).

    Args:
        embed_dim (int): dimension of the embedding space.
        context_length (int): expected length of input sequences (used for validation).
        num_layers (int): number of SelectiveSSMBlock layers to stack.
        state_dim (int): hidden state dimension for each SSM block.
        conv_dim (int): kernel size for the local convolution in each SSM block.
        expand_factor (int): expansion factor for the inner dimension of each block.
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        embed_dim,
        context_length,
        num_layers=4,
        state_dim=16,
        conv_dim=4,
        expand_factor=2,
        dropout=0.1,
    ):
        super(SSM_Backbone, self).__init__()

        self.embed_dim = embed_dim
        self.context_length = context_length
        self.num_layers = num_layers

        self._create_networks(
            embed_dim=embed_dim,
            num_layers=num_layers,
            state_dim=state_dim,
            conv_dim=conv_dim,
            expand_factor=expand_factor,
            dropout=dropout,
        )

        # weight initialization
        self.apply(self._init_weights)

        param_count = sum(p.numel() for p in self.parameters())
        print(
            "Created {} model with number of parameters: {}".format(
                self.__class__.__name__, param_count
            )
        )

    def _create_networks(
        self,
        embed_dim,
        num_layers,
        state_dim,
        conv_dim,
        expand_factor,
        dropout,
    ):
        """
        Helper function to create networks.
        """
        self.nets = nn.ModuleDict()

        # stacked SSM blocks
        self.nets["ssm_blocks"] = nn.Sequential(
            *[
                SelectiveSSMBlock(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    conv_dim=conv_dim,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # final layer norm
        self.nets["output_ln"] = nn.LayerNorm(embed_dim)

    def _init_weights(self, module):
        """
        Weight initializer.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                module.bias.data.zero_()

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape.
        """
        return input_shape[:-1] + [self.embed_dim]

    def forward(self, inputs):
        """
        Forward pass through the stacked SSM backbone.

        Args:
            inputs (torch.Tensor): input of shape (B, T, embed_dim).

        Returns:
            output (torch.Tensor): output of shape (B, T, embed_dim).
        """
        assert inputs.shape[-1] == self.embed_dim, (
            "Expected input embedding dimension {}, got {}".format(
                self.embed_dim, inputs.shape[-1]
            )
        )
        x = self.nets["ssm_blocks"](inputs)
        return self.nets["output_ln"](x)
