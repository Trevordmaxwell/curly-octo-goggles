"""Lightweight surrogate for Mamba selective state-space layer."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class MambaLayer(nn.Module):
    """Simplified Mamba layer capturing gated state-space updates."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_proj = nn.Linear(d_model, d_state)
        self.conv = nn.Conv1d(d_state, d_state, kernel_size=d_conv, padding=d_conv // 2)
        self.output_proj = nn.Linear(d_state, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: Tensor, *, state: Optional[Tensor] = None) -> Tensor:
        """Process ``context`` sequence while conditioning on current ``state``."""

        if context.ndim != 3:
            msg = "context must have shape (batch, length, d_model)"
            raise ValueError(msg)
        batch, length, d_model = context.shape
        if d_model != self.d_model:
            msg = "context embedding dimension must match d_model"
            raise ValueError(msg)
        if state is None:
            state = torch.zeros(batch, d_model, device=context.device, dtype=context.dtype)
        elif state.shape != (batch, d_model):
            msg = "state must have shape (batch, d_model)"
            raise ValueError(msg)

        state_tokens = self.state_proj(state).unsqueeze(-1).expand(-1, -1, length)
        context_tokens = self.state_proj(context.reshape(batch * length, d_model))
        context_tokens = context_tokens.reshape(batch, length, -1).transpose(1, 2)
        conv_out = self.conv(context_tokens + state_tokens)
        conv_out = conv_out.transpose(1, 2)
        summary = conv_out.mean(dim=1)
        candidate = self.output_proj(summary)
        gate_input = torch.cat([candidate, state], dim=-1)
        gate = self.gate(gate_input)
        updated = gate * candidate + (1.0 - gate) * state
        return self.dropout(updated)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, state_proj={self.state_proj.out_features}"
