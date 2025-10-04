"""Unified dynamics blending Mamba temporal updates with Hopfield retrieval."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mamba import MambaLayer


class UnifiedDynamics(nn.Module):
    """Dynamics ``f`` implementing simultaneous Mamba and Hopfield updates."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        *,
        beta: float = 1.0,
        gate_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be strictly positive")
        self.d_model = d_model
        self.beta = beta
        self.mamba = MambaLayer(d_model, d_state, d_conv)
        self.query_proj = nn.Linear(d_model, d_model)
        if gate_type not in {"sigmoid", "tanh"}:
            raise ValueError("gate_type must be 'sigmoid' or 'tanh'")
        self._gate_type = gate_type
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        z: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
        *,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Blend Mamba and Hopfield updates into a single step."""

        mamba_update = self.mamba(x_context, state=z)
        hopfield_update = self.hopfield_update(z, memory_patterns)
        gate_raw = self.gate_linear(torch.cat([mamba_update, hopfield_update], dim=-1))
        if self._gate_type == "sigmoid":
            gate_values = torch.sigmoid(gate_raw)
        else:  # tanh gating gets rescaled into [0, 1]
            gate_values = 0.5 * (torch.tanh(gate_raw) + 1.0)
        blended = gate_values * mamba_update + (1.0 - gate_values) * hopfield_update
        z_next = z + self.out_proj(blended)
        if return_components:
            return z_next, {
                "mamba": mamba_update,
                "hopfield": hopfield_update,
                "gate": gate_values,
                "blended": blended,
            }
        return z_next

    def hopfield_update(self, z: Tensor, memory_patterns: Tensor) -> Tensor:
        """Modern Hopfield retrieval update."""

        if z.ndim != 2:
            raise ValueError("z must have shape (batch, d_model)")
        if memory_patterns.ndim != 2:
            raise ValueError("memory_patterns must have shape (memories, d_model)")
        similarities = self.beta * (self.query_proj(z) @ memory_patterns.t())
        attention_weights = F.softmax(similarities, dim=-1)
        return attention_weights @ memory_patterns

    def mamba_update(self, z: Tensor, x_context: Tensor) -> Tensor:
        """Wrapper to expose the Mamba update for testing."""

        return self.mamba(x_context, state=z)

    def is_contraction(
        self,
        z: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
        *,
        epsilon: float = 1e-2,
    ) -> Tuple[bool, float]:
        """Check whether the dynamics are contractive around ``z``."""

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        with torch.no_grad():
            delta = epsilon * torch.randn_like(z)
            z_perturbed = z + delta
            f_z = self.forward(z, x_context, memory_patterns)
            f_z_pert = self.forward(z_perturbed, x_context, memory_patterns)
            if isinstance(f_z, tuple):
                f_z = f_z[0]
            if isinstance(f_z_pert, tuple):
                f_z_pert = f_z_pert[0]
            numerator = torch.norm(f_z_pert - f_z, dim=-1)
            denominator = torch.norm(delta, dim=-1).clamp_min(1e-12)
            lipschitz = torch.mean(numerator / denominator).item()
        return lipschitz < 1.0, lipschitz
