"""Unified energy functional blending Hopfield, DEQ, and regularisation terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class EnergyHyperParameters:
    """Hyper-parameters controlling the unified energy functional."""

    beta: float = 1.0
    alpha: float = 1.0
    lambda_l2: float = 1e-2
    lambda_smooth: float = 1e-3

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.beta <= 0:
            raise ValueError("beta must be strictly positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        if self.lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")


class UnifiedEnergyFunction(nn.Module):
    """Energy function unifying Hopfield retrieval and DEQ equilibrium."""

    def __init__(self, d_model: int, hyper: Optional[EnergyHyperParameters] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.hyper = hyper or EnergyHyperParameters()

    def hopfield_energy(self, z: Tensor, memory_patterns: Tensor) -> Tensor:
        """Modern Hopfield energy: :math:`-\log\sum_i \exp(\beta \langle z, m_i\rangle)`."""

        self._validate_latent(z, memory_patterns)
        beta = self.hyper.beta
        logits = beta * z @ memory_patterns.t()
        energy = -torch.logsumexp(logits, dim=-1)
        return energy

    def consistency_energy(self, z: Tensor, z_next: Tensor) -> Tensor:
        """DEQ consistency term ``||z - f(z)||^2``."""

        if z.shape != z_next.shape:
            msg = "z_next must match z in shape"
            raise ValueError(msg)
        residual = z - z_next
        return torch.sum(residual * residual, dim=-1)

    def regularization_energy(self, z: Tensor, grad_z: Optional[Tensor] = None) -> Tensor:
        """Regularisation combining L2 penalty and optional smoothness term."""

        l2 = torch.sum(z * z, dim=-1)
        smoothness = torch.zeros_like(l2)
        if grad_z is not None:
            smoothness = torch.sum(grad_z * grad_z, dim=-1)
        return self.hyper.lambda_l2 * l2 + self.hyper.lambda_smooth * smoothness

    def forward(
        self,
        z: Tensor,
        z_next: Tensor,
        memory_patterns: Tensor,
        *,
        compute_grad: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total energy and per-term diagnostics."""

        hopfield = self.hopfield_energy(z, memory_patterns).mean()
        consistency = self.hyper.alpha * self.consistency_energy(z, z_next).mean()

        grad_z = None
        if compute_grad and z.requires_grad:
            grad_z = torch.autograd.grad(hopfield, z, create_graph=True, retain_graph=True)[0]

        regularization = self.regularization_energy(z, grad_z).mean()
        total = hopfield + consistency + regularization
        components = {
            "hopfield": hopfield,
            "consistency": consistency,
            "regularization": regularization,
            "total": total,
        }
        return total, components

    def energy_gradient(
        self,
        z: Tensor,
        memory_patterns: Tensor,
        z_next: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute gradient of total energy w.r.t. ``z``.

        Parameters
        ----------
        z:
            Current latent state.
        memory_patterns:
            Hopfield-style memory bank.
        z_next:
            Optional next state (typically ``f(z)``). If omitted the method
            defaults to a self-consistency evaluation.
        """

        if z_next is not None and z_next.shape != z.shape:
            msg = "z_next must match z in shape"
            raise ValueError(msg)

        z_req = z.detach().requires_grad_(True)
        if z_next is None:
            z_next_input = z_req
        else:
            z_next_input = z_next.detach()
        with torch.enable_grad():
            energy, _ = self.forward(
                z_req,
                z_next_input,
                memory_patterns,
                compute_grad=True,
            )
        grad = torch.autograd.grad(energy, z_req, create_graph=False, retain_graph=False)[0]
        return grad

    def _validate_latent(self, z: Tensor, memory_patterns: Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("z must be 2D with shape (batch, d_model)")
        if memory_patterns.ndim != 2:
            raise ValueError("memory_patterns must be 2D")
        if z.size(-1) != self.d_model:
            raise ValueError("z dimension must match d_model")
        if memory_patterns.size(-1) != self.d_model:
            raise ValueError("memory patterns must align with d_model")
