"""Energy-based training objectives."""
from __future__ import annotations

from typing import Dict

from torch import Tensor

from ..core.energy import UnifiedEnergyFunction


def energy_training_loss(
    energy_fn: UnifiedEnergyFunction,
    z: Tensor,
    z_next: Tensor,
    memory_patterns: Tensor,
) -> Dict[str, float]:
    """Compute training loss components."""

    total, components = energy_fn(z, z_next, memory_patterns, compute_grad=True)
    return {"loss": float(total.detach()), **components}
