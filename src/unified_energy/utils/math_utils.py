"""Mathematical helper utilities."""
from __future__ import annotations

import torch
from torch import Tensor


def stable_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute a numerically stable L2 norm."""

    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)
