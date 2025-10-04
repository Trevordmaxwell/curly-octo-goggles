"""Reusable building blocks for the unified architecture."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class MemoryAdapter(nn.Module):
    """Project inputs into the shared memory space."""

    def __init__(self, d_model: int, memory_dim: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim or d_model
        self.proj = nn.Linear(d_model, self.memory_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DynamicsAdapter(nn.Module):
    """Wrap dynamics modules with residual connections."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> Tensor:
        z = args[0]
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        return out_tensor
