"""Baseline fixed-point solvers used by the unified equilibrium solver."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


class FixedPointSolver:
    """Simple fixed-point iterator with optional Anderson acceleration."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def iterate(self, f: Callable[[Tensor], Tensor], z0: Tensor) -> Tensor:
        """Iteratively apply ``f`` until convergence."""

        z = z0
        for _ in range(self.max_iter):
            z_next = f(z)
            if torch.norm(z_next - z) < self.tol:
                return z_next
            z = z_next
        return z
