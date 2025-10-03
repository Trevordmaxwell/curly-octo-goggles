"""Interfaces for joint equilibrium solvers."""
from __future__ import annotations

from typing import Tuple

from torch import Tensor

from ..solvers.hybrid_solver import UnifiedEquilibriumSolver


__all__ = ["solve_equilibrium"]


def solve_equilibrium(
    solver: UnifiedEquilibriumSolver,
    z_init: Tensor,
    x_context: Tensor,
    memory_patterns: Tensor,
) -> Tuple[Tensor, dict]:
    """Convenience wrapper delegating to :class:`UnifiedEquilibriumSolver`."""

    return solver.solve(z_init, x_context, memory_patterns)
