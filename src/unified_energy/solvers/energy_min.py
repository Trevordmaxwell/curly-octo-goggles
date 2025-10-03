"""Generic energy minimisation utilities."""
from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import Tensor


class EnergyMinimisationSolver:
    """Perform gradient-based minimisation of an energy function."""

    def __init__(self, lr: float = 1e-2, max_iter: int = 200) -> None:
        self.lr = lr
        self.max_iter = max_iter

    def minimise(
        self,
        energy_fn: Callable[[Tensor], Tensor],
        grad_fn: Callable[[Tensor], Tensor],
        z0: Tensor,
    ) -> Dict[str, object]:
        """Run gradient descent until convergence criteria are met."""

        z = z0
        history = []
        for iteration in range(self.max_iter):
            energy = energy_fn(z)
            grad = grad_fn(z)
            history.append(float(energy.detach()))
            z = z - self.lr * grad
            if torch.norm(grad) < 1e-4:
                break
        return {"solution": z, "energy_history": history}
