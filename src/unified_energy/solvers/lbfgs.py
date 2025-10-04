"""Quasi-Newton solver with confidence-based early exit."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass(slots=True)
class LBFGSConfig:
    """Configuration for :class:`LBFGSSolver`."""

    max_iter: int = 30
    min_iter: int = 5
    confidence_threshold: float = 0.95
    history_size: int = 10
    tol: float = 1e-3
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.min_iter <= 0:
            raise ValueError("min_iter must be positive")
        if self.min_iter > self.max_iter:
            raise ValueError("min_iter cannot exceed max_iter")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be within (0, 1]")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")


class LBFGSSolver:
    """Solve for equilibria using an L-BFGS optimiser with heuristics."""

    def __init__(self, config: Optional[LBFGSConfig] = None) -> None:
        self.config = config or LBFGSConfig()

    def solve(
        self,
        z_init: Tensor,
        dynamics_fn: Callable[[Tensor, Optional[Tensor], Optional[Tensor]], Tensor],
        energy_fn,
        *,
        context: Optional[Tensor] = None,
        memory_patterns: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """Run the optimiser until convergence or early exit."""

        device = z_init.device
        orig_dtype = z_init.dtype
        z_param = z_init.detach().clone().to(torch.float32).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [z_param],
            max_iter=1,
            history_size=self.config.history_size,
            line_search_fn="strong_wolfe",
        )
        energy_history: list[float] = []
        final_components: Dict[str, float] = {}
        final_confidence = 0.0
        converged = False
        for iteration in range(self.config.max_iter):

            def closure() -> Tensor:
                optimizer.zero_grad()
                use_half = (
                    self.config.mixed_precision
                    and torch.cuda.is_available()
                    and iteration < int(0.7 * self.config.max_iter)
                )
                context_manager = (
                    torch.cuda.amp.autocast(dtype=torch.float16)
                    if use_half
                    else nullcontext()
                )
                with context_manager:
                    z_next = dynamics_fn(z_param, context, memory_patterns)
                    residual = z_param - z_next
                    energy, _ = energy_fn(z_param, z_next, memory_patterns)
                    loss = residual.pow(2).sum(dim=-1).mean() + energy
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                z_next = dynamics_fn(z_param, context, memory_patterns)
                energy, components = energy_fn(z_param, z_next, memory_patterns)
                residual = torch.linalg.vector_norm(z_param - z_next).item()
                grad_norm = torch.linalg.vector_norm(energy_fn.energy_gradient(z_param, memory_patterns)).item()
                energy_history.append(float(energy.detach()))
                final_components = {key: float(val) for key, val in components.items()}
                final_confidence = self._estimate_confidence(residual, grad_norm, iteration + 1)
                converged = residual < self.config.tol and grad_norm < self.config.tol
                if (
                    iteration + 1 >= self.config.min_iter
                    and final_confidence >= self.config.confidence_threshold
                    and converged
                ):
                    break
        info: Dict[str, object] = {
            "iterations": iteration + 1,
            "converged": converged,
            "confidence": final_confidence,
            "final_residual": residual,
            "final_energy_grad": grad_norm,
            "final_energy": float(energy.detach()),
            "energy_history": energy_history,
            "energy_components": final_components,
        }
        return z_param.detach().to(device=device, dtype=orig_dtype), info

    def _estimate_confidence(self, residual: float, grad_norm: float, iteration: int) -> float:
        progress = iteration / max(1, self.config.max_iter)
        residual_score = max(0.0, 1.0 - residual / (self.config.tol + 1e-8))
        grad_score = max(0.0, 1.0 - grad_norm / (self.config.tol + 1e-8))
        score = 0.6 * residual_score + 0.4 * grad_score
        confidence = 0.2 + 0.5 * score + 0.3 * progress
        return float(max(0.0, min(1.0, confidence)))
