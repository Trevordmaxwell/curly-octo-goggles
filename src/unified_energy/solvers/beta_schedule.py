"""Utilities for annealing the Hopfield/energy sharpness parameter ``beta``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass(slots=True)
class BetaAnnealingSchedule:
    """Piecewise schedule that anneals ``beta`` during DEQ iterations.

    Parameters
    ----------
    beta_start:
        Value used during the warm-up iterations. Should be small enough to
        encourage broad, exploratory retrieval.
    beta_end:
        Target value used once the warm-up period is over. Larger values make
        the retrieval landscape sharper and favour decisive convergence.
    warmup_steps:
        Number of initial iterations that keep ``beta`` fixed at
        :pyattr:`beta_start`.
    total_steps:
        Total number of iterations to plan for. If the solver executes fewer
        iterations the schedule automatically clamps to the maximum defined
        here.
    schedule:
        Ramp type used after the warm-up stage. Supported values are
        ``"linear"`` for a straight interpolation and ``"cosine"`` for the
        smooth :math:`\\sin^2` continuation described in the documentation.
    """

    beta_start: float = 0.5
    beta_end: float = 3.0
    warmup_steps: int = 3
    total_steps: int = 10
    schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.beta_start <= 0:
            raise ValueError("beta_start must be positive")
        if self.beta_end <= 0:
            raise ValueError("beta_end must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.schedule not in {"linear", "cosine"}:
            raise ValueError("schedule must be either 'linear' or 'cosine'")

    def value(self, step: int, *, total_steps: Optional[int] = None) -> float:
        """Return the annealed ``beta`` for the provided iteration step.

        Parameters
        ----------
        step:
            One-indexed iteration counter (``1`` corresponds to the first
            DEQ/Hopfield update).
        total_steps:
            Optional override for the total number of iterations. When omitted
            the value supplied during initialisation is used.
        """

        if step <= 0:
            raise ValueError("step must be one-indexed and therefore positive")

        total = self._resolve_total_steps(total_steps)
        step = min(step, total)
        warmup = min(self.warmup_steps, total)
        if step <= warmup or total == 1 or warmup >= total:
            return self.beta_start

        progress = (step - warmup) / max(1, total - warmup)
        ramp = self._ramp(progress)
        return self.beta_start + (self.beta_end - self.beta_start) * ramp

    def as_list(self, *, total_steps: Optional[int] = None) -> list[float]:
        """Materialise the schedule for diagnostic or logging purposes."""

        total = self._resolve_total_steps(total_steps)
        return [self.value(step, total_steps=total) for step in range(1, total + 1)]

    def _ramp(self, progress: float) -> float:
        progress = float(min(max(progress, 0.0), 1.0))
        if self.schedule == "linear":
            return progress
        # Smooth continuation: beta = beta0 + (beta_max - beta0) * sin^2(pi * x / 2)
        return math.sin(0.5 * math.pi * progress) ** 2

    def _resolve_total_steps(self, override: Optional[int]) -> int:
        total = self.total_steps if override is None else override
        if total <= 0:
            raise ValueError("total_steps must be positive")
        return total
