"""Curriculum and annealing schedules."""
from __future__ import annotations

from typing import Iterator


def linear_schedule(start: float, end: float, steps: int) -> Iterator[float]:
    """Yield linearly spaced values from ``start`` to ``end``."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = (end - start) / max(steps - 1, 1)
    for index in range(steps):
        yield start + index * delta
