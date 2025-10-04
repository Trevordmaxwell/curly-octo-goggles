"""Utility helpers for the unified energy framework."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time hinting only
    from .math_utils import stable_norm
    from .rolling_context import (
        ResidualTracker,
        RollingContextBuffer,
        RollingContextEngine,
        StateController,
    )
    from .visualization import plot_energy_history

__all__ = [
    "plot_energy_history",
    "ResidualTracker",
    "RollingContextBuffer",
    "RollingContextEngine",
    "StateController",
    "stable_norm",
]


def __getattr__(name: str):  # pragma: no cover - small wrapper
    if name == "plot_energy_history":
        return getattr(import_module("unified_energy.utils.visualization"), name)
    if name in {
        "ResidualTracker",
        "RollingContextBuffer",
        "RollingContextEngine",
        "StateController",
    }:
        module = import_module("unified_energy.utils.rolling_context")
        return getattr(module, name)
    if name == "stable_norm":
        return getattr(import_module("unified_energy.utils.math_utils"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
