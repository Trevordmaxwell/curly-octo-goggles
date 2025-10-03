"""Interfaces for loading pretrained components."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


def load_pretrained_state(path: str | Path) -> Dict[str, Any]:
    """Load a serialized checkpoint if it exists."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def apply_pretrained_weights(module: nn.Module, state: Dict[str, Any]) -> None:
    """Load state dictionary into ``module``."""

    module.load_state_dict(state)
