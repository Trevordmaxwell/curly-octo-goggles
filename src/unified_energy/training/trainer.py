"""Prototype training loop for the unified energy model."""
from __future__ import annotations

from typing import Iterable

from torch import Tensor

from ..models.unified import UnifiedModel


def train_epoch(model: UnifiedModel, batches: Iterable[Tensor]) -> None:
    """Run a dummy epoch to validate interfaces."""

    for batch in batches:
        z0, context, memory = batch
        model(z0, context, memory)
