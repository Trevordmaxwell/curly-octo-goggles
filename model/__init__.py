"""Minimal neural network package with configurable optimizations."""

from .config import ModelConfig, EarlyStoppingConfig
from .network import TwoLayerNN

__all__ = [
    "ModelConfig",
    "EarlyStoppingConfig",
    "TwoLayerNN",
]
