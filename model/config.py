"""Configuration dataclasses for the neural network trainer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Configuration controlling model structure and optimization features.

    Parameters
    ----------
    input_dim:
        Number of features in the input vector.
    hidden_dim:
        Size of the hidden layer. Larger values increase model capacity.
    output_dim:
        Number of output classes. The model performs multi-class
        classification with a softmax head.
    learning_rate:
        Initial learning rate used by gradient descent updates. This value is
        automatically decayed each epoch when ``learning_rate_decay`` is
        greater than zero.
    learning_rate_decay:
        Factor controlling inverse time decay. A value of ``1e-3`` will reduce
        the learning rate roughly by ``1 / (1 + 1e-3 * epoch)``.
    l2_strength:
        L2 penalty applied to the weights. This is optional and can be set to
        zero when not desired.
    use_dropout:
        Enables dropout on the hidden activations. Dropout introduces
        stochastic regularisation that can improve generalisation at the cost
        of additional computation.
    dropout_rate:
        Probability of dropping a hidden activation when ``use_dropout`` is
        enabled. Must be between ``0`` and ``1``.
    gradient_clip:
        Maximum absolute value allowed for gradient entries. This optimisation
        is permanently enabled to avoid training instabilities caused by large
        updates.
    seed:
        Optional random seed used for reproducibility. Setting the seed makes
        stochastic features such as dropout deterministic, which simplifies
        testing and experimentation.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float = 0.05
    learning_rate_decay: float = 0.0
    l2_strength: float = 0.0
    use_dropout: bool = False
    dropout_rate: float = 0.1
    gradient_clip: float = 5.0
    seed: int | None = None


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Configuration for optional early stopping during training."""

    patience: int = 20
    min_delta: float = 1e-4
