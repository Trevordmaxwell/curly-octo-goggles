"""Minimal recurrent language model for rapid experimentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class SimpleLanguageModelConfig:
    """Configuration for :class:`SimpleLanguageModel`."""

    vocab_size: int
    d_model: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    tied_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


class SimpleLanguageModel(nn.Module):
    """Compact language model using a GRU encoder and tied embeddings."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        tied_embeddings: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_size or d_model
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.embedding = nn.Embedding(vocab_size, d_model)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self._tied_embeddings = tied_embeddings and hidden_dim == d_model
        if self._tied_embeddings:
            self.output_layer.weight = self.embedding.weight

    @classmethod
    def from_config(cls, config: SimpleLanguageModelConfig) -> "SimpleLanguageModel":
        return cls(
            config.vocab_size,
            d_model=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            tied_embeddings=config.tied_embeddings,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        hidden: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Return token logits and optionally the recurrent state."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length)")
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        output, next_hidden = self.encoder(embedded, hidden)
        logits = self.output_layer(self.dropout(output))
        if return_state:
            return logits, next_hidden
        return logits

    def compute_loss(
        self,
        batch: Tuple[Tensor, Tensor],
        *,
        hidden: Optional[Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Cross-entropy loss for teacher-forced language modelling."""

        input_ids, target_ids = batch
        logits, next_hidden = self.forward(input_ids, hidden=hidden, return_state=True)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=ignore_index,
        )
        return loss, next_hidden

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Tensor:
        """Autoregressively sample tokens from the model."""

        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape (batch, length)")
        generated = prompt_ids.clone()
        for _ in range(max_length):
            logits, _ = self.forward(generated, return_state=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0 and top_k < logits.size(-1):
                kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth_values, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


__all__ = ["SimpleLanguageModel", "SimpleLanguageModelConfig"]
