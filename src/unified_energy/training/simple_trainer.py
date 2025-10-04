"""Utility trainer for :class:`SimpleLanguageModel`."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_


class SimpleLanguageModelTrainer:
    """Minimal training loop for recurrent language models."""

    def __init__(
        self,
        model,
        *,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable[Tuple[Tensor, Tensor]],
        val_loader: Optional[Iterable[Tuple[Tensor, Tensor]]] = None,
        device: Optional[torch.device | str] = None,
        grad_clip: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip
        self.ignore_index = ignore_index
        self.step = 0

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Execute a single optimisation step."""

        self.model.train()
        input_ids, target_ids = self._move_batch(batch)
        loss, _ = self.model.compute_loss(
            (input_ids, target_ids),
            ignore_index=self.ignore_index,
        )
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.step += 1
        return {
            "loss": float(loss.detach()),
            "perplexity": float(torch.exp(loss.detach()).item()),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute validation loss across the validation loader."""

        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            input_ids, target_ids = self._move_batch(batch)
            loss, _ = self.model.compute_loss(
                (input_ids, target_ids),
                ignore_index=self.ignore_index,
            )
            total_loss += float(loss.detach()) * target_ids.numel()
            total_tokens += target_ids.numel()
        if total_tokens == 0:
            return {}
        avg_loss = total_loss / total_tokens
        return {
            "val/loss": avg_loss,
            "val/perplexity": float(torch.exp(torch.tensor(avg_loss)).item()),
        }

    def train_epoch(self) -> Dict[str, float]:
        """Iterate once over the training loader."""

        metrics: Dict[str, float] = {}
        for batch in self.train_loader:
            metrics = self.train_step(batch)
        return metrics

    def _move_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        return input_ids, target_ids


__all__ = ["SimpleLanguageModelTrainer"]
