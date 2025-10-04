"""Hierarchical memory inspired by multi-level biological memory systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class HierarchicalMemoryConfig:
    """Configuration values for :class:`HierarchicalMemory`."""

    d_model: int
    l1_size: int = 1000
    l2_size: int = 10000
    l3_size: int = 500
    beta_l1: float = 2.0
    beta_l2: float = 1.0
    beta_l3: float = 0.5
    consolidation_freq: int = 1000

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        for name, value in {
            "l1_size": self.l1_size,
            "l2_size": self.l2_size,
            "l3_size": self.l3_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "beta_l1": self.beta_l1,
            "beta_l2": self.beta_l2,
            "beta_l3": self.beta_l3,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.consolidation_freq <= 0:
            raise ValueError("consolidation_freq must be positive")


class HierarchicalMemory(nn.Module):
    """Three-level memory with learned routing and consolidation."""

    def __init__(
        self,
        config: HierarchicalMemoryConfig | None = None,
        *,
        d_model: Optional[int] = None,
        l1_size: int = 1000,
        l2_size: int = 10000,
        l3_size: int = 500,
        beta_l1: float = 2.0,
        beta_l2: float = 1.0,
        beta_l3: float = 0.5,
        consolidation_freq: int = 1000,
    ) -> None:
        super().__init__()
        if config is None:
            if d_model is None:
                raise ValueError("d_model must be provided when config is None")
            config = HierarchicalMemoryConfig(
                d_model=d_model,
                l1_size=l1_size,
                l2_size=l2_size,
                l3_size=l3_size,
                beta_l1=beta_l1,
                beta_l2=beta_l2,
                beta_l3=beta_l3,
                consolidation_freq=consolidation_freq,
            )
        self.config = config
        self.d_model = config.d_model
        self.register_buffer("level_counts", torch.zeros(3, dtype=torch.long))
        self.register_buffer("level_cursors", torch.zeros(3, dtype=torch.long))
        self.register_buffer("l1_memory", torch.zeros(config.l1_size, config.d_model))
        self.register_buffer("l2_memory", torch.zeros(config.l2_size, config.d_model))
        self.register_buffer("l3_memory", torch.zeros(config.l3_size, config.d_model))
        hidden = max(8, config.d_model // 2)
        self.router_mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer(
            "betas",
            torch.tensor([config.beta_l1, config.beta_l2, config.beta_l3], dtype=torch.float32),
        )
        self._consolidation_counter = 0

    @property
    def level_sizes(self) -> Tuple[int, int, int]:
        return (self.config.l1_size, self.config.l2_size, self.config.l3_size)

    def forward(self, query: Tensor) -> Tensor:
        return self.retrieve(query)[0]

    def retrieve(self, query: Tensor, *, return_details: bool = False) -> Tuple[Tensor, Dict[str, object]]:
        if query.ndim != 2:
            raise ValueError("query must have shape (batch, d_model)")
        if query.size(-1) != self.d_model:
            raise ValueError("query dimensionality must match d_model")
        weights = self.router(query)
        outputs = []
        attentions = []
        for level, beta in enumerate(self.betas):
            patterns = self._level_patterns(level)
            if patterns.numel() == 0:
                outputs.append(torch.zeros_like(query))
                attentions.append(torch.zeros(query.size(0), 0, device=query.device, dtype=query.dtype))
                continue
            patterns = patterns.to(query.device, query.dtype)
            logits = beta.to(query.device, query.dtype) * (query @ patterns.t())
            attn = torch.softmax(logits, dim=-1)
            summary = attn @ patterns
            outputs.append(summary)
            attentions.append(attn)
        combined = torch.zeros_like(query)
        details: Dict[str, object] = {}
        for level, summary in enumerate(outputs):
            combined = combined + weights[:, level : level + 1] * summary
        if return_details:
            stacked = torch.stack(outputs)
            details["level_weights"] = weights
            details["level_outputs"] = stacked
            details["level_attentions"] = attentions
        return combined, details

    def router(self, query: Tensor) -> Tensor:
        logits = self.router_mlp(query)
        return torch.softmax(logits, dim=-1)

    def store(self, patterns: Tensor, *, level: str | None = None) -> None:
        if patterns.ndim == 1:
            patterns = patterns.unsqueeze(0)
        if patterns.ndim != 2:
            raise ValueError("patterns must be of shape (batch, d_model)")
        if patterns.size(-1) != self.d_model:
            raise ValueError("pattern dimensionality must match d_model")
        if level is None or level == "auto":
            assignments = torch.argmax(self.router(patterns), dim=-1)
        else:
            level_map = {"l1": 0, "l2": 1, "l3": 2}
            if level not in level_map:
                raise ValueError("level must be 'l1', 'l2', 'l3', or 'auto'")
            assignments = torch.full((patterns.size(0),), level_map[level], dtype=torch.long, device=patterns.device)
        for pattern, level_idx in zip(patterns, assignments):
            self._write_to_level(int(level_idx.item()), pattern)

    def consolidate(self) -> None:
        self._consolidation_counter += 1
        if self._consolidation_counter % self.config.consolidation_freq != 0:
            return
        l1_patterns = self._level_patterns(0)
        if l1_patterns.numel() == 0:
            return
        summary = l1_patterns.mean(dim=0)
        self._write_to_level(1, summary)
        l2_patterns = self._level_patterns(1)
        if l2_patterns.numel() == 0:
            return
        schema = l2_patterns.mean(dim=0)
        self._write_to_level(2, schema)

    def _write_to_level(self, level: int, pattern: Tensor) -> None:
        buffers = [self.l1_memory, self.l2_memory, self.l3_memory]
        buffer = buffers[level]
        pattern = pattern.detach().to(buffer.device, buffer.dtype)
        cursor = int(self.level_cursors[level].item())
        buffer[cursor] = pattern
        cursor = (cursor + 1) % self.level_sizes[level]
        self.level_cursors[level] = torch.tensor(cursor, device=self.level_cursors.device, dtype=torch.long)
        count = min(self.level_sizes[level], int(self.level_counts[level].item()) + 1)
        self.level_counts[level] = torch.tensor(count, device=self.level_counts.device, dtype=torch.long)

    def _level_patterns(self, level: int) -> Tensor:
        buffers = [self.l1_memory, self.l2_memory, self.l3_memory]
        count = int(self.level_counts[level].item())
        if count == 0:
            return buffers[level][:0]
        return buffers[level][:count]
