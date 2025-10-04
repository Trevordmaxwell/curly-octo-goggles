"""Hybrid backbone interleaving Mamba layers with sparse attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ..core.mamba import MambaLayer


@dataclass(slots=True)
class HybridBackboneConfig:
    """Configuration parameters for :class:`HybridBackbone`."""

    d_model: int
    num_layers: int = 6
    attention_every: int = 4
    attention_type: str = "local"
    window_size: int = 256
    num_heads: int = 4
    dropout: float = 0.0
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.attention_every <= 0:
            raise ValueError("attention_every must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.attention_type not in {"local", "global"}:
            msg = "attention_type must be either 'local' or 'global'"
            raise ValueError(msg)
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")


class HybridBackbone(nn.Module):
    """Interleave lightweight Mamba updates with occasional attention blocks."""

    def __init__(
        self,
        config: HybridBackboneConfig | None = None,
        *,
        d_model: Optional[int] = None,
        num_layers: int = 6,
        attention_every: int = 4,
        attention_type: str = "local",
        window_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if config is None:
            if d_model is None:
                raise ValueError("d_model must be provided when config is None")
            config = HybridBackboneConfig(
                d_model=d_model,
                num_layers=num_layers,
                attention_every=attention_every,
                attention_type=attention_type,
                window_size=window_size,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        self.config = config
        d_state = max(1, config.d_model // 2)
        self.mamba_layers = nn.ModuleList(
            [MambaLayer(config.d_model, d_state, 3) for _ in range(config.num_layers)]
        )
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.norm_mamba = nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        self.norm_attention = nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        self._mask_cache: Dict[Tuple[int, torch.device], Tensor] = {}

    def forward(
        self,
        context: Tensor,
        *,
        initial_state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Process ``context`` and optionally return the final latent state."""

        if context.ndim != 3:
            raise ValueError("context must have shape (batch, length, d_model)")
        batch, length, d_model = context.shape
        if d_model != self.config.d_model:
            raise ValueError("context embedding dimension does not match configuration")
        state = initial_state
        sequence = context
        for layer_index, layer in enumerate(self.mamba_layers, start=1):
            state = layer(sequence, state=state)
            sequence = self.norm_mamba(sequence + state.unsqueeze(1))
            if layer_index % self.config.attention_every == 0:
                attn_out = self._apply_attention(sequence)
                sequence = self.norm_attention(sequence + self.attn_dropout(attn_out))
        if return_state:
            if state is None:
                state = torch.zeros(batch, d_model, device=context.device, dtype=context.dtype)
            return sequence, state
        return sequence

    def _apply_attention(self, sequence: Tensor) -> Tensor:
        if self.config.attention_type == "global":
            attn_out, _ = self.attention(sequence, sequence, sequence)
            return attn_out
        mask = self._local_attention_mask(sequence.size(1), sequence.device, sequence.dtype)
        attn_out, _ = self.attention(sequence, sequence, sequence, attn_mask=mask)
        return attn_out

    def _local_attention_mask(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (length, device)
        if key not in self._mask_cache:
            mask = torch.full((length, length), float("-inf"), device=device, dtype=torch.float32)
            idx = torch.arange(length, device=device)
            dist = (idx[:, None] - idx[None, :]).abs()
            mask = mask.masked_fill(dist <= self.config.window_size, 0.0)
            self._mask_cache[key] = mask
        mask = self._mask_cache[key]
        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)
        return mask
