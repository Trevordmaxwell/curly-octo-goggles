"""Rolling context utilities for EMMA-style pipelines.

This module provides standalone helpers for building an "always-on"
rolling context window with optional state management.  The utilities
are designed to be drop-in friendly so existing pipelines can adopt a
sliding buffer without modifying core training/inference code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")  # token or feature representation
S = TypeVar("S")  # hidden/state type
Y = TypeVar("Y")  # model output type


class RollingContextBuffer(Generic[T]):
    """Maintain a rolling window with configurable overlap.

    Parameters
    ----------
    window_size:
        Maximum number of elements kept in the buffer.  This is the
        "K" parameter in the design note.
    overlap_size:
        Number of most recent elements to retain across steps.  If not
        provided, a third of the window size is used.
    """

    def __init__(self, window_size: int, overlap_size: Optional[int] = None):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap_size is None:
            overlap_size = max(1, window_size // 3)
        if not 0 < overlap_size <= window_size:
            raise ValueError("overlap_size must be in (0, window_size]")

        self.window_size = window_size
        self.overlap_size = overlap_size
        self._buffer: Deque[T] = deque(maxlen=window_size)

    def extend(self, items: Iterable[T]) -> None:
        """Append items into the buffer."""

        for item in items:
            self._buffer.append(item)

    def snapshot(self) -> Tuple[T, ...]:
        """Return the current window contents as a tuple."""

        return tuple(self._buffer)

    def slide(self) -> Tuple[T, ...]:
        """Slide the buffer, keeping only the overlap.

        Returns
        -------
        tuple
            The retained overlap after the slide.  This can be helpful
            for diagnostics or tests.
        """

        if not self._buffer:
            return tuple()

        retained = list(self._buffer)[-self.overlap_size :]
        self._buffer.clear()
        self._buffer.extend(retained)
        return tuple(retained)

    def reset(self) -> None:
        """Clear the buffer entirely."""

        self._buffer.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


@dataclass
class StateController(Generic[S]):
    r"""Manage a state vector with decay and norm clamping.

    Parameters
    ----------
    state:
        Initial state to keep alive across ticks.
    decay:
        Exponential decay factor applied every step (``None`` to
        disable).
    norm_clip:
        Maximum :math:`\ell_2` norm.  Values above this threshold are
        rescaled.  ``None`` disables clipping.
    """

    state: S
    decay: Optional[float] = None
    norm_clip: Optional[float] = None
    _norm_fn: Callable[[S], float] = field(default=lambda state: 0.0, repr=False)
    _scale_fn: Callable[[S, float], S] = field(default=lambda state, scale: state, repr=False)

    def update(self, new_state: S) -> S:
        state = new_state
        if self.decay is not None:
            state = self._scale_fn(state, self.decay)
        if self.norm_clip is not None and self.norm_clip > 0:
            norm = self._norm_fn(state)
            if norm > self.norm_clip:
                scale = self.norm_clip / (norm + 1e-8)
                state = self._scale_fn(state, scale)
        self.state = state
        return state

    def reset(self, new_state: Optional[S] = None) -> None:
        if new_state is None:
            new_state = self.state
        self.state = new_state


class RollingContextEngine(Generic[T, S, Y]):
    """High-level helper that wires the buffer and state controller.

    The engine accepts a ``step_fn`` that implements the actual model
    logic.  The function is called with the current window and the
    latest state and must return a tuple ``(output, new_state)``.
    """

    def __init__(
        self,
        step_fn: Callable[[Sequence[T], S], Tuple[Y, S]],
        *,
        initial_state: S,
        window_size: int,
        overlap_size: Optional[int] = None,
        decay: Optional[float] = None,
        norm_clip: Optional[float] = None,
        norm_fn: Optional[Callable[[S], float]] = None,
        scale_fn: Optional[Callable[[S, float], S]] = None,
    ) -> None:
        self.buffer = RollingContextBuffer[T](window_size, overlap_size)
        self.state_controller = StateController[S](
            initial_state,
            decay=decay,
            norm_clip=norm_clip,
            _norm_fn=norm_fn or (lambda _: 0.0),
            _scale_fn=scale_fn or (lambda state, scale: state),
        )
        self._step_fn = step_fn

    def feed(self, items: Iterable[T]) -> Tuple[Y, S, Tuple[T, ...]]:
        """Feed new items through the rolling window.

        Returns the step output, the updated state and the overlap
        retained for the next call.  The overlap is returned so that
        downstream systems can log diagnostics such as boundary
        perplexity or residuals.
        """

        self.buffer.extend(items)
        window = self.buffer.snapshot()
        output, state = self._step_fn(window, self.state_controller.state)
        updated_state = self.state_controller.update(state)
        overlap = self.buffer.slide()
        return output, updated_state, overlap

    def reset(self, *, new_state: Optional[S] = None) -> None:
        """Reset both buffer and state."""

        self.buffer.reset()
        self.state_controller.reset(new_state)


class ResidualTracker:
    """Lightweight diagnostics container for DEQ-style residuals."""

    def __init__(self) -> None:
        self.boundary_residuals: List[float] = []
        self.mid_residuals: List[float] = []

    def log(self, *, boundary: float, mid: float) -> None:
        self.boundary_residuals.append(boundary)
        self.mid_residuals.append(mid)

    def summary(self) -> dict:
        return {
            "boundary_mean": float(sum(self.boundary_residuals) / max(len(self.boundary_residuals), 1)),
            "mid_mean": float(sum(self.mid_residuals) / max(len(self.mid_residuals), 1)),
        }


__all__ = [
    "RollingContextBuffer",
    "RollingContextEngine",
    "ResidualTracker",
    "StateController",
]
