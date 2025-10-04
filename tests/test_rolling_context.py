import math
import pathlib
import sys
from typing import List, Sequence, Tuple

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from unified_energy.utils.rolling_context import (
    ResidualTracker,
    RollingContextBuffer,
    RollingContextEngine,
    StateController,
)


def test_buffer_sliding_behavior():
    buffer = RollingContextBuffer[int](window_size=6, overlap_size=2)
    buffer.extend(range(4))
    assert buffer.snapshot() == (0, 1, 2, 3)
    buffer.extend([4, 5, 6])
    assert buffer.snapshot() == (1, 2, 3, 4, 5, 6)
    overlap = buffer.slide()
    assert overlap == (5, 6)
    assert buffer.snapshot() == (5, 6)


def test_state_controller_applies_decay_and_clamp():
    def norm_fn(state: List[float]) -> float:
        return math.sqrt(sum(v * v for v in state))

    def scale_fn(state: List[float], scale: float) -> List[float]:
        return [v * scale for v in state]

    controller = StateController[List[float]](
        [10.0, 0.0],
        decay=0.5,
        norm_clip=5.0,
        _norm_fn=norm_fn,
        _scale_fn=scale_fn,
    )

    new_state = controller.update([20.0, 0.0])
    assert pytest.approx(norm_fn(new_state), rel=1e-5) == 5.0


def test_engine_integration_cycle():
    history: List[Tuple[Sequence[int], Sequence[int]]] = []

    def step_fn(window: Sequence[int], state: Sequence[int]):
        history.append((tuple(window), tuple(state)))
        new_state = tuple(window[-2:]) if len(window) >= 2 else tuple(window)
        return sum(window), new_state

    engine = RollingContextEngine[int, Tuple[int, ...], int](
        step_fn,
        initial_state=tuple(),
        window_size=5,
        overlap_size=2,
    )

    output, state, overlap = engine.feed([1, 2, 3])
    assert output == 6
    assert state == (2, 3)
    assert overlap == (2, 3)

    output, state, overlap = engine.feed([4, 5])
    assert output == 2 + 3 + 4 + 5
    assert state == (4, 5)
    assert overlap == (4, 5)

    assert history[0][0] == (1, 2, 3)
    assert history[1][0] == (2, 3, 4, 5)
    assert history[1][1] == (2, 3)


def test_residual_tracker_summary_handles_empty_logs():
    tracker = ResidualTracker()
    assert tracker.summary() == {"boundary_mean": 0.0, "mid_mean": 0.0}
    tracker.log(boundary=1.0, mid=3.0)
    tracker.log(boundary=3.0, mid=5.0)
    assert tracker.summary() == {"boundary_mean": 2.0, "mid_mean": 4.0}
