import importlib.util
import math
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "unified_energy" / "solvers" / "beta_schedule.py"
spec = importlib.util.spec_from_file_location("beta_schedule", MODULE_PATH)
assert spec is not None and spec.loader is not None
beta_module = importlib.util.module_from_spec(spec)
sys.modules.setdefault("beta_schedule", beta_module)
spec.loader.exec_module(beta_module)
BetaAnnealingSchedule = beta_module.BetaAnnealingSchedule


def test_linear_schedule_progression() -> None:
    schedule = BetaAnnealingSchedule(
        beta_start=0.4,
        beta_end=1.6,
        warmup_steps=2,
        total_steps=6,
        schedule="linear",
    )
    values = [schedule.value(step) for step in range(1, 7)]
    assert values[0] == pytest.approx(0.4)
    assert values[1] == pytest.approx(0.4)
    assert values[-1] == pytest.approx(1.6)
    assert all(values[i] <= values[i + 1] + 1e-6 for i in range(len(values) - 1))


def test_cosine_schedule_matches_reference_formula() -> None:
    schedule = BetaAnnealingSchedule(
        beta_start=0.5,
        beta_end=2.5,
        warmup_steps=1,
        total_steps=5,
        schedule="cosine",
    )
    expected = []
    for step in range(1, 6):
        if step <= 1:
            expected.append(0.5)
        else:
            progress = (step - 1) / (5 - 1)
            expected.append(0.5 + 2.0 * (math.sin(0.5 * math.pi * progress) ** 2))
    values = [schedule.value(step) for step in range(1, 6)]
    for val, ref in zip(values, expected):
        assert val == pytest.approx(ref)


def test_value_validates_inputs() -> None:
    schedule = BetaAnnealingSchedule()
    with pytest.raises(ValueError):
        schedule.value(0)
    with pytest.raises(ValueError):
        schedule.value(1, total_steps=0)
