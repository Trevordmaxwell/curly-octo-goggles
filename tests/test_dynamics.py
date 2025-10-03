import pytest

torch = pytest.importorskip("torch")

from unified_energy.core.dynamics import UnifiedDynamics


def test_dynamics_output_shape() -> None:
    dynamics = UnifiedDynamics(d_model=6, d_state=8, d_conv=3)
    z = torch.randn(2, 6)
    context = torch.randn(2, 5, 6)
    memory = torch.randn(10, 6)

    z_next = dynamics(z, context, memory)
    assert z_next.shape == z.shape


def test_contraction_estimate_returns_bool_and_value() -> None:
    dynamics = UnifiedDynamics(d_model=4, d_state=6, d_conv=3)
    z = torch.randn(2, 4)
    context = torch.randn(2, 7, 4)
    memory = torch.randn(5, 4)

    is_contractive, lipschitz = dynamics.is_contraction(z, context, memory)
    assert isinstance(is_contractive, bool)
    assert isinstance(lipschitz, float)
