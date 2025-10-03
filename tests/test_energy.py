import pytest

torch = pytest.importorskip("torch")

from unified_energy.core.energy import EnergyHyperParameters, UnifiedEnergyFunction


def test_energy_components_sum_to_total() -> None:
    hyper = EnergyHyperParameters(beta=1.1, alpha=0.7, lambda_l2=0.02, lambda_smooth=0.01)
    energy_fn = UnifiedEnergyFunction(d_model=5, hyper=hyper)
    z = torch.randn(4, 5, requires_grad=True)
    z_next = torch.randn(4, 5)
    memory = torch.randn(8, 5)

    total, components = energy_fn(z, z_next, memory, compute_grad=True)
    reconstructed = sum(float(components[key].detach()) for key in ("hopfield", "consistency", "regularization"))
    assert pytest.approx(float(total.detach())) == reconstructed


def test_energy_gradient_matches_autograd() -> None:
    energy_fn = UnifiedEnergyFunction(d_model=3)
    z = torch.randn(2, 3, requires_grad=True)
    memory = torch.randn(6, 3)

    z_next = torch.randn(2, 3)
    grad = energy_fn.energy_gradient(z, memory, z_next=z_next)
    loss, _ = energy_fn(z, z_next, memory, compute_grad=True)
    loss.backward()
    assert torch.allclose(grad, z.grad, atol=1e-5)


def test_invalid_shapes_raise() -> None:
    energy_fn = UnifiedEnergyFunction(d_model=4)
    z = torch.randn(2, 4)
    memory = torch.randn(5, 3)

    with pytest.raises(ValueError):
        energy_fn(z, z, memory)
