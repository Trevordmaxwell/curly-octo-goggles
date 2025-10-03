import pytest

torch = pytest.importorskip("torch")

from unified_energy.models import UnifiedMambaHopfieldDEQ


def test_unified_model_forward_shapes() -> None:
    model = UnifiedMambaHopfieldDEQ(
        vocab_size=32,
        d_model=16,
        d_state=8,
        d_conv=3,
        n_layers=2,
        memory_size=32,
        max_iterations=5,
    )
    input_ids = torch.randint(0, 32, (2, 5))

    logits, diagnostics = model(input_ids, return_diagnostics=True)

    assert logits.shape == (2, 5, 32)
    assert diagnostics is not None
    assert "solver_info" in diagnostics


def test_update_memory_queue_advances() -> None:
    model = UnifiedMambaHopfieldDEQ(
        vocab_size=16,
        d_model=8,
        d_state=4,
        d_conv=3,
        n_layers=1,
        memory_size=4,
        max_iterations=3,
    )
    z = torch.randn(3, 8)
    initial_cursor = int(model.memory_cursor.item())
    model.update_memory(z, should_store=torch.tensor([True, False, True], dtype=torch.bool))
    updated_cursor = int(model.memory_cursor.item())
    assert updated_cursor != initial_cursor
