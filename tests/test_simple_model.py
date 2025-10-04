import pytest

torch = pytest.importorskip("torch")

from unified_energy.models.simple import SimpleLanguageModel
from unified_energy.training.simple_trainer import SimpleLanguageModelTrainer


def test_simple_language_model_forward_shape() -> None:
    model = SimpleLanguageModel(vocab_size=20, d_model=12, hidden_size=16, num_layers=1)
    input_ids = torch.randint(0, 20, (3, 5))
    logits = model(input_ids)
    assert logits.shape == (3, 5, 20)


def test_simple_language_model_trainer_step_and_eval() -> None:
    torch.manual_seed(0)
    vocab_size = 10
    seq_len = 6
    model = SimpleLanguageModel(vocab_size=vocab_size, d_model=8, hidden_size=8, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    data = [
        (
            torch.randint(0, vocab_size, (2, seq_len)),
            torch.randint(0, vocab_size, (2, seq_len)),
        )
        for _ in range(3)
    ]
    trainer = SimpleLanguageModelTrainer(
        model,
        optimizer=optimizer,
        train_loader=data,
        val_loader=data,
        device="cpu",
    )
    metrics = trainer.train_step(data[0])
    assert "loss" in metrics
    val_metrics = trainer.evaluate()
    assert "val/loss" in val_metrics
