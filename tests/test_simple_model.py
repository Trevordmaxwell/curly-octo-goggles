import pytest

torch = pytest.importorskip("torch")

from unified_energy.data import ByteTokenizer, create_byte_lm_dataloaders
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


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello simple model"
    tokens = tokenizer.encode(text)
    assert tokens.dtype == torch.long
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_byte_dataloaders_create_batches(tmp_path) -> None:
    corpus = "simple language modelling with byte tokens. " * 4
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    train_loader, val_loader, tokenizer = create_byte_lm_dataloaders(
        path,
        seq_len=8,
        batch_size=4,
        val_fraction=0.2,
    )
    batch_inputs, batch_targets = next(iter(train_loader))
    assert batch_inputs.shape[1] == 8
    assert batch_targets.shape == batch_inputs.shape
    assert tokenizer.vocab_size == 256
    if val_loader is not None:
        val_inputs, _ = next(iter(val_loader))
        assert val_inputs.shape[1] == 8


def test_simple_language_model_generate_topk() -> None:
    torch.manual_seed(0)
    model = SimpleLanguageModel(vocab_size=32, d_model=16, hidden_size=16, num_layers=1)
    prompt = torch.randint(0, 32, (1, 5))
    generated = model.generate(prompt, max_length=3, temperature=0.8, top_k=5)
    assert generated.shape == (1, 8)
