from model import EarlyStoppingConfig, ModelConfig, TwoLayerNN


def make_xor_dataset():
    x = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [0, 1, 1, 0]
    return x, y


def test_model_learns_xor_without_dropout():
    x, y = make_xor_dataset()
    config = ModelConfig(
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        learning_rate=0.2,
        learning_rate_decay=1e-3,
        l2_strength=1e-3,
        use_dropout=False,
        gradient_clip=5.0,
        seed=42,
    )
    model = TwoLayerNN(config)
    history = model.fit(x, y, epochs=2000, early_stopping=EarlyStoppingConfig(patience=200, min_delta=1e-5))

    assert history.losses[-1] < 0.03
    predictions = model.predict(x)
    assert sum(int(p == t) for p, t in zip(predictions, y)) / len(y) > 0.95


def test_model_learns_with_dropout_and_decay():
    x, y = make_xor_dataset()
    config = ModelConfig(
        input_dim=2,
        hidden_dim=16,
        output_dim=2,
        learning_rate=0.3,
        learning_rate_decay=5e-3,
        l2_strength=1e-4,
        use_dropout=True,
        dropout_rate=0.2,
        gradient_clip=5.0,
        seed=123,
    )
    model = TwoLayerNN(config)
    history = model.fit(x, y, epochs=3000, early_stopping=EarlyStoppingConfig(patience=300, min_delta=1e-4))

    assert history.losses[0] > history.losses[-1]
    assert history.losses[-1] < 0.1
    assert sum(int(p == t) for p, t in zip(model.predict(x), y)) / len(y) > 0.9
