import pytest

torch = pytest.importorskip("torch")

from unified_energy.tasks import AssociativeRecallConfig, AssociativeRecallDataset


def test_associative_recall_batch_shapes() -> None:
    config = AssociativeRecallConfig(vocab_size=20, key_length=2, value_length=2, sequence_length=10, batch_size=4)
    dataset = AssociativeRecallDataset(config)
    inputs, targets = dataset.sample_batch()
    assert inputs.shape == (config.batch_size, config.sequence_length)
    assert targets.shape == (config.batch_size, config.sequence_length)
    # Ensure answer tokens placed at tail
    assert torch.all((targets[:, -config.value_length :] != config.pad_token) | (targets[:, -config.value_length :] == config.pad_token))

