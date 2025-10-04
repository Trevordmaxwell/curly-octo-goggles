import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"


def _env_with_src() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)
    return env


def test_train_and_generate_workflow(tmp_path) -> None:
    corpus = "byte level testing corpus " * 8
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")

    checkpoint_path = tmp_path / "model.pt"

    train_cmd = [
        sys.executable,
        "scripts/train_simple.py",
        "--text-path",
        str(corpus_path),
        "--seq-len",
        "16",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--val-fraction",
        "0.0",
        "--device",
        "cpu",
        "--save-path",
        str(checkpoint_path),
    ]
    subprocess.run(
        train_cmd,
        cwd=REPO_ROOT,
        env=_env_with_src(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint_path.exists()

    generate_cmd = [
        sys.executable,
        "scripts/generate_simple.py",
        "--checkpoint",
        str(checkpoint_path),
        "--prompt",
        "hello",
        "--max-length",
        "4",
        "--temperature",
        "0.8",
        "--device",
        "cpu",
    ]
    result = subprocess.run(
        generate_cmd,
        cwd=REPO_ROOT,
        env=_env_with_src(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() != ""
