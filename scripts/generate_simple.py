"""Generate text from a trained simple language model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from unified_energy.data import ByteTokenizer
from unified_energy.models.simple import SimpleLanguageModel, SimpleLanguageModelConfig


@dataclass
class LoadedModel:
    model: SimpleLanguageModel
    tokenizer: Optional[ByteTokenizer]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tokens with the simple LM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string for generation")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file containing the prompt",
    )
    parser.add_argument(
        "--prompt-ids",
        type=int,
        nargs="*",
        default=None,
        help="Explicit token ids (useful when the tokenizer is synthetic)",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokens to append")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k filtering")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(path: str, device: Optional[str] = None) -> LoadedModel:
    raw = torch.load(path, map_location=device or "cpu")
    config = SimpleLanguageModelConfig(**raw["config"])
    model = SimpleLanguageModel.from_config(config)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    tokenizer_name = raw.get("tokenizer", "synthetic")
    tokenizer = ByteTokenizer() if tokenizer_name == "byte" else None
    return LoadedModel(model=model.to(device or "cpu"), tokenizer=tokenizer)


def prepare_prompt(args: argparse.Namespace, tokenizer: Optional[ByteTokenizer]) -> torch.Tensor:
    if tokenizer is not None:
        prompt_text = args.prompt
        if args.prompt_file:
            prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
        if prompt_text is None:
            prompt_text = ""
        tokens = tokenizer.encode(prompt_text)
        if tokens.numel() == 0:
            tokens = torch.tensor([0], dtype=torch.long)
        return tokens.unsqueeze(0)
    if args.prompt_ids is None:
        raise ValueError("Prompt ids required for synthetic checkpoints")
    ids = torch.tensor(args.prompt_ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = load_checkpoint(args.checkpoint, device=device)
    prompt_ids = prepare_prompt(args, loaded.tokenizer).to(device)
    with torch.no_grad():
        generated = loaded.model.generate(
            prompt_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    generated = generated.cpu()
    if loaded.tokenizer is not None:
        text = loaded.tokenizer.decode(generated[0])
        print(text)
    else:
        print("Generated token ids:", generated.tolist())


if __name__ == "__main__":
    main()
