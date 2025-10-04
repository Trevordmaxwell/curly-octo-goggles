"""Download or convert corpora for the simple language model."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATASETS = {
    "tiny-shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "tiny-shakespeare-100k": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small text corpus")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny-shakespeare",
        choices=sorted(DATASETS.keys()),
        help="Identifier of the builtin corpus",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Directory to store the downloaded corpus",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional override for the output filename",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    return parser.parse_args()


def download_corpus(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Corpus already exists at {destination}, skipping download.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())
    print("Download complete.")


def main() -> None:
    args = parse_args()
    url = DATASETS[args.dataset]
    out_dir = Path(args.out_dir)
    filename = args.filename or f"{args.dataset}.txt"
    destination = out_dir / filename
    download_corpus(url, destination, force=args.force)


if __name__ == "__main__":
    main()
