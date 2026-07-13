#!/bin/bash
# Helper launcher so you can double-click to open the 4D shape viewer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
DEFAULT_REPO="$HOME/curly-octo-goggles"

if [[ ! -f "$REPO_DIR/scripts/visualize_hyper_shapes.py" ]]; then
  if [[ -f "$DEFAULT_REPO/scripts/visualize_hyper_shapes.py" ]]; then
    REPO_DIR="$DEFAULT_REPO"
  else
    echo "Could not locate scripts/visualize_hyper_shapes.py."
    echo "Please edit visualize_hyper_shapes.command to point at your repo."
    exit 1
  fi
fi

cd "$REPO_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH."
  exit 1
fi

echo "Launching 4D hyper-shape viewer..."
python3 scripts/visualize_hyper_shapes.py "$@"
