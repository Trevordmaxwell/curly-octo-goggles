#!/usr/bin/env python
"""Launch the interactive 4D shape viewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_energy.visualization import HyperShapeViewer, build_demo_shapes, load_shapes_from_json


def _parse_axis_order(value: str | None) -> tuple[str, str, str, str]:
    if value is None:
        return ("x", "y", "z", "w")
    cleaned_value = value.replace(" ", "").lower()
    cleaned = tuple(cleaned_value)
    if len(cleaned) != 4 or set(cleaned) != {"x", "y", "z", "w"}:
        raise argparse.ArgumentTypeError("Axis order must be a permutation of 'xyzw'.")
    return cleaned[0], cleaned[1], cleaned[2], cleaned[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes-file",
        type=Path,
        help="Path to a JSON file containing shape definitions. "
        "If omitted a set of demo shapes is used.",
    )
    parser.add_argument(
        "--drop-axis",
        choices=("x", "y", "z", "w"),
        default="w",
        help="Axis that will be hidden from the 3D projection.",
    )
    parser.add_argument(
        "--axis-order",
        type=str,
        help="Permutation of 'xyzw' describing how axes are labeled.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the demo shapes.",
    )
    parser.add_argument(
        "--offset-limit",
        type=float,
        help="Max absolute translation allowed by the sliders.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    axis_order = _parse_axis_order(args.axis_order)
    if args.shapes_file:
        shapes = load_shapes_from_json(args.shapes_file)
    else:
        shapes = build_demo_shapes(scale=args.scale)

    viewer = HyperShapeViewer(
        shapes,
        drop_axis=args.drop_axis,
        axis_labels=axis_order,
        offset_limit=args.offset_limit,
    )
    viewer.show()


if __name__ == "__main__":
    main()
