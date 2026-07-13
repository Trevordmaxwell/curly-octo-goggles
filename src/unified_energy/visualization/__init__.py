"""Visualization helpers including the interactive 4D shape viewer."""

from .hyper_shapes import (
    HyperShapeViewer,
    Shape4D,
    build_demo_shapes,
    load_shapes_from_json,
    ShapeCollection4D,
)

__all__ = [
    "Shape4D",
    "ShapeCollection4D",
    "HyperShapeViewer",
    "build_demo_shapes",
    "load_shapes_from_json",
]
