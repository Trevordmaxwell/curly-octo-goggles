import json
from pathlib import Path

import numpy as np
import pytest

from unified_energy.visualization.hyper_shapes import (
    Shape4D,
    ShapeCollection4D,
    build_demo_shapes,
    load_shapes_from_json,
)


def test_shape4d_validates_input() -> None:
    shape = Shape4D("valid", np.zeros((3, 4)))
    assert shape.points.shape == (3, 4)

    with pytest.raises(ValueError):
        Shape4D("bad_dims", np.zeros((2, 3)))

    with pytest.raises(ValueError):
        Shape4D("too_few", np.zeros((1, 4)))


def test_projection_respects_offsets() -> None:
    points = np.array([[0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 1.0, 1.0]])
    shape = Shape4D("unit", points)
    scene = ShapeCollection4D([shape])

    projected, axes, drop = scene.project("w", offsets={"x": 1.0, "z": -0.5})
    assert axes == ("x", "y", "z")
    assert np.allclose(projected["unit"][0], np.array([1.0, 0.0, -0.5]))
    assert np.allclose(drop["unit"], np.array([0.5, 1.0]))


def test_load_shapes_from_json(tmp_path: Path) -> None:
    data = [
        {"name": "custom", "points": [[0, 0, 0, 0], [1, 1, 1, 1]], "color": "#ff0000"},
        {"points": [[0, 0, 0, 0], [0, 0, 2, 0]]},
    ]
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(data))

    shapes = load_shapes_from_json(path)
    assert [shape.name for shape in shapes] == ["custom", "shape_1"]
    assert shapes[0].color == "#ff0000"
    assert shapes[1].points.shape == (2, 4)


def test_demo_shapes_cover_three_examples() -> None:
    shapes = build_demo_shapes(scale=1.5)
    assert len(shapes) == 3
    assert {shape.name for shape in shapes} == {"tesseract", "simplex", "hyper-cross"}
