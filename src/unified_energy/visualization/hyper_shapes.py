"""Interactive utilities for visualizing simple collections of 4D shapes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import itertools
import json

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AxisName = str
DEFAULT_AXES: tuple[AxisName, AxisName, AxisName, AxisName] = ("x", "y", "z", "w")


@dataclass(frozen=True)
class Shape4D:
    """Container for a single 4D shape represented by its vertices."""

    name: str
    points: np.ndarray
    color: str = "C0"

    def __post_init__(self) -> None:
        arr = np.asarray(self.points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"{self.name}: expected points with shape (n, 4); got {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(f"{self.name}: expected at least two vertices")
        object.__setattr__(self, "points", arr)


class ShapeCollection4D:
    """Utility wrapper that provides slicing, projections, and bounds."""

    def __init__(self, shapes: Sequence[Shape4D], axis_labels: Sequence[AxisName] = DEFAULT_AXES) -> None:
        if not shapes:
            raise ValueError("ShapeCollection4D requires at least one shape.")
        if len(axis_labels) != 4:
            raise ValueError("axis_labels must contain exactly four entries.")
        if len(set(axis_labels)) != 4:
            raise ValueError("axis_labels must list unique axes.")
        self.shapes = list(shapes)
        self.axis_labels = tuple(axis_labels)
        self._axis_to_index = {axis: idx for idx, axis in enumerate(self.axis_labels)}

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mins = np.stack([shape.points.min(axis=0) for shape in self.shapes])
        maxs = np.stack([shape.points.max(axis=0) for shape in self.shapes])
        return mins.min(axis=0), maxs.max(axis=0)

    def _offset_vector(self, offsets: Mapping[AxisName, float] | None = None) -> np.ndarray:
        vector = np.zeros(4, dtype=float)
        if offsets:
            for axis, value in offsets.items():
                if axis not in self._axis_to_index:
                    raise ValueError(f"Unknown axis {axis!r}")
                vector[self._axis_to_index[axis]] = float(value)
        return vector

    def project(
        self, drop_axis: AxisName, offsets: Mapping[AxisName, float] | None = None
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[AxisName, AxisName, AxisName],
        dict[str, np.ndarray],
    ]:
        if drop_axis not in self._axis_to_index:
            raise ValueError(f"Unknown axis {drop_axis!r}")
        keep_axes = tuple(axis for axis in self.axis_labels if axis != drop_axis)
        index = [self._axis_to_index[axis] for axis in keep_axes]
        drop_index = self._axis_to_index[drop_axis]
        offset_vector = self._offset_vector(offsets)

        projected: dict[str, np.ndarray] = {}
        drop_values: dict[str, np.ndarray] = {}
        for shape in self.shapes:
            transformed = shape.points + offset_vector
            projected[shape.name] = transformed[:, index]
            drop_values[shape.name] = transformed[:, drop_index]
        return projected, keep_axes, drop_values

    def estimate_offset_limit(self) -> float:
        mins, maxs = self.bounds()
        span = float(np.max(maxs - mins))
        return span if span > 0 else 1.0

    def padded_axis_limits(self, axis: AxisName, padding: float = 0.0) -> tuple[float, float]:
        if axis not in self._axis_to_index:
            raise ValueError(f"Unknown axis {axis!r}")
        mins, maxs = self.bounds()
        idx = self._axis_to_index[axis]
        base_min, base_max = float(mins[idx]), float(maxs[idx])
        if base_min == base_max:
            base_min -= 1.0
            base_max += 1.0
        return base_min - padding, base_max + padding


class HyperShapeViewer:
    """Matplotlib-based viewer with sliders for moving shapes along each axis."""

    def __init__(
        self,
        shapes: Sequence[Shape4D],
        drop_axis: AxisName = "w",
        axis_labels: Sequence[AxisName] = DEFAULT_AXES,
        offset_limit: float | None = None,
    ) -> None:
        self.scene = ShapeCollection4D(shapes, axis_labels=axis_labels)
        if drop_axis not in self.scene.axis_labels:
            raise ValueError(f"drop_axis must be one of {self.scene.axis_labels}")
        self.drop_axis = drop_axis
        self.offsets = {axis: 0.0 for axis in self.scene.axis_labels}
        self.offset_limit = offset_limit or self.scene.estimate_offset_limit()

        self._plt = None
        self._Slider = None
        self._RadioButtons = None
        self._Normalize = None
        self._ScalarMappable = None
        self._cmap = None
        self._plane_faces: list[Poly3DCollection] = []
        self._axis_ranges_cache: dict[AxisName, tuple[float, float]] = {}
        self._fig = None
        self._ax = None
        self._sliders = {}
        self._artists: dict[str, dict[str, Any]] = {}
        self._axis_selector = None
        self._colorbar = None
        self._mappable = None
        self._color_norm = None
        self._info_text = None

    def _load_matplotlib(self) -> None:
        if self._plt is not None:
            return
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider
        from matplotlib import cm
        from matplotlib.colors import Normalize

        self._plt = plt
        self._Slider = Slider
        self._RadioButtons = RadioButtons
        self._Normalize = Normalize
        self._ScalarMappable = cm.ScalarMappable
        self._cmap = cm.get_cmap("viridis")

    def _setup_widgets(self) -> None:
        self._load_matplotlib()
        assert (
            self._plt is not None
            and self._Slider is not None
            and self._RadioButtons is not None
            and self._Normalize is not None
            and self._ScalarMappable is not None
            and self._cmap is not None
        )

        self._fig = self._plt.figure(figsize=(9, 7))
        self._ax = self._fig.add_axes([0.05, 0.32, 0.9, 0.65], projection="3d")
        slider_height = 0.035
        slider_spacing = 0.01
        slider_left = 0.1
        slider_width = 0.65
        bottom = 0.02
        self._sliders = {}
        for idx, axis in enumerate(self.scene.axis_labels):
            slider_ax = self._fig.add_axes(
                [
                    slider_left,
                    bottom + idx * (slider_height + slider_spacing),
                    slider_width,
                    slider_height,
                ]
            )
            slider = self._Slider(
                slider_ax,
                f"{axis.upper()} offset",
                -self.offset_limit,
                self.offset_limit,
                valinit=0.0,
            )
            slider.on_changed(self._handle_slider_change)
            self._sliders[axis] = slider

        radio_ax = self._fig.add_axes([0.8, bottom, 0.15, 0.18])
        self._axis_selector = self._RadioButtons(
            radio_ax,
            [axis.upper() for axis in self.scene.axis_labels],
            active=self.scene.axis_labels.index(self.drop_axis),
        )
        self._axis_selector.on_clicked(self._handle_axis_change)
        self._info_text = self._fig.text(0.5, 0.27, "", ha="center", va="center", fontsize=9)

    def _init_plot(self) -> None:
        assert self._ax is not None
        projected, axes, drop_values = self.scene.project(self.drop_axis, self.offsets)
        self._configure_axes(axes)
        self._refresh_color_mapping()
        assert self._color_norm is not None

        self._artists = {}
        for idx, shape in enumerate(self.scene.shapes):
            coords = projected[shape.name]
            scatter = self._ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=drop_values[shape.name],
                cmap=self._cmap,
                vmin=self._color_norm.vmin,
                vmax=self._color_norm.vmax,
                s=55,
                edgecolors="black",
                linewidths=0.5,
                depthshade=False,
                alpha=0.9,
            )
            centroid = coords.mean(axis=0)
            label = self._ax.text(
                centroid[0],
                centroid[1],
                centroid[2],
                shape.name,
                fontsize=8,
                ha="center",
                va="center",
                color="black",
            )
            self._artists[shape.name] = {"scatter": scatter, "label": label}
        self._ax.legend(loc="upper right")
        self._update_info_text()

    def _handle_slider_change(self, _value: float) -> None:
        for axis, slider in self._sliders.items():
            self.offsets[axis] = float(slider.val)
        self._update_scene(reset_axes=False)

    def _handle_axis_change(self, label: str) -> None:
        axis = label.lower()
        if axis == self.drop_axis:
            return
        self.drop_axis = axis
        self._update_scene(reset_axes=True)

    def _update_scene(self, reset_axes: bool) -> None:
        assert self._fig is not None and self._ax is not None
        projected, axes, drop_values = self.scene.project(self.drop_axis, self.offsets)
        if reset_axes:
            self._configure_axes(axes)
        self._refresh_color_mapping()
        assert self._color_norm is not None

        for shape_name, artists in self._artists.items():
            scatter = artists["scatter"]
            label = artists["label"]
            coords = projected[shape_name]
            scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
            scatter.set_array(drop_values[shape_name])
            scatter.set_clim(self._color_norm.vmin, self._color_norm.vmax)
            centroid = coords.mean(axis=0)
            label.set_position((centroid[0], centroid[1]))
            label.set_3d_properties(centroid[2])

        self._fig.canvas.draw_idle()
        self._update_info_text()

    def show(self) -> None:
        """Launch the matplotlib window."""

        if self._fig is None:
            self._setup_widgets()
            self._init_plot()
        assert self._plt is not None
        self._plt.show()

    def _configure_axes(self, axes: tuple[AxisName, AxisName, AxisName]) -> None:
        assert self._ax is not None
        self._ax.set_xlabel(axes[0].upper())
        self._ax.set_ylabel(axes[1].upper())
        self._ax.set_zlabel(axes[2].upper())
        self._ax.set_title("Interactive 4D shape viewer")
        self._ax.set_box_aspect((1, 1, 1))
        mins, maxs = self.scene.bounds()
        padding = self.offset_limit * 1.1
        for setter, axis in zip(
            (self._ax.set_xlim, self._ax.set_ylim, self._ax.set_zlim), axes, strict=True
        ):
            idx = self.scene.axis_labels.index(axis)
            axis_min = float(mins[idx]) - padding
            axis_max = float(maxs[idx]) + padding
            if axis_min == axis_max:
                axis_min -= 1.0
                axis_max += 1.0
            setter((axis_min, axis_max))
            self._axis_ranges_cache[axis] = (axis_min, axis_max)

        self._ax.view_init(elev=24, azim=35)
        self._ax.set_facecolor("#fdfdfd")
        if hasattr(self._ax.xaxis, "pane"):
            for pane in (self._ax.xaxis.pane, self._ax.yaxis.pane, self._ax.zaxis.pane):
                pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
                pane.set_edgecolor("#dddddd")
        grid_color = "#cccccc"
        self._ax.xaxis._axinfo["grid"]["color"] = grid_color
        self._ax.yaxis._axinfo["grid"]["color"] = grid_color
        self._ax.zaxis._axinfo["grid"]["color"] = grid_color
        self._update_planes(axes)

    def _refresh_color_mapping(self) -> None:
        assert (
            self._Normalize is not None
            and self._ScalarMappable is not None
            and self._cmap is not None
            and self._ax is not None
        )

        padding = self.offset_limit
        vmin, vmax = self.scene.padded_axis_limits(self.drop_axis, padding)
        self._color_norm = self._Normalize(vmin=vmin, vmax=vmax)
        self._mappable = self._ScalarMappable(norm=self._color_norm, cmap=self._cmap)
        if self._colorbar is None:
            self._colorbar = self._plt.colorbar(self._mappable, ax=self._ax, pad=0.11, shrink=0.8)
        else:
            self._colorbar.update_normal(self._mappable)
        self._colorbar.set_label(f"{self.drop_axis.upper()} axis (color)")

    def _update_info_text(self) -> None:
        if self._info_text is None:
            return
        self._info_text.set_text(
            f"Colors encode the {self.drop_axis.upper()} axis (currently hidden). "
            "Adjust sliders to translate shapes; rotate the plot with the mouse."
        )

    def _update_planes(self, axes: tuple[AxisName, AxisName, AxisName]) -> None:
        assert self._ax is not None
        for face in self._plane_faces:
            face.remove()
        self._plane_faces = []

        try:
            x_range = self._axis_ranges_cache[axes[0]]
            y_range = self._axis_ranges_cache[axes[1]]
            z_range = self._axis_ranges_cache[axes[2]]
        except KeyError:
            return

        x0, x1 = x_range
        y0, y1 = y_range
        z0, z1 = z_range

        faces = [
            # Bottom/top Z planes
            [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
            ],
            [
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ],
            # Y planes
            [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y0, z1),
                (x0, y0, z1),
            ],
            [
                (x0, y1, z0),
                (x1, y1, z0),
                (x1, y1, z1),
                (x0, y1, z1),
            ],
            # X planes
            [
                (x0, y0, z0),
                (x0, y1, z0),
                (x0, y1, z1),
                (x0, y0, z1),
            ],
            [
                (x1, y0, z0),
                (x1, y1, z0),
                (x1, y1, z1),
                (x1, y0, z1),
            ],
        ]

        colors = ["#e1ecff", "#e1ecff", "#f2f2ff", "#f2f2ff", "#e8fff0", "#e8fff0"]
        for face, color in zip(faces, colors, strict=True):
            poly = Poly3DCollection(
                [face],
                facecolors=color,
                alpha=0.15,
                linewidths=0.3,
                edgecolors="#c0c0c0",
            )
            self._ax.add_collection3d(poly)
            self._plane_faces.append(poly)


def build_demo_shapes(scale: float = 1.0) -> list[Shape4D]:
    """Return a small set of canonical 4D shapes for quick exploration."""

    tesseract = _make_tesseract(edge=scale)
    simplex = _make_simplex(scale=scale * 0.7)
    cross = _make_cross(radius=scale * 1.4)
    return [tesseract, simplex, cross]


def _make_tesseract(edge: float) -> Shape4D:
    coords = np.array(list(itertools.product([-edge, edge], repeat=4)), dtype=float)
    return Shape4D("tesseract", coords, color="C0")


def _make_simplex(scale: float) -> Shape4D:
    origin = np.zeros((1, 4), dtype=float)
    basis = np.eye(4, dtype=float) * scale
    coords = np.concatenate([origin, basis], axis=0)
    return Shape4D("simplex", coords, color="C1")


def _make_cross(radius: float) -> Shape4D:
    coords = []
    for axis_index in range(4):
        basis = np.zeros(4)
        basis[axis_index] = radius
        coords.append(basis.copy())
        coords.append(-basis)
    return Shape4D("hyper-cross", np.array(coords), color="C2")


def load_shapes_from_json(path: str | Path) -> list[Shape4D]:
    """Load user-provided shape definitions from a JSON file."""

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "shapes" in payload:
        entries = payload["shapes"]
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = [payload]

    shapes: list[Shape4D] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or "points" not in entry:
            raise ValueError("Each shape entry must be an object with a 'points' field.")
        name = entry.get("name", f"shape_{idx}")
        color = entry.get("color", f"C{idx % 10}")
        shapes.append(Shape4D(name=name, points=np.asarray(entry["points"], dtype=float), color=color))

    if not shapes:
        raise ValueError(f"No valid shapes found in {path}.")
    return shapes


__all__ = [
    "Shape4D",
    "ShapeCollection4D",
    "HyperShapeViewer",
    "build_demo_shapes",
    "load_shapes_from_json",
]
