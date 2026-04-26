from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch
from torch import nn

try:
    from qcnn import ClassicalCNN, ClassicalMLP, configure_matplotlib_pdf_fonts
    from qcnn.article_training import build_figure_2_model_spec
    from qcnn.model_spec import ModelSpec, instantiate_model
except ModuleNotFoundError:  # pragma: no cover - enables geometry-only local testing.
    ClassicalCNN = nn.Module
    ClassicalMLP = nn.Module
    ModelSpec = Any

    def configure_matplotlib_pdf_fonts(matplotlib_module: Any | None = None) -> None:
        return None

    def _missing_qcnn(*args, **kwargs):
        raise ModuleNotFoundError(
            "qcnn package is required for resolving Figure 2 models. "
            "Geometry/layout helpers can still be imported without it."
        )

    build_figure_2_model_spec = _missing_qcnn
    instantiate_model = _missing_qcnn

try:
    from prepare_translated_mnist_baselines_data import Figure2TrainingDefaults
except ModuleNotFoundError:
    try:
        from run.prepare_translated_mnist_baselines_data import Figure2TrainingDefaults
    except ModuleNotFoundError:  # pragma: no cover - enables geometry-only local testing.
        Figure2TrainingDefaults = Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_CLASSICAL_MODEL_KINDS: tuple[str, str] = ("classical_cnn", "classical_mlp")
DEFAULT_BASE_BLOCK_SIZE_INCHES = 4.0
DEFAULT_BASE_FONT_SIZE_POINTS = 10.0
DEFAULT_CONTENT_PADDING_INCHES = 0.060

_BLOCK_FACE_COLORS = {
    "input": "#f7fafc",
    "reshape": "#edf2f7",
    "conv": "#dbeafe",
    "pool": "#d1fae5",
    "activation": "#fef3c7",
    "regularization": "#fce7f3",
    "linear": "#dcfce7",
}
_BLOCK_EDGE_COLOR = "#334155"
_ARROW_COLOR = "#64748b"
_DEFAULT_MINIMUM_BLOCK_GAP = 0.020
_DEFAULT_BASE_FONT_SCALE = 1.00
_DEFAULT_TEXT_GAP = 0.006
_DEFAULT_TEXT_VERTICAL_PADDING = 0.002
_LAYOUT_SOLVER_EPSILON = 1e-6


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def scaled(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def scaled(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)


@dataclass(frozen=True)
class Bounds2D:
    left: float
    right: float
    bottom: float
    top: float

    @classmethod
    def from_points(cls, points: Sequence[Vec2]) -> "Bounds2D":
        if not points:
            raise ValueError("Bounds2D.from_points requires at least one point.")
        return cls(
            left=min(point.x for point in points),
            right=max(point.x for point in points),
            bottom=min(point.y for point in points),
            top=max(point.y for point in points),
        )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def shifted(self, *, dx: float = 0.0, dy: float = 0.0) -> "Bounds2D":
        return Bounds2D(
            left=self.left + dx,
            right=self.right + dx,
            bottom=self.bottom + dy,
            top=self.top + dy,
        )

    def expanded(self, *, dx: float = 0.0, dy: float = 0.0) -> "Bounds2D":
        return Bounds2D(
            left=self.left - dx,
            right=self.right + dx,
            bottom=self.bottom - dy,
            top=self.top + dy,
        )


@dataclass(frozen=True)
class BoxDimensions:
    width: float
    height: float
    depth: float

    def scaled(self, value: float) -> "BoxDimensions":
        return BoxDimensions(
            width=self.width * value,
            height=self.height * value,
            depth=self.depth * value,
        )


@dataclass(frozen=True)
class ObliqueProjection:
    angle_degrees: float = 45.0
    foreshortening: float = 0.50

    @property
    def depth_vector(self) -> Vec2:
        radians = math.radians(self.angle_degrees)
        return Vec2(
            self.foreshortening * math.cos(radians),
            self.foreshortening * math.sin(radians),
        )

    def project(self, point: Vec3) -> Vec2:
        depth_vector = self.depth_vector
        return Vec2(
            point.x + depth_vector.x * point.z,
            point.y + depth_vector.y * point.z,
        )


@dataclass(frozen=True)
class Box3D:
    origin: Vec3
    size: BoxDimensions

    _VERTEX_OFFSETS: Mapping[str, Vec3] = field(
        default_factory=lambda: {
            "front_bottom_left": Vec3(0.0, 0.0, 0.0),
            "front_bottom_right": Vec3(1.0, 0.0, 0.0),
            "front_top_left": Vec3(0.0, 1.0, 0.0),
            "front_top_right": Vec3(1.0, 1.0, 0.0),
            "back_bottom_left": Vec3(0.0, 0.0, 1.0),
            "back_bottom_right": Vec3(1.0, 0.0, 1.0),
            "back_top_left": Vec3(0.0, 1.0, 1.0),
            "back_top_right": Vec3(1.0, 1.0, 1.0),
        },
        init=False,
        repr=False,
        compare=False,
    )

    _FACE_VERTEX_NAMES: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "front": (
                "front_bottom_left",
                "front_bottom_right",
                "front_top_right",
                "front_top_left",
            ),
            "top": (
                "front_top_left",
                "front_top_right",
                "back_top_right",
                "back_top_left",
            ),
            "right": (
                "front_bottom_right",
                "back_bottom_right",
                "back_top_right",
                "front_top_right",
            ),
        },
        init=False,
        repr=False,
        compare=False,
    )

    _EDGE_VERTEX_NAMES: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "front_bottom": ("front_bottom_left", "front_bottom_right"),
            "front_top": ("front_top_left", "front_top_right"),
            "right_front_vertical": ("front_bottom_right", "front_top_right"),
            "lower_right_depth": ("front_bottom_right", "back_bottom_right"),
            "upper_right_depth": ("front_top_right", "back_top_right"),
        },
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_projected_center(
        cls,
        *,
        projected_center: Vec2,
        size: BoxDimensions,
        projection: ObliqueProjection,
    ) -> "Box3D":
        depth_vector = projection.depth_vector
        return cls(
            origin=Vec3(
                projected_center.x - 0.5 * size.width - 0.5 * depth_vector.x * size.depth,
                projected_center.y - 0.5 * size.height - 0.5 * depth_vector.y * size.depth,
                0.0,
            ),
            size=size,
        )

    @property
    def center(self) -> Vec3:
        return Vec3(
            self.origin.x + 0.5 * self.size.width,
            self.origin.y + 0.5 * self.size.height,
            self.origin.z + 0.5 * self.size.depth,
        )

    def vertex(self, name: str) -> Vec3:
        if name not in self._VERTEX_OFFSETS:
            raise KeyError(f"Unknown box vertex {name!r}.")
        offset = self._VERTEX_OFFSETS[name]
        return Vec3(
            self.origin.x + offset.x * self.size.width,
            self.origin.y + offset.y * self.size.height,
            self.origin.z + offset.z * self.size.depth,
        )

    def face_vertices(self, face_name: str) -> tuple[Vec3, ...]:
        if face_name not in self._FACE_VERTEX_NAMES:
            raise KeyError(f"Unknown box face {face_name!r}.")
        return tuple(self.vertex(vertex_name) for vertex_name in self._FACE_VERTEX_NAMES[face_name])

    def face_center(self, face_name: str) -> Vec3:
        vertices = self.face_vertices(face_name)
        return Vec3(
            sum(vertex.x for vertex in vertices) / len(vertices),
            sum(vertex.y for vertex in vertices) / len(vertices),
            sum(vertex.z for vertex in vertices) / len(vertices),
        )

    def edge_vertices(self, edge_name: str) -> tuple[Vec3, Vec3]:
        if edge_name not in self._EDGE_VERTEX_NAMES:
            raise KeyError(f"Unknown box edge {edge_name!r}.")
        start_name, end_name = self._EDGE_VERTEX_NAMES[edge_name]
        return self.vertex(start_name), self.vertex(end_name)

    def edge_midpoint(self, edge_name: str) -> Vec3:
        start, end = self.edge_vertices(edge_name)
        return Vec3(
            0.5 * (start.x + end.x),
            0.5 * (start.y + end.y),
            0.5 * (start.z + end.z),
        )

    def project(self, projection: ObliqueProjection) -> "ProjectedBox":
        projected_vertices = {
            name: projection.project(self.vertex(name))
            for name in self._VERTEX_OFFSETS
        }
        return ProjectedBox(
            source=self,
            projection=projection,
            vertices=projected_vertices,
        )


@dataclass(frozen=True)
class ProjectedBox:
    source: Box3D
    projection: ObliqueProjection
    vertices: Mapping[str, Vec2]

    _FACE_VERTEX_NAMES: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "front": (
                "front_bottom_left",
                "front_bottom_right",
                "front_top_right",
                "front_top_left",
            ),
            "top": (
                "front_top_left",
                "front_top_right",
                "back_top_right",
                "back_top_left",
            ),
            "right": (
                "front_bottom_right",
                "back_bottom_right",
                "back_top_right",
                "front_top_right",
            ),
        },
        init=False,
        repr=False,
        compare=False,
    )
    _EDGE_VERTEX_NAMES: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "front_bottom": ("front_bottom_left", "front_bottom_right"),
            "front_top": ("front_top_left", "front_top_right"),
            "right_front_vertical": ("front_bottom_right", "front_top_right"),
            "lower_right_depth": ("front_bottom_right", "back_bottom_right"),
            "upper_right_depth": ("front_top_right", "back_top_right"),
        },
        init=False,
        repr=False,
        compare=False,
    )

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> "ProjectedBox":
        return ProjectedBox(
            source=self.source,
            projection=self.projection,
            vertices={
                name: Vec2(point.x + dx, point.y + dy)
                for name, point in self.vertices.items()
            },
        )

    @property
    def projected_center(self) -> Vec2:
        vertices = tuple(self.vertices.values())
        return Vec2(
            sum(vertex.x for vertex in vertices) / len(vertices),
            sum(vertex.y for vertex in vertices) / len(vertices),
        )

    @property
    def bounds(self) -> Bounds2D:
        return Bounds2D.from_points(tuple(self.vertices.values()))

    def face_polygon(self, face_name: str) -> tuple[Vec2, ...]:
        if face_name not in self._FACE_VERTEX_NAMES:
            raise KeyError(f"Unknown box face {face_name!r}.")
        return tuple(self.vertices[vertex_name] for vertex_name in self._FACE_VERTEX_NAMES[face_name])

    def face_center(self, face_name: str) -> Vec2:
        polygon = self.face_polygon(face_name)
        return Vec2(
            sum(vertex.x for vertex in polygon) / len(polygon),
            sum(vertex.y for vertex in polygon) / len(polygon),
        )

    def edge_points(self, edge_name: str) -> tuple[Vec2, Vec2]:
        if edge_name not in self._EDGE_VERTEX_NAMES:
            raise KeyError(f"Unknown box edge {edge_name!r}.")
        start_name, end_name = self._EDGE_VERTEX_NAMES[edge_name]
        return self.vertices[start_name], self.vertices[end_name]

    def edge_midpoint(self, edge_name: str) -> Vec2:
        start, end = self.edge_points(edge_name)
        return Vec2(
            0.5 * (start.x + end.x),
            0.5 * (start.y + end.y),
        )

    def top_caption_anchor(self) -> Vec2:
        top_polygon = self.face_polygon("top")
        top_edges = tuple(
            (start, end)
            for start, end in zip(top_polygon, top_polygon[1:] + top_polygon[:1], strict=False)
        )
        start, end = max(
            top_edges,
            key=lambda edge: 0.5 * (edge[0].y + edge[1].y),
        )
        return Vec2(
            0.5 * (start.x + end.x),
            self.bounds.top,
        )

    def bottom_caption_anchor(self) -> Vec2:
        front_center = self.face_center("front")
        return Vec2(
            front_center.x,
            self.bounds.bottom,
        )

    def anchor_point(self, anchor: "AnchorRef") -> Vec2:
        if anchor.target == "box_center":
            return self.projected_center
        if anchor.target == "face_center":
            return self.face_center(anchor.name)
        if anchor.target == "edge_midpoint":
            return self.edge_midpoint(anchor.name)
        if anchor.target == "top_caption":
            return self.top_caption_anchor()
        if anchor.target == "bottom_caption":
            return self.bottom_caption_anchor()
        if anchor.target == "vertex":
            return self.vertices[anchor.name]
        raise ValueError(f"Unsupported anchor target {anchor.target!r}.")

    def edge_angle_degrees(self, edge_name: str, ax: "Axes") -> float:
        start, end = self.edge_points(edge_name)
        display_start = ax.transData.transform((start.x, start.y))
        display_end = ax.transData.transform((end.x, end.y))
        return math.degrees(
            math.atan2(
                display_end[1] - display_start[1],
                display_end[0] - display_start[0],
            )
        )


@dataclass(frozen=True)
class AnchorRef:
    target: str
    name: str = ""


@dataclass(frozen=True)
class BlockLabel:
    text: str
    anchor: AnchorRef
    offset: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    fontsize: float = 6.0
    ha: str = "center"
    va: str = "center"
    color: str = "#0f172a"
    rotation_degrees: float = 0.0
    rotation_edge: str | None = None
    rotation_mode: str = "fixed"
    role: str = "extra_label"
    zorder: float = 5.5


@dataclass(frozen=True)
class Size2D:
    width: float
    height: float


@dataclass(frozen=True)
class PlacedShape:
    kind: str
    vertices: tuple[Vec2, ...]
    name: str = ""

    @property
    def bounds(self) -> Bounds2D:
        return Bounds2D.from_points(self.vertices)

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> "PlacedShape":
        return PlacedShape(
            kind=self.kind,
            vertices=tuple(Vec2(vertex.x + dx, vertex.y + dy) for vertex in self.vertices),
            name=self.name,
        )


@dataclass(frozen=True)
class PlacedLabel:
    spec: BlockLabel
    position: Vec2
    rotation_degrees: float
    size: Size2D
    shape: PlacedShape

    @property
    def bounds(self) -> Bounds2D:
        return self.shape.bounds

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> "PlacedLabel":
        return PlacedLabel(
            spec=self.spec,
            position=Vec2(self.position.x + dx, self.position.y + dy),
            rotation_degrees=self.rotation_degrees,
            size=self.size,
            shape=self.shape.shifted(dx=dx, dy=dy),
        )


@dataclass(frozen=True)
class DiagramBlock:
    kind: str
    title: str
    detail: str = ""
    caption: str = ""
    output_label: str = ""
    geometry: str = "plate"
    primary_size: float = 1.0
    secondary_size: float = 1.0
    explicit_box: BoxDimensions | None = None
    caption_position: str = "auto"
    extra_labels: tuple[BlockLabel, ...] = ()


@dataclass(frozen=True)
class BlockStyle:
    face_color: str
    edge_color: str = _BLOCK_EDGE_COLOR
    top_lighten: float = 0.28
    side_darken: float = 0.18
    shadow_offset: Vec2 = field(default_factory=lambda: Vec2(0.010, -0.010))
    shadow_depth_extra: Vec2 = field(default_factory=lambda: Vec2(0.015, 0.0))
    shadow_color: str = "#cbd5e1"
    shadow_alpha: float = 0.22
    linewidth: float = 1.0


@dataclass(frozen=True)
class DiagramTheme:
    block_styles: Mapping[str, BlockStyle]
    default_block_style: BlockStyle = field(
        default_factory=lambda: BlockStyle(face_color="#f8fafc")
    )
    arrow_color: str = _ARROW_COLOR
    baseline_color: str = "#e2e8f0"

    def style_for(self, kind: str) -> BlockStyle:
        return self.block_styles.get(kind, self.default_block_style)


@dataclass(frozen=True)
class GeometryScaleConfig:
    feature_channel_width_range: tuple[float, float] = (0.022, 0.070)
    feature_spatial_extent_range: tuple[float, float] = (0.095, 0.240)
    vector_width_range: tuple[float, float] = (0.030, 0.082)
    vector_height_range: tuple[float, float] = (0.090, 0.140)
    vector_depth_fraction: float = 0.34
    plate_sizes: Mapping[str, BoxDimensions] = field(
        default_factory=lambda: {
            "reshape": BoxDimensions(width=0.017, height=0.090, depth=0.020),
            "regularization": BoxDimensions(width=0.016, height=0.082, depth=0.020),
            "activation": BoxDimensions(width=0.015, height=0.082, depth=0.020),
            "default": BoxDimensions(width=0.015, height=0.082, depth=0.020),
        }
    )


@dataclass(frozen=True)
class DiagramLayoutConfig:
    projection: ObliqueProjection = field(default_factory=ObliqueProjection)
    minimum_block_gap: float = _DEFAULT_MINIMUM_BLOCK_GAP
    text_gap_x: float = _DEFAULT_TEXT_GAP
    text_gap_y: float = _DEFAULT_TEXT_VERTICAL_PADDING


@dataclass(frozen=True)
class DiagramScaleStatistics:
    feature_spatial_values: tuple[float, ...]
    feature_channel_values: tuple[float, ...]
    vector_values: tuple[float, ...]

    @classmethod
    def from_blocks(cls, blocks: Sequence[DiagramBlock]) -> "DiagramScaleStatistics":
        return cls(
            feature_spatial_values=tuple(
                max(block.primary_size, 1.0)
                for block in blocks
                if block.geometry == "feature_map"
            ),
            feature_channel_values=tuple(
                max(block.secondary_size, 1.0)
                for block in blocks
                if block.geometry == "feature_map"
            ),
            vector_values=tuple(
                max(block.primary_size, 1.0)
                for block in blocks
                if block.geometry == "vector"
            ),
        )

    @property
    def max_feature_spatial(self) -> float:
        return max(self.feature_spatial_values, default=32.0)

    @property
    def max_feature_channels(self) -> float:
        return max(self.feature_channel_values, default=64.0)

    @property
    def min_vector(self) -> float:
        return min(self.vector_values, default=10.0)

    @property
    def max_vector(self) -> float:
        return max(self.vector_values, default=128.0)


@dataclass(frozen=True)
class BlockTemplate:
    block: DiagramBlock
    projected_box: ProjectedBox
    labels: tuple[PlacedLabel, ...]
    shapes: tuple[PlacedShape, ...]
    reference_center_x: float

    @property
    def bounds(self) -> Bounds2D:
        all_bounds = [shape.bounds for shape in self.shapes]
        return Bounds2D(
            left=min(bounds.left for bounds in all_bounds),
            right=max(bounds.right for bounds in all_bounds),
            bottom=min(bounds.bottom for bounds in all_bounds),
            top=max(bounds.top for bounds in all_bounds),
        )

    @property
    def body_shape(self) -> PlacedShape:
        return self.shapes[0]

    def shifted(self, new_center_x: float) -> "PlacedBlock":
        dx = new_center_x - self.reference_center_x
        return PlacedBlock(
            block=self.block,
            projected_box=self.projected_box.shifted(dx),
            labels=tuple(label.shifted(dx) for label in self.labels),
            shapes=tuple(shape.shifted(dx=dx) for shape in self.shapes),
        )


@dataclass(frozen=True)
class PlacedBlock:
    block: DiagramBlock
    projected_box: ProjectedBox
    labels: tuple[PlacedLabel, ...]
    shapes: tuple[PlacedShape, ...]

    @property
    def bounds(self) -> Bounds2D:
        all_bounds = [shape.bounds for shape in self.shapes]
        return Bounds2D(
            left=min(bounds.left for bounds in all_bounds),
            right=max(bounds.right for bounds in all_bounds),
            bottom=min(bounds.bottom for bounds in all_bounds),
            top=max(bounds.top for bounds in all_bounds),
        )

    @property
    def body_shape(self) -> PlacedShape:
        return self.shapes[0]

    @property
    def connection_y(self) -> float:
        return self.projected_box.projected_center.y


@dataclass(frozen=True)
class RenderedDiagram:
    blocks: tuple[PlacedBlock, ...]
    font_scale: float
    box_scale: float

    @property
    def bounds(self) -> Bounds2D:
        return Bounds2D(
            left=min(block.bounds.left for block in self.blocks),
            right=max(block.bounds.right for block in self.blocks),
            bottom=min(block.bounds.bottom for block in self.blocks),
            top=max(block.bounds.top for block in self.blocks),
        )


@dataclass(frozen=True)
class _PreparedFigureS1Panel:
    rendered: RenderedDiagram
    width: float
    height: float
    figsize: tuple[float, float]


def _rectangle_vertices_from_bounds(bounds: Bounds2D) -> tuple[Vec2, ...]:
    return (
        Vec2(bounds.left, bounds.bottom),
        Vec2(bounds.right, bounds.bottom),
        Vec2(bounds.right, bounds.top),
        Vec2(bounds.left, bounds.top),
    )


def _rotate_point(point: Vec2, angle_degrees: float) -> Vec2:
    radians = math.radians(angle_degrees)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    return Vec2(
        point.x * cos_angle - point.y * sin_angle,
        point.x * sin_angle + point.y * cos_angle,
    )


def _aligned_rectangle_local_vertices(
    size: Size2D,
    *,
    ha: str,
    va: str,
) -> tuple[Vec2, ...]:
    if ha == "left":
        left, right = 0.0, size.width
    elif ha == "center":
        left, right = -0.5 * size.width, 0.5 * size.width
    elif ha == "right":
        left, right = -size.width, 0.0
    else:
        raise ValueError(f"Unsupported horizontal alignment {ha!r}.")

    if va in {"baseline", "bottom"}:
        bottom, top = 0.0, size.height
    elif va in {"center", "center_baseline"}:
        bottom, top = -0.5 * size.height, 0.5 * size.height
    elif va == "top":
        bottom, top = -size.height, 0.0
    else:
        raise ValueError(f"Unsupported vertical alignment {va!r}.")

    return (
        Vec2(left, bottom),
        Vec2(right, bottom),
        Vec2(right, top),
        Vec2(left, top),
    )


def _text_shape_from_anchor(
    *,
    text: str,
    anchor: Vec2,
    size: Size2D,
    ha: str,
    va: str,
    rotation_degrees: float,
    role: str,
) -> PlacedShape:
    local_vertices = _aligned_rectangle_local_vertices(size, ha=ha, va=va)
    return PlacedShape(
        kind=role,
        vertices=tuple(
            anchor + _rotate_point(local_vertex, rotation_degrees)
            for local_vertex in local_vertices
        ),
        name=text,
    )


def _block_body_shape(projected_box: ProjectedBox) -> PlacedShape:
    return PlacedShape(
        kind="block_body",
        vertices=_rectangle_vertices_from_bounds(projected_box.bounds),
        name="block_body",
    )


def _polygon_axes(vertices: Sequence[Vec2]) -> tuple[Vec2, ...]:
    axes: list[Vec2] = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        edge = end - start
        if abs(edge.x) <= _LAYOUT_SOLVER_EPSILON and abs(edge.y) <= _LAYOUT_SOLVER_EPSILON:
            continue
        axes.append(Vec2(-edge.y, edge.x))
    return tuple(axes)


def _project_vertices_onto_axis(vertices: Sequence[Vec2], axis: Vec2) -> tuple[float, float]:
    projections = tuple(vertex.x * axis.x + vertex.y * axis.y for vertex in vertices)
    return min(projections), max(projections)


def _dx_interval_for_axis_overlap(
    first: PlacedShape,
    second: PlacedShape,
    axis: Vec2,
) -> tuple[float, float] | None:
    first_min, first_max = _project_vertices_onto_axis(first.vertices, axis)
    second_min, second_max = _project_vertices_onto_axis(second.vertices, axis)
    axis_dx = axis.x

    if abs(axis_dx) <= _LAYOUT_SOLVER_EPSILON:
        if first_max < second_min - _LAYOUT_SOLVER_EPSILON:
            return None
        if second_max < first_min - _LAYOUT_SOLVER_EPSILON:
            return None
        return -math.inf, math.inf

    left = (first_min - second_max) / axis_dx
    right = (first_max - second_min) / axis_dx
    return min(left, right), max(left, right)


def _horizontal_overlap_interval_for_shapes(
    first: PlacedShape,
    second: PlacedShape,
) -> tuple[float, float] | None:
    lower_bound = -math.inf
    upper_bound = math.inf
    for axis in _polygon_axes(first.vertices) + _polygon_axes(second.vertices):
        interval = _dx_interval_for_axis_overlap(first, second, axis)
        if interval is None:
            return None
        lower_bound = max(lower_bound, interval[0])
        upper_bound = min(upper_bound, interval[1])
        if lower_bound > upper_bound + _LAYOUT_SOLVER_EPSILON:
            return None
    return lower_bound, upper_bound


def _shapes_overlap_at_current_offset(first: PlacedShape, second: PlacedShape) -> bool:
    interval = _horizontal_overlap_interval_for_shapes(first, second)
    if interval is None:
        return False
    return interval[0] < -_LAYOUT_SOLVER_EPSILON and interval[1] > _LAYOUT_SOLVER_EPSILON


def _required_horizontal_separation_between_shapes(
    first: PlacedShape,
    second: PlacedShape,
    *,
    minimum_gap: float = 0.0,
    epsilon: float = _LAYOUT_SOLVER_EPSILON,
) -> float:
    interval = _horizontal_overlap_interval_for_shapes(first, second)
    if interval is None:
        return 0.0
    _, upper_bound = interval
    adjusted_upper_bound = upper_bound + minimum_gap
    if adjusted_upper_bound < 0.0 - epsilon:
        return 0.0
    return max(0.0, adjusted_upper_bound) + epsilon


class ArchitectureDiagramRenderer:
    """Render architecture diagrams in physical inches from explicit geometry."""

    def __init__(
        self,
        *,
        theme: DiagramTheme | None = None,
        geometry_config: GeometryScaleConfig | None = None,
        layout_config: DiagramLayoutConfig | None = None,
        base_block_size_inches: float = DEFAULT_BASE_BLOCK_SIZE_INCHES,
        base_font_size_points: float = DEFAULT_BASE_FONT_SIZE_POINTS,
    ) -> None:
        self.theme = theme or DiagramTheme(
            block_styles={
                kind: BlockStyle(face_color=color)
                for kind, color in _BLOCK_FACE_COLORS.items()
            }
        )
        self.geometry_config = geometry_config or GeometryScaleConfig()
        self.layout_config = layout_config or DiagramLayoutConfig()
        self.base_block_size_inches = base_block_size_inches
        self.base_font_size_points = base_font_size_points

    def _scale_length(self, value: float) -> float:
        return value * self.base_block_size_inches

    def _scale_offset(self, offset: Vec2) -> Vec2:
        return offset.scaled(self.base_block_size_inches)

    def render(
        self,
        ax: "Axes",
        blocks: Sequence[DiagramBlock],
        *,
        base_font_scale: float = _DEFAULT_BASE_FONT_SCALE,
    ) -> RenderedDiagram:
        if not blocks:
            raise ValueError("ArchitectureDiagramRenderer requires at least one block.")

        ax.figure.canvas.draw()
        statistics = DiagramScaleStatistics.from_blocks(blocks)
        templates = tuple(
            self._build_block_template(
                ax,
                block,
                statistics=statistics,
                box_scale=self.base_block_size_inches,
                font_scale=base_font_scale,
            )
            for block in blocks
        )
        return self._layout_templates(
            templates,
            box_scale=self.base_block_size_inches,
            font_scale=base_font_scale,
        )

    def draw(
        self,
        ax: "Axes",
        rendered: RenderedDiagram,
        *,
        transform=None,
    ) -> None:
        coordinate_transform = transform or ax.transData
        for current, nxt in zip(rendered.blocks, rendered.blocks[1:], strict=False):
            start, end = self._arrow_segment(current, nxt)
            ax.annotate(
                "",
                xy=(end.x, end.y),
                xycoords=coordinate_transform,
                xytext=(start.x, start.y),
                textcoords=coordinate_transform,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": self.theme.arrow_color,
                    "linewidth": 1.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                    "mutation_scale": 8.5,
                    "connectionstyle": "arc3,rad=0.0",
                },
            )

        for placed_block in rendered.blocks:
            self._draw_block(ax, placed_block, transform=coordinate_transform)
            for label in placed_block.labels:
                ax.text(
                    label.position.x,
                    label.position.y,
                    label.spec.text,
                    transform=coordinate_transform,
                    ha=label.spec.ha,
                    va=label.spec.va,
                    rotation=label.rotation_degrees,
                    rotation_mode="anchor",
                    fontsize=label.spec.fontsize,
                    color=label.spec.color,
                    zorder=label.spec.zorder,
                )

    def _layout_templates(
        self,
        templates: Sequence[BlockTemplate],
        *,
        box_scale: float,
        font_scale: float,
    ) -> RenderedDiagram:
        if not templates:
            raise ValueError("_layout_templates requires at least one template.")

        centers: list[float] = [0.0]
        for previous_template, current_template in zip(templates, templates[1:], strict=False):
            previous_center = centers[-1]
            separation = self._required_center_separation(previous_template, current_template)
            centers.append(previous_center + separation)

        placed_blocks = tuple(
            template.shifted(center_x)
            for template, center_x in zip(templates, centers, strict=True)
        )
        return RenderedDiagram(
            blocks=placed_blocks,
            font_scale=font_scale,
            box_scale=box_scale,
        )

    def _required_center_separation(
        self,
        previous_template: BlockTemplate,
        current_template: BlockTemplate,
    ) -> float:
        maximum_required = 0.0
        for previous_shape in previous_template.shapes:
            for current_shape in current_template.shapes:
                pair_minimum_gap = 0.0
                if (
                    previous_shape.kind != "block_body"
                    or current_shape.kind != "block_body"
                ):
                    pair_minimum_gap = self._scale_length(self.layout_config.text_gap_x)
                maximum_required = max(
                    maximum_required,
                    _required_horizontal_separation_between_shapes(
                        previous_shape,
                        current_shape,
                        minimum_gap=pair_minimum_gap,
                    ),
                )

        body_to_body_required = _required_horizontal_separation_between_shapes(
            previous_template.body_shape,
            current_template.body_shape,
        )
        return max(
            maximum_required,
            body_to_body_required + self._scale_length(self.layout_config.minimum_block_gap),
        )

    def _build_block_template(
        self,
        ax: "Axes",
        block: DiagramBlock,
        *,
        statistics: DiagramScaleStatistics,
        box_scale: float,
        font_scale: float,
    ) -> BlockTemplate:
        size = self._resolve_box_dimensions(block, statistics=statistics).scaled(box_scale)
        center = Vec2(0.0, 0.0)
        box = Box3D.from_projected_center(
            projected_center=center,
            size=size,
            projection=self.layout_config.projection,
        )
        projected_box = box.project(self.layout_config.projection)
        labels = tuple(
            self._measure_label(ax, projected_box, label_spec)
            for label_spec in self._default_labels(block, font_scale=font_scale)
        ) + tuple(
            self._measure_label(
                ax,
                projected_box,
                self._scale_extra_label(label_spec, font_scale=font_scale),
            )
            for label_spec in block.extra_labels
        )
        shapes = (_block_body_shape(projected_box),) + tuple(label.shape for label in labels)
        return BlockTemplate(
            block=block,
            projected_box=projected_box,
            labels=labels,
            shapes=shapes,
            reference_center_x=center.x,
        )

    def _default_labels(self, block: DiagramBlock, *, font_scale: float) -> tuple[BlockLabel, ...]:
        labels: list[BlockLabel] = []
        caption = block.caption or block.title
        if caption:
            labels.append(self._caption_label(block, caption, font_scale=font_scale))
        if block.output_label:
            labels.append(self._output_edge_label(block.output_label, font_scale=font_scale))
        return tuple(labels)

    def _caption_label(self, block: DiagramBlock, caption: str, *, font_scale: float) -> BlockLabel:
        label_fontsize = self.base_font_size_points * font_scale
        placement = block.caption_position
        if placement == "auto":
            placement = "below" if block.geometry == "plate" else "above"
        if placement == "below":
            return BlockLabel(
                text=caption,
                anchor=AnchorRef("bottom_caption"),
                offset=Vec2(0.0, -self._scale_length(self.layout_config.text_gap_y)),
                fontsize=label_fontsize,
                va="top",
                role="caption",
                zorder=5.0,
            )
        if placement == "above":
            return BlockLabel(
                text=caption,
                anchor=AnchorRef("top_caption"),
                offset=Vec2(0.0, self._scale_length(self.layout_config.text_gap_y)),
                fontsize=label_fontsize,
                va="bottom",
                role="caption",
                zorder=5.0,
            )
        raise ValueError(f"Unsupported caption placement {placement!r}.")

    def _output_edge_label(self, text: str, *, font_scale: float) -> BlockLabel:
        return BlockLabel(
            text=text,
            anchor=AnchorRef("edge_midpoint", "lower_right_depth"),
            offset=self._scale_offset(Vec2(0.014, -0.028)),
            fontsize=self.base_font_size_points * font_scale,
            rotation_edge="lower_right_depth",
            rotation_mode="edge",
            role="output_label",
            zorder=5.8,
        )

    def _scale_extra_label(self, label: BlockLabel, *, font_scale: float) -> BlockLabel:
        return BlockLabel(
            text=label.text,
            anchor=label.anchor,
            offset=self._scale_offset(label.offset),
            fontsize=label.fontsize * font_scale,
            ha=label.ha,
            va=label.va,
            color=label.color,
            rotation_degrees=label.rotation_degrees,
            rotation_edge=label.rotation_edge,
            rotation_mode=label.rotation_mode,
            role=label.role,
            zorder=label.zorder,
        )

    def _measure_label(
        self,
        ax: "Axes",
        projected_box: ProjectedBox,
        label_spec: BlockLabel,
    ) -> PlacedLabel:
        anchor_point = projected_box.anchor_point(label_spec.anchor)
        position = anchor_point + label_spec.offset
        rotation_degrees = label_spec.rotation_degrees
        if label_spec.rotation_mode == "edge":
            if label_spec.rotation_edge is None:
                raise ValueError("rotation_mode='edge' requires rotation_edge.")
            rotation_degrees = projected_box.edge_angle_degrees(label_spec.rotation_edge, ax)
        unrotated_bounds = _measure_text_bbox_data(
            ax,
            x=position.x,
            y=position.y,
            text=label_spec.text,
            fontsize=label_spec.fontsize,
            ha=label_spec.ha,
            va=label_spec.va,
            rotation=0.0,
        )
        size = Size2D(width=unrotated_bounds.width, height=unrotated_bounds.height)
        shape = _text_shape_from_anchor(
            text=label_spec.text,
            anchor=position,
            size=size,
            ha=label_spec.ha,
            va=label_spec.va,
            rotation_degrees=rotation_degrees,
            role=label_spec.role,
        )
        return PlacedLabel(
            spec=label_spec,
            position=position,
            rotation_degrees=rotation_degrees,
            size=size,
            shape=shape,
        )

    def _resolve_box_dimensions(
        self,
        block: DiagramBlock,
        *,
        statistics: DiagramScaleStatistics,
    ) -> BoxDimensions:
        if block.explicit_box is not None:
            return block.explicit_box

        geometry = block.geometry
        config = self.geometry_config
        if geometry == "feature_map":
            side_extent = _interpolate_log_scale(
                max(block.primary_size, 1.0),
                minimum_value=1.0,
                maximum_value=statistics.max_feature_spatial,
                minimum_output=config.feature_spatial_extent_range[0],
                maximum_output=config.feature_spatial_extent_range[1],
            )
            width = _interpolate_log_scale(
                max(block.secondary_size, 1.0),
                minimum_value=1.0,
                maximum_value=statistics.max_feature_channels,
                minimum_output=config.feature_channel_width_range[0],
                maximum_output=config.feature_channel_width_range[1],
            )
            return BoxDimensions(width=width, height=side_extent, depth=side_extent)
        if geometry == "vector":
            width = _interpolate_log_scale(
                max(block.primary_size, 1.0),
                minimum_value=statistics.min_vector,
                maximum_value=statistics.max_vector,
                minimum_output=config.vector_width_range[0],
                maximum_output=config.vector_width_range[1],
            )
            height = _interpolate_log_scale(
                max(block.primary_size, 1.0),
                minimum_value=statistics.min_vector,
                maximum_value=statistics.max_vector,
                minimum_output=config.vector_height_range[0],
                maximum_output=config.vector_height_range[1],
            )
            return BoxDimensions(width=width, height=height, depth=height * config.vector_depth_fraction)
        if geometry == "plate":
            return config.plate_sizes.get(block.kind, config.plate_sizes["default"])
        raise ValueError(f"Unsupported block geometry {geometry!r}.")

    def _arrow_segment(self, current: PlacedBlock, nxt: PlacedBlock) -> tuple[Vec2, Vec2]:
        start = current.projected_box.face_center("right")
        target = nxt.projected_box.projected_center

        next_front_left_bottom = nxt.projected_box.vertices["front_bottom_left"]
        next_front_left_top = nxt.projected_box.vertices["front_top_left"]
        left_edge_x = next_front_left_bottom.x
        left_edge_bottom = min(next_front_left_bottom.y, next_front_left_top.y)
        left_edge_top = max(next_front_left_bottom.y, next_front_left_top.y)

        if target.x <= start.x:
            end_y = target.y
        else:
            interpolation = (left_edge_x - start.x) / (target.x - start.x)
            interpolation = max(0.0, min(1.0, interpolation))
            end_y = start.y + interpolation * (target.y - start.y)
        end_y = max(left_edge_bottom, min(left_edge_top, end_y))
        return start, Vec2(left_edge_x, end_y)

    def _draw_block(self, ax: "Axes", placed_block: PlacedBlock, *, transform) -> None:
        from matplotlib.patches import Polygon

        projected_box = placed_block.projected_box
        style = self.theme.style_for(placed_block.block.kind)
        face_color = style.face_color
        top_color = _shade_hex_color(face_color, toward="white", weight=style.top_lighten)
        side_color = _shade_hex_color(face_color, toward="black", weight=style.side_darken)

        shadow_bounds = projected_box.bounds
        shadow_offset = self._scale_offset(style.shadow_offset)
        shadow_depth_extra = self._scale_offset(style.shadow_depth_extra)
        shadow_polygon = (
            Vec2(shadow_bounds.left, shadow_bounds.bottom)
            + shadow_offset,
            Vec2(shadow_bounds.right, shadow_bounds.bottom)
            + shadow_offset,
            Vec2(shadow_bounds.right, shadow_bounds.bottom)
            + shadow_offset
            + shadow_depth_extra,
            Vec2(shadow_bounds.left, shadow_bounds.bottom)
            + shadow_offset
            + shadow_depth_extra,
        )
        ax.add_patch(
            Polygon(
                [(point.x, point.y) for point in shadow_polygon],
                closed=True,
                transform=transform,
                facecolor=style.shadow_color,
                edgecolor="none",
                alpha=style.shadow_alpha,
                zorder=1.0,
            )
        )

        for face_name, fill_color, zorder in (
            ("right", side_color, 2.5),
            ("front", face_color, 3.0),
            ("top", top_color, 4.0),
        ):
            polygon = projected_box.face_polygon(face_name)
            ax.add_patch(
                Polygon(
                    [(point.x, point.y) for point in polygon],
                    closed=True,
                    transform=transform,
                    facecolor=fill_color,
                    edgecolor=style.edge_color,
                    linewidth=style.linewidth,
                    zorder=zorder,
                )
            )


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import failure depends on env.
        raise ImportError(
            "plot_classical_baseline_architectures.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure S1."
        ) from exc
    return plt


def resolve_current_classical_figure_2_model_spec(
    model_kind: str,
    *,
    defaults: Figure2TrainingDefaults | None = None,
) -> ModelSpec:
    if model_kind not in DEFAULT_CLASSICAL_MODEL_KINDS:
        raise ValueError(
            "Figure S1 only supports current Figure 2 classical baselines "
            f"{DEFAULT_CLASSICAL_MODEL_KINDS}, got {model_kind!r}."
        )

    resolved_defaults = defaults or Figure2TrainingDefaults()
    return build_figure_2_model_spec(model_kind, defaults=resolved_defaults)


def resolve_current_classical_figure_2_model(
    model_kind: str,
    *,
    defaults: Figure2TrainingDefaults | None = None,
) -> tuple[ModelSpec, nn.Module]:
    spec = resolve_current_classical_figure_2_model_spec(model_kind, defaults=defaults)
    return spec, instantiate_model(spec)


def _square_size_label(value: int | tuple[int, int]) -> str:
    if isinstance(value, int):
        return f"{value}x{value}"
    if len(value) != 2:
        raise ValueError(f"Expected a 2D size tuple, got {value!r}.")
    return f"{value[0]}x{value[1]}"


def _describe_module_block(module: nn.Module) -> DiagramBlock:
    if isinstance(module, nn.Conv2d):
        return DiagramBlock(
            kind="conv",
            title="Conv2d",
            detail=f"{module.in_channels}->{module.out_channels}\n{_square_size_label(module.kernel_size)}",
            caption=(
                rf"Conv"
                "\n"
                rf"${module.in_channels}\to {module.out_channels}$"
                "\n"
                f"{_square_size_label(module.kernel_size)}"
            ),
            geometry="feature_map",
            primary_size=float(module.out_channels),
            secondary_size=float(module.out_channels),
        )
    if isinstance(module, nn.Linear):
        return DiagramBlock(
            kind="linear",
            title="Linear",
            detail=f"{module.in_features}->{module.out_features}",
            caption=rf"Linear" "\n" rf"${module.in_features}\to {module.out_features}$",
            geometry="vector",
            primary_size=float(module.out_features),
            secondary_size=float(module.out_features),
        )
    if isinstance(module, (nn.ReLU, nn.GELU)):
        return DiagramBlock(
            kind="activation",
            title=type(module).__name__,
            caption=type(module).__name__,
            geometry="plate",
        )
    if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        stride = module.stride if module.stride is not None else module.kernel_size
        pool_name = type(module).__name__
        pool_short_name = "AvgPool" if isinstance(module, nn.AvgPool2d) else "MaxPool"
        return DiagramBlock(
            kind="pool",
            title=pool_name,
            detail=f"{_square_size_label(module.kernel_size)}\nstride {_square_size_label(stride)}",
            caption=(
                f"{pool_short_name}\n"
                f"{_square_size_label(module.kernel_size)}"
            ),
            geometry="feature_map",
        )
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return DiagramBlock(
            kind="pool",
            title="AdaptiveAvgPool2d",
            detail=f"output {_square_size_label(module.output_size)}",
            caption=f"AdaptiveAvgPool\n{_square_size_label(module.output_size)}",
            geometry="feature_map",
        )
    if isinstance(module, nn.Dropout):
        return DiagramBlock(
            kind="regularization",
            title="Dropout",
            detail=f"p={module.p:.2f}",
            caption=f"Dropout\n{module.p:.2f}",
            geometry="plate",
        )
    if isinstance(module, nn.Flatten):
        return DiagramBlock(
            kind="reshape",
            title="Flatten",
            detail=f"start_dim={module.start_dim}",
            caption="Flatten",
            geometry="plate",
            caption_position="above",
        )
    if isinstance(module, nn.ZeroPad2d):
        pad = module.padding
        if isinstance(pad, int):
            pad_text = str(pad)
        else:
            pad_text = "x".join(str(value) for value in pad)
        return DiagramBlock(
            kind="reshape",
            title="ZeroPad2d",
            detail=f"padding {pad_text}",
            caption=f"ZeroPad\n{pad_text}",
            geometry="plate",
        )
    raise TypeError(f"Figure S1 does not know how to visualize module type {type(module).__name__}.")


def _format_feature_map_output_label(activations: torch.Tensor) -> str:
    if activations.ndim != 4:
        raise ValueError(
            "Expected CNN feature-map activations with shape [B, C, X, Y], "
            f"got {tuple(activations.shape)}."
        )
    _, channels, height, width = activations.shape
    return f"{height}x{width}x{channels}"


def _format_vector_output_label(activations: torch.Tensor) -> str:
    if activations.ndim != 2:
        raise ValueError(
            "Expected vector activations with shape [B, F], "
            f"got {tuple(activations.shape)}."
        )
    return str(int(activations.shape[1]))


def extract_cnn_diagram_blocks(model: ClassicalCNN) -> list[DiagramBlock]:
    if not isinstance(model, ClassicalCNN):
        raise TypeError(f"extract_cnn_diagram_blocks expects ClassicalCNN, got {type(model).__name__}.")

    blocks = [
        DiagramBlock(
            kind="input",
            title="Input",
            detail=f"1x{model.image_size}x{model.image_size}\ngrayscale",
            caption="Input",
            output_label=f"{model.image_size}x{model.image_size}x1",
            geometry="feature_map",
            primary_size=float(model.image_size),
            secondary_size=1.0,
        )
    ]
    with torch.no_grad():
        model.eval()
        activations = torch.zeros(1, model.image_size, model.image_size, dtype=torch.float32)
        feature_activations = activations.unsqueeze(1)
        for module in model.features:
            feature_activations = module(feature_activations)
            base_block = _describe_module_block(module)
            output_label = ""
            primary_size = base_block.primary_size
            secondary_size = base_block.secondary_size
            if base_block.geometry == "feature_map":
                output_label = _format_feature_map_output_label(feature_activations)
                primary_size = float(feature_activations.shape[-1])
                secondary_size = float(feature_activations.shape[1])
            blocks.append(
                DiagramBlock(
                    kind=base_block.kind,
                    title=base_block.title,
                    detail=base_block.detail,
                    caption=base_block.caption,
                    output_label=output_label,
                    geometry=base_block.geometry,
                    primary_size=primary_size,
                    secondary_size=secondary_size,
                )
            )

        vector_activations = feature_activations.flatten(start_dim=1)
        dropout_output = model.dropout(vector_activations)
        blocks.append(_describe_module_block(model.dropout))
        classifier_block = _describe_module_block(model.classifier)
        logits = model.classifier(dropout_output)
        blocks.append(
            DiagramBlock(
                kind=classifier_block.kind,
                title=classifier_block.title,
                detail=classifier_block.detail,
                caption=classifier_block.caption,
                output_label=_format_vector_output_label(logits),
                geometry=classifier_block.geometry,
                primary_size=float(logits.shape[1]),
                secondary_size=float(logits.shape[1]),
            )
        )
    return blocks


def extract_mlp_diagram_blocks(model: ClassicalMLP) -> list[DiagramBlock]:
    if not isinstance(model, ClassicalMLP):
        raise TypeError(f"extract_mlp_diagram_blocks expects ClassicalMLP, got {type(model).__name__}.")

    blocks = [
        DiagramBlock(
            kind="input",
            title="Input",
            detail=f"1x{model.image_size}x{model.image_size}\ngrayscale",
            caption="Input",
            output_label=f"{model.image_size}x{model.image_size}x1",
            geometry="feature_map",
            primary_size=float(model.image_size),
            secondary_size=float(model.image_size),
        ),
    ]
    with torch.no_grad():
        model.eval()
        activations = torch.zeros(1, model.image_size, model.image_size, dtype=torch.float32)

        flatten_block = _describe_module_block(model.flatten)
        vector_activations = model.flatten(activations)
        blocks.append(
            DiagramBlock(
                kind=flatten_block.kind,
                title=flatten_block.title,
                detail=flatten_block.detail,
                caption=flatten_block.caption,
                output_label=_format_vector_output_label(vector_activations),
                geometry=flatten_block.geometry,
                caption_position=flatten_block.caption_position,
                primary_size=float(vector_activations.shape[1]),
                secondary_size=float(vector_activations.shape[1]),
            )
        )

        input_layer_block = _describe_module_block(model.input_layer)
        vector_activations = model.input_layer(vector_activations)
        blocks.append(
            DiagramBlock(
                kind=input_layer_block.kind,
                title=input_layer_block.title,
                detail=input_layer_block.detail,
                caption=input_layer_block.caption,
                output_label=_format_vector_output_label(vector_activations),
                geometry=input_layer_block.geometry,
                caption_position=input_layer_block.caption_position,
                primary_size=float(vector_activations.shape[1]),
                secondary_size=float(vector_activations.shape[1]),
            )
        )
        blocks.append(_describe_module_block(model.input_activation))
        vector_activations = model.input_activation(vector_activations)
        blocks.append(_describe_module_block(model.input_dropout))
        vector_activations = model.input_dropout(vector_activations)

        if model.expansion_layer is not None:
            if model.expansion_activation is None or model.expansion_dropout is None:
                raise ValueError("ClassicalMLP expansion stage is partially defined.")
            expansion_block = _describe_module_block(model.expansion_layer)
            vector_activations = model.expansion_layer(vector_activations)
            blocks.append(
                DiagramBlock(
                    kind=expansion_block.kind,
                    title=expansion_block.title,
                    detail=expansion_block.detail,
                    caption=expansion_block.caption,
                    output_label=_format_vector_output_label(vector_activations),
                    geometry=expansion_block.geometry,
                    caption_position=expansion_block.caption_position,
                    primary_size=float(vector_activations.shape[1]),
                    secondary_size=float(vector_activations.shape[1]),
                )
            )
            blocks.append(_describe_module_block(model.expansion_activation))
            vector_activations = model.expansion_activation(vector_activations)
            blocks.append(_describe_module_block(model.expansion_dropout))
            vector_activations = model.expansion_dropout(vector_activations)

        if model.hidden_layer is not None:
            if model.hidden_activation is None or model.hidden_dropout is None:
                raise ValueError("ClassicalMLP hidden stage is partially defined.")
            hidden_block = _describe_module_block(model.hidden_layer)
            vector_activations = model.hidden_layer(vector_activations)
            blocks.append(
                DiagramBlock(
                    kind=hidden_block.kind,
                    title=hidden_block.title,
                    detail=hidden_block.detail,
                    caption=hidden_block.caption,
                    output_label=_format_vector_output_label(vector_activations),
                    geometry=hidden_block.geometry,
                    caption_position=hidden_block.caption_position,
                    primary_size=float(vector_activations.shape[1]),
                    secondary_size=float(vector_activations.shape[1]),
                )
            )
            blocks.append(_describe_module_block(model.hidden_activation))
            vector_activations = model.hidden_activation(vector_activations)
            blocks.append(_describe_module_block(model.hidden_dropout))
            vector_activations = model.hidden_dropout(vector_activations)

        for layer, activation, dropout in zip(
            model.extra_hidden_layers,
            model.extra_hidden_activations,
            model.extra_hidden_dropouts,
            strict=True,
        ):
            layer_block = _describe_module_block(layer)
            vector_activations = layer(vector_activations)
            blocks.append(
                DiagramBlock(
                    kind=layer_block.kind,
                    title=layer_block.title,
                    detail=layer_block.detail,
                    caption=layer_block.caption,
                    output_label=_format_vector_output_label(vector_activations),
                    geometry=layer_block.geometry,
                    caption_position=layer_block.caption_position,
                    primary_size=float(vector_activations.shape[1]),
                    secondary_size=float(vector_activations.shape[1]),
                )
            )
            blocks.append(_describe_module_block(activation))
            vector_activations = activation(vector_activations)
            blocks.append(_describe_module_block(dropout))
            vector_activations = dropout(vector_activations)

        classifier_block = _describe_module_block(model.classifier)
        logits = model.classifier(vector_activations)
        blocks.append(
            DiagramBlock(
                kind=classifier_block.kind,
                title=classifier_block.title,
                detail=classifier_block.detail,
                caption=classifier_block.caption,
                output_label=_format_vector_output_label(logits),
                geometry=classifier_block.geometry,
                primary_size=float(logits.shape[1]),
                secondary_size=float(logits.shape[1]),
            )
        )
    return blocks


def _blend_channel(channel: int, target: int, weight: float) -> int:
    return max(0, min(255, int(round(channel * (1.0 - weight) + target * weight))))


def _shade_hex_color(hex_color: str, *, toward: str, weight: float) -> str:
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}.")
    target = 255 if toward == "white" else 0
    channels = tuple(int(normalized[index:index + 2], 16) for index in range(0, 6, 2))
    return "#" + "".join(
        f"{_blend_channel(channel, target, weight):02x}"
        for channel in channels
    )


def _interpolate_log_scale(
    value: float,
    *,
    minimum_value: float,
    maximum_value: float,
    minimum_output: float,
    maximum_output: float,
) -> float:
    if minimum_value <= 0 or maximum_value <= 0:
        raise ValueError("Log-scale interpolation requires positive bounds.")
    if maximum_value <= minimum_value:
        return 0.5 * (minimum_output + maximum_output)
    position = (math.log(value) - math.log(minimum_value)) / (
        math.log(maximum_value) - math.log(minimum_value)
    )
    position = max(0.0, min(1.0, position))
    return minimum_output + position * (maximum_output - minimum_output)


def _measure_text_bbox_data(
    ax: "Axes",
    *,
    x: float,
    y: float,
    text: str,
    fontsize: float,
    ha: str,
    va: str,
    rotation: float = 0.0,
) -> Bounds2D:
    text_artist = ax.text(
        x,
        y,
        text,
        transform=ax.transData,
        ha=ha,
        va=va,
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=fontsize,
        alpha=0.0,
    )
    try:
        renderer = ax.figure.canvas.get_renderer()
        bbox_display = text_artist.get_window_extent(renderer=renderer)
        inverted = ax.transData.inverted()
        (x0, y0) = inverted.transform((bbox_display.x0, bbox_display.y0))
        (x1, y1) = inverted.transform((bbox_display.x1, bbox_display.y1))
        left, right = sorted((x0, x1))
        bottom, top = sorted((y0, y1))
        return Bounds2D(left=left, right=right, bottom=bottom, top=top)
    finally:
        text_artist.remove()

def _new_square_measurement_axes(
    *,
    layout_side: float = 2.0 * DEFAULT_BASE_BLOCK_SIZE_INCHES,
):
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(layout_side, layout_side))
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()
    half_side = 0.5 * layout_side
    ax.set_xlim(-half_side, half_side)
    ax.set_ylim(-half_side, half_side)
    ax.set_aspect("equal", adjustable="box")
    return figure, ax


def _shift_rendered_diagram(
    rendered: RenderedDiagram,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> RenderedDiagram:
    return RenderedDiagram(
        blocks=tuple(
            PlacedBlock(
                block=placed_block.block,
                projected_box=placed_block.projected_box.shifted(dx=dx, dy=dy),
                labels=tuple(label.shifted(dx=dx, dy=dy) for label in placed_block.labels),
                shapes=tuple(shape.shifted(dx=dx, dy=dy) for shape in placed_block.shapes),
            )
            for placed_block in rendered.blocks
        ),
        font_scale=rendered.font_scale,
        box_scale=rendered.box_scale,
    )


def _prepare_figure_s1_panel(
    *,
    blocks: Sequence[DiagramBlock],
    renderer: ArchitectureDiagramRenderer,
    base_font_scale: float,
) -> _PreparedFigureS1Panel:
    measurement_figure, measurement_ax = _new_square_measurement_axes(
        layout_side=2.0 * renderer.base_block_size_inches
    )
    try:
        rendered = renderer.render(
            measurement_ax,
            blocks,
            base_font_scale=base_font_scale,
        )
    finally:
        measurement_figure.clf()
        _require_matplotlib().close(measurement_figure)
    bounds = rendered.bounds
    padding = DEFAULT_CONTENT_PADDING_INCHES
    width = bounds.width + 2.0 * padding
    height = bounds.height + 2.0 * padding
    shifted = _shift_rendered_diagram(
        rendered,
        dx=padding - bounds.left,
        dy=padding - bounds.bottom,
    )
    return _PreparedFigureS1Panel(
        rendered=shifted,
        width=width,
        height=height,
        figsize=(width, height),
    )


def _draw_prepared_architecture_diagram(
    *,
    prepared_panel: _PreparedFigureS1Panel,
    panel_title: str,
    renderer: ArchitectureDiagramRenderer,
) -> "Figure":
    plt = _require_matplotlib()

    figure, ax = plt.subplots(figsize=prepared_panel.figsize)
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.set_xlim(0.0, prepared_panel.width)
    ax.set_ylim(0.0, prepared_panel.height)
    ax.set_aspect("equal", adjustable="box")

    renderer.draw(ax, prepared_panel.rendered, transform=ax.transData)
    del panel_title
    return figure


def plot_article_figure_s1a(
    *,
    defaults: Figure2TrainingDefaults | None = None,
    figsize: tuple[float, float] | None = None,
    minimum_block_gap: float = _DEFAULT_MINIMUM_BLOCK_GAP,
    base_font_scale: float = _DEFAULT_BASE_FONT_SCALE,
    renderer: ArchitectureDiagramRenderer | None = None,
) -> "Figure":
    diagram_renderer = renderer or ArchitectureDiagramRenderer(
        layout_config=DiagramLayoutConfig(minimum_block_gap=minimum_block_gap)
    )
    if figsize is not None:
        raise ValueError(
            "Figure S1 now uses content-sized layout; explicit figsize is no longer supported."
        )

    _, model = resolve_current_classical_figure_2_model("classical_cnn", defaults=defaults)
    prepared_panel = _prepare_figure_s1_panel(
        blocks=extract_cnn_diagram_blocks(model),
        renderer=diagram_renderer,
        base_font_scale=base_font_scale,
    )
    return _draw_prepared_architecture_diagram(
        prepared_panel=prepared_panel,
        panel_title="Classical CNN",
        renderer=diagram_renderer,
    )


def plot_article_figure_s1b(
    *,
    defaults: Figure2TrainingDefaults | None = None,
    figsize: tuple[float, float] | None = None,
    minimum_block_gap: float = _DEFAULT_MINIMUM_BLOCK_GAP,
    base_font_scale: float = _DEFAULT_BASE_FONT_SCALE,
    renderer: ArchitectureDiagramRenderer | None = None,
) -> "Figure":
    diagram_renderer = renderer or ArchitectureDiagramRenderer(
        layout_config=DiagramLayoutConfig(minimum_block_gap=minimum_block_gap)
    )
    if figsize is not None:
        raise ValueError(
            "Figure S1 now uses content-sized layout; explicit figsize is no longer supported."
        )

    _, model = resolve_current_classical_figure_2_model("classical_mlp", defaults=defaults)
    prepared_panel = _prepare_figure_s1_panel(
        blocks=extract_mlp_diagram_blocks(model),
        renderer=diagram_renderer,
        base_font_scale=base_font_scale,
    )
    return _draw_prepared_architecture_diagram(
        prepared_panel=prepared_panel,
        panel_title="Classical MLP",
        renderer=diagram_renderer,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Figure S1 classical-architecture schematics from the current "
            "Figure 2 model definitions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where classical_cnn_architecture.pdf and "
            "classical_mlp_architecture.pdf will be written."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn_figure = plot_article_figure_s1a()
    cnn_output_path = output_dir / "classical_cnn_architecture.pdf"
    cnn_figure.savefig(cnn_output_path)
    plt.close(cnn_figure)

    mlp_figure = plot_article_figure_s1b()
    mlp_output_path = output_dir / "classical_mlp_architecture.pdf"
    mlp_figure.savefig(mlp_output_path)
    plt.close(mlp_figure)

    print(cnn_output_path)
    print(mlp_output_path)


if __name__ == "__main__":
    main()
