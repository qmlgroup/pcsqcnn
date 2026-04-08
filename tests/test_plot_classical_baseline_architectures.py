from pathlib import Path

import pytest
from qcnn import ClassicalMLP

from .script_loading import load_run_script

plot_figure_s1 = load_run_script(
    "plot_classical_baseline_architectures_module",
    "plot_classical_baseline_architectures.py",
)


def _axis_aligned_shape(
    left: float,
    right: float,
    *,
    bottom: float = 0.0,
    top: float = 1.0,
    kind: str = "test_shape",
) -> object:
    bounds = plot_figure_s1.Bounds2D(left=left, right=right, bottom=bottom, top=top)
    return plot_figure_s1.PlacedShape(
        kind=kind,
        vertices=plot_figure_s1._rectangle_vertices_from_bounds(bounds),
        name=kind,
    )


def _rotated_shape(
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_degrees: float,
    kind: str = "rotated_label",
) -> object:
    return plot_figure_s1._text_shape_from_anchor(
        text=kind,
        anchor=plot_figure_s1.Vec2(center_x, center_y),
        size=plot_figure_s1.Size2D(width=width, height=height),
        ha="center",
        va="center",
        rotation_degrees=rotation_degrees,
        role=kind,
    )


def _render_model_layout(model_kind: str) -> object:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(
            minimum_block_gap=plot_figure_s1._DEFAULT_MINIMUM_BLOCK_GAP
        )
    )
    measurement_figure, measurement_ax = plot_figure_s1._new_square_measurement_axes()
    try:
        _, model = plot_figure_s1.resolve_current_classical_figure_2_model(model_kind)
        extractor = (
            plot_figure_s1.extract_cnn_diagram_blocks
            if model_kind == "classical_cnn"
            else plot_figure_s1.extract_mlp_diagram_blocks
        )
        return renderer.render(
            measurement_ax,
            extractor(model),
            base_font_scale=1.0,
        )
    finally:
        plot_figure_s1._require_matplotlib().close(measurement_figure)


def _prepare_model_panel(model_kind: str) -> object:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(
            minimum_block_gap=plot_figure_s1._DEFAULT_MINIMUM_BLOCK_GAP
        )
    )
    _, model = plot_figure_s1.resolve_current_classical_figure_2_model(model_kind)
    extractor = (
        plot_figure_s1.extract_cnn_diagram_blocks
        if model_kind == "classical_cnn"
        else plot_figure_s1.extract_mlp_diagram_blocks
    )
    return plot_figure_s1._prepare_figure_s1_panel(
        blocks=extractor(model),
        renderer=renderer,
        base_font_scale=1.0,
    )


def test_resolve_current_classical_figure_2_model_specs_uses_current_saved_contract() -> None:
    cnn_spec = plot_figure_s1.resolve_current_classical_figure_2_model_spec("classical_cnn")
    mlp_spec = plot_figure_s1.resolve_current_classical_figure_2_model_spec("classical_mlp")

    assert cnn_spec.module == "qcnn.classic"
    assert cnn_spec.class_name == "ClassicalCNN"
    assert cnn_spec.constructor_kwargs == {
        "image_size": 32,
        "num_classes": 10,
        "base_channels": 16,
        "dropout": 0.1,
    }
    assert mlp_spec.module == "qcnn.classic"
    assert mlp_spec.class_name == "ClassicalMLP"
    assert mlp_spec.constructor_kwargs == {
        "image_size": 32,
        "num_classes": 10,
        "hidden_widths": (29, 116, 116),
        "dropout": 0.1,
    }


def test_extract_cnn_diagram_blocks_reflects_current_model_definition() -> None:
    _, model = plot_figure_s1.resolve_current_classical_figure_2_model("classical_cnn")

    blocks = plot_figure_s1.extract_cnn_diagram_blocks(model)

    assert [block.title for block in blocks] == [
        "Input",
        "Conv2d",
        "ReLU",
        "Conv2d",
        "ReLU",
        "AvgPool2d",
        "Conv2d",
        "ReLU",
        "Conv2d",
        "ReLU",
        "AvgPool2d",
        "AdaptiveAvgPool2d",
        "Dropout",
        "Linear",
    ]
    assert blocks[0].detail == "1x32x32\ngrayscale"
    assert blocks[1].detail == "1->16\n3x3"
    assert blocks[3].detail == "16->32\n3x3"
    assert blocks[5].detail == "2x2\nstride 2x2"
    assert blocks[6].detail == "32->48\n3x3"
    assert blocks[8].detail == "48->64\n3x3"
    assert blocks[11].detail == "output 1x1"
    assert blocks[12].detail == "p=0.10"
    assert blocks[13].detail == "64->10"
    assert blocks[12].caption == "Dropout\n0.10"
    assert [block.caption for block in blocks if block.title == "AvgPool2d"] == [
        "AvgPool\n2x2",
        "AvgPool\n2x2",
    ]
    assert [block.caption for block in blocks if block.title == "Conv2d"] == [
        "Conv\n$1\\to 16$\n3x3",
        "Conv\n$16\\to 32$\n3x3",
        "Conv\n$32\\to 48$\n3x3",
        "Conv\n$48\\to 64$\n3x3",
    ]


def test_extract_mlp_diagram_blocks_reflects_current_model_definition() -> None:
    _, model = plot_figure_s1.resolve_current_classical_figure_2_model("classical_mlp")

    blocks = plot_figure_s1.extract_mlp_diagram_blocks(model)

    assert [block.title for block in blocks] == [
        "Input",
        "Flatten",
        "Linear",
        "GELU",
        "Dropout",
        "Linear",
        "GELU",
        "Dropout",
        "Linear",
        "GELU",
        "Dropout",
        "Linear",
    ]
    assert blocks[0].detail == "1x32x32\ngrayscale"
    assert blocks[0].primary_size == 32.0
    assert blocks[0].secondary_size == 32.0
    assert blocks[1].detail == "start_dim=1"
    assert blocks[1].caption_position == "above"
    assert [block.detail for block in blocks if block.title == "Linear"] == [
        "1024->29",
        "29->116",
        "116->116",
        "116->10",
    ]
    assert [block.detail for block in blocks if block.title == "Dropout"] == [
        "p=0.10",
        "p=0.10",
        "p=0.10",
    ]
    assert [block.caption for block in blocks if block.title == "Dropout"] == [
        "Dropout\n0.10",
        "Dropout\n0.10",
        "Dropout\n0.10",
    ]
    assert [block.caption for block in blocks if block.title == "Linear"] == [
        "Linear\n$1024\\to 29$",
        "Linear\n$29\\to 116$",
        "Linear\n$116\\to 116$",
        "Linear\n$116\\to 10$",
    ]


def test_extract_mlp_diagram_blocks_handles_extra_hidden_layers_generically() -> None:
    model = ClassicalMLP(
        image_size=8,
        num_classes=5,
        hidden_widths=(7, 9, 11, 13),
        dropout=0.25,
    )

    blocks = plot_figure_s1.extract_mlp_diagram_blocks(model)

    assert [block.detail for block in blocks if block.title == "Linear"] == [
        "64->7",
        "7->9",
        "9->11",
        "11->13",
        "13->5",
    ]
    assert [block.detail for block in blocks if block.title == "Dropout"] == [
        "p=0.25",
        "p=0.25",
        "p=0.25",
        "p=0.25",
    ]


def test_required_horizontal_separation_between_shapes_axis_aligned_rectangles() -> None:
    first = _axis_aligned_shape(0.0, 1.0)
    second = _axis_aligned_shape(0.75, 1.75)

    separation = plot_figure_s1._required_horizontal_separation_between_shapes(first, second)

    assert separation == pytest.approx(0.25 + plot_figure_s1._LAYOUT_SOLVER_EPSILON)


def test_required_horizontal_separation_between_shapes_returns_zero_for_disjoint_shapes() -> None:
    first = _axis_aligned_shape(0.0, 1.0)
    second = _axis_aligned_shape(1.25, 2.25)

    separation = plot_figure_s1._required_horizontal_separation_between_shapes(first, second)

    assert separation == 0.0


def test_required_horizontal_separation_between_shapes_honors_minimum_gap() -> None:
    first = _axis_aligned_shape(0.0, 1.0)
    second = _axis_aligned_shape(1.25, 2.25)

    separation = plot_figure_s1._required_horizontal_separation_between_shapes(
        first,
        second,
        minimum_gap=0.40,
    )

    assert separation == pytest.approx(0.15 + plot_figure_s1._LAYOUT_SOLVER_EPSILON)


def test_required_horizontal_separation_between_shapes_treats_touching_shapes_as_epsilon() -> None:
    first = _axis_aligned_shape(0.0, 1.0)
    second = _axis_aligned_shape(1.0, 2.0)

    separation = plot_figure_s1._required_horizontal_separation_between_shapes(first, second)

    assert separation == pytest.approx(plot_figure_s1._LAYOUT_SOLVER_EPSILON)


def test_required_horizontal_separation_between_shapes_handles_rotated_shape() -> None:
    first = _axis_aligned_shape(0.0, 1.0, bottom=0.0, top=1.0)
    second = _rotated_shape(
        center_x=0.9,
        center_y=0.5,
        width=0.6,
        height=0.2,
        rotation_degrees=37.0,
    )

    separation = plot_figure_s1._required_horizontal_separation_between_shapes(first, second)
    shifted_second = second.shifted(dx=separation)

    assert separation > 0.0
    assert not plot_figure_s1._shapes_overlap_at_current_offset(first, shifted_second)


def test_projected_box_top_caption_anchor_uses_upper_visible_top_edge() -> None:
    projected_box = plot_figure_s1.Box3D(
        origin=plot_figure_s1.Vec3(0.0, 0.0, 0.0),
        size=plot_figure_s1.BoxDimensions(width=0.3, height=0.2, depth=0.1),
    ).project(plot_figure_s1.ObliqueProjection())

    anchor = projected_box.anchor_point(plot_figure_s1.AnchorRef("top_caption"))
    top_polygon = projected_box.face_polygon("top")
    top_edges = tuple(
        (start, end)
        for start, end in zip(top_polygon, top_polygon[1:] + top_polygon[:1], strict=False)
    )
    expected_start, expected_end = max(
        top_edges,
        key=lambda edge: 0.5 * (edge[0].y + edge[1].y),
    )

    assert anchor.x == pytest.approx(0.5 * (expected_start.x + expected_end.x))
    assert anchor.y == pytest.approx(projected_box.bounds.top)


def test_projected_box_bottom_caption_anchor_uses_front_center_x_and_block_bottom() -> None:
    projected_box = plot_figure_s1.Box3D(
        origin=plot_figure_s1.Vec3(0.0, 0.0, 0.0),
        size=plot_figure_s1.BoxDimensions(width=0.3, height=0.2, depth=0.1),
    ).project(plot_figure_s1.ObliqueProjection())

    anchor = projected_box.anchor_point(plot_figure_s1.AnchorRef("bottom_caption"))

    assert anchor.x == pytest.approx(projected_box.face_center("front").x)
    assert anchor.y == pytest.approx(projected_box.bounds.bottom)


def test_caption_label_above_uses_top_caption_anchor_and_layout_gap() -> None:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(text_gap_y=0.017)
    )
    block = plot_figure_s1.DiagramBlock(
        kind="conv",
        title="Conv2d",
        caption_position="above",
        geometry="feature_map",
    )

    label = renderer._caption_label(block, "Conv2d", font_scale=1.0)

    assert label.anchor == plot_figure_s1.AnchorRef("top_caption")
    assert label.va == "bottom"
    assert label.offset == plot_figure_s1.Vec2(
        0.0, 0.017 * plot_figure_s1.DEFAULT_BASE_BLOCK_SIZE_INCHES
    )


def test_caption_label_below_uses_bottom_caption_anchor_and_layout_gap() -> None:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(text_gap_y=0.017)
    )
    block = plot_figure_s1.DiagramBlock(
        kind="reshape",
        title="Flatten",
        caption_position="below",
        geometry="plate",
    )

    label = renderer._caption_label(block, "Flatten", font_scale=1.0)

    assert label.anchor == plot_figure_s1.AnchorRef("bottom_caption")
    assert label.va == "top"
    assert label.offset == plot_figure_s1.Vec2(
        0.0, -0.017 * plot_figure_s1.DEFAULT_BASE_BLOCK_SIZE_INCHES
    )


def test_top_caption_measured_bottom_sits_above_block_top_by_text_gap() -> None:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(text_gap_y=0.013)
    )
    block = plot_figure_s1.DiagramBlock(
        kind="conv",
        title="Conv2d",
        caption="Conv2d",
        caption_position="above",
        geometry="feature_map",
        primary_size=32.0,
        secondary_size=16.0,
    )
    measurement_figure, measurement_ax = plot_figure_s1._new_square_measurement_axes()
    try:
        template = renderer._build_block_template(
            measurement_ax,
            block,
            statistics=plot_figure_s1.DiagramScaleStatistics.from_blocks((block,)),
            box_scale=1.0,
            font_scale=1.0,
        )
    finally:
        plot_figure_s1._require_matplotlib().close(measurement_figure)

    caption_label = next(label for label in template.labels if label.spec.role == "caption")
    assert caption_label.bounds.bottom == pytest.approx(
        template.projected_box.bounds.top
        + renderer.layout_config.text_gap_y * renderer.base_block_size_inches
    )


def test_bottom_caption_measured_top_sits_below_block_bottom_by_text_gap() -> None:
    renderer = plot_figure_s1.ArchitectureDiagramRenderer(
        layout_config=plot_figure_s1.DiagramLayoutConfig(text_gap_y=0.013)
    )
    block = plot_figure_s1.DiagramBlock(
        kind="reshape",
        title="Flatten",
        caption="Flatten",
        caption_position="below",
        geometry="plate",
    )
    measurement_figure, measurement_ax = plot_figure_s1._new_square_measurement_axes()
    try:
        template = renderer._build_block_template(
            measurement_ax,
            block,
            statistics=plot_figure_s1.DiagramScaleStatistics.from_blocks((block,)),
            box_scale=1.0,
            font_scale=1.0,
        )
    finally:
        plot_figure_s1._require_matplotlib().close(measurement_figure)

    caption_label = next(label for label in template.labels if label.spec.role == "caption")
    assert caption_label.bounds.top == pytest.approx(
        template.projected_box.bounds.bottom
        - renderer.layout_config.text_gap_y * renderer.base_block_size_inches
    )


def test_measure_text_bbox_data_returns_physical_units_and_scales_with_fontsize() -> None:
    measurement_figure, measurement_ax = plot_figure_s1._new_square_measurement_axes()
    try:
        small = plot_figure_s1._measure_text_bbox_data(
            measurement_ax,
            x=0.0,
            y=0.0,
            text="Linear\n1024->29",
            fontsize=10.0,
            ha="center",
            va="center",
        )
        large = plot_figure_s1._measure_text_bbox_data(
            measurement_ax,
            x=0.0,
            y=0.0,
            text="Linear\n1024->29",
            fontsize=14.0,
            ha="center",
            va="center",
        )
    finally:
        plot_figure_s1._require_matplotlib().close(measurement_figure)

    assert small.width > 0.0
    assert small.height > 0.0
    assert large.width > small.width
    assert large.height > small.height


@pytest.mark.parametrize("model_kind", ["classical_cnn", "classical_mlp"])
def test_neighbor_shapes_do_not_overlap_after_layout(model_kind: str) -> None:
    rendered = _render_model_layout(model_kind)

    for current, nxt in zip(rendered.blocks, rendered.blocks[1:], strict=False):
        for current_shape in current.shapes:
            for next_shape in nxt.shapes:
                assert not plot_figure_s1._shapes_overlap_at_current_offset(current_shape, next_shape)


@pytest.mark.parametrize("model_kind", ["classical_cnn", "classical_mlp"])
def test_neighbor_block_bodies_respect_minimum_gap(model_kind: str) -> None:
    rendered = _render_model_layout(model_kind)
    minimum_body_gap = (
        plot_figure_s1._DEFAULT_MINIMUM_BLOCK_GAP
        * plot_figure_s1.DEFAULT_BASE_BLOCK_SIZE_INCHES
    )

    for current, nxt in zip(rendered.blocks, rendered.blocks[1:], strict=False):
        body_gap = nxt.body_shape.bounds.left - current.body_shape.bounds.right
        assert body_gap >= minimum_body_gap - 1e-9


@pytest.mark.parametrize("model_kind", ["classical_cnn", "classical_mlp"])
def test_neighbor_text_shapes_respect_minimum_horizontal_gap(model_kind: str) -> None:
    rendered = _render_model_layout(model_kind)
    minimum_text_gap = plot_figure_s1._DEFAULT_TEXT_GAP * plot_figure_s1.DEFAULT_BASE_BLOCK_SIZE_INCHES

    for current, nxt in zip(rendered.blocks, rendered.blocks[1:], strict=False):
        for current_shape in current.shapes:
            for next_shape in nxt.shapes:
                if current_shape.kind == "block_body" and next_shape.kind == "block_body":
                    continue
                remaining = plot_figure_s1._required_horizontal_separation_between_shapes(
                    current_shape,
                    next_shape,
                    minimum_gap=minimum_text_gap,
                )
                assert remaining <= 2.0 * plot_figure_s1._LAYOUT_SOLVER_EPSILON


def test_plot_article_figure_s1_panels_render_successfully() -> None:
    cnn_panel = _prepare_model_panel("classical_cnn")
    mlp_panel = _prepare_model_panel("classical_mlp")
    cnn_figure = plot_figure_s1.plot_article_figure_s1a()
    mlp_figure = plot_figure_s1.plot_article_figure_s1b()
    try:
        cnn_size = tuple(float(value) for value in cnn_figure.get_size_inches())
        mlp_size = tuple(float(value) for value in mlp_figure.get_size_inches())
        assert cnn_size == pytest.approx(cnn_panel.figsize)
        assert mlp_size == pytest.approx(mlp_panel.figsize)
        assert cnn_size[0] > 0.0
        assert mlp_size[0] > 0.0
        assert cnn_size[1] > 0.0
        assert mlp_size[1] > 0.0
        assert cnn_size[1] != pytest.approx(mlp_size[1])
    finally:
        plt = plot_figure_s1._require_matplotlib()
        plt.close(cnn_figure)
        plt.close(mlp_figure)


def test_prepared_figure_s1_panel_adds_only_content_padding() -> None:
    prepared = _prepare_model_panel("classical_cnn")
    bounds = prepared.rendered.bounds

    assert bounds.left == pytest.approx(plot_figure_s1.DEFAULT_CONTENT_PADDING_INCHES)
    assert bounds.bottom == pytest.approx(plot_figure_s1.DEFAULT_CONTENT_PADDING_INCHES)
    assert prepared.width - bounds.right == pytest.approx(
        plot_figure_s1.DEFAULT_CONTENT_PADDING_INCHES
    )
    assert prepared.height - bounds.top == pytest.approx(
        plot_figure_s1.DEFAULT_CONTENT_PADDING_INCHES
    )


@pytest.mark.parametrize(
    "plotter",
    [plot_figure_s1.plot_article_figure_s1a, plot_figure_s1.plot_article_figure_s1b],
)
def test_plot_article_figure_s1_rejects_explicit_figsize(plotter) -> None:
    with pytest.raises(ValueError, match="content-sized layout"):
        plotter(figsize=(6.0, 2.0))


def test_parse_args_accepts_output_dir_override() -> None:
    args = plot_figure_s1.parse_args(
        [
            "--output-dir",
            "/tmp/custom-figs",
        ]
    )

    assert args.output_dir == Path("/tmp/custom-figs")
