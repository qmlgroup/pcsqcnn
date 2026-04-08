import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pennylane = pytest.importorskip("pennylane")

from qcnn import (
    PCSQCNN,
    PCSQCNNNoQFT,
    move_active_qubit_to_condition,
)
from qcnn.quantum import _apply_fourier_junction_2d, _apply_iqft_2d, _apply_qft_2d


def _load_reference_module():
    module_path = Path(__file__).resolve().parents[1] / "pennylane" / "pcs_qcnn_reference.py"
    module_name = "pcs_qcnn_reference_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE_MODULE = _load_reference_module()
PennyLanePCSQCNN = REFERENCE_MODULE.PennyLanePCSQCNN


def _make_torch_model(
    *,
    image_size: int,
    num_classes: int,
    feature_qubits: int,
    quantum_layers: int,
    use_reduced_fourier_junction: bool = True,
    reduce_readout_to_feature_distribution: bool = False,
) -> PCSQCNN:
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        return PCSQCNN(
            image_size=image_size,
            num_classes=num_classes,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
            use_reduced_fourier_junction=use_reduced_fourier_junction,
            reduce_readout_to_feature_distribution=reduce_readout_to_feature_distribution,
        )


def _make_reference_config(
    model: PCSQCNN,
    *,
    include_dtype: bool = True,
) -> dict[str, object]:
    config: dict[str, object] = {
        "image_size": model.image_size,
        "num_classes": model.num_classes,
        "feature_qubits": model.feature_qubits,
        "quantum_layers": model.quantum_layers,
        "brightness_range": list(model.brightness_range),
        "shot_budget": model.shot_budget,
        "reduce_readout_to_feature_distribution": model.reduce_readout_to_feature_distribution,
        "use_reduced_fourier_junction": model.use_reduced_fourier_junction,
        "multiplexer_init_scale": model.multiplexer_init_scale,
    }
    if include_dtype:
        config["dtype"] = str(next(model.parameters()).dtype)
    return config


def _torch_quantum_readout(model: PCSQCNN, images: torch.Tensor) -> torch.Tensor:
    state = model.encoder(images)
    current_layout = model.layer_layouts[0]

    if model.use_reduced_fourier_junction:
        state = _apply_qft_2d(state)
        for layer_idx, multiplexer in enumerate(model.multiplexers):
            state = multiplexer(state)
            if layer_idx < model.quantum_layers - 1:
                state, current_layout = _apply_fourier_junction_2d(state, current_layout)
            else:
                state = _apply_iqft_2d(state)
    else:
        for layer_idx, multiplexer in enumerate(model.multiplexers):
            state = _apply_qft_2d(state)
            state = multiplexer(state)
            state = _apply_iqft_2d(state)
            if layer_idx < model.quantum_layers - 1:
                state, current_layout = move_active_qubit_to_condition(
                    state,
                    current_layout,
                    x_qubits_to_condition=1,
                    y_qubits_to_condition=1,
                )

    return model.measurement(state)


@pytest.mark.parametrize(
    ("image_size", "feature_qubits", "quantum_layers", "use_reduced_fourier_junction"),
    [
        (4, 1, 1, True),
        (4, 1, 1, False),
        (8, 1, 2, True),
        (8, 2, 2, True),
    ],
)
def test_pennylane_reference_matches_torch_quantum_readout_and_logits(
    image_size: int,
    feature_qubits: int,
    quantum_layers: int,
    use_reduced_fourier_junction: bool,
) -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=image_size,
        num_classes=5,
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
        use_reduced_fourier_junction=use_reduced_fourier_junction,
    )
    reference = PennyLanePCSQCNN.from_torch_state(
        _make_reference_config(model),
        model.state_dict(),
    )

    images = torch.rand(3, image_size, image_size, dtype=torch.float32)
    torch_readout = _torch_quantum_readout(model, images).detach().cpu().numpy()
    torch_logits = model(images).detach().cpu().numpy()

    reference_readout = reference.quantum_readout(images.detach().cpu().numpy())
    reference_logits = reference.predict_logits(images.detach().cpu().numpy())

    assert reference_readout.shape == torch_readout.shape
    assert reference_logits.shape == torch_logits.shape
    assert pytest.approx(reference_readout, abs=1e-6, rel=1e-6) == torch_readout
    assert pytest.approx(reference_logits, abs=1e-6, rel=1e-6) == torch_logits
    assert not hasattr(reference, "source_use_reduced_fourier_junction")


def test_pennylane_reference_accepts_saved_constructor_kwargs_without_dtype() -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=4,
        num_classes=3,
        feature_qubits=1,
        quantum_layers=1,
    )

    reference = PennyLanePCSQCNN.from_torch_state(
        _make_reference_config(model, include_dtype=False),
        model.state_dict(),
    )

    images = torch.rand(2, 4, 4, dtype=torch.float32)
    torch_logits = model(images).detach().cpu().numpy()
    reference_logits = reference.predict_logits(images.detach().cpu().numpy())

    assert pytest.approx(reference_logits, abs=1e-6, rel=1e-6) == torch_logits


def test_pennylane_reference_rejects_reduced_readout_config() -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=4,
        num_classes=3,
        feature_qubits=2,
        quantum_layers=1,
        reduce_readout_to_feature_distribution=True,
    )

    with pytest.raises(ValueError, match="full readout"):
        PennyLanePCSQCNN.from_torch_state(
            _make_reference_config(model),
            model.state_dict(),
        )


def test_pennylane_reference_rejects_no_qft_torch_checkpoint() -> None:
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        model = PCSQCNNNoQFT(
            image_size=4,
            num_classes=3,
            feature_qubits=1,
            quantum_layers=1,
        )

    with pytest.raises(ValueError, match="PCSQCNNNoQFT"):
        PennyLanePCSQCNN.from_torch_state(
            {
                "image_size": model.image_size,
                "num_classes": model.num_classes,
                "feature_qubits": model.feature_qubits,
                "quantum_layers": model.quantum_layers,
                "brightness_range": list(model.brightness_range),
                "shot_budget": model.shot_budget,
            },
            model.state_dict(),
        )

def test_prepare_mnist_test_dataset_resizes_and_normalizes_images(monkeypatch, tmp_path: Path) -> None:
    class FakeMnist:
        def __init__(self, *, root: str, train: bool, download: bool) -> None:
            assert root == str(tmp_path)
            assert train is False
            assert download is False
            self.data = torch.tensor(
                [
                    [[0, 255], [128, 64]],
                    [[255, 255], [0, 0]],
                ],
                dtype=torch.uint8,
            )
            self.targets = torch.tensor([3, 7], dtype=torch.long)

    monkeypatch.setattr(REFERENCE_MODULE, "MNIST", FakeMnist)

    dataset = REFERENCE_MODULE.prepare_mnist_test_dataset(
        root=tmp_path,
        image_size=4,
        download=False,
    )

    assert dataset.images.shape == (2, 4, 4)
    assert dataset.labels.tolist() == [3, 7]
    assert dataset.images.dtype == torch.float32
    assert float(dataset.images.min().item()) >= 0.0
    assert float(dataset.images.max().item()) <= 1.0


def test_prepare_mnist_test_dataset_places_images_on_zero_canvas(monkeypatch, tmp_path: Path) -> None:
    class FakeMnist:
        def __init__(self, *, root: str, train: bool, download: bool) -> None:
            assert root == str(tmp_path)
            assert train is False
            assert download is False
            self.data = torch.tensor([[[255, 255], [255, 255]]], dtype=torch.uint8)
            self.targets = torch.tensor([5], dtype=torch.long)

    monkeypatch.setattr(REFERENCE_MODULE, "MNIST", FakeMnist)

    dataset = REFERENCE_MODULE.prepare_mnist_test_dataset(
        root=tmp_path,
        image_size=4,
        scaled_image_size=2,
        max_offset=0,
        seed=0,
        download=False,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert dataset.labels.tolist() == [5]
    assert torch.allclose(dataset.images[0], expected)


def test_prepare_mnist_test_dataset_clips_translation_offset_to_canvas_slack(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeMnist:
        def __init__(self, *, root: str, train: bool, download: bool) -> None:
            assert root == str(tmp_path)
            assert train is False
            assert download is False
            self.data = torch.arange(2 * 2 * 2, dtype=torch.uint8).reshape(2, 2, 2)
            self.targets = torch.tensor([1, 9], dtype=torch.long)

    monkeypatch.setattr(REFERENCE_MODULE, "MNIST", FakeMnist)

    clipped = REFERENCE_MODULE.prepare_mnist_test_dataset(
        root=tmp_path,
        image_size=4,
        scaled_image_size=2,
        max_offset=2,
        seed=13,
        download=False,
    )
    reference = REFERENCE_MODULE.prepare_mnist_test_dataset(
        root=tmp_path,
        image_size=4,
        scaled_image_size=2,
        max_offset=1,
        seed=13,
        download=False,
    )

    assert torch.equal(clipped.images, reference.images)
    assert torch.equal(clipped.labels, reference.labels)


def test_pennylane_reference_builtin_qft_convention_preserves_one_layer_parity() -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=4,
        num_classes=3,
        feature_qubits=1,
        quantum_layers=1,
    )
    reference = PennyLanePCSQCNN.from_torch_state(
        _make_reference_config(model),
        model.state_dict(),
    )
    images = torch.rand(2, 4, 4, dtype=torch.float32)

    torch_readout = _torch_quantum_readout(model, images).detach().cpu().numpy()
    torch_logits = model(images).detach().cpu().numpy()
    reference_readout = reference.quantum_readout(images.detach().cpu().numpy())
    reference_logits = reference.predict_logits(images.detach().cpu().numpy())

    assert pytest.approx(reference_readout, abs=1e-6, rel=1e-6) == torch_readout
    assert pytest.approx(reference_logits, abs=1e-6, rel=1e-6) == torch_logits


def test_evaluate_predictions_returns_transparent_arrays() -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=4,
        num_classes=3,
        feature_qubits=1,
        quantum_layers=1,
    )
    reference = PennyLanePCSQCNN.from_torch_state(
        _make_reference_config(model),
        model.state_dict(),
    )
    images = torch.rand(5, 4, 4, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    evaluation_result = REFERENCE_MODULE.evaluate_predictions(reference, loader)

    assert evaluation_result["logits"].shape == (5, 3)
    assert evaluation_result["predicted_labels"].shape == (5,)
    assert evaluation_result["true_labels"].shape == (5,)
    assert evaluation_result["confusion_matrix"].shape == (3, 3)
    assert isinstance(evaluation_result["accuracy"], float)
    assert 0.0 <= evaluation_result["accuracy"] <= 1.0


def test_evaluate_predictions_accepts_progress_factory() -> None:
    torch.manual_seed(0)
    model = _make_torch_model(
        image_size=4,
        num_classes=3,
        feature_qubits=1,
        quantum_layers=1,
    )
    reference = PennyLanePCSQCNN.from_torch_state(
        _make_reference_config(model),
        model.state_dict(),
    )
    images = torch.rand(5, 4, 4, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    progress_calls: list[object] = []

    def progress_factory(batch_iterator):
        progress_calls.append(batch_iterator)
        for batch in batch_iterator:
            yield batch

    evaluation_result = REFERENCE_MODULE.evaluate_predictions(
        reference,
        loader,
        progress_factory=progress_factory,
    )

    assert progress_calls == [loader]
    assert evaluation_result["logits"].shape == (5, 3)
    assert evaluation_result["predicted_labels"].shape == (5,)
    assert evaluation_result["true_labels"].shape == (5,)
    assert evaluation_result["confusion_matrix"].shape == (3, 3)
    assert isinstance(evaluation_result["accuracy"], float)
    assert 0.0 <= evaluation_result["accuracy"] <= 1.0


def test_plot_confusion_matrix_builds_axis() -> None:
    confusion_ax = REFERENCE_MODULE.plot_confusion_matrix(
        np.array([[3, 1], [2, 4]], dtype=np.int64)
    )

    assert confusion_ax.get_title() == "Confusion Matrix"


def test_pennylane_reference_draw_mentions_explicit_circuit_stages() -> None:
    reference = PennyLanePCSQCNN(
        image_size=4,
        num_classes=2,
        feature_qubits=1,
        quantum_layers=2,
    )

    circuit_text = reference.draw(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ]
    )

    assert "|Ψ⟩" in circuit_text or "StatePrep" in circuit_text
    assert "QFT" in circuit_text
    assert "QFT†" in circuit_text or "Adjoint(QFT)" in circuit_text
    assert "QubitUnitary" in circuit_text or "U(" in circuit_text
    assert "●" in circuit_text or "Controlled" in circuit_text or "ctrl" in circuit_text.lower()
