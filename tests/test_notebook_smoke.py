import importlib
import importlib.util
from pathlib import Path


def test_can_import_package_and_marimo_notebook() -> None:
    package = importlib.import_module("qcnn")
    project_root = Path(__file__).resolve().parents[1]

    for module_name, notebook_path in (
        ("train_qcnn", project_root / "notebooks" / "train_qcnn.py"),
        ("inspect_artifact", project_root / "notebooks" / "inspect_artifact.py"),
        (
            "inspect_pennylane_artifact",
            project_root / "pennylane" / "inspect_pennylane_artifact.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(module_name, notebook_path)
        assert spec is not None
        assert spec.loader is not None

        notebook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notebook)

        assert hasattr(notebook, "app")
    assert package.__name__ == "qcnn"


def test_legacy_pcsqcnn_artifact_surface_is_removed() -> None:
    package = importlib.import_module("qcnn")
    project_root = Path(__file__).resolve().parents[1]
    removed_symbols = (
        "save_" + "pcsqcnn_artifact",
        "load_" + "pcsqcnn_artifact",
        "Loaded" + "PCSQCNNArtifact",
        "from_" + "torch_artifact",
        "normalize_" + "training_history",
        "Functional" + "LossCollector",
        "Functional" + "MetricCollector",
        "MISSING_" + "PARAMETER_STATS_LINE",
    )

    assert not hasattr(package, removed_symbols[0])
    assert not hasattr(package, removed_symbols[1])
    assert not hasattr(package, removed_symbols[2])
    assert not hasattr(package, removed_symbols[4])
    assert not hasattr(package, removed_symbols[5])
    assert not hasattr(package, removed_symbols[6])
    assert not hasattr(package, removed_symbols[7])

    offenders: list[str] = []
    for path in project_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for symbol in removed_symbols:
            if symbol in text:
                offenders.append(f"{path.relative_to(project_root)}:{symbol}")

    assert offenders == []
