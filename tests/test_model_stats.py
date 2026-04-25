from qcnn.classic import ClassicalCNN, ClassicalMLP
from qcnn.hybrid import PCSQCNN, PCSQCNNNoQFT
from qcnn.model_stats import (
    collect_trainable_layer_parameter_stats,
    format_trainable_parameter_stats_line,
)


def test_format_trainable_parameter_stats_line_for_classical_mlp() -> None:
    model = ClassicalMLP(image_size=4, num_classes=10)

    assert format_trainable_parameter_stats_line(model) == "c799 c9024 c35532 c1890 Q0 C47245"


def test_format_trainable_parameter_stats_line_for_classical_cnn() -> None:
    model = ClassicalCNN(image_size=4, num_classes=10)

    assert format_trainable_parameter_stats_line(model) == "c160 c4640 c13872 c27712 c650 Q0 C47034"


def test_format_trainable_parameter_stats_line_for_pcsqcnn() -> None:
    model = PCSQCNN(image_size=4, num_classes=10, feature_qubits=1, quantum_layers=1)

    assert format_trainable_parameter_stats_line(model) == "q64 c330 Q64 C330"


def test_format_trainable_parameter_stats_line_for_pcsqcnn_no_qft() -> None:
    model = PCSQCNNNoQFT(image_size=4, num_classes=10, feature_qubits=1, quantum_layers=1)

    assert format_trainable_parameter_stats_line(model) == "q64 c330 Q64 C330"


def test_collect_trainable_layer_parameter_stats_skips_parameterless_layers() -> None:
    model = ClassicalMLP(image_size=4, num_classes=10)

    assert [stat.layer_name for stat in collect_trainable_layer_parameter_stats(model)] == [
        "input_layer",
        "expansion_layer",
        "hidden_layer",
        "classifier",
    ]
