import inspect

from qcnn import article, classic, data, hybrid, layout, model, quantum, serialization, visualization


def test_qcnn_modules_expose_nontrivial_module_docstrings() -> None:
    modules = [article, classic, data, hybrid, layout, model, quantum, serialization, visualization]

    for module in modules:
        doc = inspect.getdoc(module)

        assert doc is not None
        assert len(doc) > 120

def test_critical_internal_helpers_document_contract_markers() -> None:
    helper_markers = {
        layout._move_one_active_x_qubit_to_condition: ("Contract:", "Formula:", "most-significant"),
        layout._move_one_active_y_qubit_to_condition: ("Contract:", "Formula:", "most-significant"),
        quantum._apply_qft_2d: ("Contract:", "Formula:", "FFT"),
        quantum._apply_iqft_2d: ("Contract:", "Formula:", "inverse"),
        quantum._apply_fourier_junction_1d: ("Contract:", "Formula:", "IQFT"),
        quantum._apply_fourier_junction_2d: (
            "Returns:",
            "move_active_qubit_to_condition",
            "Notes:",
        ),
    }

    for helper, markers in helper_markers.items():
        doc = inspect.getdoc(helper)

        assert doc is not None, helper.__name__
        for marker in markers:
            assert marker in doc, helper.__name__
