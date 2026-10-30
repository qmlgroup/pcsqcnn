from __future__ import annotations

import math

from qcnn import HurQCNN
from qcnn.model_spec import instantiate_model
from qcnn.tuning import SearchConfig, search_grid
from tests.script_loading import load_run_script


tune = load_run_script("tune_hur_qcnn_translated_mnist_module", "tune_hur_qcnn_translated_mnist.py")


def test_default_search_matches_the_qcsqcnn_procedure() -> None:
    qcs_tune = load_run_script("tune_qcsqcnn_translated_mnist_module_for_hur", "tune_qcsqcnn_translated_mnist.py")
    defaults = tune.HurQcnnTuningDefaults()
    qcs_defaults = qcs_tune.QcsQcnnTuningDefaults()

    assert search_grid(defaults) == search_grid(qcs_defaults)
    assert len(search_grid(defaults)) == 12
    for field in ("validation_per_class", "rung_epochs", "rung_survivors", "selection_window", "train_batch_size"):
        assert getattr(defaults, field) == getattr(qcs_defaults, field)
    assert defaults.output_directory == "hur_qcnn_tuning/su4_l1"


def test_model_spec_applies_the_grid_initialization_scale() -> None:
    defaults = tune.HurQcnnTuningDefaults()

    spec = tune.build_model_spec(defaults, SearchConfig(learning_rate=1e-3, init_scale=0.05))
    model = instantiate_model(spec)

    assert isinstance(model, HurQCNN)
    assert spec.constructor_kwargs["angle_init_scale"] == 0.05
    assert spec.constructor_kwargs["filters"] == 1 and spec.constructor_kwargs["image_size"] == 32
    assert max(value for layer in model.layers for value in layer.angles.flatten().tolist()) < 0.05
    assert math.isclose(defaults.init_scales[-1], 2 * math.pi)
