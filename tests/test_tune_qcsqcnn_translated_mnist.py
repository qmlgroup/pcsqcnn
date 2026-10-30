from __future__ import annotations

import math

import pytest

from tests.script_loading import load_run_script


tune = load_run_script("tune_qcsqcnn_translated_mnist_module", "tune_qcsqcnn_translated_mnist.py")


def test_default_grid_covers_learning_rates_and_init_scales() -> None:
    defaults = tune.QcsQcnnTuningDefaults()

    grid = tune.search_grid(defaults)

    assert len(grid) == 12
    assert {config.learning_rate for config in grid} == {1e-3, 3e-3, 1e-2, 3e-2}
    assert {config.init_scale for config in grid} == {0.05, 0.5, 2.0 * math.pi}
    assert defaults.sublayers == 8 and defaults.validation_per_class == 100
    assert defaults.rung_epochs == (300, 1000, 2000) and defaults.rung_survivors == (12, 4, 2)


def test_successive_halving_promotes_the_best_configurations() -> None:
    configs = [tune.SearchConfig(learning_rate=rate, init_scale=1.0) for rate in (1.0, 2.0, 3.0, 4.0)]
    calls: list[tuple[float, int]] = []

    def train_to(config, epochs):
        calls.append((config.learning_rate, epochs))
        # Larger learning rates look better early, but the runner-up overtakes at the last rung.
        if epochs == 3 and config.learning_rate == 3.0:
            return [0.0, 0.99]
        return [0.0, config.learning_rate / 10]

    rungs, selected = tune.successive_halving(
        configs,
        rung_epochs=(1, 2, 3),
        rung_survivors=(4, 2, 2),
        train_to=train_to,
        window=1,
    )

    assert calls == [(1.0, 1), (2.0, 1), (3.0, 1), (4.0, 1), (4.0, 2), (3.0, 2), (4.0, 3), (3.0, 3)]
    assert [rung["ranking"] for rung in rungs][1] == ["lr4_init1", "lr3_init1"]
    assert rungs[2]["ranking"] == ["lr3_init1", "lr4_init1"]
    assert selected.learning_rate == 3.0


def test_selection_score_averages_the_last_window() -> None:
    assert tune.selection_score([0.1, 0.2, 0.3, 0.5], window=2) == pytest.approx(0.4)


def test_successive_halving_validates_the_schedule() -> None:
    configs = [tune.SearchConfig(learning_rate=1.0, init_scale=1.0)]

    with pytest.raises(ValueError, match="strictly increasing"):
        tune.successive_halving(configs, rung_epochs=(2, 1), rung_survivors=(1, 1), train_to=None, window=1)
    with pytest.raises(ValueError, match="only 1 configurations"):
        tune.successive_halving(configs, rung_epochs=(1,), rung_survivors=(2,), train_to=None, window=1)
