"""Validation-based successive-halving search over learning rate and initialization.

The search is shared by the baseline tuning scripts in ``run/``. Selection
never reads the test split:

- the seed-``search_seed`` translated-MNIST training set is split into
  training and ``validation_per_class`` validation images per class with
  ``stratified_validation_split``;
- the grid is Adam learning rate times the upper endpoint of the uniform
  initialization interval of the quantum parameters;
- successive halving trains the best ``rung_survivors[r]`` configurations of
  the previous rung (all configurations at the first rung) to
  ``rung_epochs[r]`` epochs, ranking by the mean validation accuracy of the
  last ``selection_window`` evaluations (every ``evaluation_interval_epochs``).

Every configuration is checkpointed after each rung (model, optimizer, data
order, history), so an interrupted search resumes where it stopped.
``search_summary.json`` is written last and marks the search as complete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import torch
from torch.utils.data import DataLoader

from qcnn.article_training import ArticleTrainingDefaults
from qcnn.data import prepare_mnist_splits, stratified_validation_split
from qcnn.model import ImageClassifierRunner
from qcnn.model_spec import ModelSpec, instantiate_model


@dataclass(frozen=True)
class SearchDefaults(ArticleTrainingDefaults):
    """Translated-MNIST data, grid, and successive-halving schedule of a search."""

    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    train_batch_size: int = 256
    validation_batch_size: int = 1600
    search_seed: int = 0
    validation_per_class: int = 100
    learning_rates: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
    init_scales: tuple[float, ...] = (0.05, 0.5, 2.0 * math.pi)
    rung_epochs: tuple[int, ...] = (300, 1000, 2000)
    rung_survivors: tuple[int, ...] = (12, 4, 2)
    evaluation_interval_epochs: int = 10
    selection_window: int = 5
    output_directory: str = ""


@dataclass(frozen=True)
class SearchConfig:
    """One point of the learning-rate and initialization-scale grid."""

    learning_rate: float
    init_scale: float

    @property
    def name(self) -> str:
        return f"lr{self.learning_rate:g}_init{self.init_scale:.4g}"


def search_grid(defaults: SearchDefaults) -> tuple[SearchConfig, ...]:
    return tuple(
        SearchConfig(learning_rate=learning_rate, init_scale=init_scale)
        for learning_rate in defaults.learning_rates
        for init_scale in defaults.init_scales
    )


def selection_score(validation_accuracy: Sequence[float], *, window: int) -> float:
    """Mean of the last ``window`` validation accuracies."""

    if not validation_accuracy:
        raise ValueError("validation_accuracy must not be empty.")
    return statistics.fmean(validation_accuracy[-window:])


def successive_halving(
    configs: Sequence[SearchConfig],
    *,
    rung_epochs: Sequence[int],
    rung_survivors: Sequence[int],
    train_to: Callable[[SearchConfig, int], Sequence[float]],
    window: int,
) -> tuple[list[dict[str, object]], SearchConfig]:
    """Run successive halving and return the per-rung rankings and the winner.

    ``train_to(config, epochs)`` must train ``config`` until it has seen
    ``epochs`` epochs in total and return its full validation-accuracy history.
    Rung ``r`` trains the best ``rung_survivors[r]`` configurations of the
    previous rung (all configurations at the first rung) to
    ``rung_epochs[r]`` epochs.
    """

    if len(rung_epochs) != len(rung_survivors) or not rung_epochs:
        raise ValueError("rung_epochs and rung_survivors must be non-empty and of equal length.")
    if list(rung_epochs) != sorted(set(rung_epochs)):
        raise ValueError(f"rung_epochs must be strictly increasing, got {list(rung_epochs)}.")
    if list(rung_survivors) != sorted(rung_survivors, reverse=True) or rung_survivors[-1] < 1:
        raise ValueError(f"rung_survivors must be positive and non-increasing, got {list(rung_survivors)}.")
    if rung_survivors[0] > len(configs):
        raise ValueError(f"The first rung keeps {rung_survivors[0]} of only {len(configs)} configurations.")

    ranking = list(configs)
    rungs: list[dict[str, object]] = []
    for epochs, survivors in zip(rung_epochs, rung_survivors, strict=True):
        contestants = ranking[:survivors]
        scores = {config: selection_score(train_to(config, epochs), window=window) for config in contestants}
        ranking = sorted(contestants, key=lambda config: scores[config], reverse=True)
        rungs.append(
            {
                "epochs": epochs,
                "ranking": [config.name for config in ranking],
                "scores": {config.name: scores[config] for config in ranking},
            }
        )
    return rungs, ranking[0]


class ResumableSearchTrainer:
    """Train search configurations incrementally with per-configuration checkpoints.

    ``build_model_spec(config)`` returns the model spec of one grid point; the
    initialization scale must already be applied to it.
    """

    def __init__(
        self,
        defaults: SearchDefaults,
        *,
        build_model_spec: Callable[[SearchConfig], ModelSpec],
        state_directory: Path,
        device: torch.device,
        log: Callable[[str], None] = print,
    ) -> None:
        self.defaults = defaults
        self.build_model_spec = build_model_spec
        self.state_directory = state_directory
        self.device = device
        self.log = log
        splits = prepare_mnist_splits(
            root=defaults.data_root,
            samples_per_class=defaults.samples_per_class,
            image_size=defaults.image_size,
            scaled_image_size=defaults.scaled_image_size,
            max_offset=defaults.max_offset,
            seed=defaults.search_seed,
            download=defaults.download,
        )
        training, validation = stratified_validation_split(
            splits.train,
            validation_per_class=defaults.validation_per_class,
            seed=defaults.search_seed,
        )
        self.training = training.to(device)
        self.validation = validation.to(device)
        self.histories: dict[str, dict[str, list[float]]] = {}

    def _state_path(self, config: SearchConfig) -> Path:
        return self.state_directory / f"{config.name}.pt"

    def train_to(self, config: SearchConfig, epochs: int) -> list[float]:
        defaults = self.defaults
        torch.manual_seed(defaults.search_seed)
        model = instantiate_model(self.build_model_spec(config)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator()
        generator.manual_seed(defaults.search_seed)
        history: dict[str, list[float]] = {
            "epoch": [],
            "validation_accuracy": [],
            "validation_loss": [],
            "train_accuracy": [],
            "train_loss": [],
        }
        completed = 0
        path = self._state_path(config)
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["generator"])
            history = state["history"]
            completed = state["epochs"]

        if completed < epochs:
            started = time.perf_counter()
            runner = ImageClassifierRunner(model=model, optimizer=optimizer, device=self.device)
            fitted = runner.fit(
                DataLoader(self.training, batch_size=defaults.train_batch_size, shuffle=True, generator=generator),
                DataLoader(self.validation, batch_size=defaults.validation_batch_size, shuffle=False),
                num_epochs=epochs - completed,
                test_evaluation_interval_epochs=defaults.evaluation_interval_epochs,
            )
            history["epoch"].extend(completed + epoch for epoch in fitted.test_epoch)
            history["validation_accuracy"].extend(fitted.test_metrics["accuracy"])
            history["validation_loss"].extend(fitted.test_loss)
            history["train_accuracy"].extend(fitted.train_metrics["accuracy"])
            history["train_loss"].extend(fitted.train_loss)
            self.state_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "history": history,
                    "epochs": epochs,
                    "learning_rate": config.learning_rate,
                    "init_scale": config.init_scale,
                },
                path,
            )
            score = selection_score(history["validation_accuracy"], window=defaults.selection_window)
            self.log(
                f"{time.strftime('%H:%M')} {config.name}: epochs {completed}->{epochs} "
                f"in {time.perf_counter() - started:.0f}s, validation score {score:.4f}, "
                f"train acc {history['train_accuracy'][-1]:.4f}"
            )
        self.histories[config.name] = history
        # A resumed configuration may already be past this rung; score it on the rung's epochs only.
        return [
            accuracy
            for epoch, accuracy in zip(history["epoch"], history["validation_accuracy"], strict=True)
            if epoch <= epochs
        ]


def run_search(
    defaults: SearchDefaults,
    *,
    build_model_spec: Callable[[SearchConfig], ModelSpec],
    model_fields: Mapping[str, object],
    rebuild: bool = False,
) -> None:
    """Run (or resume) the search and write ``search_summary.json``.

    ``model_fields`` describes the searched model in the summary.
    """

    output_directory = Path(defaults.artifacts_root) / defaults.output_directory
    summary_path = output_directory / "search_summary.json"
    if rebuild and output_directory.exists():
        shutil.rmtree(output_directory)
    if summary_path.is_file():
        print(f"Search already complete: {summary_path}")
        return

    device = torch.device(defaults.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if defaults.torch_matmul_precision is not None:
        torch.set_float32_matmul_precision(defaults.torch_matmul_precision)
    trainer = ResumableSearchTrainer(
        defaults,
        build_model_spec=build_model_spec,
        state_directory=output_directory / "states",
        device=device,
        log=lambda line: print(line, flush=True),
    )
    rungs, selected = successive_halving(
        search_grid(defaults),
        rung_epochs=defaults.rung_epochs,
        rung_survivors=defaults.rung_survivors,
        train_to=trainer.train_to,
        window=defaults.selection_window,
    )
    summary = {
        **model_fields,
        "search_seed": defaults.search_seed,
        "validation_per_class": defaults.validation_per_class,
        "learning_rates": list(defaults.learning_rates),
        "init_scales": list(defaults.init_scales),
        "rung_epochs": list(defaults.rung_epochs),
        "rung_survivors": list(defaults.rung_survivors),
        "selection_window": defaults.selection_window,
        "rungs": rungs,
        "selected": {
            "name": selected.name,
            "learning_rate": selected.learning_rate,
            "init_scale": selected.init_scale,
        },
        "histories": trainer.histories,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Selected {selected.name}; summary written to {summary_path}", flush=True)
