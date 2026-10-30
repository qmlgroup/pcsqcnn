"""Helpers for formatting trainable parameter statistics by module."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from qcnn.henderson import HendersonBlockLayer2D
from qcnn.hur import HurConvolutionLayer, HurPoolingLayer
from qcnn.qcs import QCSIndexLayer2D
from qcnn.quantum import ModeMultiplexer2D

_QUANTUM_LAYER_TYPES = (
    ModeMultiplexer2D,
    HendersonBlockLayer2D,
    QCSIndexLayer2D,
    HurConvolutionLayer,
    HurPoolingLayer,
)


@dataclass(frozen=True)
class TrainableLayerParameterStat:
    """One trainable leaf-module entry in a formatted parameter summary."""

    kind: str
    layer_name: str
    parameter_count: int


def _module_kind(module: nn.Module) -> str:
    return (
        "q"
        if isinstance(module, _QUANTUM_LAYER_TYPES)
        or getattr(module, "_qcnn_quantum_parameter_module", False)
        else "c"
    )


def collect_trainable_layer_parameter_stats(model: nn.Module) -> list[TrainableLayerParameterStat]:
    """Return trainable direct-parameter counts for leaf modules in registration order."""

    stats: list[TrainableLayerParameterStat] = []
    for module_name, module in model.named_modules():
        parameter_count = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad
        )
        if parameter_count == 0:
            continue
        layer_name = module_name or "model"
        stats.append(
            TrainableLayerParameterStat(
                kind=_module_kind(module),
                layer_name=layer_name,
                parameter_count=parameter_count,
            )
        )
    return stats


def format_trainable_parameter_stats_line(model: nn.Module) -> str:
    """Format trainable per-layer counts and classical/quantum totals on one line."""

    stats = collect_trainable_layer_parameter_stats(model)
    quantum_total = sum(stat.parameter_count for stat in stats if stat.kind == "q")
    classical_total = sum(stat.parameter_count for stat in stats if stat.kind == "c")
    tokens = [f"{stat.kind}{stat.parameter_count}" for stat in stats]
    tokens.append(f"Q{quantum_total}")
    tokens.append(f"C{classical_total}")
    return " ".join(tokens)
