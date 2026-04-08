"""Generic serializable model specifications for qcnn training flows."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model-construction specification.

    Attributes:
        module: Import path of the Python module that exports the model class.
        class_name: Name of the class inside ``module``.
        constructor_kwargs: Keyword arguments passed to the constructor.
    """

    module: str
    class_name: str
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    """Convert a ``ModelSpec`` into a plain mapping."""

    return asdict(spec)


def model_spec_from_mapping(data: Mapping[str, Any]) -> ModelSpec:
    """Construct a ``ModelSpec`` from a mapping payload."""

    module = data.get("module")
    class_name = data.get("class_name")
    constructor_kwargs = data.get("constructor_kwargs", {})
    if not isinstance(module, str) or not module:
        raise ValueError("model_spec.module must be a non-empty string.")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("model_spec.class_name must be a non-empty string.")
    if not isinstance(constructor_kwargs, Mapping):
        raise ValueError("model_spec.constructor_kwargs must be a mapping.")
    return ModelSpec(
        module=module,
        class_name=class_name,
        constructor_kwargs=dict(constructor_kwargs),
    )


def resolve_model_class(spec: ModelSpec) -> type[nn.Module]:
    """Resolve the model class referenced by a ``ModelSpec``."""

    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise ValueError(f"Could not import model module {spec.module!r}.") from exc

    try:
        model_class = getattr(module, spec.class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Could not find model class {spec.class_name!r} in module {spec.module!r}."
        ) from exc

    if not isinstance(model_class, type):
        raise ValueError(
            f"Resolved model symbol {spec.module}.{spec.class_name} is not a class."
        )
    return model_class


def instantiate_model(spec: ModelSpec) -> nn.Module:
    """Instantiate a model from a ``ModelSpec`` and validate its type."""

    model_class = resolve_model_class(spec)
    model = model_class(**dict(spec.constructor_kwargs))
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Constructed object from {spec.module}.{spec.class_name} is not an nn.Module."
        )
    return model
