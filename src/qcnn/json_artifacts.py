"""Exact JSON storage of saved results and figure payloads.

Training and evaluation scripts save their results with ``torch.save``. The
plot scripts need only the data in those files, so the published figure
inputs are stored as JSON with the encoding below and read back into the same
objects:

- tensors keep their dtype and shape, ``{"__tensor__": {"dtype", "shape", "data"}}``
  with the values flattened in row-major order;
- dataclass instances such as ``TrainingHistory`` are rebuilt from their
  fields, ``{"__dataclass__": "module.QualName", "fields": {...}}``;
- tuples stay tuples, ``{"__tuple__": [...]}``, and mappings with non-string
  keys keep their keys, ``{"__items__": [[key, value], ...]}``.

Floating-point values round-trip exactly: float32 values are stored as the
doubles that represent them, and non-finite values use the ``NaN`` and
``Infinity`` tokens of Python's ``json`` module.

``load_payload`` reads either format, so code that loads a ``.pt`` path also
accepts a ``.json`` export with the same stem in its place.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import importlib
import json
from pathlib import Path
from typing import Any

import torch

TENSOR_KEY = "__tensor__"
DATACLASS_KEY = "__dataclass__"
TUPLE_KEY = "__tuple__"
ITEMS_KEY = "__items__"
_TAG_KEYS = {TENSOR_KEY, DATACLASS_KEY, TUPLE_KEY, ITEMS_KEY}


def to_json_data(obj: Any) -> Any:
    """Convert ``obj`` into JSON-compatible data that ``from_json_data`` inverts exactly."""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        if tensor.is_complex():
            raise TypeError("Complex tensors are not supported by the JSON payload encoding.")
        return {
            TENSOR_KEY: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "data": tensor.reshape(-1).tolist(),
            }
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return {
            DATACLASS_KEY: f"{cls.__module__}.{cls.__qualname__}",
            "fields": {field.name: to_json_data(getattr(obj, field.name)) for field in dataclasses.fields(obj)},
        }
    if isinstance(obj, tuple):
        return {TUPLE_KEY: [to_json_data(value) for value in obj]}
    if isinstance(obj, list):
        return [to_json_data(value) for value in obj]
    if isinstance(obj, Mapping):
        if all(isinstance(key, str) for key in obj) and not _TAG_KEYS & set(obj):
            return {key: to_json_data(value) for key, value in obj.items()}
        return {ITEMS_KEY: [[to_json_data(key), to_json_data(value)] for key, value in obj.items()]}
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot encode object of type {type(obj).__module__}.{type(obj).__qualname__} as JSON payload data.")


def _resolve_class(qualified_name: str) -> type:
    module_name, _, qualname = qualified_name.rpartition(".")
    # Nested qualnames (Outer.Inner) are resolved attribute by attribute from the longest importable module.
    parts = qualified_name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            target: Any = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            target = getattr(target, attribute)
        if not (isinstance(target, type) and dataclasses.is_dataclass(target)):
            raise TypeError(f"{qualified_name} is not a dataclass.")
        return target
    raise ImportError(f"Cannot import {qualified_name} (module {module_name!r}, name {qualname!r}).")


def from_json_data(data: Any) -> Any:
    """Invert ``to_json_data``."""

    if isinstance(data, list):
        return [from_json_data(value) for value in data]
    if not isinstance(data, dict):
        return data
    if TENSOR_KEY in data:
        spec = data[TENSOR_KEY]
        dtype = getattr(torch, spec["dtype"])
        return torch.tensor(spec["data"], dtype=dtype).reshape(spec["shape"])
    if DATACLASS_KEY in data:
        cls = _resolve_class(data[DATACLASS_KEY])
        return cls(**{name: from_json_data(value) for name, value in data["fields"].items()})
    if TUPLE_KEY in data:
        return tuple(from_json_data(value) for value in data[TUPLE_KEY])
    if ITEMS_KEY in data:
        return {from_json_data(key): from_json_data(value) for key, value in data[ITEMS_KEY]}
    return {key: from_json_data(value) for key, value in data.items()}


def save_json_payload(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` to ``path`` in the JSON payload encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_data(obj), separators=(",", ":")), encoding="utf-8")
    return path


def load_json_payload(path: str | Path) -> Any:
    """Read an object written by ``save_json_payload``."""

    return from_json_data(json.loads(Path(path).read_text(encoding="utf-8")))


def resolve_payload_path(path: str | Path) -> Path:
    """Return ``path``, or its ``.json`` export when only the export exists."""

    path = Path(path)
    if not path.exists() and path.suffix != ".json":
        json_path = path.with_suffix(".json")
        if json_path.exists():
            return json_path
    return path


def load_payload(path: str | Path) -> Any:
    """Load a payload saved with ``torch.save`` or its JSON export in the same place."""

    resolved = resolve_payload_path(path)
    if resolved.suffix == ".json":
        return load_json_payload(resolved)
    return torch.load(resolved, map_location="cpu", weights_only=False)
