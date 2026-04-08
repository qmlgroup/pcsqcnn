from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def load_run_script(module_name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
