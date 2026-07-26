"""Public access to the five model-visible tool modules.

Tool modules are loaded lazily so importing one tool does not initialize every
implementation or create circular imports between shared analysis helpers.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from src.tools.modules import TOOL_MODULES

__all__ = list(TOOL_MODULES)


def __getattr__(name: str) -> ModuleType:
    """Resolve a public tool name to its implementation package."""

    module_path = TOOL_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    globals()[name] = module
    return module
