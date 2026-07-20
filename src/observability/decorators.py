"""Reusable observability decorators."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from src.observability.trace_manager import TraceManager

F = TypeVar("F", bound=Callable[..., Any])


def traced_tool(name: str, tracer: TraceManager) -> Callable[[F], F]:
    """Wrap a tool function in a TraceManager span."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(name, {"tool_name": name}):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
