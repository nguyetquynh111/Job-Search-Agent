"""Agent orchestration exceptions."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool returns an invalid or incomplete artifact."""


__all__ = ["ToolExecutionError"]
