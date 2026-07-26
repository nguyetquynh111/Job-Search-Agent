"""Controller helpers for starting and resuming job-search agent runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from langgraph.types import Command

from src.agent.state import AgentState, create_initial_state, state_value


class AgentController:
    """Small public controller around a compiled LangGraph workflow."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def invoke_new_run(
        self,
        state: AgentState | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a new graph run until the human-review pause."""

        return invoke_new_run(self.app, state=state, thread_id=thread_id)

    def resume_run(self, thread_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
        """Resume the graph with human-review feedback."""

        return resume_run(self.app, thread_id, feedback)


def create_sqlite_checkpointer(
    db_path: str | Path,
) -> tuple[Any, AbstractContextManager[Any]]:
    """Create a SQLite checkpointer and keep its context manager alive."""

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = SqliteSaver.from_conn_string(str(path))
    return context.__enter__(), context


def create_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer."""

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def invoke_new_run(
    app: Any,
    state: AgentState | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a new graph run until the human-review pause."""

    run_state = state or create_initial_state(thread_id=thread_id)
    config = {"configurable": {"thread_id": state_value(run_state, "thread_id")}}
    return app.invoke(run_state, config=config)


def resume_run(app: Any, thread_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
    """Resume the graph once with all review decisions."""

    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=feedback), config=config)


__all__ = [
    "AgentController",
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
    "invoke_new_run",
    "resume_run",
]
