"""Cached graph resources for Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.config import get_config


@dataclass
class GraphBundle:
    """Long-lived graph resources for a Streamlit process."""

    app: object
    tracer: object
    checkpointer_context: object


@st.cache_resource(show_spinner=False)
def get_graph_bundle(checkpoint_db: str) -> GraphBundle:
    """Build and cache the graph, SQLite checkpointer, and tracer."""

    # Import the workflow stack lazily. This lets the setup/upload page render
    # even in a partially installed development environment.
    from src.agent.graph import build_agent_graph, create_sqlite_checkpointer
    from src.observability.trace_manager import TraceManager
    from src.tools.registry import load_tool_registry

    tracer = TraceManager()
    tools = load_tool_registry()
    checkpointer, context = create_sqlite_checkpointer(checkpoint_db)
    app = build_agent_graph(tools=tools, checkpointer=checkpointer, tracer=tracer)
    return GraphBundle(app=app, tracer=tracer, checkpointer_context=context)


def configured_graph_bundle() -> GraphBundle:
    """Return the graph bundle for current environment settings."""

    config = get_config()
    return get_graph_bundle(str(config.checkpoint_db))
