"""Durable candidate-memory evidence."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot


def render_memory(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">Review side effect</div><h2>Memory update and propagation</h2></div>
          <span class="source-tag">JSONMemoryStore</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    destination = _display_path(snapshot)
    st.markdown(
        f"""
        <div class="memory-summary">
          <div><span>Memory destination</span><strong>{escape(destination)}</strong></div>
          <div><span>Stored facts</span><strong>{len(snapshot.memory_facts) if snapshot.evidence_available.get("memory") else "Unavailable"}</strong></div>
          <div><span>New in this run</span><strong>{len(snapshot.new_memory_fact_ids) if snapshot.new_memory_fact_ids else "Not evidenced"}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not snapshot.evidence_available.get("memory"):
        st.info(NOT_AVAILABLE)
        return
    if not snapshot.memory_facts:
        st.info("The selected memory artifact is an empty JSON array.")
        return
    for fact in snapshot.memory_facts:
        _fact_card(fact, snapshot)
    with st.expander("Raw memory artifact", expanded=False):
        st.json(snapshot.memory_facts, expanded=1)


def _fact_card(fact: dict[str, Any], snapshot: RunSnapshot) -> None:
    provenance = fact.get("provenance", {})
    fact_id = str(fact.get("fact_id") or "Fact ID unavailable")
    is_new = snapshot.fact_is_new_for_run(fact)
    badge = "New in selected run" if is_new else "Stored memory"
    tone = "new" if is_new else "stored"
    propagation = _propagation_targets(fact_id, snapshot.memory_propagation)
    st.markdown(
        f"""
        <article class="memory-card">
          <div class="memory-card__header">
            <span class="memory-badge memory-badge--{tone}">{escape(badge)}</span>
            <span class="id-tag">{escape(fact_id)}</span>
          </div>
          <h3>{escape(str(fact.get("canonical_value") or NOT_AVAILABLE))}</h3>
          <div class="memory-card__grid">
            <div><span>Fact type</span><strong>{escape(str(fact.get("fact_type") or NOT_AVAILABLE))}</strong></div>
            <div><span>Provenance</span><strong>{escape(str(provenance.get("source") or NOT_AVAILABLE))}</strong></div>
            <div><span>Review round</span><strong>{escape(str(provenance.get("review_round") or NOT_AVAILABLE))}</strong></div>
            <div><span>Timestamp</span><strong>{escape(str(fact.get("created_at") or NOT_AVAILABLE))}</strong></div>
          </div>
          <p><strong>Original statement:</strong> {escape(str(provenance.get("original_statement") or NOT_AVAILABLE))}</p>
          <p><strong>Applied to other Top 3 resumes:</strong> {escape(", ".join(propagation) if propagation else NOT_AVAILABLE)}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _propagation_targets(
    fact_id: str, actions: list[dict[str, Any]]
) -> list[str]:
    targets: list[str] = []
    for action in actions:
        ids = [
            *action.get("memory_fact_ids", []),
            *action.get("memory_evidence_ids", []),
            *action.get("new_memory_fact_ids", []),
        ]
        if fact_id in ids and action.get("job_id"):
            targets.append(str(action["job_id"]))
    return list(dict.fromkeys(targets))


def _display_path(snapshot: RunSnapshot) -> str:
    if snapshot.memory_file is None:
        return NOT_AVAILABLE
    try:
        return str(snapshot.memory_file.resolve().relative_to(snapshot.repo_root))
    except (OSError, ValueError):
        return str(snapshot.memory_file)

