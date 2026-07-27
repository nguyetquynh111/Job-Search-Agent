"""Presentation-safe run state used by the Streamlit control center.

The models in this module contain no workflow logic. They normalize either the
existing agent state or durable repository artifacts for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NOT_AVAILABLE = "Not available in the selected run"


@dataclass
class JobArtifacts:
    """Real files and structured metadata available for one selected job."""

    job_id: str
    directory: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)
    page_counts: dict[str, int | None] = field(default_factory=dict)
    fit_analysis: dict[str, Any] = field(default_factory=dict)
    change_log: list[dict[str, Any]] = field(default_factory=list)
    review_decision: dict[str, Any] = field(default_factory=dict)
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    tailoring_result: dict[str, Any] = field(default_factory=dict)
    cover_letter_result: dict[str, Any] = field(default_factory=dict)
    after_is_draft: bool = False

    def path(self, key: str) -> Path | None:
        """Return an existing artifact path, never a speculative path."""

        path = self.files.get(key)
        return path if path is not None and path.is_file() else None


@dataclass
class RunSnapshot:
    """Evidence-aware view of a live or existing agent run."""

    mode: str
    repo_root: Path
    run_id: str
    run_dir: Path | None
    status: str
    phase: str
    read_only: bool
    jobs_loaded: int | None = None
    filtered_jobs: list[dict[str, Any]] = field(default_factory=list)
    rejected_jobs: list[dict[str, Any]] = field(default_factory=list)
    ranked_jobs: list[dict[str, Any]] = field(default_factory=list)
    top_3_job_ids: list[str] = field(default_factory=list)
    jobs_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    fit_analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, JobArtifacts] = field(default_factory=dict)
    memory_facts: list[dict[str, Any]] = field(default_factory=list)
    new_memory_fact_ids: list[str] = field(default_factory=list)
    memory_file: Path | None = None
    memory_propagation: list[dict[str, Any]] = field(default_factory=list)
    review_history: list[dict[str, Any]] = field(default_factory=list)
    review_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    interrupt_payload: dict[str, Any] = field(default_factory=dict)
    cover_letter_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    trace_url: str | None = None
    trace_public: bool = False
    trace_ingest_confirmed: bool = False
    observation_count: int = 0
    trace_export_error: str | None = None
    trace_debug_status: str | None = None
    agent_decisions: list[dict[str, Any]] = field(default_factory=list)
    output_manifest: dict[str, Any] = field(default_factory=dict)
    evidence_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    scoring_weights: dict[str, float] = field(default_factory=dict)
    top_selection_source: str | None = None
    evidence_available: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted_count(self) -> int | None:
        """Return a count only when accepted-job evidence exists."""

        if self.evidence_available.get("filtering"):
            return len(self.filtered_jobs or self.ranked_jobs)
        return None

    @property
    def rejected_count(self) -> int | None:
        """Return a count only when rejection evidence exists."""

        if self.evidence_available.get("rejections"):
            return len(self.rejected_jobs)
        return None

    @property
    def human_pause_used(self) -> int | None:
        """Return pause use only when a real interrupt or review record exists."""

        if self.interrupt_payload or self.review_history or self.review_decisions:
            return 1
        if self.status == "COMPLETED" and self.output_manifest:
            return 1
        return None

    @property
    def is_waiting_for_review(self) -> bool:
        """Whether the live graph is at its real LangGraph interrupt."""

        return (
            not self.read_only
            and self.status == "WAITING_FOR_REVIEW"
            and bool(self.interrupt_payload.get("resumes"))
        )

    @property
    def trace_status(self) -> str:
        if self.trace_ingest_confirmed and self.trace_url:
            return f"Confirmed ({self.observation_count} observations)"
        if self.trace_export_error:
            return "Tracing export error"
        if self.trace_events:
            return f"{len(self.trace_events)} local events"
        return NOT_AVAILABLE

    def job(self, job_id: str) -> dict[str, Any]:
        """Return real job details from the run or repository input catalog."""

        return self.jobs_by_id.get(job_id, {"job_id": job_id})

    def artifact(self, job_id: str) -> JobArtifacts:
        """Return an empty artifact record rather than raising in the UI."""

        return self.artifacts.get(job_id, JobArtifacts(job_id=job_id))

    def fact_is_new_for_run(self, fact: dict[str, Any]) -> bool:
        """Whether durable run evidence attributes a fact to this review."""

        return str(fact.get("fact_id", "")) in set(self.new_memory_fact_ids)


def display_value(value: Any, *, fallback: str = NOT_AVAILABLE) -> str:
    """Render missing values consistently without inventing placeholders."""

    if value is None or value == "" or value == [] or value == {}:
        return fallback
    return str(value)
