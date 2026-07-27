"""Defensive, read-only loading of existing Job Search Agent artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.services.run_state import JobArtifacts, RunSnapshot


class ArtifactLoadError(RuntimeError):
    """Raised when a selected run cannot be resolved safely."""


def repository_root() -> Path:
    """Resolve the repository root from this app-local module."""

    return Path(__file__).resolve().parents[2]


def discover_existing_runs(repo_root: Path | None = None) -> list[str]:
    """Return run names with complete/richer artifact sets first, then newest."""

    root = (repo_root or repository_root()).resolve()
    outputs = root / "outputs"
    if not outputs.is_dir():
        return []
    candidates = [
        path
        for path in outputs.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("run-")
            or path.name.startswith("job-search-live-")
            or path.name.startswith("ui-live-")
        )
    ]
    candidates.sort(
        key=lambda path: (_artifact_completeness(path), path.stat().st_mtime),
        reverse=True,
    )
    return [path.name for path in candidates]


def load_existing_run(
    run_id: str,
    repo_root: Path | None = None,
) -> RunSnapshot:
    """Load a prior run without mutating it or recomputing agent results."""

    root = (repo_root or repository_root()).resolve()
    run_dir = _safe_run_directory(root, run_id)
    manifest, manifest_present = _read_json_with_presence(
        run_dir / "run_manifest.json", {}
    )
    ranked_raw, ranked_present = _read_json_with_presence(
        run_dir / "ranked_jobs.json", []
    )
    rejected_raw, rejected_present = _read_json_with_presence(
        run_dir / "rejected_jobs.json", []
    )
    traces_raw, traces_present = _read_json_with_presence(
        run_dir / "trace_events.json", []
    )
    memory_raw, memory_present = _read_json_with_presence(run_dir / "memory.json", [])

    ranked_jobs = _unwrap_list(ranked_raw, "ranked_jobs")
    rejected_jobs = _unwrap_list(rejected_raw, "rejected_jobs")
    trace_events = _unwrap_list(traces_raw, "trace_events")
    memory_facts = memory_raw if isinstance(memory_raw, list) else []

    jobs_by_id, preferences, evidence_lookup, input_warnings = _load_repository_inputs(
        root
    )
    for item in ranked_jobs:
        job = item.get("job", item) if isinstance(item, dict) else {}
        if isinstance(job, dict) and job.get("job_id"):
            jobs_by_id[str(job["job_id"])] = job
    for item in rejected_jobs:
        job = item.get("job", {}) if isinstance(item, dict) else {}
        if isinstance(job, dict) and job.get("job_id"):
            jobs_by_id[str(job["job_id"])] = job

    top_ids, selection_source = _selected_ids(
        run_dir=run_dir,
        manifest=manifest if isinstance(manifest, dict) else {},
        ranked_raw=ranked_raw,
    )
    artifacts = {
        job_id: _load_job_artifacts(run_dir / job_id, job_id)
        for job_id in top_ids
    }
    for job_id, record in artifacts.items():
        job_details = _read_json(record.files.get("job_details"), {})
        if isinstance(job_details, dict) and job_details.get("job_id"):
            jobs_by_id[job_id] = job_details
        _add_job_evidence(evidence_lookup, jobs_by_id.get(job_id))

    fit_analyses = {
        job_id: record.fit_analysis
        for job_id, record in artifacts.items()
        if record.fit_analysis
    }
    review_decisions = {
        job_id: record.review_decision
        for job_id, record in artifacts.items()
        if record.review_decision
    }
    review_history = _dedupe_dicts(
        entry
        for record in artifacts.values()
        for entry in record.revision_history
        if isinstance(entry, dict)
    )
    cover_letter_results = {
        job_id: {
            "job_id": job_id,
            "output_pdf_path": str(path),
            "page_count": record.page_counts.get("cover_letter"),
        }
        for job_id, record in artifacts.items()
        if (path := record.path("cover_letter")) is not None
    }
    trace_url = _trace_url(run_dir, manifest)
    phase, status = _infer_existing_stage(
        manifest_present=manifest_present,
        artifacts=artifacts,
        review_decisions=review_decisions,
    )
    new_fact_ids, propagation = _review_memory_evidence(review_history)
    warnings = [*input_warnings]
    if selection_source == "artifact_folders":
        warnings.append(
            "Selected-job folders were found, but automatic Top 3 selection metadata "
            "is absent; these folders are not presented as ranking proof."
        )
    if not manifest_present:
        warnings.append(
            "This is an incomplete artifact set; final run checks remain unavailable."
        )

    jobs_loaded = len(jobs_by_id) if jobs_by_id else None
    return RunSnapshot(
        mode="Existing Run",
        repo_root=root,
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        phase=phase,
        read_only=True,
        jobs_loaded=jobs_loaded,
        filtered_jobs=[
            item.get("job", item)
            for item in ranked_jobs
            if isinstance(item, dict)
        ],
        rejected_jobs=rejected_jobs,
        ranked_jobs=ranked_jobs,
        top_3_job_ids=top_ids,
        jobs_by_id=jobs_by_id,
        preferences=preferences,
        fit_analyses=fit_analyses,
        artifacts=artifacts,
        memory_facts=memory_facts,
        new_memory_fact_ids=new_fact_ids,
        memory_file=(run_dir / "memory.json") if memory_present else None,
        memory_propagation=propagation,
        review_history=review_history,
        review_decisions=review_decisions,
        cover_letter_results=cover_letter_results,
        trace_events=trace_events,
        trace_id=_string_or_none(
            manifest.get("trace_id") if isinstance(manifest, dict) else None
        ),
        trace_url=trace_url,
        trace_public=manifest.get("trace_public") is True,
        trace_ingest_confirmed=manifest.get("trace_ingest_confirmed") is True,
        observation_count=int(manifest.get("observation_count") or 0),
        trace_export_error=_string_or_none(manifest.get("trace_export_error")),
        trace_debug_status=_string_or_none(manifest.get("trace_debug_status")),
        output_manifest=manifest if isinstance(manifest, dict) else {},
        evidence_lookup=evidence_lookup,
        scoring_weights=_scoring_weights(),
        top_selection_source=selection_source,
        evidence_available={
            "manifest": manifest_present,
            "filtering": ranked_present,
            "rejections": rejected_present,
            "ranking": ranked_present,
            "trace": traces_present,
            "memory": memory_present,
            "review": bool(review_decisions or review_history),
        },
        warnings=warnings,
    )


def snapshot_from_live_state(
    state: dict[str, Any],
    *,
    repo_root: Path | None = None,
    trace_events: list[dict[str, Any]] | None = None,
    trace_url: str | None = None,
) -> RunSnapshot:
    """Normalize an actual graph result for display without changing it."""

    root = (repo_root or repository_root()).resolve()
    run_id = str(state.get("run_id") or "live-run")
    run_dir_candidate = root / "outputs" / run_id
    run_dir = run_dir_candidate if run_dir_candidate.is_dir() else None
    interrupt_payload = _extract_interrupt_payload(state)
    ranked_jobs = list(state.get("ranked_jobs", []))
    rejected_jobs = list(state.get("rejected_jobs", []))
    top_ids = list(state.get("top_3_job_ids", []))
    jobs_by_id = {
        str(item["job_id"]): dict(item)
        for item in state.get("jobs", [])
        if isinstance(item, dict) and item.get("job_id")
    }
    preferences = dict(
        (state.get("candidate_profile") or {}).get("preferences", {})
    )
    evidence_lookup = _evidence_from_live_state(state)
    for job_id in top_ids:
        _add_job_evidence(evidence_lookup, jobs_by_id.get(job_id))

    artifacts: dict[str, JobArtifacts] = {}
    for job_id in top_ids:
        directory = (run_dir_candidate / job_id)
        record = (
            _load_job_artifacts(directory, job_id)
            if directory.is_dir()
            else JobArtifacts(job_id=job_id)
        )
        fit = (state.get("fit_analyses") or {}).get(job_id, {})
        tailoring = (state.get("tailoring_results") or {}).get(job_id, {})
        cover = (state.get("cover_letter_results") or {}).get(job_id, {})
        if isinstance(fit, dict) and fit:
            record.fit_analysis = fit
        if isinstance(tailoring, dict):
            record.tailoring_result = tailoring
            record.change_log = list(tailoring.get("change_log", record.change_log))
            _attach_declared_file(record, "resume_after", tailoring.get("output_pdf_path"))
            if tailoring.get("page_count") is not None:
                record.page_counts["resume_after"] = tailoring.get("page_count")
        if isinstance(cover, dict):
            record.cover_letter_result = cover
            _attach_declared_file(record, "cover_letter", cover.get("output_pdf_path"))
            if cover.get("page_count") is not None:
                record.page_counts["cover_letter"] = cover.get("page_count")
        artifacts[job_id] = record

    memory_file_value = state.get("memory_file")
    memory_file = Path(memory_file_value) if memory_file_value else None
    memory_facts = list(state.get("memory_facts", []))
    if not memory_facts and memory_file and memory_file.is_file():
        loaded = _read_json(memory_file, [])
        memory_facts = loaded if isinstance(loaded, list) else []

    review_history = list(state.get("review_history", []))
    derived_ids, derived_propagation = _review_memory_evidence(review_history)
    new_ids = list(state.get("new_memory_fact_ids", [])) or derived_ids
    propagation = list(state.get("memory_propagation_actions", []))
    if not propagation:
        propagation = derived_propagation
    status = str(state.get("status") or "RUNNING")
    phase = str(state.get("phase") or "INITIALIZE")
    if interrupt_payload and status == "WAITING_FOR_REVIEW":
        status = "WAITING_FOR_REVIEW"
        phase = "HUMAN_REVIEW"

    return RunSnapshot(
        mode="Live Run",
        repo_root=root,
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        phase=phase,
        read_only=False,
        jobs_loaded=len(state.get("jobs", [])) or None,
        filtered_jobs=list(state.get("filtered_jobs", [])),
        rejected_jobs=rejected_jobs,
        ranked_jobs=ranked_jobs,
        top_3_job_ids=top_ids,
        jobs_by_id=jobs_by_id,
        preferences=preferences,
        fit_analyses=dict(state.get("fit_analyses", {})),
        artifacts=artifacts,
        memory_facts=memory_facts,
        new_memory_fact_ids=new_ids,
        memory_file=memory_file if memory_file and memory_file.is_file() else None,
        memory_propagation=propagation,
        review_history=review_history,
        review_decisions=dict(state.get("review_decisions", {})),
        interrupt_payload=interrupt_payload,
        cover_letter_results=dict(state.get("cover_letter_results", {})),
        trace_events=trace_events or [],
        trace_id=_string_or_none(state.get("trace_id")),
        trace_url=(
            trace_url or _string_or_none(state.get("trace_url"))
            if state.get("trace_ingest_confirmed") is True
            else None
        ),
        trace_public=state.get("trace_public") is True,
        trace_ingest_confirmed=state.get("trace_ingest_confirmed") is True,
        observation_count=int(state.get("observation_count") or 0),
        trace_export_error=_string_or_none(state.get("trace_export_error")),
        trace_debug_status=_string_or_none(state.get("trace_debug_status")),
        agent_decisions=list(state.get("agent_decisions", [])),
        output_manifest=dict(state.get("output_manifest", {})),
        evidence_lookup=evidence_lookup,
        scoring_weights=_scoring_weights(),
        top_selection_source="live_state" if top_ids else None,
        evidence_available={
            "manifest": bool(state.get("output_manifest")),
            "filtering": "filtered_jobs" in state,
            "rejections": "rejected_jobs" in state,
            "ranking": bool(ranked_jobs),
            "trace": bool(trace_events),
            "memory": bool(memory_file and memory_file.is_file()),
            "review": bool(interrupt_payload or state.get("review_history")),
        },
        errors=list(state.get("errors", [])),
    )


def _safe_run_directory(root: Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ArtifactLoadError("Invalid run identifier.")
    outputs = (root / "outputs").resolve()
    candidate = (outputs / run_id).resolve()
    if candidate.parent != outputs or not candidate.is_dir():
        raise ArtifactLoadError(f"Existing run not found: {run_id}")
    return candidate


def _artifact_completeness(run_dir: Path) -> int:
    """Prefer the most useful real demo run without claiming it is complete."""

    score = 0
    if (run_dir / "run_manifest.json").is_file():
        score += 100
    job_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
    score += min(3, sum((path / "fit_analysis.json").is_file() for path in job_dirs)) * 3
    score += min(
        3,
        sum(
            (path / "resume_after.pdf").is_file()
            or (path / "resume_draft.pdf").is_file()
            for path in job_dirs
        ),
    ) * 5
    score += min(
        3, sum((path / "cover_letter.pdf").is_file() for path in job_dirs)
    ) * 7
    return score


def _read_json_with_presence(path: Path, default: Any) -> tuple[Any, bool]:
    if not path.is_file():
        return default, False
    return _read_json(path, default), True


def _read_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _unwrap_list(value: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get(key, [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _selected_ids(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    ranked_raw: Any,
) -> tuple[list[str], str | None]:
    manifest_ids = manifest.get("job_ids")
    if isinstance(manifest_ids, list) and manifest_ids:
        return [str(item) for item in manifest_ids[:3]], "manifest"
    if isinstance(ranked_raw, dict):
        explicit = ranked_raw.get("top_3_job_ids")
        if isinstance(explicit, list) and explicit:
            return [str(item) for item in explicit[:3]], "ranked_artifact"
    folders = sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir()
        and (
            (path / "fit_analysis.json").is_file()
            or (path / "resume_before.pdf").is_file()
        )
    )
    return folders[:3], "artifact_folders" if folders else None


def _load_job_artifacts(job_dir: Path, job_id: str) -> JobArtifacts:
    record = JobArtifacts(
        job_id=job_id,
        directory=job_dir if job_dir.is_dir() else None,
    )
    if not job_dir.is_dir():
        return record
    aliases = {
        "job_details": ("job_details.json",),
        "resume_before": ("resume_before.pdf",),
        "resume_after": ("resume_after.pdf", "resume_draft.pdf"),
        "resume_before_tex": ("resume_before.tex",),
        "resume_after_tex": ("resume_after.tex", "resume_draft.tex"),
        "cover_letter": ("cover_letter.pdf",),
        "fit_analysis_json": ("fit_analysis.json",),
        "fit_analysis_markdown": ("fit_analysis.md",),
        "change_log": ("change_log.json",),
        "review_decision": ("human_review_decision.json",),
        "revision_history": ("revision_history.json",),
    }
    for key, names in aliases.items():
        for name in names:
            path = job_dir / name
            if path.is_file():
                record.files[key] = path
                if key == "resume_after" and name == "resume_draft.pdf":
                    record.after_is_draft = True
                break
    for key in ("resume_before", "resume_after", "cover_letter"):
        record.page_counts[key] = _pdf_page_count(record.files.get(key))
    fit = _read_json(record.files.get("fit_analysis_json"), {})
    record.fit_analysis = fit if isinstance(fit, dict) else {}
    changes = _read_json(record.files.get("change_log"), [])
    record.change_log = changes if isinstance(changes, list) else []
    decision = _read_json(record.files.get("review_decision"), {})
    record.review_decision = decision if isinstance(decision, dict) else {}
    history = _read_json(record.files.get("revision_history"), [])
    record.revision_history = history if isinstance(history, list) else []
    return record


def _pdf_page_count(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def _load_repository_inputs(
    root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[str],
]:
    jobs_by_id: dict[str, dict[str, Any]] = {}
    preferences: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    try:
        from src.agent import (
            load_candidate_profile,
            load_jobs_csv,
            load_portfolio,
            load_resume_data,
        )

        jobs = load_jobs_csv(root / "data" / "jobs.csv")
        jobs_by_id = {item.job_id: item.model_dump() for item in jobs}
        profile = load_candidate_profile(root / "data" / "preferences.yaml")
        preferences = profile.preferences.model_dump()
        resume = load_resume_data(root / "data" / "resume.tex")
        portfolio = load_portfolio(root / "data" / "portfolio.txt")
        for item in [
            *profile.resume_evidence,
            *profile.master_skill_evidence,
            *profile.portfolio_evidence,
            *resume.evidence_items,
            *portfolio.evidence_items,
        ]:
            evidence[item.evidence_id] = item.model_dump()
    except Exception as exc:  # UI stays usable when optional input context is absent.
        warnings.append(f"Repository input context could not be loaded: {exc}")
    return jobs_by_id, preferences, evidence, warnings


def _add_job_evidence(
    lookup: dict[str, dict[str, Any]], job: dict[str, Any] | None
) -> None:
    if not job:
        return
    try:
        from src.agent import Job, build_job_evidence

        for item in build_job_evidence(Job.model_validate(job)):
            lookup[item.evidence_id] = item.model_dump()
    except Exception:
        return


def _evidence_from_live_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profile = state.get("candidate_profile") or {}
    portfolio = state.get("portfolio") or {}
    facts = state.get("memory_facts") or []
    items = [
        *(profile.get("resume_evidence") or []),
        *(profile.get("master_skill_evidence") or []),
        *(profile.get("portfolio_evidence") or []),
        *(portfolio.get("evidence_items") or []),
    ]
    lookup = {
        str(item["evidence_id"]): item
        for item in items
        if isinstance(item, dict) and item.get("evidence_id")
    }
    try:
        from src.review.memory import MemoryFact, memory_fact_to_evidence

        for raw in facts:
            item = memory_fact_to_evidence(MemoryFact.model_validate(raw))
            lookup[str(item["evidence_id"])] = item
    except Exception:
        pass
    return lookup


def _attach_declared_file(record: JobArtifacts, key: str, value: Any) -> None:
    if not value:
        return
    path = Path(str(value))
    if path.is_file():
        record.files[key] = path
        record.page_counts[key] = _pdf_page_count(path)


def _extract_interrupt_payload(state: dict[str, Any]) -> dict[str, Any]:
    explicit = state.get("interrupt_payload")
    if isinstance(explicit, dict) and explicit.get("resumes"):
        return explicit
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return {}
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    value = getattr(first, "value", first)
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, dict) and hasattr(value, "model_dump"):
        value = value.model_dump()
    return value if isinstance(value, dict) else {}


def _infer_existing_stage(
    *,
    manifest_present: bool,
    artifacts: dict[str, JobArtifacts],
    review_decisions: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if manifest_present:
        return "COMPLETE", "COMPLETED"
    if any(record.path("cover_letter") for record in artifacts.values()):
        return "COVER_LETTERS", "INCOMPLETE_ARTIFACTS"
    if review_decisions:
        return "REVISION", "INCOMPLETE_ARTIFACTS"
    if any(record.path("resume_after") for record in artifacts.values()):
        return "TAILOR", "INCOMPLETE_ARTIFACTS"
    if any(record.fit_analysis for record in artifacts.values()):
        return "FIT_ANALYSIS", "INCOMPLETE_ARTIFACTS"
    return "INITIALIZE", "INCOMPLETE_ARTIFACTS"


def _trace_url(run_dir: Path, manifest: dict[str, Any]) -> str | None:
    if manifest.get("trace_ingest_confirmed") is not True:
        return None
    value = manifest.get("trace_url") if isinstance(manifest, dict) else None
    if value and str(value).startswith(("https://", "http://")):
        return str(value)
    path = run_dir / "public_trace_url.txt"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text if text.startswith(("https://", "http://")) else None


def _review_memory_evidence(
    review_history: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    ids: list[str] = []
    actions: list[dict[str, Any]] = []
    for entry in review_history:
        for fact in entry.get("memory_writes", []):
            if isinstance(fact, dict) and fact.get("fact_id"):
                ids.append(str(fact["fact_id"]))
        raw_actions = entry.get("actions_taken", {})
        if isinstance(raw_actions, dict):
            for job_id, action in raw_actions.items():
                if isinstance(action, dict) and (
                    action.get("memory_fact_ids") or action.get("memory_evidence_ids")
                ):
                    actions.append({"job_id": job_id, **action})
    return list(dict.fromkeys(ids)), actions


def _scoring_weights() -> dict[str, float]:
    try:
        from src.tools.filtering_scoring.scoring import SCORING_WEIGHTS

        return dict(SCORING_WEIGHTS)
    except Exception:
        return {}


def _dedupe_dicts(items: Any) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _string_or_none(value: Any) -> str | None:
    return str(value) if value else None
