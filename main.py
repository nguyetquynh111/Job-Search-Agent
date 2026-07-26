#!/usr/bin/env python3
"""Run the production E2E workflow with live tool selection and Langfuse.

This script intentionally fails instead of falling back to the deterministic
offline selector when live model or connected tracing credentials are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import (  # noqa: E402
    build_agent_graph,
    create_initial_state,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.tool_selection import DeepInfraToolSelectionModel  # noqa: E402
from src.review.memory import (  # noqa: E402
    JSONMemoryStore,
    MemoryFact,
    MemoryProvenance,
)
from src.tracing.langfuse import TraceManager  # noqa: E402


def main() -> int:
    args = _parse_args()
    _load_dotenv(REPO_ROOT / ".env")
    run_id = f"job-search-live-{_timestamp()}"
    output_dir = _resolve_output_dir(args.output_dir)
    os.environ["OUTPUT_DIR"] = str(output_dir)

    missing = _missing_live_environment()
    model_config = _model_config()
    langfuse_config = _langfuse_config(missing)
    command = " ".join(["python", "scripts/run_production_e2e.py", *sys.argv[1:]])
    if missing:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_MISSING_LIVE_CONFIGURATION",
                    "missing_environment_variables": missing,
                    "model_configuration": model_config,
                    "langfuse_configuration_status": langfuse_config,
                    "e2e_command": command,
                },
                indent=2,
            )
        )
        return 2

    try:
        resume_pdf = _compile_original_resume(output_dir / run_id / "source_resume")
        final, tracer = _run_workflow(
            run_id=run_id,
            output_dir=output_dir,
            reject_job_id=args.reject_job_id,
        )
        summary = _build_summary(
            final=final,
            tracer=tracer,
            run_id=run_id,
            resume_pdf=resume_pdf,
            command=command,
            model_config=model_config,
            langfuse_config=_langfuse_config([]),
        )
        _remove_source_resume_directory(resume_pdf.parent)
        _remove_non_submission_output_artifacts(output_dir, run_id)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:  # noqa: BLE001 - produce a durable failure summary
        failure = {
            "status": "FAILED",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "run_id": run_id,
            "model_configuration": model_config,
            "langfuse_configuration_status": _langfuse_config([]),
            "e2e_command": command,
        }
        output_dir.joinpath(run_id).mkdir(parents=True, exist_ok=True)
        output_dir.joinpath(run_id, "live_e2e_failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False, default=str))
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    parser.add_argument("--reject-job-id", default="J028")
    return parser.parse_args()


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resolve_output_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _missing_live_environment() -> list[str]:
    required = (
        "LLM_MODEL",
        "DEEPINFRA_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    )
    return [name for name in required if not os.getenv(name, "").strip()]


def _model_config() -> dict[str, Any]:
    return {
        "selector": "DeepInfraToolSelectionModel",
        "model": os.getenv("LLM_MODEL", ""),
        "base_url": os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),
        "temperature": 0,
        "api_key_present": bool(os.getenv("DEEPINFRA_API_KEY", "").strip()),
    }


def _langfuse_config(missing: list[str]) -> dict[str, Any]:
    missing_langfuse = [name for name in missing if name.startswith("LANGFUSE_")]
    return {
        "mode": "langfuse",
        "host": (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://us.cloud.langfuse.com"
        ),
        "public_key_present": bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()),
        "secret_key_present": bool(os.getenv("LANGFUSE_SECRET_KEY", "").strip()),
        "status": "missing_credentials" if missing_langfuse else "configured",
        "missing_environment_variables": missing_langfuse,
    }


def _compile_original_resume(output_dir: Path) -> Path:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex is not installed or not on PATH.")
    tex_path = REPO_ROOT / "data" / "resume.tex"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "resume.pdf"
    result = subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0 or not pdf_path.is_file():
        tail = "\n".join((result.stdout or result.stderr).splitlines()[-12:])
        raise RuntimeError(f"Could not compile source resume PDF: {tail}")
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count != 1:
        raise RuntimeError(f"Source resume PDF must be one page; found {page_count}.")
    _remove_latex_temporary_files(output_dir)
    return pdf_path


def _remove_latex_temporary_files(directory: Path) -> None:
    for suffix in (".aux", ".log", ".out", ".toc", ".fdb_latexmk", ".fls"):
        for path in directory.glob(f"*{suffix}"):
            path.unlink()


def _remove_source_resume_directory(directory: Path) -> None:
    if directory.name == "source_resume" and directory.is_dir():
        shutil.rmtree(directory)


def _remove_non_submission_output_artifacts(output_dir: Path, run_id: str) -> None:
    run_root = output_dir / run_id
    for path in (run_root / "checkpoints.sqlite", output_dir / "memory.json"):
        if path.is_file():
            path.unlink()
    for child in output_dir.iterdir():
        if (
            child.is_dir()
            and child.name.startswith("run-")
            and child.name != run_id
            and not (child / "run_manifest.json").is_file()
        ):
            shutil.rmtree(child)


def _run_workflow(
    *,
    run_id: str,
    output_dir: Path,
    reject_job_id: str,
) -> tuple[dict[str, Any], TraceManager]:
    run_root = output_dir / run_id
    memory_file = run_root / "memory.json"
    _seed_conflicting_memory(memory_file)
    checkpointer, context = create_sqlite_checkpointer(run_root / "checkpoints.sqlite")
    tracer = TraceManager()
    selector = DeepInfraToolSelectionModel(
        configured_model_name=os.environ["LLM_MODEL"]
    )
    try:
        app = build_agent_graph(
            checkpointer=checkpointer,
            tracer=tracer,
            tool_selection_model=selector,
        )
        state = create_initial_state(
            thread_id=f"thread-{run_id}",
            run_id=run_id,
            memory_file=str(memory_file),
        )
        waiting = invoke_new_run(app, state)
        interrupt_payload = waiting["__interrupt__"][0].value
        job_ids = list(interrupt_payload["resumes"])
        actual_reject = reject_job_id if reject_job_id in job_ids else job_ids[-1]
        feedback = {
            job_id: {"decision": "approve", "comment": ""} for job_id in job_ids
        }
        approved_for_conflict = next(job_id for job_id in job_ids if job_id != actual_reject)
        feedback[approved_for_conflict] = {
            "decision": "approve",
            "comment": "I have 5 years of experience in data engineering.",
        }
        feedback[actual_reject] = {
            "decision": "reject",
            "comment": "Add LangGraph. I have used it in previous projects.",
        }
        return resume_run(app, f"thread-{run_id}", feedback), tracer
    finally:
        context.__exit__(None, None, None)


def _seed_conflicting_memory(memory_file: Path) -> None:
    store = JSONMemoryStore(memory_file)
    old_fact = MemoryFact(
        fact_id="mem-seeded-old-experience",
        fact_type="experience",
        canonical_value="2 years of experience in data engineering",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=1,
            original_statement="I have 2 years of experience in data engineering.",
            related_job_id="J000",
        ),
        created_at="2026-01-01T00:00:00+00:00",
        active=True,
    )
    store.reset()
    store.append_many([old_fact], run_id=None)


def _build_summary(
    *,
    final: dict[str, Any],
    tracer: TraceManager,
    run_id: str,
    resume_pdf: Path,
    command: str,
    model_config: dict[str, Any],
    langfuse_config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(
        final.get("output_manifest", {}).get("output_root")
        or Path(os.environ["OUTPUT_DIR"]) / run_id
    )
    trace_path = root / "trace_events.json"
    trace_events = json.loads(trace_path.read_text(encoding="utf-8"))
    memory = json.loads(Path(final["memory_file"]).read_text(encoding="utf-8"))
    conflict_records = list(
        {
            conflict["conflict_id"]: conflict
            for fact in memory
            for conflict in fact.get("conflicts", [])
        }.values()
    )
    page_counts = _page_counts(final)
    return {
        "status": final["status"],
        "run_id": run_id,
        "model_configuration": model_config,
        "langfuse_configuration_status": {
            **langfuse_config,
            "trace_manager_status": tracer.status_message,
            "trace_id": final.get("trace_id"),
            "trace_url": final.get("trace_url"),
        },
        "model_selected_tool_call_sequence": [
            {
                "phase": decision.get("phase"),
                "selected_tool": decision.get("selected_tool"),
                "available_tools": decision.get("available_tools"),
                "validation_result": decision.get("result"),
                "job_id": decision.get("job_id"),
                "rationale": decision.get("rationale"),
            }
            for decision in final.get("agent_decisions", [])
        ],
        "memory_conflict_records": conflict_records,
        "memory_conflict_trace_events": [
            event
            for event in trace_events
            if event.get("name") == "memory.conflict_handling"
        ],
        "revision_memory_application": _revision_memory_application(
            trace_events, final
        ),
        "top_3_job_ids": final.get("top_3_job_ids"),
        "job_output_folders": final["output_manifest"]["job_directories"],
        "pdf_page_counts": page_counts,
        "source_resume_pdf": {
            "path": _display_path(resume_pdf),
            "exists": resume_pdf.is_file(),
            "page_count": len(PdfReader(str(resume_pdf)).pages),
        },
        "e2e_command": command,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _page_counts(final: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for job_id in final["top_3_job_ids"]:
        resume = final["tailoring_results"][job_id]
        letter = final["cover_letter_results"][job_id]
        before = Path(resume["output_pdf_path"]).with_name("resume_before.pdf")
        counts[job_id] = {
            "resume_before_pdf": len(PdfReader(str(before)).pages),
            "resume_after_pdf": len(PdfReader(resume["output_pdf_path"]).pages),
            "cover_letter_pdf": len(PdfReader(letter["output_pdf_path"]).pages),
        }
    return counts


def _revision_memory_application(
    trace_events: list[dict[str, Any]],
    final: dict[str, Any],
) -> dict[str, Any]:
    new_ids = set(final.get("new_memory_fact_ids", []))
    revision_jobs = set(final.get("review_history", [{}])[0].get("rejected_job_ids", []))
    matching_dispatches: list[dict[str, Any]] = []
    for event in trace_events:
        if event.get("name") != "tool_registry.dispatch":
            continue
        input_payload = event.get("input") or {}
        arguments = input_payload.get("arguments") or {}
        if input_payload.get("tool_name") != "resume_tailoring":
            continue
        job_id = (arguments.get("job") or {}).get("job_id")
        if job_id not in revision_jobs:
            continue
        evidence_ids = [
            item.get("evidence_id")
            for item in arguments.get("candidate_evidence", [])
            if item.get("evidence_id") in new_ids
        ]
        if evidence_ids:
            matching_dispatches.append(
                {
                    "job_id": job_id,
                    "new_memory_evidence_ids_in_revision_arguments": evidence_ids,
                    "revision_feedback": arguments.get("revision_feedback"),
                }
            )
    return {
        "new_memory_fact_ids": sorted(new_ids),
        "rejected_revision_jobs": sorted(revision_jobs),
        "revision_dispatches_containing_new_memory": matching_dispatches,
    }


if __name__ == "__main__":
    raise SystemExit(main())
