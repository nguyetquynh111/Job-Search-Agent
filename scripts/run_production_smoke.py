#!/usr/bin/env python3
"""Run and preserve one mandatory real production workflow demonstration."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypdf import PdfReader

from scripts.preflight import run_preflight
from src.agent.graph import (
    build_agent_graph,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry


def main() -> int:
    repo_root = REPO_ROOT
    output_dir = Path(
        os.getenv("PRODUCTION_SMOKE_OUTPUT_DIR", "outputs/production-smoke")
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    errors = run_preflight(
        repo_root=repo_root,
        output_dir=output_dir,
        require_langfuse=True,
    )
    if errors:
        raise RuntimeError("Production smoke preflight failed: " + "; ".join(errors))
    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Production smoke output must start empty: {output_dir}. "
            "Move the prior evidence directory before rerunning."
        )

    os.environ["OUTPUT_DIR"] = str(output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"production-smoke-{timestamp}"
    thread_id = f"thread-{run_id}"
    tracer = TraceManager()
    if not tracer.enabled:
        raise RuntimeError(
            f"Connected Langfuse tracing is required: {tracer.status_message}"
        )
    checkpoint_path = output_dir / "checkpoints.sqlite"
    memory_path = output_dir / "memory.json"
    checkpointer, context = create_sqlite_checkpointer(checkpoint_path)
    try:
        app = build_agent_graph(
            tools=load_tool_registry(),
            checkpointer=checkpointer,
            tracer=tracer,
        )
        state = create_initial_state(
            jobs_path=str(repo_root / "data/jobs.csv"),
            candidate_profile_path=str(repo_root / "data/preferences.yaml"),
            resume_path=str(repo_root / "data/resume.tex"),
            portfolio_path=str(repo_root / "data/portfolio.txt"),
            memory_file=str(memory_path),
            run_id=run_id,
            thread_id=thread_id,
        )
        waiting = invoke_new_run(app, state)
        interrupts = waiting.get("__interrupt__", [])
        if len(interrupts) != 1:
            raise RuntimeError(
                f"Expected exactly one review interrupt, found {len(interrupts)}"
            )
        payload = interrupts[0].value
        feedback = {
            job_id: {"decision": "approve", "comment": ""}
            for job_id in payload["resumes"]
        }
        rejected_job_id = (
            "J028" if "J028" in payload["resumes"] else next(iter(payload["resumes"]))
        )
        feedback[rejected_job_id] = {
            "decision": "reject",
            "comment": "Add LangGraph. I have used it in previous projects.",
        }
        final = resume_run(app, thread_id, feedback)
    finally:
        context.__exit__(None, None, None)

    if final.get("status") != "COMPLETED":
        raise RuntimeError(
            "Production workflow did not complete: "
            + json.dumps(final.get("errors", []), indent=2)
        )
    if not tracer.trace_url:
        raise RuntimeError("Langfuse connected but no trace URL was returned")
    controller_generations = [
        event
        for event in tracer.events
        if event.name == "agent_controller_llm"
        and event.observation_type == "GENERATION"
        and event.status == "OK"
    ]
    if not controller_generations:
        raise RuntimeError("No successful real controller generation was traced")
    if not all(
        decision.get("decision_source") == "llm"
        for decision in final["agent_decisions"]
    ):
        raise RuntimeError("Production run contained a non-LLM controller decision")
    review_interrupt_count = sum(
        event.name == "human_review_pause" for event in tracer.events
    )
    if review_interrupt_count != 1:
        raise RuntimeError(
            f"Expected one review pause, observed {review_interrupt_count}"
        )
    if final["revision_round"] < 1:
        raise RuntimeError("The rejected resume did not require a revision round")

    page_counts: dict[str, int] = {}
    artifacts: list[str] = []
    for job_id in final["top_3_job_ids"]:
        for label, result in (
            ("resume", final["tailoring_results"][job_id]),
            ("cover_letter", final["cover_letter_results"][job_id]),
        ):
            pdf_path = Path(result["output_pdf_path"])
            page_count = len(PdfReader(str(pdf_path)).pages)
            if page_count != 1:
                raise RuntimeError(f"{pdf_path} has {page_count} pages")
            page_counts[f"{job_id}:{label}"] = page_count
            artifacts.extend([result["output_tex_path"], str(pdf_path)])
        fit_paths = final["fit_analysis_artifacts"][job_id]
        artifacts.extend(fit_paths.values())
    trace_export = output_dir / "trace_export.json"
    trace_export.write_text(
        json.dumps([asdict(event) for event in tracer.events], indent=2, default=str),
        encoding="utf-8",
    )
    revision_memory_reuse = [
        {
            "job_id": decision["target_job_id"],
            "memory_evidence_ids": [
                item["evidence_id"]
                for item in decision["arguments"].get("candidate_evidence", [])
                if item["evidence_id"].startswith("mem-")
            ],
        }
        for decision in final["agent_decisions"]
        if decision["selected_tool"] == "tailor_resume"
        and decision["arguments"].get("revision_feedback")
    ]
    if not revision_memory_reuse or not any(
        item["memory_evidence_ids"] for item in revision_memory_reuse
    ):
        raise RuntimeError("No revised resume reused the same-run memory fact")
    report = {
        "run_id": run_id,
        "thread_id": thread_id,
        "status": final["status"],
        "controller": "real DeepInfra LLM",
        "model": os.environ["LLM_MODEL"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pdflatex": subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[0],
        "langfuse_trace_url": tracer.trace_url,
        "trace_export_path": str(trace_export),
        "checkpoint_path": str(checkpoint_path),
        "memory_path": str(memory_path),
        "top_3_job_ids": final["top_3_job_ids"],
        "rejected_job_id": rejected_job_id,
        "review_interrupt_count": review_interrupt_count,
        "revision_round": final["revision_round"],
        "new_memory_fact_ids": final["new_memory_fact_ids"],
        "revision_memory_reuse": revision_memory_reuse,
        "review_history": final["review_history"],
        "page_counts": page_counts,
        "artifacts": artifacts,
    }
    report_path = output_dir / "production_run.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tracer.flush()
    print(json.dumps(report, indent=2))
    print(f"Production evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
