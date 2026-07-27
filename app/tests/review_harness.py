"""Streamlit test harness for the real-interrupt review presentation."""

from pathlib import Path

from app.components.human_review import render_human_review
from app.services.run_state import RunSnapshot

jobs = {
    f"J00{index}": {
        "job_id": f"J00{index}",
        "title": f"Role {index}",
        "company": f"Company {index}",
    }
    for index in range(1, 4)
}
resumes = {
    job_id: {
        "job_title": job["title"],
        "company": job["company"],
        "fit_analysis": {},
        "change_log": [],
        "resume_pdf_path": "",
    }
    for job_id, job in jobs.items()
}
snapshot = RunSnapshot(
    mode="Live Run",
    repo_root=Path.cwd(),
    run_id="review-harness",
    run_dir=None,
    status="WAITING_FOR_REVIEW",
    phase="HUMAN_REVIEW",
    read_only=False,
    top_3_job_ids=list(jobs),
    jobs_by_id=jobs,
    interrupt_payload={"resumes": resumes},
)

render_human_review(snapshot)

