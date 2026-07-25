# Output

Contract: `AnalyzeFitInput` → `FitAnalysisOutput` (defined in `src/schemas/fit_analysis.py`, frozen).

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | The analyzed job's ID. Always equals `input.job.job_id`. |
| `relevant_experience` | list of `EvidenceClaim` | Resume experience that matches the job. |
| `seniority` | list of `EvidenceClaim` | Seniority match (candidate level/years vs the job's expectation). |
| `education` | list of `EvidenceClaim` | Education match. |
| `aligned_skills` | list of `EvidenceClaim` | Job-required skills **already present on the resume**. |
| `evidenced_missing_skills` | list of `EvidenceClaim` | Job-required skills **not on the resume but evidenced elsewhere** (portfolio, master skills, or memory). |
| `genuine_gaps` | list of `EvidenceClaim` | Job-required skills with **no supporting evidence anywhere**. |
| `project_analysis` | list of `EvidenceClaim` | Relevance notes for the current resume projects (and the explicit "no swap" statement when applicable). |
| `project_swap` | `ProjectSwap` or null | At most one recommended project substitution, or null. |

Each `EvidenceClaim` is `{claim, evidence_ids, confidence (0..1), notes}`. Every `evidence_ids`
entry must be an ID that exists in `input.evidence_items`; the tool drops any that do not.

## The three skill buckets (disjoint)

A required skill appears in **exactly one** bucket:

- `aligned_skills` — required **and** already on the resume.
- `evidenced_missing_skills` — required, **not on the resume, but evidenced** in the portfolio,
  master skills, or memory. **These are the only skills the resume-tailoring tool may add**, so
  each claim names the skill and cites the evidence ID(s) that justify it.
- `genuine_gaps` — required with **no supporting evidence** in resume, portfolio, master skills,
  or memory.

Absence of a skill from the resume alone is **never** sufficient to call it a genuine gap — the
tool checks the portfolio, master skills, and memory first. Missing evidence must not be treated
as a gap. Resume **education** and **experience** lines count as evidence too, so a degree in a
required field grounds that skill rather than leaving it a gap.

Every bucket is grounded: an `aligned_skills` claim always cites at least one **resume-sourced**
evidence ID, and an `evidenced_missing_skills` claim always cites at least one real evidence ID.
A claim that cannot meet its bucket's bar is demoted during post-validation, so no pass marker is
ever asserted without a citation.

**Category skills.** A required *capability* (e.g. `agents`, `APIs`, `cloud deployment`, `deep
learning`) is satisfied by evidence of a concrete member *tool* (`AutoGen`, `FastAPI`, `Docker`,
`PyTorch`). When a claim is satisfied this way, its `evidence_ids` point at the member's evidence and
the claim text names the concrete tool, e.g. `agents: required by the job … but evidenced in
portfolio via AutoGen.` (rendered `❌ agents (used in "No-Code LLM Chatbot Builder" via AutoGen)`).
Tailoring should add the **named member tool**, not the bare capability word.

## `project_swap` identifier convention (contract for the tailoring tool)

The resume-tailoring tool is not built yet, so this tool defines the swap identifier convention.
**Tailoring must read this section.** Implemented behind `swap.build_project_swap` so it can change
in one place.

| Field | Value |
| --- | --- |
| `remove_project` | The **exact name** as it appears in `input.current_resume_projects`. |
| `add_project` | The **exact `PROJECT_NAME`** from `portfolio.txt`. |
| `evidence_ids` | The portfolio project's evidence IDs; the `P0x` ID is recoverable as `portfolio-P0x`. |

Worked example (real values from `data/resume.tex` and `data/portfolio.txt`):

```json
{
  "remove_project": "Cardiovascular Flow and Stenosis Analysis",
  "add_project": "AI CRM Platform",
  "rationale": "'AI CRM Platform' is a stronger match for this job than 'Cardiovascular Flow and Stenosis Analysis'.",
  "evidence_ids": ["portfolio-P02"]
}
```

Here `add_project` is P02's `PROJECT_NAME`; the ID `P02` is recovered from `evidence_ids`
(`portfolio-P02`). Only one swap is emitted even when two resume slots are weak; a second weak
slot is surfaced as a `project_analysis` claim, never a second swap (the schema allows one).

## Confidence

Confidence is assigned by a **documented, deterministic rule** from the *kind* of evidence cited,
not chosen arbitrarily (see `src/tools/fit_analysis/README.md`): corroboration by multiple
independent sources ranks above a single portfolio entry, which ranks above a single master-skill
or memory fact.
