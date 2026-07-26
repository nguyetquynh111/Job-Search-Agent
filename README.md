# Job Search Agent

A single-agent job-search workflow that ranks jobs, creates evidence-backed fit
analyses, tailors resume, human review, and produce cover-letter.

## Requirements

- Python 3.12 (the pinned environment is tested with Python 3.12.11);
- `pdflatex` on `PATH`;
- a DeepInfra API key and an accessible model identifier for a real run;
- optional Langfuse credentials for remote traces.

`pdflatex` is a system program, not a Python package, so `pip` cannot install it.
Use a complete TeX distribution to ensure the resume packages (`fullpage`,
`titlesec`, `enumitem`, `hyperref`, `babel`, and `tabularx`) are present:

- macOS: install [MacTeX](https://tug.org/mactex/);
- Linux/Unix: install [TeX Live](https://tug.org/texlive/quickinstall.html);
- Windows: use the [TeX Live Windows installer](https://tug.org/texlive/windows.html).

Verify the installation before running the app:

```bash
pdflatex --version
```

## Quick Start

Run these commands from the repository root.

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Configure the Environment

Open `.env` and set at least these values:

```env
LLM_MODEL=provider/model-id
DEEPINFRA_API_KEY=your-deepinfra-key
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
```

Use a model that supports structured output through DeepInfra's OpenAI-compatible
API. A production run intentionally fails its preflight check when `LLM_MODEL`,
`DEEPINFRA_API_KEY`, or `pdflatex` is missing.

These settings are optional:

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com
OUTPUT_DIR=outputs
```

Remote traces are always public and trace payloads are sanitized before export.
Never commit `.env`.

Check the complete production configuration:

```bash
python scripts/preflight.py --require-langfuse
```

This checks Python 3.12 and imported dependencies, `pdflatex`, every required
LaTeX package, a real TeX compilation, DeepInfra credentials, complete Langfuse
credentials, writable output storage, and all four checked-in input fixtures.

## Verify the Checkout

The main suite runs offline. Fast graph-wiring tests use valid in-memory PDF
fixtures, while a separate production-like smoke test uses the real
`pdflatex`, real TeX edits, real PDF page inspection, real approved-resume text
extraction, SQLite checkpoints, and the JSON memory store. That smoke test is
skipped with an explicit reason when `pdflatex` is not installed.

```bash
python -m pip check
python -m pytest -q
```

On a machine without `pdflatex`, the expected result is `180 passed, 1 skipped`.
With `pdflatex` and the required LaTeX packages installed, all 181 tests run.

Run the real artifact smoke test directly with:

```bash
python -m pytest -q src/tests/integration/test_real_pdflatex_end_to_end.py
```

The test smoke intentionally uses the explicit offline controller policy so it
is deterministic in CI; controller-specific tests separately exercise LLM-owned
selection among multiple valid actions and rejection of invalid actions. A
submission demonstration must use the configured LLM controller.

Run the mandatory connected production demonstration with:

```bash
python scripts/preflight.py --require-langfuse
python scripts/run_production_smoke.py | tee outputs/production-smoke.log
```

The production-smoke output directory must be empty at the start. The command
uses the real DeepInfra controller, real tool implementations, SQLite
checkpointing, persistent memory, one review interrupt with one rejected resume,
real `pdflatex`, page inspection, all six final PDFs, and one connected Langfuse
trace. It writes `production_run.json` and `trace_export.json` alongside the
artifacts and fails instead of skipping any required production dependency.

## Run the Application

```bash
python -m streamlit run src/app/app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

For the shortest reproducible run, upload the checked-in examples:

| App field | Example file |
| --- | --- |
| Job listings | `data/jobs.csv` |
| Preferences | `data/preferences.yaml` |
| Resume | `data/resume.tex` |
| Portfolio | `data/portfolio.txt` |

Then:

1. Select **Start search**.
2. Wait for filtering, deterministic scoring, fit analysis, and all three draft
   resumes to finish.
3. At the single review page, approve or reject each resume. A rejection requires
   actionable feedback.
4. Submit all three decisions together.
5. Download the final application packages from the Results page.

Rejected resumes may receive at most two internal revision attempts. The workflow
does not pause for a second review. If the feedback cannot be satisfied within the
limit, the run ends as `FAILED_REVIEW` and cover letters are not generated.

## Input Contracts

### Job listings (`.csv`)

The minimum columns are:

```text
title, company, description
```

Recommended columns are:

```text
job_id, title, company, industry_domain, location, remote,
description, required_skills, years_experience_required,
company_details, url, salary_min, salary_max
```

Common spreadsheet headings such as `Job Title`, `Required Skills`, and `URL` are
normalized automatically. Separate multiple skills with semicolons or commas.
When `job_id` is missing, IDs are generated in row order.

### Preferences (`.yaml`)

The example below uses the supported assignment-style structure:

```yaml
candidate:
  years_of_experience: 4

preferences:
  target_job_titles:
    - Machine Learning Engineer
  preferred_locations:
    - Remote, US
  remote_only: false
  excluded_companies: []
  job_types:
    - full-time
  min_salary: 100000
  excluded_keywords: []

master_skills:
  - Python
  - PyTorch
```

Aliases such as `target_titles`, `locations`, `remote`, and
`companies_to_exclude` are also accepted.

### Resume (`.tex`)

Tailoring edits the uploaded source instead of regenerating the document. Start
from `data/resume.tex` or preserve its edit markers:

```text
% AGENT-EDIT-TARGET: summary
% AGENT-EDIT-TARGET: experience-bullet-1
% AGENT-EDIT-TARGET: experience-bullet-2
% AGENT-EDIT-TARGET: skills
% AGENT-SWAP-TARGET: project-N | PORTFOLIO-ID: PNN
```

The first four markers are required for the allowed summary, exactly-two-bullet,
and evidenced-skill edits. Project markers identify existing projects that may be
swapped only for a project present in the portfolio.

### Portfolio (`.txt`)

Plain-text projects may be separated with blank lines, or may use the structured
format in `data/portfolio.txt`. Structured records should include a unique
`PROJECT_ID`, `PROJECT_NAME`, `SUMMARY`, and `TECH_STACK`; additional fields
improve evidence matching and project-swap quality.

## Workflow

```text
initialize
    |
    v
single LLM controller <----> tool registry
    |                         filter_jobs
    |                         score_jobs
    |                         analyze_fit
    |                         tailor_resume
    |                         generate_cover_letter
    v
all three draft resumes
    |
    v
one human-review interrupt
    |
    +--> memory write and affected revisions
    |
    v
finalize resumes --> three cover letters --> complete
```

`SingleAgentController` receives the current workflow snapshot and only the tools
allowed in the current phase. It returns a structured tool choice and target job.
Python assembles Pydantic-validated arguments from checkpointed candidate and job
evidence. Deterministic phase guards enforce prerequisites without replacing the
model's tool-selection decision.

Production runs require the configured LLM controller. Tests inject the
deterministic policy directly without changing runtime configuration.

## Outputs

The default `outputs/` layout is:

```text
outputs/
  checkpoints.sqlite
  memory.json
  <job-id>/
    fit_analysis.json
    fit_analysis.md
    resume.tex
    resume.pdf
    resume-revision-1.tex       # only when revised
    resume-revision-1.pdf
    cover-letter.tex
    cover-letter.pdf
```

Every accepted PDF is inspected programmatically and must contain exactly one
page. Resume changes include before text, after text, reason, and evidence IDs.
The Results page can download each PDF or all approved application packages as a
ZIP.

`outputs/memory.json` and `outputs/checkpoints.sqlite` persist across app restarts.
Use the app's reset action when you want a clean workspace. Generated outputs,
uploaded temporary files, checkpoints, and `.env` should not be committed.

## Evidence and Safety Rules

- Deterministic scoring uses the resume, portfolio, master skills, and persistent
  memory.
- Fit-analysis claims cite evidence IDs from candidate or job sources.
- A known evidence ID is accepted only when its text or structured tags
  semantically support the exact claim. Case, common abbreviations, and the
  curated aliases in `src/tools/skill_matching.py` are normalized; unrelated,
  vague, negated, or contradictory evidence is rejected.
- Evidenced-but-missing resume skills are separated from genuine gaps.
- Genuine gaps are reported and never inserted into a resume or cover letter.
- Resume edits are limited to the summary, exactly two marked experience bullets,
  evidenced skills, and a justified portfolio project swap.
- Cover letters use the approved resume plus validated candidate, portfolio,
  master-skill, memory, job-posting, and company evidence.
- The workflow stops when an artifact is missing, unreadable, fails compilation,
  or is not exactly one page.
- Project additions require an exact `PROJECT_NAME` match in portfolio evidence.
  Reviewer-introduced memory is stored only after an explicit first-person fact
  assertion is revalidated against the verbatim feedback.

## Tracing

With valid Langfuse credentials, the run creates one root trace named
`job_search_agent_run`. Nested observations cover initialization, agent
decisions, LLM generations, tool calls, the human-review pause, memory writes,
revisions, LaTeX compilation, one-page checks, and completion.

The observation hierarchy uses one outer span for each model-visible tool and
explicitly named substeps:

```text
job_search_agent_run
  agent_controller
    agent_controller_llm
  fit_analysis_job_N
    fit_analysis.analysis_pipeline
      fit_analysis_llm
    fit_analysis.write_artifacts
  tailor_resume_job_N
    resume_tailoring.edit_content
    resume_tailoring.compile_pdf
    resume_tailoring.validate_page_count
  human_review
    human_review_pause
    human_review_feedback
    memory_write
    review_revision
      tailor_resume_job_N
  generate_cover_letter_job_N
    cover_letter.generate_content
    cover_letter.compile_pdf
    cover_letter.validate_page_count
```

Model name, parameters, token usage, available actions, selected tool and job,
decision reason, workflow state, evidence IDs, compilation command/result, page
count, review decisions, and memory provenance are retained locally as well as
sent in the sanitized remote payload. Checkpointed trace and parent IDs are
reused after the review process restarts.

If Langfuse is unavailable, the workflow continues with local no-op tracing. The
Results page shows the connection state and links to the remote trace when one is
available. Remote traces are public, and trace payloads are sanitized before
export.

## Packaging Verification

Generated artifacts, `.env`, caches, temporary LaTeX files, local SQLite files,
and memory files remain ignored. After committing intended source and test
changes, verify that no implementation depends on an untracked file:

```bash
scripts/verify_clean_checkout.sh
```

The command creates a detached clean worktree, checks required files, installs
the pinned environment, verifies imports and dependency consistency, and runs
the complete test suite.

## Repository Layout

```text
data/                 Reproducible sample inputs
src/agent/            Single controller, LangGraph workflow, and phase policy
src/app/              Streamlit entrypoint and pages
src/memory/           Persistent candidate memory and provenance extraction
src/observability/    End-to-end trace management and Langfuse integration
src/review/           Human-review validation
src/schemas/          Pydantic input and output contracts
src/tests/            Unit and end-to-end tests
src/tools/            Five tool implementations and central registry
src/ui/               Reusable Streamlit components and session helpers
```

## Troubleshooting

### `pdflatex is not installed or not on PATH`

Install a complete TeX distribution, open a new terminal, and run
`pdflatex --version`. On macOS, the executable is commonly installed under
`/Library/TeX/texbin`; ensure that directory is on `PATH`.

### `Runtime preflight failed`

Read the complete message. It reports all missing requirements at once. Set
`LLM_MODEL` and `DEEPINFRA_API_KEY`, confirm `.env` is in the repository root,
and verify `pdflatex`.

### A resume fails before compilation

Compare the uploaded file with `data/resume.tex`. The tailoring tool needs the
four `AGENT-EDIT-TARGET` markers and must be able to locate the marked
`\resumeItem` and skills content without changing the rest of the template.

### A PDF has more than one page

The workflow rejects it instead of silently accepting it. Shorten the source
resume content or choose a smaller set of evidenced portfolio details; the tool
does not alter margins, font size, or layout to force a one-page result.

### Streamlit opens the wrong Python environment

Use module execution from the activated virtual environment:

```bash
python -m streamlit run src/app/app.py
```

Confirm the executables with `python --version` and
`python -m pip --version`.
