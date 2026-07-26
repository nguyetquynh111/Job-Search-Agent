# Job Search Agent

A single-agent job-search application that filters and ranks jobs, generates evidence-backed fit analyses, tailors resumes, supports human review, and produces cover letters.

## Requirements

* Conda;
* Python 3.12;
* `pdflatex`;
* DeepInfra API credentials;
* Langfuse credentials for production tracing.

Install the command-line TeX distribution:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
pdflatex --version
```

## Environment Setup

```bash
conda create -n job-search-agent python=3.12 -y
conda activate job-search-agent

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Configure `.env`:

```env
LLM_MODEL=provider/model-id
DEEPINFRA_API_KEY=your-deepinfra-key
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://us.cloud.langfuse.com

OUTPUT_DIR=outputs
```

Langfuse tracing is required for production runs. Create project API keys in
Langfuse, set the three `LANGFUSE_*` variables above, and the completed run
manifest will include the remote trace ID and public/share URL when Langfuse
returns one. Tests use mocked or disabled tracing and never fabricate a public
URL.

## Architecture

The Streamlit app is a UI wrapper around core agent modules. Core code imports
`src.config`, agent orchestration, schemas, tools, review, tracing, and utilities;
it does not import Streamlit or app configuration.

```text
UI / CLI
   |
   v
Agent Controller
   |
   v
Workflow Graph
   |
   v
Structured Tool Request
   |
   v
Tool Registry / Executor
   |-- filtering
   |-- scoring
   |-- fit_analysis
   |-- resume_tailoring
   `-- cover_letter
   |
   v
Human Review Pause
   |
   v
Memory + Final Outputs
```

Responsibilities:

* `src.agent.controller` starts and resumes compiled graph runs.
* `src.agent.graph` owns workflow order and LangGraph nodes.
* `src.agent.state` owns persisted graph state and initial state creation.
* `src.tools.registry` validates structured tool calls and dispatches to public
  tool entrypoints.
* `src.tools.filtering_scoring.scoring` performs deterministic code-based
  scoring, stable ranking, and automatic Top 3 selection. Numeric scores are not
  accepted from model text.
* `src.review.human_review` creates the one human review pause after three draft
  resumes are ready.
* `src.review.memory` stores validated review-derived memory facts.
* `src.tracing.langfuse` records run, tool dispatch, tool result, LLM generation,
  review, memory, and final status observations.

Compatibility imports preserved for older callers:

```python
import src.agent.controller
import src.agent.graph
import src.agent.state
import src.schemas.jobs
import src.tools.registry
```

## Run the Application

```bash
conda activate job-search-agent
python -m streamlit run app/app.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

## Application Workflow

The graph performs initialization, then asks the configured chat model to select
the next registered tool using the registry's structured tool schemas. The graph
validates the selected tool name and arguments, enforces the legal workflow order,
and dispatches through the registry. Scoring remains deterministic Python code:
the model may select the `scoring` tool but does not provide or edit score
values. After three draft resumes are ready, the graph interrupts exactly once
for human review. Rejected drafts can trigger bounded resume revisions and memory
updates; approved Top 3 jobs continue to cover-letter generation.

Pipeline order is fixed as: load candidate profile, portfolio, preferences,
resume, and memory; filter jobs; deterministically score accepted jobs;
automatically select the Top 3; run fit analysis for each Top 3 job; tailor one
resume for each Top 3 job; pause once for human review; revise rejected resumes
for at most two rounds; finalize approved resumes; generate cover letters as the
last tool.

Human review happens in one interrupt payload containing all three Top 3 resume
drafts, fit analyses, change logs, project swaps, and evidence citations. Each
resume is approved or rejected independently. Approved resumes are not
regenerated; rejected resumes are revised automatically using the submitted
comments, with a hard limit of two revision attempts. Every attempted revision is
recorded in `review_history.json` and the job's `revision_history.json` with the
review round, revision round, feedback received, actions taken, accepted changes,
rejected or skipped changes, evidence used, and deterministic feedback checks.

### Filtering Rules

`run_filtering_tool` preserves accepted and rejected jobs separately. Each
rejected job carries one or more specific reasons. The rules are evaluated in a
stable order:

* company exclusion list, compared case- and whitespace-insensitively;
* remote-only preference, when enabled;
* preferred locations, while allowing remote-eligible roles;
* minimum years of experience above the candidate's configured experience;
* excluded title/description keywords;
* target-title alignment against AI/ML target role vocabulary.

Final run outputs include `filtered_jobs.json` and `rejected_jobs.json` at the
run root, in addition to the trace event log.

### Scoring Formula

`run_scoring_tool` is deterministic Python code. The model may select the tool,
but score values are calculated only inside the tool. Candidate evidence is built
from the resume, every portfolio project, the master skills list, and persistent
memory facts; scoring is not limited to the current resume text.

Scores use a 0-100 weighted sum:

```text
score =
  55 * skill_match
+ 25 * experience_alignment
+ 15 * industry_domain_alignment
+  5 * location_alignment
```

`skill_match` is the fraction of required skills with candidate evidence,
`experience_alignment` compares candidate years to the role minimum,
`industry_domain_alignment` measures overlap with curated domain terms, and
`location_alignment` gives light credit to remote-eligible roles. Jobs are sorted
by score descending; ties preserve the accepted-job input order, making Top 3
selection deterministic and automatic. Final run outputs include
`ranked_jobs.json`.

## Input Formats

### Job Listings

CSV files must include:

```text
job_id, title, company, industry_domain, location, remote,
description, required_skills, years_experience_required,
company_details, url, salary_min, salary_max
```

### Preferences

Preferences are provided as YAML:

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

### Resume

The resume must be a LaTeX file.

Regenerate the original committed PDF with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory data data/resume.tex
```

### Portfolio

The portfolio is a plain-text file. Structured projects should include:

```text
PROJECT_ID
PROJECT_NAME
SUMMARY
TECH_STACK
```

## Validation

Run the test suite:

```bash
pytest -q
```

## Outputs

Generated files are written to:

```text
outputs/
  checkpoints.sqlite
  <run-id>/
    run_manifest.json
    memory.json
    trace_events.json
    filtered_jobs.json
    rejected_jobs.json
    ranked_jobs.json
    review_history.json
    <job-id>/
      job_details.json
      fit_analysis.json
      fit_analysis.md
      resume_before.pdf
      resume_after.tex
      resume_after.pdf
      cover_letter.tex
      cover_letter.pdf
      change_log.json
      human_review_decision.json
      revision_history.json
```

## Tracing

With Langfuse configured, each run creates one root trace named:

```text
job_search_agent_run
```

The trace covers:

* controller decisions;
* LLM generations;
* registry tool dispatch and tool results;
* fit analysis;
* resume tailoring;
* human review;
* memory updates;
* memory conflict handling;
* revisions;
* LaTeX compilation;
* PDF page validation;
* cover-letter generation.

Open the Langfuse trace from the `trace_url` field in
`outputs/<run-id>/run_manifest.json` or from the Streamlit observability panel.
If tracing is explicitly enabled without credentials, startup fails before the
run begins.
