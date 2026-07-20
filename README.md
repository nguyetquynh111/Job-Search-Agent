# Job Search Agent

Foundational single-agent codebase for a university group assignment. The workflow and shared Pydantic contracts are scaffolded, but the five business tools are intentionally not implemented yet.

> Development status: the application cannot complete an end-to-end run until the five documented tool contracts are implemented by their assigned developers.

## Setup

```bash
conda create -n job_search python==3.12
conda activate job_search
pip install -r requirements.txt
cp .env.example .env
streamlit run src/app/app.py
```

## Input Files

The app opens on the setup page and requires four uploads before a search starts:

- job listings as `.csv`;
- preferences as `.yaml`;
- resume as `.tex`;
- portfolio as `.txt`.

The jobs CSV accepts both schema-style headers such as `job_id` and `title` and
spreadsheet-style headers such as `Job Title`, `Required Skills`, and `URL`.
Preferences YAML may contain `target_titles`, `locations`, `remote`, `job_types`,
`min_salary`, and `excluded_keywords`. In the portfolio text file, separate projects
with a blank line and optionally include a line such as `Technologies: Python, SQL`.

## Repository Layout

```text
src/
  app/                 Streamlit entrypoint and pages
  agent/               LangGraph workflow and controller
  memory/              Persistent candidate memory
  observability/       Local and Langfuse tracing helpers
  review/              Human-review payload and resume logic
  schemas/             Pydantic contracts for graph state and tools
  tests/               Pytest test cases and fixtures
  tools/               Tool design documents and future runtime registry
data/                  Demo input fixtures
outputs/               Generated artifacts, memory, and checkpoint database
README.md              Project documentation
```

## Architecture

```text
Streamlit UI
    |
    v
LangGraph checkpointer (SQLite for app, memory saver for tests)
    |
    v
initialize -> agent_controller -> execute_tool <----------------+
                  |              |                              |
                  |              +-> phase guard + registry      |
                  |                                             |
                  +-> one structured decision per tool call      |
                                                                |
prepare_review -> human_review_interrupt -> process_feedback ----+
                                      |        |
                                      |        v
                                      |   update_memory -> revision_controller
                                      |        |
                                      +--------+-> finalize_resumes
                                                    |
                                                    v
                                          generate_cover_letters
                                                    |
                                                    v
                                                 complete
```

The system has exactly one controller, `SingleAgentController`, and exactly five model-visible tools:

- `filter_jobs`
- `score_jobs`
- `analyze_fit`
- `tailor_resume`
- `generate_cover_letter`

Deterministic graph nodes enforce phase order; the controller only selects the next allowed tool and arguments.

When `DEEPINFRA_API_KEY` and `LLM_MODEL` are configured, the controller attempts a structured DeepInfra model decision call. `DEEPINFRA_BASE_URL` defaults to `https://api.deepinfra.com/v1/openai`. If the LLM is unavailable, local runs fall back to the deterministic controller path and record concise decision summaries without hidden chain-of-thought.

## Observability

Langfuse Cloud observability is used to inspect one complete job-search workflow across input loading, agent decisions, tool execution, human review, revisions, memory persistence, and artifact generation. The root trace name is `job_search_agent_run`, and the same trace ID is reused for all related spans in that submitted run.

Langfuse is optional for local development. If credentials are missing, invalid, the SDK is unavailable, authentication fails, or the network is unavailable, the app falls back to local no-op tracing and the job-search workflow continues.

The Langfuse Cloud project is named `Job-Search`. Create API keys in the Langfuse dashboard by opening the `Job-Search` project, going to project settings, and creating API credentials. Put the values in `.env` using the US Cloud host:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

The committed `.env.example` contains blank key placeholders and the required US host. Never commit a real `.env` file.

The Streamlit sidebar reports one of:

- `Observability: Langfuse connected`
- `Observability: Langfuse unavailable`
- `Observability: Local no-op tracing`

After a run with valid credentials, open the Langfuse `Job-Search` project dashboard and filter traces by the name `job_search_agent_run`. Nested spans should share the run trace ID and include safe metadata such as `run_id`, `session_id`, `tool_name`, `job_id`, `company`, `match_score`, `result_count`, `review_round`, `status`, `duration_ms`, and `error_type`. Resume contents, cover-letter contents, API keys, full prompts, and large job descriptions are intentionally not sent.

## Tool Design Handoff

No business-tool implementation is included at this stage. Each tool has a documentation-only folder containing its README and detailed input/output contract:

| Tool | Documentation | Contract |
| --- | --- | --- |
| `filter_jobs` | [`src/tools/filtering/`](src/tools/filtering/) | `FilterJobsInput` to `FilterJobsOutput` |
| `score_jobs` | [`src/tools/scoring/`](src/tools/scoring/) | `ScoreJobsInput` to `ScoreJobsOutput` |
| `analyze_fit` | [`src/tools/fit_analysis/`](src/tools/fit_analysis/) | `AnalyzeFitInput` to `FitAnalysisOutput` |
| `tailor_resume` | [`src/tools/resume_tailoring/`](src/tools/resume_tailoring/) | `TailorResumeInput` to `TailorResumeOutput` |
| `generate_cover_letter` | [`src/tools/cover_letter/`](src/tools/cover_letter/) | `GenerateCoverLetterInput` to `GenerateCoverLetterOutput` |

The shared Pydantic contract references remain in `src/schemas/` so independently developed tools can integrate against the same boundaries. The future implementation modules will be loaded by `src/tools/registry.py`; until those modules exist, startup is expected to fail with a missing-tool message.

## Human Review

LangGraph pauses once all three selected resumes are tailored. The interrupt payload contains all three resumes together:

```python
{
    "review_round": 1,
    "max_revision_rounds": 2,
    "resumes": {
        "J001": {
            "job_title": "...",
            "company": "...",
            "fit_analysis": {...},
            "change_log": [...],
            "resume_pdf_path": "..."
        }
    }
}
```

The Review page requires one approve/reject decision per selected job and resumes the graph with `Command(resume=feedback)`. Rejected resumes are revised in the same review phase. If resumes remain rejected after two revision rounds, the run status becomes `FAILED_REVIEW` and cover letters are not generated.

## Persistent Memory

Candidate memory is stored in `outputs/memory.json` by default. The file and its
parent directory are created automatically when missing. Review comments are scanned
for durable candidate facts, such as skills or technologies. Editing preferences such
as “make this shorter” are ignored.

When a new fact is written:

- the JSON file is updated;
- `memory_facts` in the graph state is updated;
- revisions in the same run receive the new memory as evidence;
- a `persist_memory` trace span is recorded.


## Tests

The existing integration tests describe the target workflow behavior. The full suite is expected to fail at tool loading until all five implementation modules are supplied. After implementation, run:

```bash
pytest
```

Test cases live in `src/tests/`. The intended coverage includes phase-policy enforcement, scoring/top-three gating, review interrupt payload shape, revision limit, cover-letter approval guard, JSON memory persistence, same-run memory availability, full end-to-end completion, Streamlit rerun safety, and root trace reuse. Each tool developer should also add contract-focused unit tests described in that tool's README.
