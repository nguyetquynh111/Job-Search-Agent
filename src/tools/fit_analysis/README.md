# Fit Analysis Tool

Explains how well the candidate fits **one** selected job, using only the evidence supplied to it.
The controller calls it once per Top-3 job, before any resume tailoring.

- Input fields: [INPUT.md](INPUT.md)
- Output fields + skill-bucket semantics + swap contract: [OUTPUT.md](OUTPUT.md)
- Contract: `AnalyzeFitInput` → `FitAnalysisOutput`
- Entry point: `src.tools.implementations.analyze_fit.analyze_fit(inp) -> FitAnalysisOutput`
- Implementation: `src/tools/implementations/analyze_fit.py` + the private helper package
  `src/tools/implementations/fit_analysis/`.

## How it works

Hybrid: a deterministic core (pure, fully unit-tested) plus optional LLM reasoning, unified by a
post-validation pass that runs on **both** paths.

1. **Sanitize + index** (`sanitize.py`, `evidence_index.py`). Job text is stripped of mojibake
   (see below). Every evidence item is indexed as `canonical_skill -> [(evidence_id, source_kind)]`.
   Source kinds: `resume`, `master_skills`, `portfolio`, `memory`. Memory is first-class evidence.
2. **Deterministic pre-pass** (`prepass.py`, `swap.py`). Computes the three disjoint skill buckets,
   relevant experience, seniority, education, project ranking, and the single project swap — purely
   from evidence. This is both the LLM's grounding and the complete offline fallback.
3. **LLM reasoning** (`reasoning.py`, `prompts.py`, `llm_client.py`). Sends the sanitized job,
   candidate, portfolio, and evidence index to the model and requests strict JSON matching
   `FitAnalysisOutput` at `temperature=0`, retrying once with the validation error appended. The
   model reasons only over supplied facts and never emits a numeric match score. The LLM owns the
   contrastive **relevant-experience** narrative and the **education** claims.
4. **Deterministic facts override.** Candidate facts are computed deterministically and override the
   model so it cannot rewrite them and correctness holds on both paths: the **three skill buckets**
   (an evidence lookup with category→member resolution, see below), **seniority** (a candidate-years
   vs required-years comparison), and the **project section** (`project_analysis` + `project_swap`,
   both from one shared ranking so they can never disagree — a project recommended for removal is, by
   construction, the one ranked weakest and is reported as weak, never "aligns well"). On the LLM path
   the model then writes the constrained *prose* for the fixed seniority/swap outcomes.
5. **Deterministic fallback** (required). When no model is configured, or the LLM call fails after
   its retry, the analysis is built entirely from the pre-pass. The chosen path (`llm` / `fallback`)
   is logged and recorded in the span.
6. **Post-validation** (`postvalidate.py`). Drops cited evidence IDs that do not exist; demotes an
   `evidenced_missing` skill with no valid evidence to `genuine_gaps`; enforces bucket disjointness;
   nulls a swap referencing a non-existent project; forces `job_id`; assigns confidence by the
   documented rule; and forces each claim's verdict (see below). Every repair is logged.

## Verdict markers and rendering

`FitAnalysisOutput` is frozen, so each claim carries a verdict in its `notes` as a
`verdict=match|partial|mismatch|missing` prefix (`verdict.py`). The finer verdict is kept in the
JSON, but the **rendered markdown uses only ✅ and ❌** (matching the assignment example, which marks
a partial alignment with ❌ and distinguishes it by wording): `match`→✅; `partial`/`mismatch`/unknown
→❌ (**never a positive default** — a seniority shortfall renders ❌, not ✅). Seniority is `match` if
candidate years ≥ required, `partial` within 60%, else `mismatch`. Missing-but-evidenced skills print
their human-readable source inline (`agents (used in "No-Code LLM Chatbot Builder" via AutoGen)`) with
the evidence IDs after; the group's marker is configurable via
`render_fit_analysis(..., missing_marker=...)` — default `❌`, or `➕` to distinguish it visually.

## Evidence rules

- Every `EvidenceClaim.evidence_ids` entry must exist in `input.evidence_items`. This is **not**
  enforced by the schema, so the tool enforces it in post-validation.
- A project swap may only propose a project present in `input.portfolio_projects`.
- Absence of evidence in the resume alone is never a genuine gap — the portfolio, master skills, and
  memory are checked first. A skill evidenced solely by a **memory** fact is valid support for
  `evidenced_missing_skills`.

## Confidence rule (deterministic)

Confidence is a function of the *kinds* of evidence cited, not an arbitrary value. Assigned in
post-validation so the LLM and fallback paths are identical (`confidence.py`):

| Claim | Condition | Confidence |
| --- | --- | --- |
| aligned skill | on resume + ≥1 corroborating source | 0.90 |
| aligned skill | resume only | 0.80 |
| evidenced-missing | ≥2 independent sources | 0.85 |
| evidenced-missing | single portfolio entry | 0.70 |
| evidenced-missing | single master-skills entry | 0.65 |
| evidenced-missing | single memory fact | 0.60 |
| genuine gap | no evidence anywhere | 0.75 |
| experience / seniority / education | base 0.80, +0.10 if ≥2 evidence IDs | 0.80–0.90 |

Ordering encodes: multiple independent sources > single portfolio entry > single memory/master fact.

## Alias map rationale

`aliases.py` holds an **explicit**, hand-curated map of surface-form variants to canonical skills
(`ml`→`machine learning`, `cv`→`computer vision`, `k8s`→`kubernetes`, `rag`→`retrieval-augmented
generation`, `torch`→`pytorch`, …). It is intentionally not fuzzy/edit-distance matching, which would
produce false matches on short tokens. The seed set covers this project's AI/ML vocabulary; unknown
tokens pass through normalized (lower-cased, whitespace-collapsed). Extend the map in one place.

## Skill categories (capability → tool)

Postings name **capabilities** (`agents`, `APIs`, `cloud deployment`, `vector databases`, `MLOps`,
`deep learning`) while resumes/portfolios name concrete **tools**. `aliases.CATEGORY_SKILLS` is an
explicit, documented capability→member map so a required category is satisfied by evidence of any
member — e.g. `agents ← AutoGen, LangGraph, LangChain`; `APIs ← FastAPI, REST, WebSocket, GraphQL`;
`cloud deployment ← Docker, Kubernetes, AWS SageMaker, Google Cloud`; `vector databases ← Pinecone,
Weaviate, FAISS, …`; `deep learning ← PyTorch, TensorFlow, CNN, …`. Membership lists are the instances
observed across `data/portfolio.txt`, `data/resume.tex`, and `data/jobs.csv` plus standard members;
**explicit map only, no fuzzy/embedding matching**. Rules:

- A category match cites the member's **real** evidence ID and the rendered citation **names the
  concrete tool** so the claim is defensible: `❌ agents (used in "No-Code LLM Chatbot Builder" via
  AutoGen)`. If a member is on the resume the category is **aligned**; if only in portfolio/master/
  memory it is **evidenced-missing**; if **no** member is evidenced anywhere it stays a **genuine gap**
  (never invented). Combined labels like `Docker/Kubernetes` or `GenAI/LLMs` are split on `/` and
  `and`, and resolve if any part resolves.

Skill bucketing is therefore a deterministic evidence lookup and, like seniority and the project
section, **overrides the model** — the LLM cannot turn an evidenced skill into a gap or vice versa.

## Swap identifier convention and threshold

`remove_project` = exact name from `current_resume_projects`; `add_project` = exact `PROJECT_NAME`
from `portfolio.txt`; the `P0x` ID is recoverable from `evidence_ids` (`portfolio-P0x`). Isolated in
`swap.build_project_swap`. **The tailoring tool owner must agree to this convention** — it is
documented as the contract in [OUTPUT.md](OUTPUT.md) with a worked example.

A swap is recommended **only** when the best external candidate beats the incumbent it would replace
by at least `swap.SWAP_MIN_MARGIN = 2.0`. The score scale is one distinctive-skill match = 2.0, so a
swap must add at least one full distinctive dimension (a required skill or a domain/industry axis) the
outgoing project lacks. Below that margin the tool recommends **no swap** and states the current
projects are already the best available — a barely-better project is worse than none. Project scoring
uses the same category/slash expansion as the skill buckets, so a project's FastAPI counts toward a
job's "APIs".

## Running standalone

```bash
# from repo root, with PYTHONPATH set to the repo root
python - <<'PY'
from src.data_loader import (load_candidate_profile, load_jobs_csv,
                             load_portfolio, load_resume_data)
from src.schemas.fit_analysis import AnalyzeFitInput
from src.tools.implementations.analyze_fit import analyze_fit
from src.tools.implementations.fit_analysis.render import write_fit_analysis

jobs = load_jobs_csv("data/jobs.csv")
profile = load_candidate_profile("data/preferences.yaml")
resume = load_resume_data("data/resume.tex")
portfolio = load_portfolio("data/portfolio.txt")
profile = profile.model_copy(update={
    "skills": resume.skills, "education": resume.education,
    "experience": resume.experience, "resume_projects": resume.projects,
    "resume_evidence": resume.evidence_items, "portfolio_evidence": portfolio.evidence_items})
evidence = [*profile.resume_evidence, *profile.master_skill_evidence, *portfolio.evidence_items]

job = next(j for j in jobs if j.job_id == "J030")
out = analyze_fit(AnalyzeFitInput(
    job=job, candidate_profile=profile, evidence_items=evidence,
    current_resume_projects=profile.resume_projects, portfolio_projects=portfolio.projects))
print(write_fit_analysis(out, job))  # writes outputs/J030/fit_analysis.{md,json}
PY
```

`write_fit_analysis` writes `fit_analysis.md` and `fit_analysis.json` into
`<output_dir>/<job_id>/` (from `src/config.py`; default `outputs/`). The per-job folder is shared
with other tools: it is created if absent and never cleared — only these two files are written.
A generated sample lives at `outputs/J030/`.

Run the tests: `pytest src/tests/test_fit_analysis_*.py` (offline; no credentials or network).

## Verifying the LLM path (before a demo)

The LLM path uses DeepInfra's OpenAI-compatible endpoint. Required environment variables (values
live in `.env`, never commit them):

- `LLM_MODEL` — the model id (blank ⇒ the tool runs the deterministic fallback).
- `DEEPINFRA_API_KEY` — the API credential.
- `DEEPINFRA_BASE_URL` — defaults to `https://api.deepinfra.com/v1/openai`.

Confirm a live call works before a demo run:

```bash
python -c "from src.tools.implementations.fit_analysis import llm_client; \
print('configured:', llm_client.model_configured()); \
print(llm_client.complete('Reply with the single word OK.', 'ping'))"
```

When the tool runs, the chosen path is unmistakable in the logs (`... used the LLM path.` /
`... used the deterministic fallback path.`) and in the span metadata (`path`, `llm_used`, `model`,
`system_prompt`). Requires `langchain-openai` (already in `requirements.txt`).

## Verifying on Python 3.12

Graded runs use the Python 3.12 conda env from the top-level README. Run these yourself in **Anaconda
Prompt** (the module was developed and tested against 3.12; it uses no 3.13/3.14-only syntax):

```bat
conda create -n job_search python=3.12 -y
conda activate job_search
cd path\to\Job-Search-Agent
pip install -r requirements.txt

:: full fit-analysis suite (offline; no credentials or network needed)
pytest src\tests\test_fit_analysis_*.py -q

:: one live LLM call (needs LLM_MODEL + DEEPINFRA_API_KEY in .env)
python -c "from src.tools.implementations.fit_analysis import llm_client; print('configured:', llm_client.model_configured()); print(llm_client.complete('Reply with the single word OK.', 'ping'))"
```

`langchain-openai` resolves from `requirements.txt` via PyPI — no manual install step. If `conda`
cannot reach its package servers on your network, any Python 3.12 (e.g. from python.org) plus
`pip install -r requirements.txt` in a venv works identically.

## Notes for other owners

**Note for orchestration owner.** Orchestration already wraps each tool call in a span
(`graph.py`: `with trace_manager.span(span_name, ...): raw_output = spec.func(input_model)`), so this
tool is already a nested span in the run trace. To also nest this tool's *internal* detail span
(evidence-index summary, LLM-vs-fallback path, repairs) under the same trace, pass the run tracer
into the call. Exact, non-breaking one-line change at that call site:

```python
raw_output = spec.func(input_model, tracer=trace_manager) if tool_name == "analyze_fit" else spec.func(input_model)
```

The `tracer=` parameter is keyword-only and optional, so this does not affect other tools. Do not
make this change on my behalf — it is orchestration's decision.

**Suggested upstream fix (data_loader owner).** `data/jobs.csv` contains bullet characters that
decode as replacement characters (mojibake). This tool sanitizes job text defensively in
`sanitize.py`, but a cleaner long-term fix is to normalize the CSV text in `src/data_loader.py` at
load time so every tool benefits.

## Note for the team — `outputs/` must be committed

The assignment requires the public repo to contain an `outputs/` folder with **one folder per job
for at least three jobs** (job details, resume-before PDF, resume-after PDF, cover-letter PDF, and
the fit analysis) plus the agent's `outputs/memory.json`. The current `.gitignore` ignores all of
that (`outputs/*`, keeping only `.gitkeep`), so none of the per-job deliverables would be committed.

`.gitignore` is a shared file — I have **not** edited it. Proposed targeted change (raise with the
group): un-ignore the per-job directories and the memory file while keeping runtime state ignored.

```diff
 outputs/*
 !outputs/.gitkeep
+# Assignment deliverables: per-job output folders + the agent's memory file.
+!outputs/*/
+!outputs/memory.json
+# Keep runtime state out of the public repo (still ignored).
+outputs/checkpoints.sqlite
+outputs/**/*.tmp
```

`!outputs/*/` re-includes the per-job directories (one level under `outputs/`) and their files;
`outputs/checkpoints.sqlite` stays ignored because `outputs/*` still matches that top-level file and
no negation re-includes it. A generated sample already lives at `outputs/J030/`.
