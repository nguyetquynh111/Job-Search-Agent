# Filtering Tool

This tool removes jobs that do not match the candidate's required preferences. It
runs before scoring, is fully deterministic, never mutates a job, and logs a
clear reason for every rejection.

- See [INPUT.md](INPUT.md) for input fields.
- See [OUTPUT.md](OUTPUT.md) for output fields.
- Contract: `FilterJobsInput` to `FilterJobsOutput`.
- Implementation: `src/tools/implementations/filter_jobs.py`
  (rules in `src/tools/implementations/filtering/rules.py`).

## Rules

Each rule is a pure function and is **inert unless its backing preference is
set**, so the same code serves every candidate. A job is *accepted* only when it
passes every active rule; a *rejected* job carries **one reason per rule it
failed** (all failures are collected, not just the first). Every input job
appears in exactly one output list.

| # | Rule | Preference | Rejects when |
| - | --- | --- | --- |
| 1 | Company exclusion | `excluded_companies` | Company matches the exclusion list (case- and spacing-insensitive). |
| 2 | Remote-only | `remote_only` | Set to true and the job is not remote-eligible. |
| 3 | Location preference | `preferred_locations` | Onsite job whose location shares no city/state token with a preferred location. |
| 4 | Experience level | `years_of_experience` | Job's minimum required years exceeds the candidate's (under-qualified). |
| 5 | Excluded keywords | `excluded_keywords` | Title or description contains an excluded keyword. |
| 6 | Target-title relevance | `target_job_titles` | Title shares no AI/ML or target-role term with the candidate's goals. |

Rules 1–4 are the four required filters. Rules 5–6 are additional
preference-driven filters because `CandidatePreferences` also carries excluded
keywords and target job titles (Section 2.4).

### Details worth knowing

- **Remote-eligible** = `job.remote is True` or the location text contains
  "remote". Remote-eligible jobs always pass the location rule (they can be
  worked from any home base).
- **US-wide postings** ("United States", "Multiple U.S. locations", …) pass the
  location rule when the candidate lists any remote/US preference.
- **Experience** only filters *under-qualification*; the dataset has no reliable
  upper bound, so an over-qualified match is left for scoring to weigh.
- **Title relevance** matches significant tokens from the target titles plus a
  curated AI/ML role vocabulary (ai, ml, machine learning, computer vision, nlp,
  generative, mlops, agentic, …), so genuine AI/ML roles pass while clearly
  unrelated titles (e.g. "Senior Database Administrator") are rejected.
