# Scoring Tool

This tool scores accepted jobs, ranks them, and selects up to three jobs for fit
analysis. The score is computed by **code, never by a model** (Section 3.2), so
the same input always produces the same result.

- See [INPUT.md](INPUT.md) for input fields.
- See [OUTPUT.md](OUTPUT.md) for output fields.
- Contract: `ScoreJobsInput` to `ScoreJobsOutput`.
- Implementation: `src/tools/implementations/score_jobs.py`
  (formula in `scoring/formula.py`, profile index in `scoring/profile_index.py`).

## Whole-profile evidence

Every job is scored against the *entire* candidate profile, not just the resume:
the resume skills, the master skills list, **every** portfolio project's tech
stack, and any saved memory facts are folded into one `CandidateIndex`. Skill
comparison reuses the shared, hand-curated alias/category maps
(`fit_analysis/aliases.py`) so that:

- surface variants match (a job asking for "ML" is satisfied by "machine
  learning"), and
- a capability requirement ("vector databases") is satisfied by a concrete
  member skill the candidate has ("Pinecone").

Each matched skill records the evidence IDs that prove it; those IDs are returned
on the `ScoredJob` and are guaranteed to exist in the input.

## Formula (signals and weights)

Each job earns four weighted sub-scores in `[0, 1]` that combine into a final
`0..100` score. Weights sum to 100, so a perfect match scores 100.

| Signal | Weight | Measurement |
| --- | --- | --- |
| Skill match | **55** | fraction of the job's required skills the candidate can evidence |
| Experience alignment | **25** | candidate years vs. the role's minimum (1.0 if met; partial credit `max(0.2, cand/req)` if under; 0.75 if unspecified) |
| Industry/domain alignment | **15** | overlap of the job's domain phrases with the candidate's demonstrated domains (floor 0.3 when a domain is listed but does not overlap) |
| Location alignment | **5** | 1.0 remote-eligible, else 0.4 (optional; small by design) |

```
score = 55 * skill_fraction
      + 25 * experience_fraction
      + 15 * domain_fraction
      +  5 * location_fraction      # clamped to [0, 100], rounded to 1 dp
```

Skill match dominates, then experience, then domain; location is a light
tie-breaker because the assignment marks it optional for scoring.

## Ranking and Top-3

Jobs are sorted by descending score. Ties keep their original input order (a
stable sort), so the ranking is reproducible. The Top-3 job IDs are selected
automatically from the head of the ranking — no human input. Each `ScoredJob`
also carries a short rationale string summarising each signal for the trace and
the report.
