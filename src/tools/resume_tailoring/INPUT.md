# Input

| Field | Type | Meaning |
| --- | --- | --- |
| `job` | `Job` | Job used for tailoring. |
| `fit_analysis` | `FitAnalysisOutput` | Fit analysis for the same job. |
| `source_resume_tex_path` | string | Path to the original LaTeX resume. |
| `candidate_evidence` | list of `EvidenceItem` | Facts allowed in the resume. |
| `revision_feedback` | string or null | Optional human feedback for a revision. |

The job ID and fit-analysis job ID must match. Revision feedback is an editing request, not proof of a new candidate fact.

