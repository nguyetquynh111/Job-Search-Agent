# Input

| Field | Type | Meaning |
| --- | --- | --- |
| `job` | `Job` | Target job and company information. |
| `approved_resume_path` | string | Path to the approved tailored resume. |
| `candidate_evidence` | list of `EvidenceItem` | Candidate facts allowed in the letter. |

The approved resume must belong to the same job. Job-description text is not evidence about the candidate.

