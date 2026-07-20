# Input

| Field | Type | Meaning |
| --- | --- | --- |
| `jobs` | list of `Job` | Jobs that passed filtering. |
| `candidate_profile` | `CandidateProfile` | Candidate preferences, skills, education, and experience. |
| `resume_evidence` | list of `EvidenceItem` | Facts from the resume. |
| `portfolio_evidence` | list of `EvidenceItem` | Facts from portfolio projects. |
| `memory_evidence` | list of `EvidenceItem` | Saved candidate facts from earlier interactions. |

Each `EvidenceItem` has an ID, source, text, tags, and optional metadata.

