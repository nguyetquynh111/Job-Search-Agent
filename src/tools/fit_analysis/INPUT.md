# Input

| Field | Type | Meaning |
| --- | --- | --- |
| `job` | `Job` | One selected job. |
| `candidate_profile` | `CandidateProfile` | Candidate information. |
| `evidence_items` | list of `EvidenceItem` | Facts that the analysis may cite. |
| `current_resume_projects` | list of strings | Projects currently on the resume. |
| `portfolio_projects` | list of `PortfolioProject` | Other projects available for comparison. |

Each portfolio project includes an ID, name, description, technologies, and evidence IDs.

