# Output

| Field | Type | Meaning |
| --- | --- | --- |
| `ranked_jobs` | list of `ScoredJob` | All jobs sorted from highest to lowest score. |
| `top_3_job_ids` | list of strings | IDs of the best one to three jobs. |

Each `ScoredJob` contains the original job, a score from 0 to 100, a short reason, and supporting evidence IDs.

The ranking must be stable, and every evidence ID must exist in the input.

