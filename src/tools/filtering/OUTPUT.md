# Output

| Field | Type | Meaning |
| --- | --- | --- |
| `accepted_jobs` | list of `Job` | Jobs that pass all filters. |
| `rejected_jobs` | list of `RejectedJob` | Jobs that fail at least one filter. |

Each `RejectedJob` contains the original job and one or more clear rejection reasons. Every input job must appear in exactly one output list.

