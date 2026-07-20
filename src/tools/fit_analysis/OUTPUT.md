# Output

The output contains:

- `job_id`: analyzed job ID.
- `relevant_experience`: matching experience.
- `seniority`: seniority match.
- `education`: education match.
- `aligned_skills`: supported matching skills.
- `evidenced_missing_skills`: skills not found in the supplied evidence.
- `genuine_gaps`: confirmed gaps.
- `project_analysis`: project relevance.
- `project_swap`: optional project replacement suggestion.

Each claim should include supporting evidence IDs and a confidence value from 0 to 1. Missing evidence must not automatically be treated as a genuine gap.

