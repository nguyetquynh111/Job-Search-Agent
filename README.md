# Job Search Agent

A Streamlit app and agent workflow for a repeatable job-search run. It filters
jobs, ranks the best matches, writes fit analyses, tailors resumes, pauses for
human review, and generates cover letters.

All candidate details in this repo are fictional.

## What You Need

- Python 3.12
- `pdflatex`
- DeepInfra credentials
- Langfuse credentials for live tracing

On macOS, install LaTeX with:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
pdflatex --version
```

## Setup

```bash
deactivate 2>/dev/null || true
conda create -n job-search-agent python=3.12 -y
conda activate job-search-agent
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```env
LLM_MODEL=your-provider/your-model
DEEPINFRA_API_KEY=your-deepinfra-api-key
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://us.cloud.langfuse.com

OUTPUT_DIR=outputs
```

Check that the environment is ready:

```bash
python tests/preflight.py
```

## Run The App

```bash
python -m streamlit run app/app.py
```

Open the Streamlit URL, usually `http://localhost:8501`.

In the app, upload:

- `data/jobs.csv`
- `data/preferences.yaml`
- `data/resume.tex`
- `data/portfolio.txt`

Then start the search, review the tailored resume drafts, and download the final
application package.

## Run The Production Workflow

Use this for a full live run with model tool selection and Langfuse tracing:

```bash
conda activate job-search-agent
python main.py
```

The runner creates a run ID automatically, such as
`job-search-live-20260726-184512`. It is used in the output folder
(`outputs/<run-id>/`), checkpoints, and tracing metadata.

The command fails fast if required live credentials or `pdflatex` are missing.

## Input Files

`jobs.csv` must include these columns:

```text
job_id,title,company,industry_domain,location,remote,description,
required_skills,years_experience_required,company_details,url,
salary_min,salary_max
```

`preferences.yaml` should include:

```yaml
candidate:
  years_of_experience: 4

preferences:
  target_job_titles:
    - Machine Learning Engineer
  preferred_locations:
    - Remote, US
  remote_only: false
  excluded_companies: []
  job_types:
    - full-time
  min_salary: 100000
  excluded_keywords: []

master_skills:
  - Python
  - PyTorch
```

`resume.tex` must be a LaTeX resume. `portfolio.txt` is plain text describing
projects, skills, and evidence the agent can cite.

## Outputs

Runs write files to `outputs/<run-id>/`, including:

- `run_manifest.json`
- `trace_events.json`
- `ranked_jobs.json`
- `rejected_jobs.json`
- one folder per selected job with fit analysis, tailored resume, cover letter,
  review decision, and revision history

## Tests

```bash
pytest -q
```
