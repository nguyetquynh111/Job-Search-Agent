# Job Search Agent

A single-agent job-search application that filters and ranks jobs, generates evidence-backed fit analyses, tailors resumes, supports human review, and produces cover letters.

## Requirements

* Conda;
* Python 3.12;
* `pdflatex`;
* DeepInfra API credentials;
* optional Langfuse credentials.

Install the command-line TeX distribution:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
pdflatex --version
```

## Environment Setup

```bash
conda create -n job-search-agent python=3.12 -y
conda activate job-search-agent

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Configure `.env`:

```env
LLM_MODEL=provider/model-id
DEEPINFRA_API_KEY=your-deepinfra-key
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com

OUTPUT_DIR=outputs
```

## Run the Application

```bash
conda activate job-search-agent
python -m streamlit run app/app.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

## Application Workflow

```text
initialize
    |
    v
single LLM controller <----> tool registry
    |                         filter_jobs
    |                         score_jobs
    |                         analyze_fit
    |                         tailor_resume
    |                         generate_cover_letter
    v
three draft resumes
    |
    v
human review
    |
    +--> revisions and memory update
    |
    v
final resumes --> cover letters --> results
```

## Input Formats

### Job Listings

CSV files must include:

```text
job_id, title, company, industry_domain, location, remote,
description, required_skills, years_experience_required,
company_details, url, salary_min, salary_max
```

### Preferences

Preferences are provided as YAML:

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

### Resume

The resume must be a LaTeX file.

### Portfolio

The portfolio is a plain-text file. Structured projects should include:

```text
PROJECT_ID
PROJECT_NAME
SUMMARY
TECH_STACK
```

## Validation

Run the test suite:

```bash
pytest
```

## Outputs

Generated files are written to:

```text
outputs/
  checkpoints.sqlite
  memory.json
  <job-id>/
    fit_analysis.json
    fit_analysis.md
    resume.tex
    resume.pdf
    resume-revision-1.tex
    resume-revision-1.pdf
    cover-letter.tex
    cover-letter.pdf
```

## Tracing

With Langfuse configured, each run creates one root trace named:

```text
job_search_agent_run
```

The trace covers:

* controller decisions;
* LLM generations;
* tool calls;
* fit analysis;
* resume tailoring;
* human review;
* memory updates;
* revisions;
* LaTeX compilation;
* PDF page validation;
* cover-letter generation.
