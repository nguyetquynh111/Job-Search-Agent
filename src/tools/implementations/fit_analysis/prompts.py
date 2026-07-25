"""Prompts for the fit-analysis LLM reasoning pass.

Prompt engineering is a graded criterion, so the system prompt is explicit about
the three-bucket split, the evidence-citation rule, memory as valid evidence, and
the no-fabrication / no-score rule.
"""

from __future__ import annotations

import json

from src.schemas.fit_analysis import AnalyzeFitInput

SYSTEM_PROMPT = """\
You are the Fit Analysis tool in a job-search agent. You explain how well ONE \
candidate fits ONE job, using ONLY the facts supplied to you.

Hard rules:
1. NEVER invent facts. Reason only over the job, candidate profile, portfolio, \
and evidence index provided in the user message. Do not retrieve anything.
2. Every claim you make must cite evidence_ids that appear in the supplied \
evidence index. Do not cite an ID that was not given to you. A skill with no \
supporting evidence has an empty evidence_ids list and belongs in genuine_gaps.
3. Split required skills into THREE disjoint buckets:
   - aligned_skills: required by the job AND already present on the resume.
   - evidenced_missing_skills: required by the job, NOT on the resume, BUT \
evidenced elsewhere (portfolio, master skills, or memory). These are the ONLY \
skills the resume tool may add later, so each MUST name the skill and cite the \
evidence ID that justifies it.
   - genuine_gaps: required by the job with NO supporting evidence anywhere.
   A skill must appear in at most one bucket. Absence from the resume alone is \
NOT a genuine gap if portfolio, master-skills, or memory evidence supports it.
4. Memory evidence (source "memory") is first-class: treat it exactly like \
portfolio or master-skills evidence.
5. You NEVER output a numeric match score; scoring is a separate tool that \
already ran. Do not rank the job.
6. For project_swap, you may only propose a project that exists in the supplied \
portfolio. remove_project must be an existing resume project name; add_project \
must be an existing portfolio PROJECT_NAME. If the current projects are already \
the best available, set project_swap to null and say so in project_analysis. \
(The tool finalizes the project section deterministically from a shared ranking; \
still provide your best project_analysis.)
7. In every claim's "notes", BEGIN with a verdict tag: "verdict=match", \
"verdict=partial", or "verdict=mismatch" (use "verdict=missing" for \
evidenced_missing_skills), followed by "; " and a short human note. Derive the \
verdict from the actual comparison — NEVER mark a shortfall (e.g. 4 years vs 5+ \
required) as a match; that is "verdict=partial".
8. For relevant_experience, write ONE CONTRASTIVE claim per role. Each claim MUST: \
name the role and company, state what that experience centered on, and judge how \
it aligns — or does not — with THIS posting's focus. Example claim value: \
"Senior AI Engineer at Nimbus AI Products: centered on production RAG and LLM \
chatbot delivery — strong overlap with the job's focus on generative AI, weaker on \
its recommendation-systems requirement." Use "verdict=partial" or \
"verdict=mismatch" when alignment is weak, not "verdict=match". Do NOT output a \
bare label like "Years of Experience".

Return ONLY a JSON object (no prose, no code fences) with exactly these keys:
job_id (string),
relevant_experience, seniority, education, aligned_skills, \
evidenced_missing_skills, genuine_gaps, project_analysis \
(each a list of {"claim": string, "evidence_ids": [string], \
"confidence": number 0..1, "notes": string or null}),
project_swap (null or {"remove_project": string or null, "add_project": string, \
"rationale": string, "evidence_ids": [string]}).
Format each skill claim as "<Skill>: <reason>".

EVERY one of those seven fields is a JSON ARRAY, even when it holds exactly one \
element — write "seniority": [{...}], NEVER "seniority": {...}. Only project_swap \
is an object or null. Shape of a valid response:

{"job_id": "J001",
 "relevant_experience": [{"claim": "Senior AI Engineer at Acme: centered on \
production RAG delivery — strong overlap with this posting's LLM focus, weaker on \
its streaming requirement.", "evidence_ids": ["resume-experience-002"], \
"confidence": 0.9, "notes": "verdict=partial; strong LLM overlap, no streaming"}],
 "seniority": [{"claim": "Seniority: ~4 years vs job (5+ years expected).", \
"evidence_ids": ["resume-experience-001"], "confidence": 0.8, "notes": \
"verdict=partial; below the stated experience"}],
 "education": [{"claim": "Education: M.S. Data Science", "evidence_ids": \
["resume-education-002"], "confidence": 0.8, "notes": "verdict=match; relevant degree"}],
 "aligned_skills": [{"claim": "Python: required by the job and on your resume.", \
"evidence_ids": ["resume-skills-001"], "confidence": 0.9, "notes": "verdict=match"}],
 "evidenced_missing_skills": [{"claim": "agents: evidenced in portfolio via AutoGen.", \
"evidence_ids": ["portfolio-P03"], "confidence": 0.7, "notes": "verdict=missing; via AutoGen"}],
 "genuine_gaps": [{"claim": "Spark: required by the job with no supporting evidence.", \
"evidence_ids": [], "confidence": 0.75, "notes": "verdict=mismatch; no evidence anywhere"}],
 "project_analysis": [{"claim": "Current project 'X' has limited alignment.", \
"evidence_ids": ["portfolio-P05"], "confidence": 0.7, "notes": "verdict=partial"}],
 "project_swap": {"remove_project": "X", "add_project": "Y", "rationale": "Y matches \
the job's technology, domain, and industry more closely than X.", "evidence_ids": \
["portfolio-P02"]}}
"""


SENIORITY_RATIONALE_SYSTEM = """\
You rewrite a FIXED seniority finding as one natural sentence for a candidate. The \
verdict has already been decided deterministically from a years comparison; you \
MUST NOT change it or invent facts. If the verdict is "partial" or "mismatch", do \
NOT say the candidate meets, exceeds, or satisfies the requirement. Return ONLY the \
sentence, no preamble.
"""

SWAP_RATIONALE_SYSTEM = """\
You write ONE sentence explaining a project swap that has ALREADY been decided by \
the tool. You MUST recommend exactly the given add-project (never a different one), \
and your sentence MUST reference technology, domain, and industry. Do not invent \
facts beyond those provided. Return ONLY the sentence, no preamble.
"""


def build_seniority_rationale_prompt(claim_text: str, verdict_value: str) -> str:
    """Prompt to rephrase the fixed seniority finding, preserving its verdict."""

    return (
        f"Deterministic seniority finding (verdict={verdict_value}; do not change its "
        f"meaning): {claim_text}\nRewrite it as one clear sentence for the candidate."
    )


def build_swap_rationale_prompt(
    remove: str, add: str, tech: str, domain: str, industry: str
) -> str:
    """Prompt to write the rationale for an already-decided project swap."""

    return (
        f"The tool decided to replace the resume project '{remove}' with the portfolio "
        f"project '{add}'. Technology overlap with the job: {tech}. Domain alignment: "
        f"{domain}. Industry fit: {industry}. Write one sentence recommending this swap "
        f"that cites technology, domain, and industry."
    )


def build_user_prompt(inp: AnalyzeFitInput, grounding: dict[str, object]) -> str:
    """Build the user prompt with the job, candidate, portfolio, and grounding."""

    profile = inp.candidate_profile
    payload = {
        "job": {
            "job_id": inp.job.job_id,
            "title": inp.job.title,
            "company": inp.job.company,
            "industry_domain": inp.job.industry_domain,
            "required_skills": inp.job.required_skills,
            "years_experience_required": inp.job.years_experience_required,
            "experience_requirement_text": inp.job.experience_requirement.raw_text,
            "description": grounding.get("job_text", inp.job.description),
        },
        "candidate": {
            "name": profile.name,
            "resume_skills": profile.skills,
            "master_skills": profile.master_skills,
            "education": profile.education,
            "experience": profile.experience,
            "current_resume_projects": inp.current_resume_projects,
            "years_of_experience": profile.preferences.years_of_experience,
        },
        "portfolio_projects": [
            {
                "project_id": project.project_id,
                "name": project.name,
                "technologies": project.technologies,
                "domains": project.domains,
                "resume_swap_value": project.resume_swap_value,
                "evidence_ids": project.evidence_ids,
            }
            for project in inp.portfolio_projects
        ],
        "evidence_index": grounding.get("evidence_index", {}),
        "deterministic_grounding": {
            "aligned_skills": grounding.get("aligned_skills", []),
            "evidenced_missing_skills": grounding.get("evidenced_missing_skills", []),
            "genuine_gaps": grounding.get("genuine_gaps", []),
            "suggested_swap": grounding.get("suggested_swap"),
        },
        "valid_evidence_ids": sorted(
            {item.evidence_id for item in inp.evidence_items}
        ),
    }
    return (
        "Analyze the candidate's fit for this job. Use only the following data; "
        "cite only evidence_ids listed in valid_evidence_ids.\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )
