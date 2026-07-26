"""Compatibility surface for the job-search agent package."""

from src.agent.controller import (
    AgentController,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.errors import ToolExecutionError
from src.agent.graph import build_agent_graph, validate_artifact_output
from src.agent.state import AgentState, Phase, RunStatus, create_initial_state
from src.agent.tool_selection import (
    DeepInfraToolSelectionModel,
    ModelToolCall,
    validate_model_tool_call,
)
from src.config import (
    DEFAULT_JOBS_PATH,
    DEFAULT_PORTFOLIO_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RESUME_PATH,
    MAX_REVISION_ROUNDS,
    get_config,
    validate_runtime_requirements,
)
from src.domain import (
    CandidatePreferences,
    CandidateProfile,
    ChangeLogEntry,
    EvidenceClaim,
    EvidenceItem,
    ExperienceRequirement,
    Job,
    Portfolio,
    PortfolioProject,
    ProjectSwap,
    RejectedJob,
    ResumeData,
    StrictBaseModel,
    company_comparison_key,
    normalize_string_list,
    parse_experience_requirement,
)
from src.utils.evidence_validation import (
    evidence_supports_keyword,
    evidence_supports_project,
    evidence_supports_skill,
    evidence_supports_statement,
    job_evidence_supports_skill,
)
from src.utils.input_loading import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
    load_resume_evidence,
)
from src.utils.job_evidence import (
    build_job_evidence,
    job_evidence_id,
    job_skill_evidence_id,
)
from src.utils.latex import escape_latex, pdf_page_count, pdflatex_command, run_pdflatex
from src.utils.output_validation import MANDATORY_JOB_FILES, write_and_validate_outputs
from src.utils.skill_matching import canonicalize, category_members, skill_in_text

__all__ = [
    "AgentController",
    "AgentState",
    "CandidatePreferences",
    "CandidateProfile",
    "ChangeLogEntry",
    "DEFAULT_JOBS_PATH",
    "DEFAULT_PORTFOLIO_PATH",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_RESUME_PATH",
    "DeepInfraToolSelectionModel",
    "EvidenceClaim",
    "EvidenceItem",
    "ExperienceRequirement",
    "Job",
    "MANDATORY_JOB_FILES",
    "MAX_REVISION_ROUNDS",
    "ModelToolCall",
    "Phase",
    "Portfolio",
    "PortfolioProject",
    "ProjectSwap",
    "RejectedJob",
    "ResumeData",
    "RunStatus",
    "StrictBaseModel",
    "ToolExecutionError",
    "build_agent_graph",
    "build_job_evidence",
    "canonicalize",
    "category_members",
    "company_comparison_key",
    "create_initial_state",
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
    "escape_latex",
    "evidence_supports_keyword",
    "evidence_supports_project",
    "evidence_supports_skill",
    "evidence_supports_statement",
    "get_config",
    "invoke_new_run",
    "job_evidence_id",
    "job_evidence_supports_skill",
    "job_skill_evidence_id",
    "load_candidate_profile",
    "load_jobs_csv",
    "load_portfolio",
    "load_resume_data",
    "load_resume_evidence",
    "normalize_string_list",
    "parse_experience_requirement",
    "pdflatex_command",
    "pdf_page_count",
    "resume_run",
    "run_pdflatex",
    "skill_in_text",
    "validate_artifact_output",
    "validate_model_tool_call",
    "validate_runtime_requirements",
    "write_and_validate_outputs",
]
