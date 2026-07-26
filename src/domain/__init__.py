"""Domain contracts shared by orchestration and tools."""

from src.domain.base import (
    ChangeLogEntry,
    EvidenceClaim,
    EvidenceItem,
    ProjectSwap,
    StrictBaseModel,
    company_comparison_key,
    normalize_string_list,
)
from src.domain.jobs import (
    ExperienceRequirement,
    Job,
    RejectedJob,
    parse_experience_requirement,
)
from src.domain.profile import (
    CandidatePreferences,
    CandidateProfile,
    Portfolio,
    PortfolioProject,
    ResumeData,
)

__all__ = [
    "CandidatePreferences",
    "CandidateProfile",
    "ChangeLogEntry",
    "EvidenceClaim",
    "EvidenceItem",
    "ExperienceRequirement",
    "Job",
    "Portfolio",
    "PortfolioProject",
    "ProjectSwap",
    "RejectedJob",
    "ResumeData",
    "StrictBaseModel",
    "company_comparison_key",
    "normalize_string_list",
    "parse_experience_requirement",
]
