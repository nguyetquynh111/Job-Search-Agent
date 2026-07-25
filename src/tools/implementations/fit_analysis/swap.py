"""Project relevance ranking and the single project-swap recommendation.

One ranking drives BOTH ``project_analysis`` and the swap, so they can never
disagree (a project proposed for removal is, by construction, ranked weakest and
is reported as the weak one). Ubiquitous skills (e.g. Python) are down-weighted
because they appear in nearly every posting and carry little signal. Every swap
rationale cites technology, domain, and industry.

Swap identifier convention (the de-facto contract, documented in
``src/tools/fit_analysis/OUTPUT.md``), isolated behind :func:`build_project_swap`:

    remove_project = exact name as it appears in ``current_resume_projects``
    add_project    = exact PROJECT_NAME from ``portfolio.txt``
    the P0x portfolio ID is recoverable from ``evidence_ids`` (``portfolio-P0x``)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.schemas.common import PortfolioProject, ProjectSwap
from src.schemas.jobs import Job
from src.tools.implementations.fit_analysis import verdict
from src.tools.implementations.fit_analysis.aliases import canonicalize, category_members

logger = logging.getLogger(__name__)

# Skills so common across postings that matching them is weak evidence of fit.
UBIQUITOUS_SKILLS = {
    "python", "git", "sql", "docker", "linux", "bash", "rest", "api",
    "json", "agile", "github", "gitlab",
}

_DISTINCTIVE_WEIGHT = 2.0
_UBIQUITOUS_WEIGHT = 0.5
_DOMAIN_WEIGHT = 2.0
_INDUSTRY_WEIGHT = 1.5

# A swap is only recommended when the best external candidate beats the incumbent
# it would replace by at least this margin. The scale is one distinctive-skill match
# (2.0), so 2.0 means the incoming project must add at least one full distinctive
# dimension (a required skill or domain/industry axis) the outgoing one lacks — a
# barely-better project is worse than none, so below this we recommend NO swap.
SWAP_MIN_MARGIN = 2.0


def normalize_name(name: str) -> str:
    """Return a comparison key for project-name matching."""

    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _matches(tokens: list[str], text: str) -> list[str]:
    """Return the tokens whose words appear (word-boundary) in text."""

    haystack = re.sub(r"[^a-z0-9]+", " ", text.lower())
    found: list[str] = []
    for token in tokens:
        needle = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
        if needle and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack):
            found.append(token)
    return found


@dataclass
class ProjectScore:
    """A portfolio project scored for relevance to one job."""

    project: PortfolioProject
    score: float
    matched_distinctive: list[str]
    matched_ubiquitous: list[str]
    domain_matches: list[str]
    industry_matches: list[str]


def score_project(
    project: PortfolioProject, job: Job, required_canonical: list[str], job_text: str
) -> ProjectScore:
    """Score a portfolio project's relevance to a job (deterministic, weighted)."""

    tech_canonical = {canonicalize(tech) for tech in project.technologies}
    distinctive: list[str] = []
    ubiquitous: list[str] = []
    for required in required_canonical:
        # A required capability matches the project directly or via a category member
        # tool (e.g. the job's "APIs" is matched by the project's FastAPI).
        hit = required if required in tech_canonical else None
        if hit is None:
            for member in sorted(category_members(required)):
                if canonicalize(member) in tech_canonical:
                    hit = member  # name the concrete member for the rationale
                    break
        if hit is None:
            continue
        if canonicalize(hit) in UBIQUITOUS_SKILLS or required in UBIQUITOUS_SKILLS:
            ubiquitous.append(hit)
        else:
            distinctive.append(hit)

    domain_text = f"{job.industry_domain} {job_text}"
    industry_text = f"{job.industry_domain} {job.company_details} {job_text}"
    domain_matches = _matches(project.domains, domain_text)
    industry_matches = _matches(project.industries, industry_text)

    swap_value_text = f"{project.resume_swap_value or ''} {' '.join(project.keywords)}".lower()
    swap_value_bonus = sum(
        0.5 for skill in distinctive if skill in swap_value_text
    )

    score = (
        _DISTINCTIVE_WEIGHT * len(distinctive)
        + _UBIQUITOUS_WEIGHT * len(ubiquitous)
        + (_DOMAIN_WEIGHT if domain_matches else 0.0)
        + (_INDUSTRY_WEIGHT if industry_matches else 0.0)
        + swap_value_bonus
    )
    return ProjectScore(
        project=project,
        score=score,
        matched_distinctive=distinctive,
        matched_ubiquitous=ubiquitous,
        domain_matches=domain_matches,
        industry_matches=industry_matches,
    )


def rank_projects(
    portfolio_projects: list[PortfolioProject],
    job: Job,
    required_canonical: list[str],
    job_text: str,
) -> list[ProjectScore]:
    """Rank all portfolio projects by relevance, with a stable tie-break."""

    scored = [
        score_project(project, job, required_canonical, job_text)
        for project in portfolio_projects
    ]
    scored.sort(key=lambda item: (-item.score, item.project.project_id))
    return scored


@dataclass
class CurrentVerdict:
    """A current resume project's relevance verdict, from the shared ranking."""

    name: str
    project: PortfolioProject | None
    score: float
    verdict: str
    distinctive: list[str]
    domain_matches: list[str]
    industry_matches: list[str]
    removed: bool = False


@dataclass
class SwapDecision:
    """The single swap recommendation plus the shared-ranking context."""

    swap: ProjectSwap | None
    rankings: list[ProjectScore]
    current_verdicts: list[CurrentVerdict] = field(default_factory=list)
    weak_slot_notes: list[str] = field(default_factory=list)
    unbound: list[str] = field(default_factory=list)


def _tech_phrase(score: ProjectScore) -> str:
    if score.matched_distinctive:
        return f"stronger technology match ({', '.join(score.matched_distinctive)})"
    if score.matched_ubiquitous:
        return "only shared general tooling (e.g. " + ", ".join(score.matched_ubiquitous) + ")"
    return "different but relevant technologies"


def _domain_phrase(score: ProjectScore) -> str:
    if score.domain_matches:
        return f"domain alignment ({', '.join(score.domain_matches)})"
    return "broader domain overlap"


def _industry_phrase(score: ProjectScore) -> str:
    if score.industry_matches:
        return f"industry fit ({', '.join(score.industry_matches)})"
    return "transferable industry experience"


def build_project_swap(
    current_names: list[str],
    portfolio_projects: list[PortfolioProject],
    job: Job,
    required_canonical: list[str],
    job_text: str,
) -> SwapDecision:
    """Recommend at most one swap and the per-project verdicts from one ranking."""

    rankings = rank_projects(portfolio_projects, job, required_canonical, job_text)
    score_by_id = {item.project.project_id: item for item in rankings}
    lookup = {normalize_name(p.name): p for p in portfolio_projects}

    current: list[CurrentVerdict] = []
    unbound: list[str] = []
    for name in current_names:
        bound = lookup.get(normalize_name(name))
        if bound is None:
            unbound.append(name)
            current.append(CurrentVerdict(name, None, 0.0, verdict.PARTIAL, [], [], []))
            logger.info("Resume project not bound to portfolio: %s", name)
            continue
        score = score_by_id[bound.project_id]
        current.append(
            CurrentVerdict(
                name=name,
                project=bound,
                score=score.score,
                verdict=verdict.MATCH if (score.matched_distinctive or score.domain_matches) else verdict.PARTIAL,
                distinctive=score.matched_distinctive,
                domain_matches=score.domain_matches,
                industry_matches=score.industry_matches,
            )
        )

    current_ids = {c.project.project_id for c in current if c.project}
    best_external = next(
        (item for item in rankings if item.project.project_id not in current_ids), None
    )
    bound_current = [c for c in current if c.project is not None]

    swap: ProjectSwap | None = None
    weak_slot_notes: list[str] = []
    if bound_current and best_external is not None:
        weakest = min(bound_current, key=lambda c: c.score)
        if best_external.score - weakest.score >= SWAP_MIN_MARGIN:
            weakest.removed = True
            weakest.verdict = verdict.MISMATCH
            add_score = best_external
            swap = ProjectSwap(
                remove_project=weakest.name,
                add_project=best_external.project.name,
                rationale=(
                    f"'{best_external.project.name}' offers {_tech_phrase(add_score)}, "
                    f"{_domain_phrase(add_score)}, and {_industry_phrase(add_score)} — a better "
                    f"match than '{weakest.name}' for this role."
                ),
                evidence_ids=list(best_external.project.evidence_ids),
            )
            for c in bound_current:
                if c is not weakest and c.score <= best_external.score / 2:
                    weak_slot_notes.append(
                        f"Resume project '{c.name}' is also a weak match, but only one swap "
                        f"is recommended per job."
                    )

    return SwapDecision(
        swap=swap,
        rankings=rankings,
        current_verdicts=current,
        weak_slot_notes=weak_slot_notes,
        unbound=unbound,
    )
