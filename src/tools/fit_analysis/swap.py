"""Project relevance ranking and the single project-swap recommendation.

One ranking drives BOTH ``project_analysis`` and the swap, so they can never
disagree (a project proposed for removal is, by construction, ranked weakest and
is reported as the weak one). Ubiquitous skills (e.g. Python) are down-weighted
because they appear in nearly every posting and carry little signal. Every swap
rationale cites technology, domain, and industry.

Swap identifier convention (the de-facto contract, documented in
the ``ProjectSwap`` schema), isolated behind :func:`build_project_swap`:

    remove_project = exact name as it appears in ``current_resume_projects``
    add_project    = exact PROJECT_NAME from ``portfolio.txt``
    the P0x portfolio ID is recoverable from ``evidence_ids`` (``portfolio-P0x``)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.domain import PortfolioProject, ProjectSwap
from src.domain import Job
import src.tools.fit_analysis.rules as rules
from src.utils.skill_matching import canonicalize, category_members

logger = logging.getLogger(__name__)

# Common skills carry less weight.
UBIQUITOUS_SKILLS = {
    "python",
    "git",
    "sql",
    "docker",
    "linux",
    "bash",
    "rest",
    "api",
    "json",
    "agile",
    "github",
    "gitlab",
}

_DISTINCTIVE_WEIGHT = 2.0
_UBIQUITOUS_WEIGHT = 0.5
_DOMAIN_WEIGHT = 2.0
_INDUSTRY_WEIGHT = 1.5

# Require one full distinctive-skill gain before suggesting a swap.
SWAP_MIN_MARGIN = 2.0


def normalize_name(name: str) -> str:
    """Return a comparison key for project-name matching."""

    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


# Ignore grammatical filler during phrase matching.
_PHRASE_STOPWORDS = {
    "and",
    "or",
    "of",
    "for",
    "the",
    "a",
    "an",
    "with",
    "in",
    "to",
    "on",
}

# Ignore generic words that cannot prove domain alignment alone.
_WEAK_PHRASE_WORDS = {
    "ai",
    "data",
    "deep",
    "real",
    "time",
    "web",
    "system",
    "platform",
    "analysis",
    "engineering",
    "computer",
    "learning",
    "model",
    "science",
    "technology",
    "application",
    "service",
    "intelligence",
    "general",
    "support",
    "based",
    "using",
    "product",
    "solution",
    "software",
    "development",
    "digital",
    "tool",
    "framework",
    "quality",
}


def _stem(word: str) -> str:
    """Strip a trailing plural 's' so "systems" and "system" compare equal."""

    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _content_words(phrase: str) -> list[str]:
    """Return a phrase's stemmed content words, without grammatical glue."""

    words = [w for w in re.split(r"[^a-z0-9]+", phrase.lower()) if w]
    return [_stem(w) for w in words if w not in _PHRASE_STOPWORDS]


def _matches(tokens: list[str], text: str) -> list[str]:
    """Return the tokens whose content words sufficiently overlap ``text``.

    Whole-phrase containment was too strict to be defensible: a portfolio domain of
    "Recommendation and Ranking" scored zero against a posting asking for
    "recommendation systems", so a genuinely relevant project lost to an unrelated
    one. A phrase now matches when at least half of its content words appear
    (word-boundary, plural-stemmed) AND at least one matched word carries real
    signal, which keeps shared filler words from counting as alignment.
    """

    haystack = {_stem(w) for w in re.split(r"[^a-z0-9]+", text.lower()) if w}
    found: list[str] = []
    for token in tokens:
        words = _content_words(token)
        if not words:
            continue
        hit = [word for word in words if word in haystack]
        needed = (len(words) + 1) // 2
        if len(hit) >= needed and any(word not in _WEAK_PHRASE_WORDS for word in hit):
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
        # A concrete tool can satisfy a broader required skill.
        hit = required if required in tech_canonical else None
        if hit is None:
            for member in sorted(category_members(required)):
                if canonicalize(member) in tech_canonical:
                    hit = member  # Use the concrete tool in the rationale.
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

    swap_value_text = (
        f"{project.resume_swap_value or ''} {' '.join(project.keywords)}".lower()
    )
    swap_value_bonus = sum(0.5 for skill in distinctive if skill in swap_value_text)

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
        return (
            "only shared general tooling (e.g. "
            + ", ".join(score.matched_ubiquitous)
            + ")"
        )
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
            current.append(CurrentVerdict(name, None, 0.0, rules.PARTIAL, [], [], []))
            logger.info("Resume project not bound to portfolio: %s", name)
            continue
        score = score_by_id[bound.project_id]
        current.append(
            CurrentVerdict(
                name=name,
                project=bound,
                score=score.score,
                verdict=rules.MATCH
                if (score.matched_distinctive or score.domain_matches)
                else rules.PARTIAL,
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
            weakest.verdict = rules.MISMATCH
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


def score_for(decision: SwapDecision, project_id: str) -> float:
    """Return a project's relevance score from an existing ranking."""

    for item in decision.rankings:
        if item.project.project_id == project_id:
            return item.score
    return 0.0


def resolve_portfolio_project(
    reference: str | None, portfolio_projects: list[PortfolioProject]
) -> PortfolioProject | None:
    """Resolve a portfolio project by ``project_id`` or by name, or return None."""

    if not reference:
        return None
    literal = reference.strip().lower()
    for project in portfolio_projects:
        if project.project_id.strip().lower() == literal:
            return project
    key = normalize_name(reference)
    for project in portfolio_projects:
        if normalize_name(project.name) == key:
            return project
    return None


def validate_proposed_swap(
    decision: SwapDecision,
    remove_project: str | None,
    add_project: str | None,
    current_names: list[str],
    portfolio_projects: list[PortfolioProject],
) -> tuple[PortfolioProject | None, str | None, str | None]:
    """Validate an externally proposed swap against the portfolio and the ranking.

    Returns ``(add, resolved_remove_name, rejection_reason)``. A proposal is accepted
    only when the add-project exists in the portfolio, is not already on the resume,
    the remove-project is a current resume project, and the add beats the outgoing
    project by :data:`SWAP_MIN_MARGIN` under the same ranking the deterministic path
    uses -- so an accepted proposal is never weaker than what the tool would pick.
    """

    add = resolve_portfolio_project(add_project, portfolio_projects)
    if add is None:
        return None, None, f"add_project {add_project!r} is not in the portfolio"

    current_by_key = {normalize_name(name): name for name in current_names}
    if normalize_name(add.name) in current_by_key:
        return (
            None,
            None,
            f"add_project {add.name!r} is already a current resume project",
        )

    resolved_remove = current_by_key.get(normalize_name(remove_project or ""))
    if resolved_remove is None:
        return (
            None,
            None,
            f"remove_project {remove_project!r} is not a current resume project",
        )

    outgoing = next(
        (
            c
            for c in decision.current_verdicts
            if c.name == resolved_remove and c.project
        ),
        None,
    )
    if outgoing is None or outgoing.project is None:
        return (
            None,
            None,
            f"remove_project {resolved_remove!r} has no portfolio counterpart to score",
        )

    margin = score_for(decision, add.project_id) - score_for(
        decision, outgoing.project.project_id
    )
    if margin < SWAP_MIN_MARGIN:
        return (
            None,
            None,
            (
                f"{add.name!r} does not beat {resolved_remove!r} by the required margin "
                f"(delta={margin:.2f}, minimum={SWAP_MIN_MARGIN})"
            ),
        )
    return add, resolved_remove, None


def choose_swap(
    decision: SwapDecision,
    remove_name: str,
    add: PortfolioProject,
    rationale: str | None = None,
) -> SwapDecision:
    """Return a SwapDecision reflecting an externally chosen (remove, add) pair.

    The chosen outgoing project is re-marked as the weak one on the SAME shared
    ranking, so ``project_analysis`` and the swap still cannot disagree: whatever
    project is recommended for removal is reported as the weak slot, never as
    "aligns well".
    """

    add_score = next(
        (
            item
            for item in decision.rankings
            if item.project.project_id == add.project_id
        ),
        None,
    )
    for entry in decision.current_verdicts:
        if entry.project is None:
            continue
        if entry.name == remove_name:
            entry.removed = True
            entry.verdict = rules.MISMATCH
        else:
            entry.removed = False
            entry.verdict = (
                rules.MATCH
                if (entry.distinctive or entry.domain_matches)
                else rules.PARTIAL
            )

    if add_score is None:
        return decision
    default_rationale = (
        f"'{add.name}' offers {_tech_phrase(add_score)}, {_domain_phrase(add_score)}, and "
        f"{_industry_phrase(add_score)} — a better match than '{remove_name}' for this role."
    )
    decision.swap = ProjectSwap(
        remove_project=remove_name,
        add_project=add.name,
        rationale=rationale or default_rationale,
        evidence_ids=list(add.evidence_ids),
    )
    decision.weak_slot_notes = [
        f"Resume project '{c.name}' is also a weak match, but only one swap "
        f"is recommended per job."
        for c in decision.current_verdicts
        if c.project is not None
        and c.name != remove_name
        and c.score <= add_score.score / 2
    ]
    return decision
