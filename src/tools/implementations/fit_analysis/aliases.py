"""Explicit skill alias map and canonicalization (no fuzzy matching).

The map is intentionally small and hand-curated for this project's AI/ML
vocabulary. Surface-form variants are collapsed to a single canonical token so
that job requirements, resume skills, portfolio tech stacks, master skills, and
memory facts can be compared exactly. Unknown tokens pass through normalized
(lower-cased, whitespace-collapsed) rather than being force-matched.
"""

from __future__ import annotations

import re

# canonical -> variant surface forms (canonical itself is always included).
CANONICAL_ALIASES: dict[str, set[str]] = {
    "machine learning": {"ml", "machine-learning"},
    "computer vision": {"cv", "vision"},
    "natural language processing": {"nlp"},
    "large language model": {"llm", "llms", "large language models"},
    "generative ai": {"genai", "gen ai", "generative-ai"},
    "retrieval-augmented generation": {"rag"},
    "pytorch": {"torch"},
    "tensorflow": {"tf"},
    "scikit-learn": {"sklearn", "scikit learn", "scikitlearn"},
    "kubernetes": {"k8s"},
    "amazon web services": {"aws"},
    "google cloud": {"gcp", "google cloud platform"},
    "postgresql": {"postgres", "psql"},
    "mlops": {"ml ops", "ml-ops"},
    "opencv": {"open cv"},
    "fastapi": {"fast api"},
}

# Category -> concrete member skills that satisfy it. A required skill naming a
# CAPABILITY (what postings ask for) is satisfied by evidence of any member TOOL
# (what portfolios/resumes name). Explicit map only — no fuzzy/embedding matching.
# Members are the instances observed across data/portfolio.txt, data/resume.tex, and
# data/jobs.csv, plus standard well-known members of each category. Member display
# names are canonicalized (see ``canonicalize``) when matched against evidence.
_AGENTS = {"AutoGen", "LangGraph", "LangChain", "CrewAI"}
_APIS = {"FastAPI", "REST", "WebSocket", "GraphQL", "Flask"}
_CLOUD = {"Docker", "Kubernetes", "AWS SageMaker", "AWS", "Google Cloud", "Azure"}
_VECTOR_DBS = {"Pinecone", "Weaviate", "FAISS", "Chroma", "pgvector", "Milvus"}
_MLOPS = {"MLflow", "Airflow", "CI/CD", "Kubeflow"}
_DEEP_LEARNING = {"PyTorch", "TensorFlow", "CNN", "Vision Transformer", "EfficientNet"}

CATEGORY_SKILLS: dict[str, set[str]] = {
    "agents": _AGENTS,
    "ai agents": _AGENTS,
    "agent workflows": _AGENTS,
    "agentic ai": _AGENTS,
    "agent frameworks": _AGENTS,
    "apis": _APIS,
    "api": _APIS,
    "cloud deployment": _CLOUD,
    "cloud": _CLOUD,
    "cloud platforms": _CLOUD,
    "cloud platform": _CLOUD,
    "model deployment": _CLOUD,
    "vector databases": _VECTOR_DBS,
    "vector database": _VECTOR_DBS,
    "mlops": _MLOPS,
    "deep learning": _DEEP_LEARNING,
}

# variant surface form -> canonical token (built once).
_VARIANT_TO_CANONICAL: dict[str, str] = {}
for _canonical, _variants in CANONICAL_ALIASES.items():
    _VARIANT_TO_CANONICAL[_canonical] = _canonical
    for _variant in _variants:
        _VARIANT_TO_CANONICAL[_variant] = _canonical


def _normalize(token: str) -> str:
    """Lower-case a token and collapse internal whitespace."""

    return re.sub(r"\s+", " ", token.strip().lower())


def canonicalize(token: str) -> str:
    """Return the canonical form of a skill token, or its normalized surface form."""

    normalized = _normalize(token)
    return _VARIANT_TO_CANONICAL.get(normalized, normalized)


def category_members(canonical: str) -> set[str]:
    """Return the concrete member skills for a category, or empty if not a category."""

    return CATEGORY_SKILLS.get(canonical, set())


def variants(canonical: str) -> set[str]:
    """Return every surface form (canonical plus aliases) for a canonical token."""

    normalized = _normalize(canonical)
    root = _VARIANT_TO_CANONICAL.get(normalized, normalized)
    forms = {root}
    forms.update(CANONICAL_ALIASES.get(root, set()))
    return forms


def skill_in_text(canonical: str, text: str) -> bool:
    """Return whether any surface form of ``canonical`` appears in ``text``.

    Word boundaries are enforced so short aliases (``ml``, ``cv``, ``tf``) do not
    match inside unrelated words.
    """

    haystack = _normalize(text)
    for form in variants(canonical):
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", haystack):
            return True
    return False
