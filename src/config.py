"""Application configuration."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration sourced from environment variables."""

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    deepinfra_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPINFRA_API_KEY", "")
    )
    deepinfra_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPINFRA_BASE_URL", DEEPINFRA_BASE_URL)
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "outputs"))
    )

    @property
    def memory_file(self) -> Path:
        """Return the memory file inside the configured output directory."""

        return self.output_dir / "memory.json"

    @property
    def checkpoint_db(self) -> Path:
        """Return the checkpoint database inside the configured output directory."""

        return self.output_dir / "checkpoints.sqlite"


def get_config() -> AppConfig:
    """Return the current process configuration."""

    return AppConfig()


def validate_runtime_requirements(config: AppConfig | None = None) -> None:
    """Fail early when the documented production workflow cannot finish."""

    active = config or get_config()
    errors: list[str] = []
    if shutil.which("pdflatex") is None:
        errors.append(
            "pdflatex is not installed or not on PATH; PDF artifacts cannot be generated"
        )
    if not active.deepinfra_api_key or not active.llm_model:
        errors.append(
            "LLM_MODEL and DEEPINFRA_API_KEY are required for the single-agent controller"
        )
    missing_langfuse = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", active.langfuse_public_key),
            ("LANGFUSE_SECRET_KEY", active.langfuse_secret_key),
        )
        if not value.strip()
    ]
    if missing_langfuse:
        errors.append(
            "Langfuse credentials are required: " + ", ".join(missing_langfuse)
        )
    if errors:
        raise RuntimeError("Runtime preflight failed: " + "; ".join(errors) + ".")
