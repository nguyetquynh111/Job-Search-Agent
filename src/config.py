"""Application configuration."""

from __future__ import annotations

import os
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
