"""Compatibility wrapper for runtime configuration.

Core code imports :mod:`src.config`; this module remains for the Streamlit app
and older tests that patch the application configuration import path.
"""

from src.config.settings import (
    AppConfig,
    DEEPINFRA_BASE_URL,
    DEFAULT_JOBS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PORTFOLIO_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RESUME_PATH,
    MAX_REVISION_ROUNDS,
    TOP_JOB_COUNT,
    get_config,
    shutil,
    validate_runtime_requirements,
)

__all__ = [
    "AppConfig",
    "DEEPINFRA_BASE_URL",
    "DEFAULT_JOBS_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PORTFOLIO_PATH",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_RESUME_PATH",
    "MAX_REVISION_ROUNDS",
    "TOP_JOB_COUNT",
    "get_config",
    "shutil",
    "validate_runtime_requirements",
]
