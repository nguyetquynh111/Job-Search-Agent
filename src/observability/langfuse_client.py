"""Required Langfuse client creation."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"
STATUS_CONNECTED = "Observability: Langfuse connected"
STATUS_UNAVAILABLE = "Observability: Langfuse unavailable"
STATUS_NOOP = "Observability: Local no-op tracing"


@dataclass(frozen=True)
class LangfuseConfig:
    """Langfuse configuration sourced from environment variables."""

    public_key: str
    secret_key: str
    host: str


@dataclass(frozen=True)
class LangfuseClientStatus:
    """Langfuse client availability status."""

    enabled: bool
    message: str
    mode: str
    sdk_version: str | None = None


def load_langfuse_config() -> LangfuseConfig:
    """Load Langfuse configuration from environment variables."""

    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or DEFAULT_LANGFUSE_HOST
    ).strip()
    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Langfuse credentials are required: " + ", ".join(missing)
        )
    if not _valid_host(host):
        raise ValueError("Invalid LANGFUSE_HOST")
    return LangfuseConfig(public_key=public_key, secret_key=secret_key, host=host)


def create_langfuse_client() -> tuple[Any | None, LangfuseClientStatus]:
    """Create and authenticate the required Langfuse v2 client."""

    config = load_langfuse_config()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
        )
        if hasattr(client, "auth_check") and not client.auth_check():
            raise RuntimeError("Langfuse authentication failed")
        logger.info("Langfuse enabled")
        return client, LangfuseClientStatus(
            enabled=True,
            message=STATUS_CONNECTED,
            mode="langfuse",
            sdk_version=_langfuse_version(),
        )
    except Exception:
        logger.error("Langfuse initialization failed")
        raise RuntimeError(
            "Langfuse initialization failed; connected tracing is required"
        ) from None


def _valid_host(host: str) -> bool:
    parsed = urlparse(host)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _langfuse_version() -> str | None:
    try:
        return importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        return None
