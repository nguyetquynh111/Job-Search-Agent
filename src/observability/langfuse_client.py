"""Langfuse client creation with safe local fallback."""

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


def load_langfuse_config() -> LangfuseConfig | None:
    """Load Langfuse configuration from environment variables."""

    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    host = (os.getenv("LANGFUSE_HOST") or DEFAULT_LANGFUSE_HOST).strip()
    if not public_key or not secret_key:
        logger.info("Langfuse disabled because configuration is missing")
        return None
    if not _valid_host(host):
        raise ValueError("Invalid LANGFUSE_HOST")
    return LangfuseConfig(public_key=public_key, secret_key=secret_key, host=host)


def create_langfuse_client() -> tuple[Any | None, LangfuseClientStatus]:
    """Create and authenticate a Langfuse v2 client when configured."""

    try:
        config = load_langfuse_config()
        if config is None:
            return None, LangfuseClientStatus(
                enabled=False,
                message=STATUS_NOOP,
                mode="noop",
            )
        from langfuse import Langfuse

        client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
        )
        if hasattr(client, "auth_check") and not client.auth_check():
            logger.warning("Langfuse initialization failed; using no-op tracing")
            return None, LangfuseClientStatus(
                enabled=False,
                message=STATUS_UNAVAILABLE,
                mode="unavailable",
                sdk_version=_langfuse_version(),
            )
        logger.info("Langfuse enabled")
        return client, LangfuseClientStatus(
            enabled=True,
            message=STATUS_CONNECTED,
            mode="langfuse",
            sdk_version=_langfuse_version(),
        )
    except Exception:
        logger.warning("Langfuse initialization failed; using no-op tracing")
        return None, LangfuseClientStatus(
            enabled=False,
            message=STATUS_UNAVAILABLE,
            mode="unavailable",
            sdk_version=_langfuse_version(),
        )


def _valid_host(host: str) -> bool:
    parsed = urlparse(host)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _langfuse_version() -> str | None:
    try:
        return importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        return None
