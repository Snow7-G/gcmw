"""Application configuration with validation and secret references.

Secrets are never stored here. In production, missing required secret
references cause startup to fail.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_MODALITIES = ("text", "audio", "image", "video")


def _normalized_host(value: str) -> str:
    try:
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_loopback_url(value: str) -> bool:
    host = _normalized_host(value)
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        return ip.is_loopback or ip.is_unspecified
    except ValueError:
        # Reject legacy IPv4 forms that resolve to loopback/unspecified without DNS.
        try:
            packed = socket.inet_aton(host)
            ip = ipaddress.IPv4Address(packed)
            return ip.is_loopback or ip.is_unspecified
        except OSError:
            return False


def _is_invalid_storage_url(value: str, allowed_schemes: set[str]) -> bool:
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return True
    if parsed.scheme.lower() not in allowed_schemes:
        return True
    if not parsed.hostname:
        return True
    return bool(port is not None and port == 0)


class CloudProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = "dashscope_realtime"
    adapter: str = "cloud_realtime"
    model: str = "qwen3.5-omni-plus-realtime"
    api_base: str = "https://dashscope.aliyuncs.com/api/v1"
    api_key_env: str = "GCMW_CLOUD_API_KEY"
    transport: str = "websocket"
    modalities: tuple[str, ...] = ("text", "audio", "image", "video")
    timeout_ms: int = Field(15_000, gt=0)
    connect_timeout_ms: int = Field(5_000, gt=0)

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v):
        if not v or any(item not in SUPPORTED_MODALITIES for item in v):
            raise ValueError(
                f"modalities must be non-empty subset of {SUPPORTED_MODALITIES}"
            )
        return v


class LocalProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = "vllm"
    adapter: str = "local_omni"
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    api_base: str = "http://127.0.0.1:8001/v1"
    api_key_env: str | None = None
    modalities: tuple[str, ...] = ("text", "audio", "image", "video")
    timeout_ms: int = Field(30_000, gt=0)
    connect_timeout_ms: int = Field(5_000, gt=0)

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v):
        if not v or any(item not in SUPPORTED_MODALITIES for item in v):
            raise ValueError(
                f"modalities must be non-empty subset of {SUPPORTED_MODALITIES}"
            )
        return v


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    app_name: str = "gcmw-agent"
    # "test" behaves exactly like development: the in-memory repository and
    # admission store are legitimate there, while staging/production fail closed.
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    active_provider: Literal["mock", "cloud", "local"] = "mock"

    # NAME of the environment variable holding the device credentials
    # ([{"tenant_id":…,"device_id":…,"token":…}, …]). Secrets are never stored
    # in Settings — only their env var references.
    auth_credentials_env: str = "GCMW_DEVICE_CREDENTIALS"

    redis_url: str = "redis://127.0.0.1:6379/0"
    database_url: str = "postgresql://gcmw:gcmw@127.0.0.1:5432/gcmw"
    vector_database_url: str | None = None

    run_timeout_ms: int = Field(15_000, gt=0)
    answer_token_budget: int = Field(400, gt=0)
    max_tool_calls: int = Field(4, gt=0)
    max_agent_handoffs: int = Field(2, ge=0)
    max_revisions: int = Field(1, ge=0)

    # request rate limits (#66): requests per minute per tenant/device/session.
    # 0 disables that scope. Process-local counters (see app.api.v1.rate_limit).
    rate_limit_tenant_per_minute: int = Field(600, ge=0)
    rate_limit_device_per_minute: int = Field(300, ge=0)
    rate_limit_session_per_minute: int = Field(120, ge=0)

    model_timeout_ms: int = Field(15_000, gt=0)
    tool_timeout_ms: int = Field(5_000, gt=0)
    sse_timeout_ms: int = Field(30_000, gt=0)

    cloud: CloudProviderConfig = Field(default_factory=CloudProviderConfig)
    local: LocalProviderConfig = Field(default_factory=LocalProviderConfig)

    @model_validator(mode="after")
    def _validate_model(self):
        self.validate_for_environment()
        return self

    def validate_for_environment(self) -> None:
        missing = []
        if self.active_provider == "cloud" and not os.getenv(self.cloud.api_key_env):
            missing.append(self.cloud.api_key_env)
        if self.environment in {"production", "staging"}:
            if (
                not self.redis_url
                or _is_invalid_storage_url(self.redis_url, {"redis", "rediss"})
                or _is_loopback_url(self.redis_url)
            ):
                missing.append(
                    "GCMW_REDIS_URL (production must be valid redis/rediss remote URL, not localhost)"
                )
            if (
                not self.database_url
                or _is_invalid_storage_url(
                    self.database_url, {"postgres", "postgresql"}
                )
                or _is_loopback_url(self.database_url)
            ):
                missing.append(
                    "GCMW_DATABASE_URL (production must be valid postgres/postgresql remote URL, not localhost)"
                )
            if self.active_provider != "local":
                missing.append(
                    "GCMW_ACTIVE_PROVIDER (production/staging must be local)"
                )
            if (
                self.active_provider == "local"
                and self.local.api_key_env
                and not os.getenv(self.local.api_key_env)
            ):
                missing.append(self.local.api_key_env)
        if missing:
            raise RuntimeError(
                f"Missing/invalid required configuration in {self.environment}: {missing}"
            )

    @classmethod
    def from_env(cls) -> Settings:
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc

        def _bool(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean, got: {raw!r}")

        cloud = CloudProviderConfig(
            provider=os.getenv("GCMW_CLOUD_PROVIDER", "dashscope_realtime"),
            adapter=os.getenv("GCMW_CLOUD_ADAPTER", "cloud_realtime"),
            model=os.getenv("GCMW_CLOUD_MODEL", "qwen3.5-omni-plus-realtime"),
            api_base=os.getenv(
                "GCMW_CLOUD_API_BASE", "https://dashscope.aliyuncs.com/api/v1"
            ),
            api_key_env=os.getenv("GCMW_CLOUD_API_KEY_ENV", "GCMW_CLOUD_API_KEY"),
            transport="websocket",
            modalities=tuple(
                os.getenv("GCMW_CLOUD_MODALITIES", "text,audio,image,video").split(",")
            ),
            timeout_ms=_int("GCMW_CLOUD_TIMEOUT_MS", 15_000),
            connect_timeout_ms=_int("GCMW_CLOUD_CONNECT_TIMEOUT_MS", 5_000),
        )
        local = LocalProviderConfig(
            provider=os.getenv("GCMW_LOCAL_PROVIDER", "vllm"),
            adapter=os.getenv("GCMW_LOCAL_ADAPTER", "local_omni"),
            model=os.getenv("GCMW_LOCAL_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            api_base=os.getenv("GCMW_LOCAL_API_BASE", "http://127.0.0.1:8001/v1"),
            api_key_env=os.getenv("GCMW_LOCAL_API_KEY_ENV") or None,
            modalities=tuple(
                os.getenv("GCMW_LOCAL_MODALITIES", "text,audio,image,video").split(",")
            ),
            timeout_ms=_int("GCMW_LOCAL_TIMEOUT_MS", 30_000),
            connect_timeout_ms=_int("GCMW_LOCAL_CONNECT_TIMEOUT_MS", 5_000),
        )
        settings = cls(
            environment=os.getenv("GCMW_ENV", "development"),
            debug=_bool("GCMW_DEBUG"),
            active_provider=os.getenv("GCMW_ACTIVE_PROVIDER", "mock"),
            auth_credentials_env=os.getenv(
                "GCMW_AUTH_CREDENTIALS_ENV", "GCMW_DEVICE_CREDENTIALS"
            ),
            redis_url=os.getenv("GCMW_REDIS_URL", "redis://127.0.0.1:6379/0"),
            database_url=os.getenv(
                "GCMW_DATABASE_URL", "postgresql://gcmw:gcmw@127.0.0.1:5432/gcmw"
            ),
            vector_database_url=os.getenv("GCMW_VECTOR_DATABASE_URL") or None,
            run_timeout_ms=_int("GCMW_RUN_TIMEOUT_MS", 15_000),
            answer_token_budget=_int("GCMW_ANSWER_TOKEN_BUDGET", 400),
            max_tool_calls=_int("GCMW_MAX_TOOL_CALLS", 4),
            max_agent_handoffs=_int("GCMW_MAX_AGENT_HANDOFFS", 2),
            max_revisions=_int("GCMW_MAX_REVISIONS", 1),
            rate_limit_tenant_per_minute=_int("GCMW_RATE_LIMIT_TENANT_PER_MINUTE", 600),
            rate_limit_device_per_minute=_int("GCMW_RATE_LIMIT_DEVICE_PER_MINUTE", 300),
            rate_limit_session_per_minute=_int(
                "GCMW_RATE_LIMIT_SESSION_PER_MINUTE", 120
            ),
            model_timeout_ms=_int("GCMW_MODEL_TIMEOUT_MS", 15_000),
            tool_timeout_ms=_int("GCMW_TOOL_TIMEOUT_MS", 5_000),
            sse_timeout_ms=_int("GCMW_SSE_TIMEOUT_MS", 30_000),
            cloud=cloud,
            local=local,
        )
        return settings
