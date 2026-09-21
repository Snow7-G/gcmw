"""Agent API request/response contracts (issue #36, API surface v1).

Notes:
- v1 input is text-only; voice/image content parts arrive with the
  multimodal work (#51) under the same envelope shape;
- ``Channel`` comes from contracts.common and is a SESSION property — runs
  inherit the session channel and never carry their own;
- tenant/device identities are NOT accepted here: they derive from the
  authenticated DevicePrincipal (see api/v1/auth.py); full credential
  validation lands in #36c.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from .common import Channel
from .identity import (
    MAX_DEVICE_ID_LENGTH,
    MAX_TENANT_ID_LENGTH,
    MIN_IDENTITY_LENGTH,
)
from .run import RunState


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Channel = Channel.TEXT
    locale: str = Field("zh-CN", min_length=2, max_length=16)
    # Optional client idempotency key (voice adapter): a distributed POST that
    # times out has an UNKNOWN outcome — the server may already have committed.
    # Retrying with the SAME key replays the original session instead of
    # creating a second one. Absent → behaviour unchanged. Identity fields
    # (tenant/device/session) are NEVER accepted here: they derive from the
    # authenticated DevicePrincipal.
    idempotency_key: str | None = Field(None, min_length=1, max_length=128)


class SessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(
        ..., min_length=MIN_IDENTITY_LENGTH, max_length=MAX_TENANT_ID_LENGTH
    )
    device_id: str = Field(
        ..., min_length=MIN_IDENTITY_LENGTH, max_length=MAX_DEVICE_ID_LENGTH
    )
    channel: Channel
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ttl_s: int = Field(1800, ge=1)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field("text", pattern="^text$")  # v1: text only
    text: str = Field(..., min_length=1, max_length=4000)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, max_length=128)
    input: RunInput
    idempotency_key: str = Field(..., min_length=1, max_length=128)


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    state: RunState = RunState.ACCEPTED
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cancelled: bool = False
