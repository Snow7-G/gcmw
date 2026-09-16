"""Dual-layer SSE and run event contracts (issue #30).

Protocol summary (V2.3 §9):
- every event carries ``protocol_version``; only the pinned version is accepted;
- every event belongs to exactly one layer — ``process`` (auditable facts only:
  stage, retrieved evidence, reflection decision) or ``answer`` (gated answer
  fragments, citations, actions);
- ``data`` is key-allowlisted per event type so raw chain-of-thought, prompts,
  internal policy text or tool internals can never ride through the protocol;
- a run has exactly one terminal event (``run.completed``); ``heartbeat`` lets
  intermediaries and clients distinguish keep-alives from protocol events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .identity import (
    MAX_DEVICE_ID_LENGTH,
    MAX_TENANT_ID_LENGTH,
    MIN_IDENTITY_LENGTH,
)
from .run import RunState

# Protocol version. Adding event types is additive; breaking changes (new
# required fields, changed semantics) bump this value.
SSE_PROTOCOL_VERSION = "1.0"


class EventLayer(str, Enum):
    PROCESS = "process"
    ANSWER = "answer"


class ContentOrigin(str, Enum):
    AI_GENERATED = "ai_generated"
    APPROVED_FAQ = "approved_faq"
    HUMAN = "human"


class SSEEventType(str, Enum):
    RUN_ACCEPTED = "run.accepted"
    PROCESS_STATUS = "process.status"
    EVIDENCE_FOUND = "evidence.found"
    REFLECTION_RESULT = "reflection.result"
    HEARTBEAT = "heartbeat"
    ANSWER_DELTA = "answer.delta"
    ANSWER_COMPLETED = "answer.completed"
    RUN_COMPLETED = "run.completed"
    # Device signalling (kept for the legacy mic channel; the dual-layer QA
    # protocol treats it as process-level info and migrates it out in the
    # vertical slice #55).
    MIC_STATUS = "mic_status"


# event -> allowed layer(s). Any other combination is rejected.
EVENT_LAYER_MAP: dict[SSEEventType, frozenset[EventLayer]] = {
    SSEEventType.RUN_ACCEPTED: frozenset({EventLayer.PROCESS}),
    SSEEventType.PROCESS_STATUS: frozenset({EventLayer.PROCESS}),
    SSEEventType.EVIDENCE_FOUND: frozenset({EventLayer.PROCESS}),
    SSEEventType.REFLECTION_RESULT: frozenset({EventLayer.PROCESS}),
    SSEEventType.HEARTBEAT: frozenset({EventLayer.PROCESS}),
    SSEEventType.ANSWER_DELTA: frozenset({EventLayer.ANSWER}),
    SSEEventType.ANSWER_COMPLETED: frozenset({EventLayer.ANSWER}),
    SSEEventType.RUN_COMPLETED: frozenset({EventLayer.PROCESS}),
    SSEEventType.MIC_STATUS: frozenset({EventLayer.PROCESS}),
}

# Run-level terminal events: exactly one of these ends a run stream.
TERMINAL_EVENTS: frozenset[SSEEventType] = frozenset({SSEEventType.RUN_COMPLETED})

# Allowlisted data keys per event (V2.3 §9.3 + content_origin from §11.4).
# Unknown keys — raw CoT, prompts, internal policy, tool internals — are
# rejected at the contract boundary.
EVENT_DATA_ALLOWED_KEYS: dict[SSEEventType, frozenset[str]] = {
    SSEEventType.RUN_ACCEPTED: frozenset({"status", "message"}),
    SSEEventType.PROCESS_STATUS: frozenset({"stage", "message"}),
    SSEEventType.EVIDENCE_FOUND: frozenset({"count", "sources"}),
    SSEEventType.REFLECTION_RESULT: frozenset({"decision", "message"}),
    SSEEventType.HEARTBEAT: frozenset(),
    SSEEventType.ANSWER_DELTA: frozenset({"delta"}),
    SSEEventType.ANSWER_COMPLETED: frozenset(
        {"citations", "actions", "content_origin", "model"}
    ),
    SSEEventType.RUN_COMPLETED: frozenset({"status", "result"}),
    SSEEventType.MIC_STATUS: frozenset({"state", "status", "message"}),
}

# Never allow these keys in process-layer data regardless of the per-event map.
FORBIDDEN_DATA_KEYS: frozenset[str] = frozenset(
    {
        "chain_of_thought",
        "thinking",
        "reasoning",
        "raw",
        "prompt",
        "system_prompt",
        "policy",
        "policy_text",
        "tool_arguments",
        "tool_internals",
        "memory_raw",
    }
)


def contains_forbidden_key(node: Any) -> str | None:
    """Recursively detect a sensitive key at ANY depth.

    ``chain_of_thought`` smuggled inside ``sources``/``citations``/``actions``
    must be rejected exactly like a top-level one — the check is a deep scan
    and only ever returns the key NAME (values are never echoed).
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_DATA_KEYS:
                return str(key)
            found = contains_forbidden_key(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            found = contains_forbidden_key(item)
            if found is not None:
                return found
    return None


def is_terminal_event(event: SSEEventType) -> bool:
    """True when the event type terminates a run stream."""
    return event in TERMINAL_EVENTS


def allowed_layers(event: SSEEventType) -> frozenset[EventLayer]:
    """Allowed layers for an event type (empty map entries are a bug)."""
    return EVENT_LAYER_MAP.get(event, frozenset())


class SSEEvent(BaseModel):
    # hide_input_in_errors: validation text must never embed payload values
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    protocol_version: str = SSE_PROTOCOL_VERSION
    seq: int = Field(..., ge=1)
    tenant_id: str = Field(
        ..., min_length=MIN_IDENTITY_LENGTH, max_length=MAX_TENANT_ID_LENGTH
    )
    device_id: str = Field(
        ..., min_length=MIN_IDENTITY_LENGTH, max_length=MAX_DEVICE_ID_LENGTH
    )
    session_id: str = Field(..., min_length=1, max_length=128)
    run_id: str = Field(..., min_length=1, max_length=128)
    layer: EventLayer
    event: SSEEventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("protocol_version")
    @classmethod
    def _pin_protocol_version(cls, value: str) -> str:
        if value != SSE_PROTOCOL_VERSION:
            raise ValueError(f"unsupported SSE protocol version {value!r}")
        return value

    @model_validator(mode="after")
    def _enforce_layer_map(self) -> SSEEvent:
        layers = allowed_layers(self.event)
        if not layers:
            raise ValueError(f"event {self.event.value!r} has no layer mapping")
        if self.layer not in layers:
            raise ValueError(
                f"event {self.event.value!r} is not allowed on layer {self.layer.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_data_allowlist(self) -> SSEEvent:
        allowed = EVENT_DATA_ALLOWED_KEYS.get(self.event)
        if allowed is None:
            raise ValueError(f"event {self.event.value!r} has no data allowlist")
        # one deep scan (key NAME only — values are never echoed)
        forbidden = contains_forbidden_key(self.data)
        if forbidden is not None:
            raise ValueError(f"forbidden data key {forbidden!r} in SSE event")
        for key in self.data:
            if key not in allowed:
                raise ValueError(
                    f"data key {key!r} is not allowed for event {self.event.value!r}"
                )
        # value-domain constraints (issue #30 review hardening)
        if self.event is SSEEventType.ANSWER_DELTA:
            delta = self.data.get("delta")
            if not isinstance(delta, str) or not delta.strip():
                raise ValueError("answer.delta requires a non-empty string delta")
        if self.event is SSEEventType.ANSWER_COMPLETED:
            # UNCONDITIONAL: an answer without a provenance marker can never be
            # persisted, replayed or rendered — no "optional" path exists.
            origin = self.data.get("content_origin")
            if origin not in {o.value for o in ContentOrigin}:
                raise ValueError("answer.completed requires a valid content_origin")
        return self


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, max_length=128)
    state: RunState
    event_seq: int = Field(..., ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
