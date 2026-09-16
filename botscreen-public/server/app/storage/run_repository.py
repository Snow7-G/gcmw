"""RunRepository — the single durable authority for run state + events (#65B-2 A).

Reviewer-driven design (round 3):

- **one persistence authority**: run state AND its event stream are owned here;
  the legacy ``event_store`` module is deprecated for run events;
- **tenant + device + session bound**: Redis keys are scoped by tenant and run
  inside one hash tag (``{gcmw:run:<tenant>:<run_id>}``) so both keys land in
  the same cluster slot; Memory keys on ``(tenant_id, run_id)``; and every
  write/read/delete compares the CALLER's tenant/device/session against the
  stored record (a mismatch is "not found", never data);
- **two atomic writes with an explicit state whitelist**:
  * ``commit_transition`` — legal transitions only (rules reused from
    ``RunStateMachine``), event type + layer derived from the target state;
  * ``append_event`` — state-preserving events restricted per state:
    ``answer.delta``/``answer.completed`` only while STREAMING,
    ``evidence.found`` during retrieval/draft/verify/stream, ``reflection.result``
    while VERIFYING, ``mic_status`` at the edges. ``heartbeat`` is NEVER
    persisted (B-1 defines it as a sequence-free comment frame), and
    ``answer.completed`` is single-shot — no delta afterwards;
- **business seq == physical stream id** (``<seq>-0``), so resume uses the
  business cursor directly;
- **atomic snapshot with full invariant validation**: one Lua read returns
  state fields, retained-oldest, newest id, terminal-event validity and the
  page; Python then verifies stream-id↔seq, event identity vs the stored
  record, newest id vs ``latest_seq`` and the terminal event, and reports
  violations as explicit invariant faults;
- **answer-backed success is enforced**: ``STREAMING -> COMPLETED`` requires a
  prior, single ``answer.completed`` (the answer seal); an answer-less ending
  must use ``HANDOFF`` / ``DEGRADED`` / ``FAILED`` / ``CANCELLED`` instead.
  Relaxing this would be a protocol change requiring an ADR plus SSE/UI and
  medical-safety review — it is deliberately NOT decided in this repository
  layer.
- **bounded waits and operations**: blocking reads use async ``XREAD BLOCK``
  (``redis.asyncio``) wrapped in a repository-level timeout; a cursor beyond
  ``latest_seq`` returns immediately so the engine can classify ``cursor_ahead``;
- **retry-safe conflict codes**: Lua control codes are negative and never
  collide with a real (positive) sequence number; a seq that moved under us
  retries, state mismatch is a CAS conflict;
- **Memory parity**: expired records are purged before ``create``, and both
  writes and reads use deep copies so callers can never mutate stored history.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol

from app.api.v1.sse_stream import StreamSnapshot
from app.contracts.errors import ErrorCode
from app.contracts.events import (
    EVENT_DATA_ALLOWED_KEYS,
    SSE_PROTOCOL_VERSION,
    ContentOrigin,
    EventLayer,
    SSEEvent,
    SSEEventType,
    contains_forbidden_key,
)
from app.contracts.run import RunState
from app.orchestration.state_machine import (
    is_allowed_transition,
    is_terminal_state,
    transition_event_type,
)

DEFAULT_MAX_EVENTS_PER_RUN = 10_000
DEFAULT_RUN_TTL_S = 1800  # V2.3 session idle window
DEFAULT_SNAPSHOT_LIMIT = 500
DEFAULT_OP_TIMEOUT_S = 5.0
DEFAULT_BLOCK_GRACE_S = 5.0  # network grace added to a blocking read budget
MAX_COMMIT_RETRIES = 5

#: Lua control codes (negative so they can never look like a real seq)
_CODE_NOT_FOUND = -1
_CODE_TENANT_MISMATCH = -2
_CODE_TERMINAL = -3
_CODE_IDENTITY_MISMATCH = -4
_CODE_ORPHAN_STATE = -5
_CODE_SEQ_MOVED = -6  # retry with a fresh snapshot
_CODE_STATE_MISMATCH = -7  # CAS conflict
_CODE_STATE_NOT_ALLOWED = -8  # event/state whitelist or answer already sealed
_CODE_ANSWER_REQUIRED = -9  # STREAMING -> COMPLETED without answer.completed
_CODE_CORRUPT_TAIL = -10  # newest stream entry is malformed / misidentified
_CODE_ANSWER_SEALED = -11  # FAILED refused: the newest event is answer.completed

#: state-preserving events and the states they may be written in.
#: ``heartbeat`` is intentionally absent: B-1 defines it as a comment frame
#: without a sequence number, so it must never occupy a business seq.
STATELESS_EVENT_STATES: dict[SSEEventType, frozenset[RunState]] = {
    SSEEventType.ANSWER_DELTA: frozenset({RunState.STREAMING}),
    SSEEventType.ANSWER_COMPLETED: frozenset({RunState.STREAMING}),
    SSEEventType.EVIDENCE_FOUND: frozenset(
        {RunState.RETRIEVING, RunState.DRAFTING, RunState.VERIFYING, RunState.STREAMING}
    ),
    SSEEventType.REFLECTION_RESULT: frozenset({RunState.VERIFYING}),
    SSEEventType.MIC_STATUS: frozenset(
        {RunState.ACCEPTED, RunState.GUARDING, RunState.STREAMING}
    ),
}

#: events that seal the answer: no further delta may follow them
SEALING_EVENT_TYPES = frozenset({SSEEventType.ANSWER_COMPLETED})


class RunRepositoryFault(str, Enum):
    CAS_CONFLICT = "cas_conflict"
    ANSWER_SEALED = "answer_sealed"
    CONCURRENT_MODIFICATION = "concurrent_modification"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    INVARIANT = "invariant"
    ILLEGAL_TRANSITION = "illegal_transition"


class RunRepositoryError(RuntimeError):
    """Structured repository failure (mapped by the #36 boundary)."""

    _CODES: ClassVar[dict[RunRepositoryFault, ErrorCode]] = {
        RunRepositoryFault.CAS_CONFLICT: ErrorCode.CONFLICT_ACTIVE_RUN,
        RunRepositoryFault.CONCURRENT_MODIFICATION: ErrorCode.CONFLICT_ACTIVE_RUN,
        RunRepositoryFault.NOT_FOUND: ErrorCode.NOT_FOUND_RUN,
        RunRepositoryFault.UNAVAILABLE: ErrorCode.UNAVAILABLE_OVERLOADED,
        RunRepositoryFault.INVARIANT: ErrorCode.INTERNAL_UNKNOWN,
        RunRepositoryFault.ILLEGAL_TRANSITION: ErrorCode.CONFLICT_ACTIVE_RUN,
        # never surfaced to a client as an error: the executor catches it and
        # finalizes COMPLETED/answered (a sealed answer must not end FAILED)
        RunRepositoryFault.ANSWER_SEALED: ErrorCode.INTERNAL_UNKNOWN,
    }

    def __init__(self, fault: RunRepositoryFault, message: str = "") -> None:
        super().__init__(message or fault.value)
        self.fault = fault
        self.code = self._CODES[fault]


@dataclass(frozen=True)
class RunIdentity:
    """Tenant-bound run identity; every field must be non-empty (validated)."""

    run_id: str
    tenant_id: str
    device_id: str
    session_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("tenant_id", self.tenant_id),
            ("device_id", self.device_id),
            ("session_id", self.session_id),
        ):
            if not value or not value.strip():
                raise ValueError(f"RunIdentity.{name} must be non-empty")


def _validate_config(
    max_events: int,
    ttl_s: int | None,
    op_timeout_s: float | None = None,
    block_grace_s: float | None = None,
    snapshot_limit: int | None = None,
) -> None:
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, int)
        or max_events < 1
    ):
        raise ValueError(f"max_events must be a positive integer, got {max_events!r}")
    if ttl_s is not None and (
        isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or ttl_s < 1
    ):
        raise ValueError(f"ttl_s must be None or a positive integer, got {ttl_s!r}")
    for name, value in (
        ("op_timeout_s", op_timeout_s),
        ("block_grace_s", block_grace_s),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number, got {value!r}")
    if snapshot_limit is not None and (
        isinstance(snapshot_limit, bool)
        or not isinstance(snapshot_limit, int)
        or snapshot_limit < 1
    ):
        raise ValueError(
            f"snapshot_limit must be a positive integer, got {snapshot_limit!r}"
        )


#: maximum length accepted for identity components (matches the API contracts)
_MAX_IDENTITY_COMPONENT = 128


def _encode_component(value: str) -> str:
    """Length-checked, delimiter-safe encoding for key components.

    ``tenant``/``run`` may contain ``:``, ``{`` or ``}``; joining raw values
    would let ``("a:b", "c")`` and ``("a", "b:c")`` collapse onto the same key.
    Percent-escaping the delimiters (and ``%`` itself) keeps the mapping
    injective while both derived keys still share one hash tag.
    """
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError("key component must be a non-empty string")
    if len(value) > _MAX_IDENTITY_COMPONENT:
        raise ValueError(
            f"key component exceeds {_MAX_IDENTITY_COMPONENT} characters: {value!r}"
        )
    safe = (
        value.replace("%", "%25")
        .replace(":", "%3A")
        .replace("{", "%7B")
        .replace("}", "%7D")
    )
    return safe


def _layer_for(event_type: SSEEventType) -> EventLayer:
    """Derive the SSE layer from the event type (single authority)."""
    if event_type in {SSEEventType.ANSWER_DELTA, SSEEventType.ANSWER_COMPLETED}:
        return EventLayer.ANSWER
    return EventLayer.PROCESS


#: authoritative terminal status derived from the target state (never supplied)
_TERMINAL_STATUS: dict[RunState, str] = {
    RunState.COMPLETED: "completed",
    RunState.DEGRADED: "degraded",
    RunState.HANDOFF: "handoff",
    RunState.FAILED: "failed",
    RunState.CANCELLED: "cancelled",
}


def state_event_data(
    next_state: RunState, data: dict[str, Any] | None
) -> dict[str, Any]:
    """Authoritative state-event payload derived from ``next_state``.

    The caller may only contribute non-authoritative fields (``message`` for a
    process status, ``result`` for a terminal event). ``stage`` / ``status``
    are produced here, so a caller can never claim a status that contradicts
    the state that was actually committed.
    """
    payload = json.loads(json.dumps(data or {}))
    for key in ("stage", "status"):
        if key in payload:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"{key!r} is derived from the target state and must not be supplied",
            )
    if is_terminal_state(next_state):
        payload["status"] = _TERMINAL_STATUS[next_state]
        permitted = {"result"}
    else:
        payload["stage"] = next_state.value.lower()
        permitted = {"message"}
    extra = (
        set(payload)
        - permitted
        - ({"status"} if is_terminal_state(next_state) else {"stage"})
    )
    if extra:
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            f"unsupported state-event fields: {sorted(extra)}",
        )
    return payload


def build_event(
    identity: RunIdentity,
    seq: int,
    event_type: SSEEventType,
    data: dict[str, Any] | None,
    *,
    timestamp: Any = None,
) -> SSEEvent:
    """Construct + validate the event BEFORE any write (whitelist enforced)."""
    payload = json.loads(json.dumps(data or {}))  # detach from caller objects
    forbidden = contains_forbidden_key(payload)
    if forbidden is not None:
        # key NAME only: the offending value is never echoed
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            f"forbidden data key {forbidden!r} (checked recursively)",
        )
    allowed = EVENT_DATA_ALLOWED_KEYS.get(event_type, frozenset())
    unknown = set(payload) - set(allowed)
    if unknown:
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            f"data keys {sorted(unknown)} not allowed for {event_type.value}",
        )
    if event_type is SSEEventType.ANSWER_COMPLETED:
        origin = payload.get("content_origin")
        valid_origins = {member.value for member in ContentOrigin}
        if origin not in valid_origins:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "answer.completed requires a valid content_origin",
            )
    kwargs: dict[str, Any] = {
        "seq": seq,
        "tenant_id": identity.tenant_id,
        "device_id": identity.device_id,
        "session_id": identity.session_id,
        "run_id": identity.run_id,
        "layer": _layer_for(event_type),
        "event": event_type,
        "data": payload,
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return SSEEvent(**kwargs)


def _snapshot_from(
    *,
    state: RunState,
    events: tuple[SSEEvent, ...],
    oldest: int,
    latest: int,
    terminal_seq: int | None,
    timed_out: bool,
) -> StreamSnapshot:
    terminal_state = is_terminal_state(state)
    return StreamSnapshot(
        events=events,
        state=state,
        oldest_available_seq=oldest,
        latest_seq=latest,
        terminal_seq=terminal_seq if terminal_state else None,
        timed_out=timed_out,
    )


def _require_stateless_support(
    event_type: SSEEventType, state: RunState, *, answer_sealed: bool
) -> None:
    """Python-side safety gate (the Lua script re-checks the same rules)."""
    if event_type not in STATELESS_EVENT_STATES:
        if event_type is SSEEventType.HEARTBEAT:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "heartbeat is a comment frame and must never occupy a seq",
            )
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            f"{event_type.value} is not a state-preserving event",
        )
    allowed_states = STATELESS_EVENT_STATES[event_type]
    if state not in allowed_states:
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            f"{event_type.value} is not allowed while {state.value} "
            f"(allowed: {sorted(s.value for s in allowed_states)})",
        )
    if answer_sealed:
        raise RunRepositoryError(
            RunRepositoryFault.INVARIANT,
            "the answer is already completed — no further answer events",
        )


# ---------------------------------------------------------------------------
# In-memory implementation (deterministic; same invariants as Redis)
# ---------------------------------------------------------------------------


@dataclass
class _Record:
    identity: RunIdentity
    state: RunState
    events: list[SSEEvent] = field(default_factory=list)
    terminal_seq: int | None = None
    expires_at: float | None = None
    waiters: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def latest_seq(self) -> int:
        return self.events[-1].seq if self.events else 0


class MemoryRunRepository:
    def __init__(
        self,
        *,
        max_events: int = DEFAULT_MAX_EVENTS_PER_RUN,
        ttl_s: int | None = DEFAULT_RUN_TTL_S,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        _validate_config(max_events, ttl_s)
        self._max_events = max_events
        self._ttl_s = ttl_s
        self._monotonic = monotonic or time.monotonic
        # tenant-scoped keys: (tenant_id, run_id)
        self._records: dict[tuple[str, str], _Record] = {}
        self._lock = asyncio.Lock()

    # -- lifecycle -------------------------------------------------------------

    async def create(self, identity: RunIdentity) -> int:
        async with self._lock:
            self._purge_expired(identity)  # expired ids may be recreated
            if self._key(identity) in self._records:
                raise RunRepositoryError(
                    RunRepositoryFault.CAS_CONFLICT,
                    f"run {identity.run_id!r} already exists",
                )
            record = _Record(
                identity=identity,
                state=RunState.ACCEPTED,
                expires_at=self._deadline(),
            )
            record.events.append(
                build_event(
                    identity,
                    1,
                    SSEEventType.RUN_ACCEPTED,
                    {"status": "accepted", "message": "问题已接收"},
                ).model_copy(deep=True)
            )
            self._records[self._key(identity)] = record
            record.waiters.set()
            return 1

    async def commit_transition(
        self,
        identity: RunIdentity,
        *,
        expected_state: RunState,
        next_state: RunState,
        data: dict[str, Any] | None = None,
    ) -> int:
        async with self._lock:
            record = self._require(identity)
            if record.state is not expected_state:
                raise RunRepositoryError(
                    RunRepositoryFault.CAS_CONFLICT,
                    f"run {identity.run_id!r} is {record.state.value}, "
                    f"expected {expected_state.value}",
                )
            if is_terminal_state(record.state):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"run {identity.run_id!r} is already terminal",
                )
            if not is_allowed_transition(expected_state, next_state):
                raise RunRepositoryError(
                    RunRepositoryFault.ILLEGAL_TRANSITION,
                    f"illegal transition {expected_state.value} -> {next_state.value}",
                )
            if next_state is RunState.COMPLETED and not self._answer_sealed(record):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    "STREAMING -> COMPLETED requires a prior answer.completed",
                )
            if next_state is RunState.FAILED and self._answer_sealed(record):
                # ATOMIC sealed-answer invariant (zero-write refusal): the
                # newest event IS answer.completed, so a FAILED terminal would
                # contradict an answer the client already received. This is
                # enforced HERE — inside the commit boundary — because an
                # executor-side check-then-act leaves a TOCTOU window in which
                # a concurrent seal lands between the check and the FAILED.
                # Distinguishable on purpose: the caller finalizes COMPLETED.
                raise RunRepositoryError(
                    RunRepositoryFault.ANSWER_SEALED,
                    f"run {identity.run_id!r}: newest event is answer.completed "
                    "— FAILED would contradict a delivered answer",
                )
            event_type = SSEEventType(transition_event_type(next_state))
            seq = record.latest_seq + 1
            # authoritative stage/status come from the target state
            event = build_event(
                record.identity,
                seq,
                event_type,
                state_event_data(next_state, data),
            )
            record.events.append(event.model_copy(deep=True))
            record.state = next_state
            if is_terminal_state(next_state):
                record.terminal_seq = seq
            self._trim(record)
            self._refresh_ttl(record)
            record.waiters.set()
            return seq

    async def append_event(
        self,
        identity: RunIdentity,
        *,
        event_type: SSEEventType,
        data: dict[str, Any] | None = None,
    ) -> int:
        """State-preserving append; the event/state whitelist is enforced."""
        async with self._lock:
            record = self._require(identity)
            if is_terminal_state(record.state):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"run {identity.run_id!r} is terminal",
                )
            _require_stateless_support(
                event_type, record.state, answer_sealed=self._answer_sealed(record)
            )
            seq = record.latest_seq + 1
            event = build_event(record.identity, seq, event_type, data)
            record.events.append(event.model_copy(deep=True))
            self._trim(record)
            self._refresh_ttl(record)
            record.waiters.set()
            return seq

    async def state(self, identity: RunIdentity) -> RunState:
        async with self._lock:
            return self._require(identity).state

    async def delete(self, identity: RunIdentity) -> None:
        """Tenant+device+session authenticated delete."""
        async with self._lock:
            self._require(identity)
            self._records.pop(self._key(identity), None)

    # -- engine read interface --------------------------------------------------

    async def snapshot(
        self, identity: RunIdentity, cursor: int, timeout_s: float
    ) -> StreamSnapshot:
        async with self._lock:
            record = self._require(identity)
            if record.latest_seq > cursor or is_terminal_state(record.state):
                return self._page(record, cursor, timed_out=False)
            if cursor > record.latest_seq:
                return self._page(record, cursor, timed_out=False)
            record.waiters.clear()
        try:
            await asyncio.wait_for(record.waiters.wait(), timeout=timeout_s)
        except TimeoutError:
            pass
        async with self._lock:
            record = self._require(identity)
            if record.latest_seq > cursor:
                return self._page(record, cursor, timed_out=False)
            return self._page(record, cursor, timed_out=True)

    def _page(self, record: _Record, cursor: int, *, timed_out: bool) -> StreamSnapshot:
        events = (
            ()
            if timed_out
            else tuple(e.model_copy(deep=True) for e in record.events if e.seq > cursor)
        )
        return _snapshot_from(
            state=record.state,
            events=events,
            oldest=record.events[0].seq if record.events else 0,
            latest=record.latest_seq,
            terminal_seq=record.terminal_seq,
            timed_out=timed_out,
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _key(identity: RunIdentity) -> tuple[str, str]:
        return (identity.tenant_id, identity.run_id)

    def _deadline(self) -> float | None:
        return None if self._ttl_s is None else self._monotonic() + self._ttl_s

    def _expired(self, record: _Record) -> bool:
        return record.expires_at is not None and self._monotonic() >= record.expires_at

    def _purge_expired(self, identity: RunIdentity) -> None:
        key = self._key(identity)
        record = self._records.get(key)
        if record is not None and self._expired(record):
            del self._records[key]

    @staticmethod
    def _answer_sealed(record: _Record) -> bool:
        """Seal is proven by the REAL newest event — never a cached boolean."""
        if not record.events:
            return False
        newest = record.events[-1]
        return newest.event is SSEEventType.ANSWER_COMPLETED

    def _trim(self, record: _Record) -> None:
        while len(record.events) > self._max_events:
            record.events.pop(0)

    def _refresh_ttl(self, record: _Record) -> None:
        record.expires_at = self._deadline()

    def _require(self, identity: RunIdentity) -> _Record:
        record = self._records.get(self._key(identity))
        if record is None:
            raise RunRepositoryError(
                RunRepositoryFault.NOT_FOUND, f"run {identity.run_id!r} not found"
            )
        if (
            record.identity.tenant_id != identity.tenant_id
            or record.identity.device_id != identity.device_id
            or record.identity.session_id != identity.session_id
        ):
            # foreign/mismatched identity: absent, never data
            raise RunRepositoryError(
                RunRepositoryFault.NOT_FOUND,
                f"run {identity.run_id!r} not found for this principal",
            )
        if self._expired(record):
            del self._records[self._key(identity)]
            raise RunRepositoryError(
                RunRepositoryFault.NOT_FOUND, f"run {identity.run_id!r} expired"
            )
        return record


# ---------------------------------------------------------------------------
# Redis implementation (redis.asyncio; tenant-scoped keys, Lua-only writes)
# ---------------------------------------------------------------------------

_LUA_CREATE = """
if redis.call('EXISTS', KEYS[1]) == 1 or redis.call('EXISTS', KEYS[2]) == 1 then
  return 0
end
redis.call('HSET', KEYS[1],
  'state', 'ACCEPTED', 'tenant_id', ARGV[1], 'device_id', ARGV[2],
  'session_id', ARGV[3], 'latest_seq', '1')
redis.call('XADD', KEYS[2], 'MAXLEN', '=', ARGV[5], '1-0', 'event', ARGV[6])
if tonumber(ARGV[4]) > 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[4])
  redis.call('EXPIRE', KEYS[2], ARGV[4])
end
return 1
"""

_LUA_COMMIT = """
if redis.call('EXISTS', KEYS[1]) == 0 then return -1 end
if redis.call('HGET', KEYS[1], 'tenant_id') ~= ARGV[1] then return -2 end
if redis.call('HGET', KEYS[1], 'device_id') ~= ARGV[2] then return -4 end
if redis.call('HGET', KEYS[1], 'session_id') ~= ARGV[3] then return -4 end
if redis.call('EXISTS', KEYS[2]) == 0 then return -5 end
if redis.call('HGET', KEYS[1], 'terminal_seq') then return -3 end
if redis.call('HGET', KEYS[1], 'state') ~= ARGV[4] then return -7 end
if ARGV[6] == 'FAILED' then
  -- ATOMIC sealed-answer invariant, zero-write refusal: if the REAL newest
  -- event is answer.completed, a FAILED terminal would contradict an answer
  -- the client already received. Enforced inside this atomic window because
  -- an executor-side check-then-act leaves a TOCTOU gap. Distinguishable
  -- on purpose (-11): the caller finalizes COMPLETED instead.
  local lastf = redis.call('XREVRANGE', KEYS[2], '+', '-', 'COUNT', 1)
  if lastf[1] then
    local fieldsf = lastf[1][2]
    local payloadf = nil
    for i = 1, #fieldsf - 1, 2 do
      if fieldsf[i] == 'event' then payloadf = fieldsf[i + 1] end
    end
    if payloadf ~= nil then
      local okf, decodedf = pcall(cjson.decode, payloadf)
      if okf and type(decodedf) == 'table' and decodedf['event'] == 'answer.completed' then
        return -11
      end
    end
  end
end
if ARGV[11] == '1' then
  -- the seal must be proven by the REAL newest event, not a cached bit, and
  -- only inside this atomic window (no TOCTOU gap)
  local last = redis.call('XREVRANGE', KEYS[2], '+', '-', 'COUNT', 1)
  if not last[1] then return -9 end
  if last[1][1] ~= ARGV[5] .. '-0' then return -9 end
  local fields = last[1][2]
  local payload = nil
  for i = 1, #fields - 1, 2 do
    if fields[i] == 'event' then payload = fields[i + 1] end
  end
  if payload == nil then return -9 end
  local ok, decoded = pcall(cjson.decode, payload)
  if not ok or type(decoded) ~= 'table' then return -9 end
  if decoded['event'] ~= 'answer.completed' then return -9 end
  if tostring(decoded['seq']) ~= ARGV[5] then return -9 end
  if decoded['tenant_id'] ~= ARGV[1] or decoded['device_id'] ~= ARGV[2]
     or decoded['session_id'] ~= ARGV[3] or decoded['run_id'] ~= ARGV[12] then
    return -9
  end
  if decoded['layer'] ~= 'answer' then return -9 end
  if decoded['protocol_version'] ~= ARGV[13] then return -9 end
  local origin = decoded['data'] and decoded['data']['content_origin'] or ''
  local origin_ok = false
  for token in string.gmatch(ARGV[14], '[^,]+') do
    if token == origin then origin_ok = true end
  end
  if not origin_ok then return -9 end
end
local latest = tonumber(redis.call('HGET', KEYS[1], 'latest_seq') or '0')
if latest ~= tonumber(ARGV[5]) then return -6 end
local seq = latest + 1
redis.call('XADD', KEYS[2], 'MAXLEN', '=', ARGV[9], tostring(seq) .. '-0',
  'event', ARGV[7])
redis.call('HSET', KEYS[1], 'state', ARGV[6], 'latest_seq', tostring(seq))
if ARGV[8] == '1' then
  redis.call('HSET', KEYS[1], 'terminal_seq', tostring(seq))
end
if tonumber(ARGV[10]) > 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[10])
  redis.call('EXPIRE', KEYS[2], ARGV[10])
end
return seq
"""

_LUA_APPEND = """
if redis.call('EXISTS', KEYS[1]) == 0 then return -1 end
if redis.call('HGET', KEYS[1], 'tenant_id') ~= ARGV[1] then return -2 end
if redis.call('HGET', KEYS[1], 'device_id') ~= ARGV[2] then return -4 end
if redis.call('HGET', KEYS[1], 'session_id') ~= ARGV[3] then return -4 end
if redis.call('EXISTS', KEYS[2]) == 0 then return -5 end
if redis.call('HGET', KEYS[1], 'terminal_seq') then return -3 end
local latest = tonumber(redis.call('HGET', KEYS[1], 'latest_seq') or '0')
local last = redis.call('XREVRANGE', KEYS[2], '+', '-', 'COUNT', 1)
if last[1] then
  -- the tail must be intact and self-consistent, otherwise fail closed
  if last[1][1] ~= tostring(latest) .. '-0' then return -10 end
  local fields = last[1][2]
  local payload = nil
  for i = 1, #fields - 1, 2 do
    if fields[i] == 'event' then payload = fields[i + 1] end
  end
  if payload == nil then return -10 end
  local ok, decoded = pcall(cjson.decode, payload)
  if not ok or type(decoded) ~= 'table' then return -10 end
  if tostring(decoded['seq']) ~= tostring(latest) then return -10 end
  if decoded['tenant_id'] ~= ARGV[1] or decoded['device_id'] ~= ARGV[2]
     or decoded['session_id'] ~= ARGV[3] or decoded['run_id'] ~= ARGV[11] then
    return -10
  end
  if decoded['event'] == 'answer.completed' then return -8 end
end
local state = redis.call('HGET', KEYS[1], 'state')
local allowed = false
for token in string.gmatch(ARGV[8], '[^,]+') do
  if token == state then allowed = true end
end
if not allowed then return -8 end
local latest = tonumber(redis.call('HGET', KEYS[1], 'latest_seq') or '0')
if latest ~= tonumber(ARGV[4]) then return -6 end
local seq = latest + 1
redis.call('XADD', KEYS[2], 'MAXLEN', '=', ARGV[7], tostring(seq) .. '-0',
  'event', ARGV[5])
redis.call('HSET', KEYS[1], 'latest_seq', tostring(seq))
if tonumber(ARGV[10]) > 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[10])
  redis.call('EXPIRE', KEYS[2], ARGV[10])
end
return seq
"""

_LUA_DELETE = """
local has_state = redis.call('EXISTS', KEYS[1])
local has_stream = redis.call('EXISTS', KEYS[2])
if has_state == 0 and has_stream == 0 then return -1 end
-- symmetric orphan gate: BOTH keys must exist; either one alone is invariant
if has_state == 0 or has_stream == 0 then return -5 end
if redis.call('HGET', KEYS[1], 'tenant_id') ~= ARGV[1] then return -2 end
if redis.call('HGET', KEYS[1], 'device_id') ~= ARGV[2] then return -2 end
if redis.call('HGET', KEYS[1], 'session_id') ~= ARGV[3] then return -2 end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
return 1
"""

_LUA_SNAPSHOT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  if redis.call('EXISTS', KEYS[2]) == 1 then return redis.error_reply('ORPHAN_STREAM') end
  return redis.error_reply('NOT_FOUND')
end
if redis.call('HGET', KEYS[1], 'tenant_id') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'device_id') ~= ARGV[2]
   or redis.call('HGET', KEYS[1], 'session_id') ~= ARGV[3] then
  return redis.error_reply('NOT_FOUND')
end
if redis.call('EXISTS', KEYS[2]) == 0 then
  return redis.error_reply('ORPHAN_STATE')
end
local state = redis.call('HGET', KEYS[1], 'state')
local latest = redis.call('HGET', KEYS[1], 'latest_seq') or '0'
local terminal = redis.call('HGET', KEYS[1], 'terminal_seq') or ''
local device = redis.call('HGET', KEYS[1], 'device_id') or ''
local session = redis.call('HGET', KEYS[1], 'session_id') or ''
local oldest = ''
local first = redis.call('XRANGE', KEYS[2], '-', '+', 'COUNT', 1)
if first[1] then oldest = first[1][1] end
local newest = ''
local newest_payload = ''
local last = redis.call('XREVRANGE', KEYS[2], '+', '-', 'COUNT', 1)
if last[1] then
  newest = last[1][1]
  local last_fields = last[1][2]
  for i = 1, #last_fields - 1, 2 do
    if last_fields[i] == 'event' then newest_payload = last_fields[i + 1] end
  end
  if newest_payload == '' then return redis.error_reply('MALFORMED_ENTRY') end
end
-- the terminal event itself is returned (id + payload) so the caller can
-- validate its identity even when a resume cursor hides it from the page
local terminal_id = ''
local terminal_payload = ''
if terminal ~= '' then
  local entry = redis.call('XRANGE', KEYS[2], terminal .. '-0', terminal .. '-0',
    'COUNT', 1)
  if entry[1] then
    terminal_id = entry[1][1]
    local fields = entry[1][2]
    for i = 1, #fields - 1, 2 do
      if fields[i] == 'event' then terminal_payload = fields[i + 1] end
    end
    if terminal_payload == '' then return redis.error_reply('MALFORMED_TERMINAL') end
    local ok, decoded = pcall(cjson.decode, terminal_payload)
    if not ok or type(decoded) ~= 'table' then
      return redis.error_reply('MALFORMED_TERMINAL')
    end
    if decoded['event'] ~= 'run.completed' then
      return redis.error_reply('MALFORMED_TERMINAL')
    end
  else
    return redis.error_reply('MALFORMED_TERMINAL')
  end
end
local out = {state, latest, terminal, device, session, oldest, newest,
             newest_payload, terminal_id, terminal_payload}
local entries = redis.call('XRANGE', KEYS[2], ARGV[4], '+', 'COUNT', ARGV[5])
for i = 1, #entries do
  local fields = entries[i][2]
  local payload = nil
  for j = 1, #fields - 1, 2 do
    if fields[j] == 'event' then payload = fields[j + 1] end
  end
  if payload == nil then return redis.error_reply('MALFORMED_ENTRY') end
  out[#out + 1] = entries[i][1]
  out[#out + 1] = payload
end
return out
"""


class RedisClientDuck(Protocol):
    """redis.asyncio surface the repository relies on (all writes are Lua)."""

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...

    async def xread(
        self, streams: dict, count: int | None = None, block: int | None = None
    ) -> Any: ...

    async def delete(self, *names: str) -> int: ...


@dataclass(frozen=True)
class _Page:
    state: RunState
    latest_seq: int
    terminal_seq: int | None
    device_id: str
    session_id: str
    oldest_seq: int
    events: tuple[SSEEvent, ...]
    answer_sealed: bool = False


class RedisRunRepository:
    """Redis-backed implementation: tenant-scoped keys, Lua-only atomic writes."""

    def __init__(
        self,
        client: RedisClientDuck,
        *,
        prefix: str = "gcmw:run",
        max_events: int = DEFAULT_MAX_EVENTS_PER_RUN,
        ttl_s: int | None = DEFAULT_RUN_TTL_S,
        snapshot_limit: int = DEFAULT_SNAPSHOT_LIMIT,
        op_timeout_s: float = DEFAULT_OP_TIMEOUT_S,
        block_grace_s: float = DEFAULT_BLOCK_GRACE_S,
    ) -> None:
        _validate_config(max_events, ttl_s, op_timeout_s, block_grace_s, snapshot_limit)
        self._client = client
        self._prefix = prefix
        self._max_events = max_events
        self._ttl_s = ttl_s
        self._snapshot_limit = snapshot_limit
        # ordinary commands get a short budget; a blocking read is allowed
        # ``heartbeat_timeout + block_grace_s`` so an idle stream can reach its
        # heartbeat instead of failing as unavailable
        self._op_timeout_s = float(op_timeout_s)
        self._block_grace_s = float(block_grace_s)

    # -- helpers ---------------------------------------------------------------

    def keys(self, identity: RunIdentity) -> tuple[str, str]:
        """Tenant-scoped keys inside ONE hash tag (same cluster slot).

        Components are delimiter-escaped so distinct identities can never
        collide onto one key (``tenant="a:b", run="c"`` vs ``tenant="a",
        run="b:c"``).
        """
        tenant = _encode_component(identity.tenant_id)
        run_id = _encode_component(identity.run_id)
        tag = f"{self._prefix}:{tenant}:{run_id}"
        return (f"{{{tag}}}:state", f"{{{tag}}}:events")

    @staticmethod
    def _arg(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _strict_id_seq(stream_id: str) -> int:
        """Physical stream ids must be exactly ``<seq>-0`` (no counters)."""
        head, _, tail = stream_id.partition("-")
        if not head.isdigit() or tail != "0":
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"malformed stream id {stream_id!r} (expected <seq>-0)",
            )
        seq = int(head)
        if seq < 1:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT, f"stream id {stream_id!r} below seq 1"
            )
        return seq

    async def _call(
        self, awaitable: Any, what: str, *, budget_s: float | None = None
    ) -> Any:
        budget = self._op_timeout_s if budget_s is None else budget_s
        try:
            async with asyncio.timeout(budget):
                return await awaitable
        except TimeoutError as exc:
            raise RunRepositoryError(
                RunRepositoryFault.UNAVAILABLE, f"redis {what} timed out"
            ) from exc
        except Exception as exc:
            raise self._fault_for(exc) from exc

    @staticmethod
    def _is_redis_error(exc: Exception) -> str | None:
        message = str(exc)
        for marker in (
            "NOT_FOUND",
            "ORPHAN_STREAM",
            "ORPHAN_STATE",
            "MALFORMED_ENTRY",
            "MALFORMED_TERMINAL",
        ):
            if marker in message:
                return marker
        if "WRONGTYPE" in message:
            return "WRONGTYPE"
        return None

    def _fault_for(self, exc: Exception) -> RunRepositoryError:
        marker = self._is_redis_error(exc)
        if marker == "NOT_FOUND":
            return RunRepositoryError(RunRepositoryFault.NOT_FOUND, str(exc))
        if marker in {
            "ORPHAN_STREAM",
            "ORPHAN_STATE",
            "WRONGTYPE",
            "MALFORMED_ENTRY",
            "MALFORMED_TERMINAL",
        }:
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT, f"store invariant: {marker}"
            )
        return RunRepositoryError(
            RunRepositoryFault.UNAVAILABLE, f"redis failure: {exc}"
        )

    # -- lifecycle -------------------------------------------------------------

    async def create(self, identity: RunIdentity) -> int:
        state_key, stream_key = self.keys(identity)
        event = build_event(
            identity,
            1,
            SSEEventType.RUN_ACCEPTED,
            {"status": "accepted", "message": "问题已接收"},
        )
        outcome = await self._call(
            self._client.eval(
                _LUA_CREATE,
                2,
                state_key,
                stream_key,
                identity.tenant_id,
                identity.device_id,
                identity.session_id,
                str(self._ttl_s or 0),
                str(self._max_events),
                event.model_dump_json(),
            ),
            "create",
        )
        if int(self._arg(outcome)) != 1:
            raise RunRepositoryError(
                RunRepositoryFault.CAS_CONFLICT,
                f"run {identity.run_id!r} already exists (or orphan key present)",
            )
        return 1

    async def commit_transition(
        self,
        identity: RunIdentity,
        *,
        expected_state: RunState,
        next_state: RunState,
        data: dict[str, Any] | None = None,
    ) -> int:
        if not is_allowed_transition(expected_state, next_state):
            raise RunRepositoryError(
                RunRepositoryFault.ILLEGAL_TRANSITION,
                f"illegal transition {expected_state.value} -> {next_state.value}",
            )
        state_key, stream_key = self.keys(identity)
        for _ in range(MAX_COMMIT_RETRIES):
            page = await self._read_atomic(identity, cursor=0, limit=1)
            if is_terminal_state(page.state):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"run {identity.run_id!r} is already terminal",
                )
            require_answer = next_state is RunState.COMPLETED
            if require_answer and not page.answer_sealed:
                # a "successful" answer run must have produced exactly one
                # answer.completed first; answer-less endings belong to
                # HANDOFF / DEGRADED / FAILED / CANCELLED
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    "STREAMING -> COMPLETED requires a prior answer.completed",
                )
            seq = page.latest_seq + 1
            event = build_event(
                RunIdentity(
                    run_id=identity.run_id,
                    tenant_id=identity.tenant_id,
                    device_id=page.device_id,
                    session_id=page.session_id,
                ),
                seq,
                SSEEventType(transition_event_type(next_state)),
                state_event_data(next_state, data),
            )
            terminal = "1" if is_terminal_state(next_state) else "0"
            code = await self._eval_code(
                _LUA_COMMIT,
                (state_key, stream_key),
                (
                    identity.tenant_id,
                    identity.device_id,
                    identity.session_id,
                    expected_state.value,
                    str(page.latest_seq),
                    next_state.value,
                    event.model_dump_json(),
                    terminal,
                    str(self._max_events),
                    str(self._ttl_s or 0),
                    "1" if require_answer else "0",
                    identity.run_id,
                    SSE_PROTOCOL_VERSION,
                    ",".join(sorted(o.value for o in ContentOrigin)),
                ),
                what="commit",
            )
            if code > 0:
                return code
            if code == _CODE_SEQ_MOVED:
                continue  # another writer advanced: retry with a fresh read
            raise self._control_fault(code, expected_state)
        raise RunRepositoryError(
            RunRepositoryFault.CONCURRENT_MODIFICATION,
            f"run {identity.run_id!r}: too many concurrent modifications",
        )

    async def append_event(
        self,
        identity: RunIdentity,
        *,
        event_type: SSEEventType,
        data: dict[str, Any] | None = None,
    ) -> int:
        if event_type not in STATELESS_EVENT_STATES:
            if event_type is SSEEventType.HEARTBEAT:
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    "heartbeat is a comment frame and must never occupy a seq",
                )
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"{event_type.value} is not a state-preserving event",
            )
        state_key, stream_key = self.keys(identity)
        for _ in range(MAX_COMMIT_RETRIES):
            page = await self._read_atomic(identity, cursor=0, limit=1)
            if is_terminal_state(page.state):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"run {identity.run_id!r} is terminal",
                )
            allowed = STATELESS_EVENT_STATES[event_type]
            if page.state not in allowed:
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"{event_type.value} is not allowed while {page.state.value}",
                )
            seq = page.latest_seq + 1
            event = build_event(
                RunIdentity(
                    run_id=identity.run_id,
                    tenant_id=identity.tenant_id,
                    device_id=page.device_id,
                    session_id=page.session_id,
                ),
                seq,
                event_type,
                data,
            )
            sealing = "1" if event_type in SEALING_EVENT_TYPES else "0"
            code = await self._eval_code(
                _LUA_APPEND,
                (state_key, stream_key),
                (
                    identity.tenant_id,
                    identity.device_id,
                    identity.session_id,
                    str(page.latest_seq),
                    event.model_dump_json(),
                    "0",
                    str(self._max_events),
                    ",".join(sorted(s.value for s in allowed)),
                    sealing,
                    str(self._ttl_s or 0),
                    identity.run_id,
                ),
                what="append",
            )
            if code > 0:
                return code
            if code == _CODE_SEQ_MOVED:
                continue
            raise self._control_fault(code, page.state)
        raise RunRepositoryError(
            RunRepositoryFault.CONCURRENT_MODIFICATION,
            f"run {identity.run_id!r}: too many concurrent modifications",
        )

    async def state(self, identity: RunIdentity) -> RunState:
        page = await self._read_atomic(identity, cursor=0, limit=1)
        return page.state

    async def delete(self, identity: RunIdentity) -> None:
        state_key, stream_key = self.keys(identity)
        code = await self._eval_code(
            _LUA_DELETE,
            (state_key, stream_key),
            (identity.tenant_id, identity.device_id, identity.session_id),
            what="delete",
        )
        if code == 1:
            return
        if code in (_CODE_NOT_FOUND, _CODE_TENANT_MISMATCH):
            raise RunRepositoryError(
                RunRepositoryFault.NOT_FOUND,
                f"run {identity.run_id!r} not found for this principal",
            )
        raise self._control_fault(code, None)

    # -- reads ------------------------------------------------------------------

    async def _eval_code(
        self, script: str, keys: tuple[str, str], args: tuple[Any, ...], *, what: str
    ) -> int:
        raw = await self._call(
            self._client.eval(script, 2, keys[0], keys[1], *args), what
        )
        return int(self._arg(raw))

    @staticmethod
    def _control_fault(code: int, expected: RunState | None) -> RunRepositoryError:
        if code in (_CODE_NOT_FOUND, _CODE_TENANT_MISMATCH, _CODE_IDENTITY_MISMATCH):
            return RunRepositoryError(
                RunRepositoryFault.NOT_FOUND, "run not found for this principal"
            )
        if code == _CODE_TERMINAL:
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT, "run is already terminal"
            )
        if code == _CODE_ORPHAN_STATE:
            # -5 covers both directions: state without stream (commit/append)
            # and stream without state (delete) — never silently operated on
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "orphan key pair: state and stream disagree",
            )
        if code == _CODE_CORRUPT_TAIL:
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "stream tail is corrupt or misidentified",
            )
        if code == _CODE_ANSWER_SEALED:
            return RunRepositoryError(
                RunRepositoryFault.ANSWER_SEALED,
                "newest event is answer.completed — FAILED is refused; "
                "finalize COMPLETED instead",
            )
        if code == _CODE_ANSWER_REQUIRED:
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "STREAMING -> COMPLETED requires a prior answer.completed",
            )
        if code == _CODE_STATE_NOT_ALLOWED:
            return RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                "event not allowed in the current state (or answer sealed)",
            )
        if code == _CODE_STATE_MISMATCH:
            return RunRepositoryError(
                RunRepositoryFault.CAS_CONFLICT,
                f"expected {expected.value if expected else '?'}",
            )
        return RunRepositoryError(
            RunRepositoryFault.INVARIANT, f"unexpected repository code {code}"
        )

    async def _read_atomic(
        self, identity: RunIdentity, *, cursor: int, limit: int
    ) -> _Page:
        """One atomic Lua read + full invariant validation of the result."""
        state_key, stream_key = self.keys(identity)
        start = "-" if cursor <= 0 else f"({cursor}-0"
        raw = await self._call(
            self._client.eval(
                _LUA_SNAPSHOT,
                2,
                state_key,
                stream_key,
                identity.tenant_id,
                identity.device_id,
                identity.session_id,
                start,
                str(limit),
            ),
            "snapshot",
        )
        try:
            values = [self._arg(v) for v in raw]
            state = RunState(values[0])
            latest = int(values[1])
            terminal = int(values[2]) if values[2] else None
            device_id, session_id = values[3], values[4]
            oldest_id = values[5]
            newest_id = values[6]
            newest_payload = values[7]
            terminal_id, terminal_payload = values[8], values[9]
        except (ValueError, IndexError, KeyError) as exc:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"malformed snapshot fields for {identity.run_id!r}: {exc}",
            ) from exc
        if device_id != identity.device_id or session_id != identity.session_id:
            raise RunRepositoryError(
                RunRepositoryFault.NOT_FOUND,
                f"run {identity.run_id!r} not found for this principal",
            )
        # every physical id — including the retained-oldest one, even when it
        # sits outside the requested page — must be exactly "<seq>-0"
        retained_oldest = self._strict_id_seq(oldest_id) if oldest_id else 0
        newest_seq = self._strict_id_seq(newest_id) if newest_id else 0
        if newest_id and newest_seq != latest:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"stream newest id {newest_id!r} != hash latest_seq {latest}",
            )
        answer_sealed = False
        if newest_payload:
            newest_event = self._parse_event(newest_id or "<newest>", newest_payload)
            if newest_event.seq != latest:
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"newest event seq {newest_event.seq} != latest_seq {latest}",
                )
            self._validate_event_identity(
                newest_event, identity, device_id=device_id, session_id=session_id
            )
            answer_sealed = newest_event.event is SSEEventType.ANSWER_COMPLETED
        self._validate_terminal(
            identity=identity,
            state=state,
            latest=latest,
            terminal=terminal,
            terminal_id=terminal_id,
            terminal_payload=terminal_payload,
            device_id=device_id,
            session_id=session_id,
        )
        events: list[SSEEvent] = []
        pairs = values[10:]
        if len(pairs) % 2:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"odd snapshot payload for {identity.run_id!r}",
            )
        for index in range(0, len(pairs) - 1, 2):
            stream_id, payload = pairs[index], pairs[index + 1]
            id_seq = self._strict_id_seq(stream_id)
            event = self._parse_event(stream_id, payload)
            if id_seq != event.seq:
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"stream id {stream_id!r} != event seq {event.seq}",
                )
            self._validate_event_identity(
                event, identity, device_id=device_id, session_id=session_id
            )
            if event.event is SSEEventType.RUN_COMPLETED and not is_terminal_state(
                state
            ):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"run.completed at {event.seq} while state is {state.value}",
                )
            events.append(event)
        return _Page(
            state=state,
            latest_seq=latest,
            terminal_seq=terminal,
            device_id=device_id,
            session_id=session_id,
            oldest_seq=retained_oldest,
            events=tuple(events),
            answer_sealed=answer_sealed,
        )

    def _parse_event(self, stream_id: str, payload: str) -> SSEEvent:
        """Malformed payloads become a structured invariant, never a raw error."""
        try:
            return SSEEvent.model_validate(json.loads(payload))
        except Exception as exc:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"malformed event payload at {stream_id!r}: {exc}",
            ) from exc

    def _validate_event_identity(
        self,
        event: SSEEvent,
        identity: RunIdentity,
        *,
        device_id: str,
        session_id: str,
    ) -> None:
        if (
            event.tenant_id != identity.tenant_id
            or event.device_id != device_id
            or event.session_id != session_id
            or event.run_id != identity.run_id
        ):
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"event {event.seq} identity does not match the run record",
            )

    def _validate_terminal(
        self,
        *,
        identity: RunIdentity,
        state: RunState,
        latest: int,
        terminal: int | None,
        terminal_id: str,
        terminal_payload: str,
        device_id: str,
        session_id: str,
    ) -> None:
        """Terminal state, terminal_seq and the terminal EVENT must agree, and
        the terminal event's identity is checked even when the resume cursor
        keeps it out of the returned page."""
        if terminal is None:
            if is_terminal_state(state):
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    f"terminal state {state.value} without terminal_seq",
                )
            if terminal_id or terminal_payload:
                raise RunRepositoryError(
                    RunRepositoryFault.INVARIANT,
                    "terminal event present without terminal_seq",
                )
            return
        if not is_terminal_state(state):
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"non-terminal state {state.value} carries terminal_seq {terminal}",
            )
        if terminal != latest:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"terminal_seq {terminal} != latest_seq {latest}",
            )
        if not terminal_id or not terminal_payload:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"terminal_seq {terminal} has no terminal event",
            )
        if self._strict_id_seq(terminal_id) != terminal:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"terminal id {terminal_id!r} != terminal_seq {terminal}",
            )
        event = self._parse_event(terminal_id, terminal_payload)
        if event.event is not SSEEventType.RUN_COMPLETED:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"terminal_seq {terminal} holds {event.event.value}",
            )
        if event.seq != terminal:
            # the payload's own seq must agree with its position
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"terminal payload seq {event.seq} != terminal_seq {terminal}",
            )
        self._validate_event_identity(
            event, identity, device_id=device_id, session_id=session_id
        )

    async def snapshot(
        self, identity: RunIdentity, cursor: int, timeout_s: float
    ) -> StreamSnapshot:
        page = await self._read_atomic(
            identity, cursor=cursor, limit=self._snapshot_limit
        )
        if page.events or is_terminal_state(page.state) or cursor > page.latest_seq:
            # cursor beyond the newest seq: hand it straight to the engine so it
            # can classify cursor_ahead — never wait a heartbeat for it
            return self._to_snapshot(page, timed_out=False)
        if page.latest_seq > cursor:
            raise RunRepositoryError(
                RunRepositoryFault.INVARIANT,
                f"run {identity.run_id!r}: latest_seq {page.latest_seq} exceeds "
                f"visible events at cursor {cursor}",
            )
        block_ms = max(int(timeout_s * 1000), 1)
        _, stream_key = self.keys(identity)
        # blocking read budget = requested wait window + network grace, so the
        # SSE heartbeat can never be cut short by the command timeout
        await self._call(
            self._client.xread({stream_key: f"{cursor}-0"}, count=1, block=block_ms),
            "xread",
            budget_s=max(timeout_s, 0.0) + self._block_grace_s,
        )
        page = await self._read_atomic(
            identity, cursor=cursor, limit=self._snapshot_limit
        )
        if page.events or is_terminal_state(page.state):
            return self._to_snapshot(page, timed_out=False)
        return self._to_snapshot(page, timed_out=True)

    def _to_snapshot(self, page: _Page, *, timed_out: bool) -> StreamSnapshot:
        return _snapshot_from(
            state=page.state,
            events=() if timed_out else page.events,
            oldest=page.oldest_seq,
            latest=page.latest_seq,
            terminal_seq=page.terminal_seq,
            timed_out=timed_out,
        )


__all__ = [
    "DEFAULT_BLOCK_GRACE_S",
    "DEFAULT_MAX_EVENTS_PER_RUN",
    "DEFAULT_OP_TIMEOUT_S",
    "DEFAULT_RUN_TTL_S",
    "MAX_COMMIT_RETRIES",
    "SEALING_EVENT_TYPES",
    "STATELESS_EVENT_STATES",
    "MemoryRunRepository",
    "RedisClientDuck",
    "RedisRunRepository",
    "RunIdentity",
    "RunRepositoryError",
    "RunRepositoryFault",
    "build_event",
]
