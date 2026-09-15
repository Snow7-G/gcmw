"""Agent API router (issue #36 / #36c — B2-B wiring round, review round 2).

Architecture rules enforced here:
- ``RunRepository`` (#65B-2 slice B2-A) is the ONLY authority for run state and
  events. **Every state decision reads it** — the idempotent-replay answer, the
  one-active-run-per-session admission check, status reads, cancels and stream
  authorisation. No local mirror is consulted, so a run finished by another
  component (ManagerAgent, another worker) immediately frees its session;
- the admission service keeps lifecycle bookkeeping only (session TTL,
  idempotency keys, payload hashes, the raw question snapshot) and never owns a
  sequence number or a state;
- admission is atomic per SESSION (one lock per session id): every invariant
  here is session-scoped, so a slow storage call for one session can never
  block another session's requests (no cross-tenant head-of-line blocking).
  The service lives in the FastAPI app/lifespan scope, i.e. exactly one event
  loop owns it; multi-process admission is explicitly NOT faked in memory (see
  :mod:`app.runtime` — staging/production fail closed until a persistent
  AdmissionStore exists);
- sessions expire on an injectable clock; expiry removes session, runs,
  idempotency entries and the raw question snapshots;
- tenant/device always derive from the DevicePrincipal (default deny); run
  ownership is checked once, before the stream response starts;
- session deletion is resumable: the durable record goes first and memory drops
  the run only after storage confirmed, so memory never claims a run storage no
  longer has — a mid-way failure keeps the session so the client can retry;
- this slice (B2-B) exposes the public SSE route only: no subscriber lease, no
  disconnect->cancel propagation and no reconnect grace (all B2-C).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import JSONResponse

from app.api.v1.auth import DevicePrincipal, PrincipalDep, require_owner
from app.api.v1.errors import AppError, error_responses, request_ids
from app.api.v1.sse_stream import (
    DEFAULT_HEARTBEAT_MS,
    SnapshotReader,
    SSEStreamError,
    SSEStreamingResponse,
    StreamSnapshot,
    effective_after_seq,
    stream_engine,
    stream_error_frame,
)
from app.api.v1.stream_leases import RunLeaseRegistry
from app.contracts.api import (
    CreateRunRequest,
    CreateSessionRequest,
    RunStatusResponse,
    SessionResponse,
)
from app.contracts.common import Channel
from app.contracts.errors import ErrorCode
from app.contracts.run import RunState
from app.orchestration.state_machine import is_terminal_state
from app.runtime import readiness_report
from app.storage.run_repository import (
    MemoryRunRepository,
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
)

router = APIRouter(prefix="/api/v1")

DEFAULT_SESSION_TTL_S = 1800

#: fixed server-side heartbeat — never derived from client input
SSE_HEARTBEAT_S = DEFAULT_HEARTBEAT_MS / 1000

#: SSE responses are per-principal: no store, and no proxy buffering
SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
}

#: bounded CAS retries for cancel: a run may move under us, but the loop ends
MAX_CANCEL_ATTEMPTS = 3

_READINESS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ready", "not_ready"]},
        "checks": {"type": "object"},
        "problems": {"type": "array", "items": {"type": "string"}},
    },
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    session_id: str
    tenant_id: str
    device_id: str
    channel: Channel
    locale: str
    created_at: datetime
    ttl_s: int = DEFAULT_SESSION_TTL_S

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.ttl_s)


@dataclass
class RunSnapshot:
    """Minimal snapshot the future Manager/QA agents need to execute a run.

    Short-lived: removed on session expiry/deletion (never persisted beyond
    the store). request_id/trace_id are kept for audit correlation only.
    """

    tenant_id: str
    device_id: str
    session_id: str
    channel: Channel
    text: str
    locale: str
    request_id: str
    trace_id: str


@dataclass
class RunRecord:
    """Admission bookkeeping for one run — deliberately WITHOUT state.

    State lives in the ``RunRepository`` only; keeping a mirror here is what
    allowed a stale ``ACCEPTED`` to reject a new question after the durable run
    had already finished.
    """

    run_id: str
    session_id: str
    identity: RunIdentity
    snapshot: RunSnapshot
    payload_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class _SessionGuard:
    """Reference-counted session lock.

    ``refs`` counts every coroutine that currently HOLDS or WAITS for the lock;
    the owning service deletes the entry when it drops back to zero.
    """

    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.refs = 0


# ---------------------------------------------------------------------------
# RunAdmissionService: admission + lifecycle only (no state authority).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatusSnapshot:
    """Immutable status DTO: the state always comes from the repository."""

    run_id: str
    session_id: str
    state: RunState
    created_at: datetime

    @property
    def cancelled(self) -> bool:
        return self.state is RunState.CANCELLED


class RunAdmissionService:
    def __init__(
        self,
        repository: Any | None = None,
        clock: Any | None = None,
        executor: Any | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.repository = repository or MemoryRunRepository()
        # the #55A run executor, when the environment assembles one: it is the
        # only thing that moves a run past ACCEPTED. ``None`` (staging /
        # production, or a test that only exercises admission) keeps the old
        # behaviour — runs are admitted and simply never progress.
        self.executor = executor
        # ONE reference-counted guard per session. Idempotency keys and the
        # single-active-run rule are both session-scoped, so per-session locking
        # is sufficient for correctness while keeping unrelated sessions
        # independent — and the guard removes itself once nobody holds or waits
        # (a long-running robot must not accumulate one lock per session ever
        # created).
        self._session_guards: dict[str, _SessionGuard] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.runs: dict[str, RunRecord] = {}
        # (session_id, idempotency_key) -> (run_id, payload_hash)
        self.idempotency: dict[tuple[str, str], tuple[str, str]] = {}

    def now(self) -> datetime:
        return self._clock()

    @asynccontextmanager
    async def _session_guard(self, session_id: str) -> AsyncIterator[None]:
        """Session-scoped admission lock with a reference-counted lifetime.

        The guard is created on first use and REMOVED as soon as the last
        holder or waiter leaves (``refs`` back to zero), so the lock table
        cannot grow with the number of sessions ever created. The lookup and the
        increment contain no await, which is atomic on the single event loop
        that owns this service — therefore a queued coroutine always keeps the
        entry alive and **no second lock for the same session can ever be
        created while someone is still waiting on the first one**.
        """
        guard = self._session_guards.get(session_id)
        if guard is None:
            guard = _SessionGuard()
            self._session_guards[session_id] = guard
        guard.refs += 1
        try:
            async with guard.lock:
                yield
        finally:
            guard.refs -= 1
            if guard.refs == 0 and self._session_guards.get(session_id) is guard:
                del self._session_guards[session_id]

    # -- expiry ---------------------------------------------------------------

    def _purge_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        for run_id in [
            r for r in list(self.runs) if self.runs[r].session_id == session_id
        ]:
            # via the choke point: the run's executor task is interrupted too,
            # so an expired session cannot leave a slow task running
            self._forget_run(run_id)
        for key in [k for k in list(self.idempotency) if k[0] == session_id]:
            del self.idempotency[key]
        # NOTE: the session guard is deliberately NOT touched here. Purging
        # always runs while HOLDING that guard, so a "delete it if idle" check
        # can never fire — the guard removes itself when its last holder or
        # waiter leaves (see ``_session_guard``).

    def _session_expired(self, session: SessionRecord) -> bool:
        return self.now() >= session.expires_at

    def _expire_if_needed(self, session_id: str) -> None:
        """Called under the session lock. Expired sessions vanish entirely —
        session, runs, idempotency and the raw text snapshots included."""
        session = self.sessions.get(session_id)
        if session is not None and self._session_expired(session):
            self._purge_session(session_id)

    # -- sessions --------------------------------------------------------------

    def create_session(
        self,
        principal: DevicePrincipal,
        req: CreateSessionRequest,
        ttl_s: int = DEFAULT_SESSION_TTL_S,
    ) -> SessionResponse:
        """Create a session record.

        Ordering matters: the RESPONSE is built first, so the session contract
        is validated BEFORE anything is stored. An identity the API cannot
        serialize (it is rejected at credential load time, and this is the
        backstop) therefore leaves no half-created session behind.

        Deliberately synchronous: it is a single dict insertion of a fresh uuid
        with no check-then-act window, so no admission lock (and no await) is
        needed; the run lifecycle — which does span awaits — is locked.
        """
        record = SessionRecord(
            session_id=uuid.uuid4().hex,
            tenant_id=principal.tenant_id,
            device_id=principal.device_id,
            channel=req.channel,
            locale=req.locale,
            created_at=self.now(),
            ttl_s=ttl_s,
        )
        response = SessionResponse(  # validates against the session contract
            session_id=record.session_id,
            tenant_id=record.tenant_id,
            device_id=record.device_id,
            channel=record.channel,
            created_at=record.created_at,
            ttl_s=record.ttl_s,
        )
        self.sessions[record.session_id] = record
        return response

    async def delete_session(self, principal: DevicePrincipal, session_id: str) -> None:
        async with self._session_guard(session_id):
            session = self.sessions.get(session_id)
            if session is None:
                raise AppError(ErrorCode.NOT_FOUND_SESSION)
            if self._session_expired(session):
                self._purge_session(session_id)
                raise AppError(ErrorCode.NOT_FOUND_SESSION)
            self._require_session_owner(session, principal)
            await self._delete_runs_of(session_id)
            self._purge_session(session_id)

    async def _delete_runs_of(self, session_id: str) -> None:
        """Resumable teardown: durable delete first, memory drop only after.

        Memory therefore never claims a run that storage no longer has, and a
        mid-way failure stops immediately (nothing past the failure point is
        touched) while keeping the session, so the client can simply retry: the
        runs already deleted are gone, and the retry resumes at the first run
        that is still durable.
        """
        for record in [r for r in self.runs.values() if r.session_id == session_id]:
            try:
                await self.repository.delete(record.identity)
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.NOT_FOUND:
                    pass  # already gone (TTL or a previous attempt): idempotent
                else:
                    raise AppError(exc.code) from exc
            self._forget_run(record.run_id)

    @staticmethod
    def _require_session_owner(
        session: SessionRecord, principal: DevicePrincipal
    ) -> None:
        """Session ACL: raises unless the principal owns this session."""
        require_owner(
            principal, tenant_id=session.tenant_id, device_id=session.device_id
        )

    def _forget_run(self, run_id: str) -> None:
        """Drop a run this process can no longer reach durably.

        The executor task is interrupted HERE, at the single choke point every
        "this run is unreachable" path goes through — a deleted session must not
        leave a background task answering a question nobody can read."""
        self.runs.pop(run_id, None)
        for key in [k for k, v in list(self.idempotency.items()) if v[0] == run_id]:
            del self.idempotency[key]
        if self.executor is not None:
            self.executor.cancel(run_id)

    # -- durable state reads -----------------------------------------------------

    async def _read_durable_state(self, record: RunRecord) -> RunState:
        """Authoritative state; a vanished run is reported as NOT_FOUND."""
        try:
            return await self.repository.state(record.identity)
        except RunRepositoryError as exc:
            raise AppError(exc.code) from exc

    async def _durable_state_or_none(self, record: RunRecord) -> RunState | None:
        """Like :meth:`_read_durable_state`, but ``None`` means "gone durably".

        Only the NOT_FOUND fault is absorbed; an unavailable or inconsistent
        repository keeps raising, so a storage outage can never be mistaken for
        an absent run.
        """
        try:
            return await self.repository.state(record.identity)
        except RunRepositoryError as exc:
            if exc.fault is RunRepositoryFault.NOT_FOUND:
                return None
            raise AppError(exc.code) from exc

    # -- runs -------------------------------------------------------------------

    @staticmethod
    def payload_hash(session: SessionRecord, req: CreateRunRequest) -> str:
        canonical = json.dumps(
            {
                "text": req.input.text,
                "locale": session.locale,
                "channel": session.channel.value,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot(run: RunRecord, state: RunState) -> RunStatusSnapshot:
        return RunStatusSnapshot(
            run_id=run.run_id,
            session_id=run.session_id,
            state=state,
            created_at=run.created_at,
        )

    def session_id_for(self, run_id: str) -> str | None:
        """Session that owns ``run_id``, when this process knows the run.

        A read-only in-memory lookup (no storage access) used to charge the
        SESSION rate-limit window before a route touches anything.
        """
        record = self.runs.get(run_id)
        return record.session_id if record is not None else None

    def _session_of(self, run_id: str) -> str:
        """Session that owns this run (raises NOT_FOUND before any lock)."""
        record = self.runs.get(run_id)
        if record is None:
            raise AppError(ErrorCode.NOT_FOUND_RUN)
        return record.session_id

    def _owned_run(self, principal: DevicePrincipal, run_id: str) -> RunRecord:
        """Owned-run lookup; must be called under the session lock."""
        record = self.runs.get(run_id)
        if record is None:
            raise AppError(ErrorCode.NOT_FOUND_RUN)
        self._expire_if_needed(record.session_id)
        record = self.runs.get(run_id)
        if record is None:  # expired while we looked
            raise AppError(ErrorCode.NOT_FOUND_RUN)
        snapshot = record.snapshot
        require_owner(
            principal, tenant_id=snapshot.tenant_id, device_id=snapshot.device_id
        )
        return record

    async def create_run(
        self,
        principal: DevicePrincipal,
        req: CreateRunRequest,
        request_id: str,
        trace_id: str,
    ) -> RunStatusSnapshot:
        async with self._session_guard(req.session_id):
            self._expire_if_needed(req.session_id)
            session = self.sessions.get(req.session_id)
            if session is None:
                raise AppError(ErrorCode.NOT_FOUND_SESSION)
            self._require_session_owner(session, principal)

            payload_hash = self.payload_hash(session, req)
            key = (session.session_id, req.idempotency_key)
            existing = self.idempotency.get(key)
            if existing is not None:
                run_id, stored_hash = existing
                record = self.runs.get(run_id)
                if record is not None:
                    if stored_hash != payload_hash:
                        raise AppError(ErrorCode.CONFLICT_IDEMPOTENCY)
                    # the replay answers with the DURABLE state: a run that has
                    # since finished reports finished, not the state it had when
                    # the key was first stored
                    state = await self._durable_state_or_none(record)
                    if state is not None:
                        return self._snapshot(record, state)
                    # the old run is gone durably (TTL): forget it and create
                    self._forget_run(run_id)

            # one active (non-terminal) run per session, decided by the
            # repository — never by a local mirror
            for record in list(self.runs.values()):
                if record.session_id != session.session_id:
                    continue
                state = await self._durable_state_or_none(record)
                if state is None:
                    self._forget_run(record.run_id)
                    continue
                if not is_terminal_state(state):
                    raise AppError(ErrorCode.CONFLICT_ACTIVE_RUN)

            run_id = uuid.uuid4().hex
            identity = RunIdentity(
                run_id=run_id,
                tenant_id=session.tenant_id,
                device_id=session.device_id,
                session_id=session.session_id,
            )
            try:
                # seeds seq 1 = run.accepted inside the repository's own atomic
                # create (state + first event committed together)
                await self.repository.create(identity)
            except RunRepositoryError as exc:
                raise AppError(exc.code) from exc

            record = RunRecord(
                run_id=run_id,
                session_id=session.session_id,
                identity=identity,
                snapshot=RunSnapshot(
                    tenant_id=session.tenant_id,
                    device_id=session.device_id,
                    session_id=session.session_id,
                    channel=session.channel,
                    text=req.input.text,
                    locale=session.locale,
                    request_id=request_id,
                    trace_id=trace_id,
                ),
                payload_hash=payload_hash,
            )
            self.runs[run_id] = record
            self.idempotency[key] = (run_id, payload_hash)
            if self.executor is not None:
                # #55A: hand the admitted run to the executor. The admission
                # lock protects bookkeeping only — the executor drives the run
                # in its own task, so this call never awaits a model.
                self.executor.schedule(record)
            return self._snapshot(record, RunState.ACCEPTED)

    async def get_run(
        self, principal: DevicePrincipal, run_id: str
    ) -> RunStatusSnapshot:
        async with self._session_guard(self._session_of(run_id)):
            record = self._owned_run(principal, run_id)
            return self._snapshot(record, await self._read_durable_state(record))

    async def _cancel_record(self, record: RunRecord) -> RunState:
        """CAS-cancel under the session guard; returns the resulting state.

        Terminal runs are returned untouched (cancel is idempotent); a CAS
        conflict means the run moved under us, so the state is re-read and the
        attempt repeated — bounded by ``MAX_CANCEL_ATTEMPTS``.
        """
        for _ in range(MAX_CANCEL_ATTEMPTS):
            state = await self._read_durable_state(record)
            if is_terminal_state(state):
                return state
            try:
                # expected-state CAS: a concurrent transition (another worker, a
                # future agent) can never be overwritten silently
                await self.repository.commit_transition(
                    record.identity,
                    expected_state=state,
                    next_state=RunState.CANCELLED,
                )
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.CAS_CONFLICT:
                    continue  # the run moved: re-read and retry, bounded
                raise AppError(exc.code) from exc
            return RunState.CANCELLED
        raise AppError(ErrorCode.CONFLICT_ACTIVE_RUN)

    async def cancel_run(
        self, principal: DevicePrincipal, run_id: str
    ) -> RunStatusSnapshot:
        async with self._session_guard(self._session_of(run_id)):
            record = self._owned_run(principal, run_id)
            state = await self._cancel_record(record)
            if self.executor is not None:
                # #55A: the terminal state is committed by the CAS above; this
                # only interrupts the SLOW WORK behind it (a model call), so no
                # late answer can ever be produced after the cancel
                self.executor.cancel(run_id)
            return self._snapshot(record, state)

    async def cancel_for_disconnect(self, run_id: str) -> None:
        """Cancel a run whose LAST subscriber left (lease grace expired).

        Trusted internal path: a lease only exists for a run that was already
        authenticated AND authorised (the SSE route takes it after
        ``authorize_stream``), so no principal is re-derived from here. Failures
        are swallowed deliberately — there is no client left to report to, and
        the run simply stays active (fail-open on availability, never on
        safety) — while the durable terminal event stays single-shot.
        """
        try:
            session_id = self._session_of(run_id)
        except AppError:
            return  # already gone (session deleted/expired): nothing to cancel
        async with self._session_guard(session_id):
            record = self.runs.get(run_id)
            if record is None:
                return
            try:
                await self._cancel_record(record)
            except AppError:
                return
            if self.executor is not None:
                self.executor.cancel(run_id)

    # -- streaming ---------------------------------------------------------------

    async def authorize_stream(
        self, principal: DevicePrincipal, run_id: str
    ) -> tuple[RunIdentity, RunState]:
        """Single authentication + ownership read, BEFORE the response starts.

        Both the tenant/device check and the existence check happen here, so a
        foreign or unknown run is answered with a JSON error envelope and never
        with a half-open stream.
        """
        async with self._session_guard(self._session_of(run_id)):
            record = self._owned_run(principal, run_id)
            state = await self._read_durable_state(record)
            return record.identity, state

    def reader(self, identity: RunIdentity) -> SnapshotReader:
        """Bind the repository's atomic read interface to one run identity."""

        async def wait_page(cursor: int, timeout_s: float) -> StreamSnapshot:
            return await self.repository.snapshot(identity, cursor, timeout_s)

        return wait_page


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def get_service(request: Request) -> RunAdmissionService:
    """The lifespan-scoped service (one per application run)."""
    service = getattr(request.app.state, "agent_service", None)
    if service is None:  # pragma: no cover - the server always runs the lifespan
        raise AppError(ErrorCode.UNAVAILABLE_MAINTENANCE)
    return service


ServiceDep = Depends(get_service)


def get_leases(request: Request) -> RunLeaseRegistry:
    """The lifespan-scoped SSE lease registry (one per application run)."""
    leases = getattr(request.app.state, "stream_leases", None)
    if leases is None:  # pragma: no cover - the server always runs the lifespan
        raise AppError(ErrorCode.UNAVAILABLE_MAINTENANCE)
    return leases


LeasesDep = Depends(get_leases)


def _snapshot_response(snap: RunStatusSnapshot) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=snap.run_id,
        session_id=snap.session_id,
        state=snap.state,
        created_at=snap.created_at,
        cancelled=snap.cancelled,
    )


async def _stream_with_errors(
    source: AsyncIterator[str], *, request_id: str, trace_id: str
) -> AsyncIterator[str]:
    """Map every mid-stream failure onto ONE structured error frame.

    ``CancelledError`` (client disconnect) is deliberately not caught: B2-B has
    no cancellation policy and the B-1 engine stays side-effect free. The frame
    carries the same ErrorEnvelope as the JSON boundary and always ends the
    stream — raw exception text never reaches a client.
    """
    try:
        async for chunk in source:
            yield chunk
    except (AppError, SSEStreamError, RunRepositoryError) as exc:
        yield stream_error_frame(exc.code, request_id=request_id, trace_id=trace_id)
    except Exception:  # noqa: BLE001 - registry-only envelope, never the text
        yield stream_error_frame(
            ErrorCode.INTERNAL_UNKNOWN, request_id=request_id, trace_id=trace_id
        )


async def _stream_with_lease(
    source: AsyncIterator[str], *, leases: RunLeaseRegistry, run_id: str
) -> AsyncIterator[str]:
    """Hold one connection lease for the lifetime of a single SSE response.

    The lease is taken when streaming actually starts (the generator body runs
    on first iteration, i.e. only for an already authorised request) and always
    released in ``finally``.

    Only a CONFIRMED disconnect arms the reconnect grace: the ASGI server closes
    or cancels this body generator exactly when the client goes away
    (``GeneratorExit`` / ``CancelledError``). A stream that ends on the SERVER
    side — a terminal frame, or a structured ``stream.error`` frame after a
    storage fault — completes normally and therefore releases the lease with
    ``client_gone=False``, so an outage can never be mistaken for the user
    leaving and silently cancel the run.
    """
    leases.open(run_id)
    client_gone = False
    try:
        async for chunk in source:
            yield chunk
    except (GeneratorExit, asyncio.CancelledError):
        client_gone = True
        raise
    finally:
        leases.close(run_id, client_gone=client_gone)


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(
        ErrorCode.VALIDATION_INVALID_INPUT,
        ErrorCode.AUTH_MISSING_CREDENTIALS,
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        ErrorCode.AUTH_DEVICE_NOT_REGISTERED,
        ErrorCode.UNAVAILABLE_OVERLOADED,
        ErrorCode.INTERNAL_UNKNOWN,
        ErrorCode.RATE_LIMIT_EXCEEDED,
        ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE,
        ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT,
    ),
)
async def create_session(
    req: CreateSessionRequest,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    service: RunAdmissionService = ServiceDep,
) -> SessionResponse:
    return service.create_session(principal, req)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(
        ErrorCode.AUTH_MISSING_CREDENTIALS,
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        ErrorCode.AUTHZ_FORBIDDEN,
        ErrorCode.NOT_FOUND_SESSION,
        ErrorCode.UNAVAILABLE_OVERLOADED,
        ErrorCode.INTERNAL_UNKNOWN,
        ErrorCode.RATE_LIMIT_EXCEEDED,
        ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE,
        ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT,
    ),
)
async def delete_session(
    session_id: str,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    service: RunAdmissionService = ServiceDep,
) -> None:
    await service.delete_session(principal, session_id)


@router.post(
    "/agent/runs",
    response_model=RunStatusResponse,
    responses=error_responses(
        ErrorCode.VALIDATION_INVALID_INPUT,
        ErrorCode.AUTH_MISSING_CREDENTIALS,
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        ErrorCode.AUTHZ_FORBIDDEN,
        ErrorCode.NOT_FOUND_SESSION,
        ErrorCode.CONFLICT_ACTIVE_RUN,
        ErrorCode.CONFLICT_IDEMPOTENCY,
        ErrorCode.UNAVAILABLE_OVERLOADED,
        ErrorCode.INTERNAL_UNKNOWN,
        ErrorCode.RATE_LIMIT_EXCEEDED,
        ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE,
        ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT,
    ),
)
async def create_run(
    req: CreateRunRequest,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    service: RunAdmissionService = ServiceDep,
) -> RunStatusResponse:
    request_id, trace_id = request_ids(request)
    snap = await service.create_run(
        principal, req, request_id=request_id, trace_id=trace_id
    )
    return _snapshot_response(snap)


@router.get(
    "/agent/runs/{run_id}",
    response_model=RunStatusResponse,
    responses=error_responses(
        ErrorCode.AUTH_MISSING_CREDENTIALS,
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        ErrorCode.AUTHZ_FORBIDDEN,
        ErrorCode.NOT_FOUND_RUN,
        ErrorCode.UNAVAILABLE_OVERLOADED,
        ErrorCode.INTERNAL_UNKNOWN,
        ErrorCode.RATE_LIMIT_EXCEEDED,
    ),
)
async def get_run(
    run_id: str,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    service: RunAdmissionService = ServiceDep,
) -> RunStatusResponse:
    return _snapshot_response(await service.get_run(principal, run_id))


@router.delete(
    "/agent/runs/{run_id}",
    response_model=RunStatusResponse,
    responses=error_responses(
        ErrorCode.AUTH_MISSING_CREDENTIALS,
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        ErrorCode.AUTHZ_FORBIDDEN,
        ErrorCode.NOT_FOUND_RUN,
        ErrorCode.CONFLICT_ACTIVE_RUN,
        ErrorCode.UNAVAILABLE_OVERLOADED,
        ErrorCode.INTERNAL_UNKNOWN,
        ErrorCode.RATE_LIMIT_EXCEEDED,
        ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE,
        ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT,
    ),
)
async def cancel_run(
    run_id: str,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    service: RunAdmissionService = ServiceDep,
) -> RunStatusResponse:
    return _snapshot_response(await service.cancel_run(principal, run_id))


@router.get(
    "/agent/runs/{run_id}/events",
    # No ``response_class`` on purpose: FastAPI would stamp that class's media
    # type onto EVERY documented response, advertising the pre-stream JSON error
    # envelopes as text/event-stream. The endpoint returns an
    # ``SSEStreamingResponse`` instance (which fixes the wire media type) and
    # ``app.main`` normalises the documented media types to what each response
    # really uses.
    responses={
        200: {
            "description": (
                "SSE 事件流。`id` = run 事件序号（可直接回填 Last-Event-ID 续传），"
                "`event` = 协议事件类型，`data` = 完整 SSEEvent（protocol_version/seq/"
                "tenant_id/device_id/session_id/run_id/layer/event/data/timestamp）。"
                "空闲时发送注释帧 `: keep-alive`（无 id、不占序号）；"
                "响应开始后的故障以**单帧** `stream.error` 结束（无 id，载荷为统一错误信封）。"
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        **error_responses(
            ErrorCode.VALIDATION_INVALID_INPUT,
            ErrorCode.AUTH_MISSING_CREDENTIALS,
            ErrorCode.AUTH_INVALID_CREDENTIALS,
            ErrorCode.AUTHZ_FORBIDDEN,
            ErrorCode.NOT_FOUND_RUN,
            ErrorCode.UNAVAILABLE_OVERLOADED,
            ErrorCode.INTERNAL_UNKNOWN,
            ErrorCode.RATE_LIMIT_EXCEEDED,
        ),
    },
)
async def stream_run_events(
    run_id: str,
    request: Request,
    principal: DevicePrincipal = PrincipalDep,
    after_seq: int = Query(
        0,
        ge=0,
        description="只发送 seq 大于该值的事件（与 Last-Event-ID 取较大者）",
    ),
    last_event_id: str | None = Header(
        default=None,
        alias="Last-Event-ID",
        description="断线续传游标：客户端最后收到的 SSE `id`；非法值忽略并回落 after_seq",
    ),
    service: RunAdmissionService = ServiceDep,
    leases: RunLeaseRegistry = LeasesDep,
) -> SSEStreamingResponse:
    """Public SSE route: one atomic read stream per authenticated run.

    Order matters: authentication (``PrincipalDep``, default deny) and the
    ownership/existence check both complete before the response starts, so
    failures are plain JSON envelopes. Resume state comes from ``Last-Event-ID``
    and/or ``after_seq`` (max wins); the heartbeat window is a server constant.
    """
    identity, _state = await service.authorize_stream(principal, run_id)
    cursor = effective_after_seq(after_seq, last_event_id)
    request_id, trace_id = request_ids(request)
    engine = stream_engine(
        wait_page=service.reader(identity),
        after_seq=cursor,
        heartbeat_s=SSE_HEARTBEAT_S,
    )
    return SSEStreamingResponse(
        _stream_with_lease(
            _stream_with_errors(engine, request_id=request_id, trace_id=trace_id),
            leases=leases,
            run_id=run_id,
        ),
        headers=dict(SSE_RESPONSE_HEADERS),
    )


@router.get("/health/live")
async def live() -> dict:
    return {"status": "alive"}


@router.get(
    "/health/ready",
    responses={
        200: {
            "description": "本环境所需组件全部就绪",
            "content": {"application/json": {"schema": _READINESS_SCHEMA}},
        },
        503: {
            "description": (
                "未就绪：持久化 RunRepository 或持久化 AdmissionStore 缺失"
                "（staging/production 会因此在启动阶段失败封闭）"
            ),
            "content": {"application/json": {"schema": _READINESS_SCHEMA}},
        },
    },
)
async def ready(request: Request) -> JSONResponse:
    """Readiness is computed from the ACTUAL backends in use.

    Component status only — never session/run counts. An application that is
    not ready answers 503 instead of a cheerful 200, so an in-memory
    deployment cannot be mistaken for a production-ready one.
    """
    report = getattr(request.app.state, "readiness", None)
    if report is None:
        service = getattr(request.app.state, "agent_service", None)
        settings = getattr(request.app.state, "settings", None)
        if service is not None and settings is not None:
            report = readiness_report(settings, service.repository)
    if report is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {},
                "problems": ["application service is not running"],
            },
        )
    return JSONResponse(
        status_code=200 if report.ready else 503,
        content=report.as_dict(),
    )
