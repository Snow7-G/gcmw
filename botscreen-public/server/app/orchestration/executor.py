"""Run executor (#55A) — drives ONE admitted run through the state machine.

This is the piece that was missing: ``RunAdmissionService.create_run`` admits a
run (``ACCEPTED`` + ``run.accepted``) and returns, and until now nothing ever
moved it further. The executor is the ONLY non-test code that connects

    API admission → ManagerAgent → (MedicalQAAgent RAG → Verifier) →
    RunRepository → dual-layer SSE

and it makes no safety decision of its own:

* every state change goes through the repository's ``commit_transition`` with
  the EXPECTED state (CAS). A lost race means "the run moved elsewhere" and the
  executor stands down — but ONLY after a BOUNDED read confirms the durable
  run really is terminal. A CAS conflict against a live run, or an invariant /
  illegal transition, is an EXPLICIT fault: the executor surfaces it and then
  commits exactly one terminal ``FAILED`` (with its ``run.completed``) when the
  repository is still writable, so no admitted run is ever left as a live
  orphan;
* staged transitions are driven by the Manager's optional progress hook
  (``ManagerAgent.execute(on_stage=...)``), whose fixed stage vocabulary maps
  one-to-one onto run states. The hook carries no result data, so agent
  internals cannot leak into the stream through it;
* what reaches the client is decided by the MANAGER (only a verifier PASS
  delivers an answer) — the executor merely packages the already-verified
  result into the SSE contract's allowlisted keys.

TWO clocks, stated separately — they are NOT the same budget:

* **run budget** (``run_timeout_ms``): bounds ONE turn — the Manager's
  engagements AND every state write / SSE append this executor performs. The
  Manager and the executor share the SAME deadline, read from the clock ONCE
  at turn start;
* **terminalization grace** (``terminalization_grace_s``): bounds ONE
  terminalization PASS as a WHOLE — a single monotonic deadline taken when the
  pass begins, and every read/write inside it uses only the time that is left.
  Attempts can never stack (N × grace); when the pass deadline is up, the pass
  is over. It is independent of the run budget, so an exhausted budget can
  never starve the cleanup that prevents a live orphan.

UNKNOWN-RESULT WRITES, stated plainly: a write that either outlived its
``wait_for`` budget OR came back with the repository's own ``UNAVAILABLE`` has
an UNKNOWN outcome — a Redis/Lua write may have committed and only been slow or
noisy to report. The executor never assumes a rollback and never blindly
retries the original write (a retry could duplicate a persisted event). It
RECONCILES against the durable record first:

* a transition that durably landed is ADOPTED (no duplicate commit) and the
  turn continues until the next budget check;
* an ``answer.completed`` that durably landed is finalized CONSISTENTLY as
  ``COMPLETED/answered`` — through a RETRYING finalize (bounded attempts, each
  preceded by a durable re-read) so one transient CAS conflict or UNAVAILABLE
  can never leave a sealed answer FAILED. A sealed answer must never be
  followed by a FAILED terminal;
* anything else ends as the trigger dictates (``FAILED/deadline_exceeded`` for
  an exhausted budget, ``FAILED/refused_no_answer`` for an unavailable store).

Reconciliation reads the TRUE latest event — never page one of a paginated
read: it takes ``latest_seq`` from the record, fetches the page that must
contain it, and verifies the event's seq AND run/tenant/device/session
identity before trusting it.

``_terminalize`` is THE one bounded entry point for every "make this run
terminal" need — budget stop, fault epilogue, reconcile fallback. A CAS
conflict is retried a bounded number of times after a re-read;
``NOT_FOUND`` and an already-terminal run are a normal exit; UNAVAILABLE /
INVARIANT / a time-up / exhausted retries are reported as the fixed
``terminalization_fault`` — it never silently claims a run was sealed when it
was not. ``_stand_down`` is bounded by the same principle: a hanging
repository read can never keep the caller waiting.

Outcome mapping (fixed, public, no agent prose):

=========================  ============  ==============================
Manager outcome            Run state     ``run.completed`` result
=========================  ============  ==============================
verified (PASS)            COMPLETED     ``answered``
escalated                  HANDOFF       ``escalated_to_human``
blocked                    FAILED        ``refused_blocked``
revised (final)            FAILED        ``refused_revised``
no evidence / other        FAILED        ``refused_no_answer``
deadline / budget exceeded FAILED        ``deadline_exceeded``
=========================  ============  ==============================

A cancelled run needs no epilogue from here: the CAS cancel itself commits the
terminal transition and therefore the terminal ``run.completed`` frame.

Faults are CONSUMED, never re-raised: this coroutine runs as a fire-and-forget
background task, so an escaping exception would only land in the event loop's
"task exception was never retrieved" log with its raw text. The executor logs
FIXED fault categories (never the original message) through ``fault_sink`` —
including faults raised BY the cleanup itself — and always leaves the run
terminal when the repository allows it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..agents.manager import ManagerAgent, ManagerAgentError
from ..contracts.agent import AgentContext, Channel
from ..contracts.errors import ErrorCode
from ..contracts.events import ContentOrigin, SSEEventType
from ..contracts.run import RunState
from ..storage.run_repository import RunRepositoryError, RunRepositoryFault
from .state_machine import is_terminal_state

_LOGGER = logging.getLogger(__name__)

#: how large one ``answer.delta`` chunk is. The answer is verified verbatim
#: content, so chunking is presentation only.
_DELTA_CHUNK_CHARS = 24

#: bounded attempts inside a terminalization pass / the answered finalize.
_TERMINALIZE_ATTEMPTS = 3


#: fixed fault categories for the fault sink. NEVER an exception's own text.
class FaultCategory(str, Enum):
    STORAGE_TRANSITION = "storage_transition_fault"
    STORAGE_APPEND = "storage_append_fault"
    DEADLINE = "deadline_exceeded"
    CANCELLED = "cancelled"
    UNEXPECTED = "unexpected_fault"
    TERMINALIZATION = "terminalization_fault"


class RunExecutionFault(RuntimeError):
    """An executor fault carrying ONLY its fixed category.

    The original repository exception is deliberately not chained or embedded:
    this error crosses the fault sink boundary, and its text must never reach a
    log or a client."""

    def __init__(self, category: FaultCategory) -> None:
        super().__init__(category.value)
        self.category = category


class _TimeUp(Exception):
    """Internal: the pass's single monotonic deadline is exhausted."""


#: repository faults that mean "this run moved / ended elsewhere". The executor
#: stands down instead of fighting them — but only after a BOUNDED read
#: confirms the run is terminal (``_stand_down``). NOT_FOUND means the run
#: record is gone.
_STAND_DOWN_FAULTS = frozenset(
    {RunRepositoryFault.CAS_CONFLICT, RunRepositoryFault.NOT_FOUND}
)

#: fixed public result markers (``run.completed`` data key ``result``). A
#: refusal is a FIXED string, never composed from agent output.
RESULT_ANSWERED = "answered"
RESULT_ESCALATED = "escalated_to_human"
RESULT_BLOCKED = "refused_blocked"
RESULT_REVISED = "refused_revised"
RESULT_NO_ANSWER = "refused_no_answer"
RESULT_DEADLINE = "deadline_exceeded"

_STAGE_MESSAGES: dict[str, str] = {
    "guarding": "安全检查中",
    "routing": "问题理解中",
    "retrieving": "资料检索中",
    "drafting": "回答整理中",
    "verifying": "证据核验中",
    "streaming": "回答生成中",
}


def _stage_message(stage: str) -> dict[str, Any]:
    """The only field a caller may add to a state-transition event.

    ``stage`` itself is DERIVED by the repository from ``next_state`` — a caller
    can never claim a stage the committed state does not correspond to."""
    return {"message": _STAGE_MESSAGES[stage]}


class RunExecutor:
    """Background driver for admitted runs (one task per run)."""

    def __init__(
        self,
        *,
        repository: Any,
        manager: ManagerAgent,
        clock: Callable[[], datetime] | None = None,
        delta_chunk_chars: int = _DELTA_CHUNK_CHARS,
        run_timeout_ms: int = 15_000,
        terminalization_grace_s: float = 2.0,
        fault_sink: Callable[[FaultCategory, str], None] | None = None,
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._delta_chunk_chars = max(1, int(delta_chunk_chars))
        self._run_timeout_ms = max(1, int(run_timeout_ms))
        self._terminalization_grace_s = max(0.001, float(terminalization_grace_s))
        self._fault_sink = fault_sink or self._default_fault_sink
        self._tasks: set[asyncio.Task[None]] = set()
        #: the demo assembly attaches its internal ToolGateway here so the app
        #: lifecycle can release its worker pool (assembly owns the wiring)
        self.tool_gateway: Any | None = None

    # -- lifecycle ---------------------------------------------------------------

    def schedule(self, record: Any) -> None:
        """Start driving one admitted run in the background."""
        task = asyncio.create_task(
            self.execute(record), name=f"gcmw-run:{record.run_id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def cancel(self, run_id: str) -> bool:
        """Interrupt the in-flight task for one run (API cancel / lease expiry).

        The run's terminal state is committed by the caller's CAS; cancelling
        the task only stops the slow work behind it. Returns whether a task was
        found."""
        name = f"gcmw-run:{run_id}"
        found = False
        for task in list(self._tasks):
            if task.get_name() == name and not task.done():
                task.cancel()
                found = True
        return found

    async def shutdown(self) -> None:
        """Cancel every in-flight run task (application shutdown)."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # -- bounded I/O primitives ----------------------------------------------------

    async def _bounded(self, call: Callable[[], Any], deadline: float) -> Any:
        """Run one repository call under the REMAINING time of the pass's
        single deadline — never a fresh grace per call (attempts must not
        stack). Raises ``_TimeUp`` when there is no time left."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _TimeUp
        try:
            return await asyncio.wait_for(call(), timeout=remaining)
        except asyncio.TimeoutError:
            raise _TimeUp from None

    # -- execution ----------------------------------------------------------------

    async def execute(self, record: Any) -> None:
        """Drive one run from ``ACCEPTED`` to a terminal state.

        Faults are CONSUMED here — including faults raised BY the cleanup
        itself. This coroutine is a fire-and-forget background task, so an
        escaping exception lands in the event loop's unhandled-task log with
        its raw text; nothing may escape. Only fixed categories reach the
        fault sink, never an original message."""
        identity = record.identity
        category: FaultCategory | None = None
        try:
            await self._run_turn(record)
        except asyncio.CancelledError:
            category = FaultCategory.CANCELLED
        except RunExecutionFault as exc:
            category = exc.category
        except Exception:  # noqa: BLE001 - consumed, never re-raised
            category = FaultCategory.UNEXPECTED

        if category is None:
            return

        cleanup_fault: FaultCategory | None = None
        try:
            await self._terminalize(identity, marker=RESULT_NO_ANSWER)
        except asyncio.CancelledError:
            cleanup_fault = FaultCategory.CANCELLED
        except Exception:  # noqa: BLE001 - the way down must not leak either
            cleanup_fault = FaultCategory.UNEXPECTED
        self._emit_fault(identity, category)
        if cleanup_fault is not None:
            self._emit_fault(identity, cleanup_fault)

    async def _run_turn(self, record: Any) -> None:
        identity = record.identity
        snapshot = record.snapshot
        # ONE clock read: the Manager and this executor share the SAME deadline,
        # so neither can drift ahead of the other's notion of "out of budget"
        deadline = self._clock() + timedelta(milliseconds=self._run_timeout_ms)
        context = AgentContext(
            tenant_id=identity.tenant_id,
            device_id=identity.device_id,
            session_id=identity.session_id,
            run_id=identity.run_id,
            channel=Channel(snapshot.channel),
            normalized_input=snapshot.text,
            deadline=deadline,
        )

        # the executor's OWN view of the run state: every write CASes from it,
        # and a lost race flips `aborted`, ending the whole turn
        state = RunState.ACCEPTED
        aborted = False

        async def _stop_for_budget() -> None:
            """The budget is exhausted: abort the turn and terminalize through
            the ONE bounded entry point."""
            nonlocal aborted
            aborted = True
            await self._terminalize(identity, marker=RESULT_DEADLINE)

        async def reconcile_unknown(
            *,
            target: RunState | None,
            event_type: SSEEventType | None,
            marker: str,
        ) -> bool:
            """A write whose result is UNKNOWN — an outer timeout OR the
            repository's own UNAVAILABLE. Reconcile against the durable record;
            never assume a rollback, never retry the original write.

            Returns True only when a transition was ADOPTED (it durably landed)
            and the turn may continue; every other resolution ends the step."""
            nonlocal state, aborted
            aborted = True  # whichever way this resolves, this step is over
            pass_deadline = time.monotonic() + self._terminalization_grace_s
            try:
                if event_type is not None:
                    page = await self._bounded(
                        lambda: self._repository.snapshot(identity, 0, 0.0),
                        pass_deadline,
                    )
                    if is_terminal_state(page.state):
                        return False  # sealed elsewhere; nothing to add
                    if event_type is SSEEventType.ANSWER_COMPLETED:
                        # an unknown-result seal: check the TRUE tail — landed
                        # → consistent COMPLETED; not landed → budget seal
                        latest = await self._durable_latest_event(
                            identity, page, pass_deadline
                        )
                        if latest is not None and latest.event is event_type:
                            await self._finalize_answered(identity, pass_deadline)
                            return False
                    # everything else (and an unlanded seal) ends via the ONE
                    # terminalization entry — SAME deadline, no fresh grace
                    await self._terminalize(
                        identity, marker=marker, deadline=pass_deadline
                    )
                    return False
                # a state transition: adopt it if it durably landed
                durable = await self._bounded(
                    lambda: self._repository.state(identity), pass_deadline
                )
                if is_terminal_state(durable):
                    return False  # the run ended elsewhere; nothing to add
                if durable == target:
                    # the write DID land: adopt it — no duplicate commit — and
                    # let the turn continue to the next budget check
                    state = target
                    aborted = False
                    return True
                # the write did NOT land: the ONE terminalization entry decides
                # — including the sealed-answer invariant (a persisted
                # answer.completed ends COMPLETED/answered, never FAILED) —
                # under the SAME pass deadline
                await self._terminalize(identity, marker=marker, deadline=pass_deadline)
                return False
            except _TimeUp:
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return False
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.NOT_FOUND:
                    return False  # the record is gone: nothing to reconcile
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return False

        async def advance(target: RunState, data: dict[str, Any] | None) -> bool:
            """One CAS transition from the executor's current state.

            A CAS loss stands down only when a BOUNDED read confirms the
            durable run really is terminal; anything else (including a lost
            race against a LIVE run) is an explicit fault. An unknown-result
            write (outer timeout OR repository UNAVAILABLE) is RECONCILED."""
            nonlocal state, aborted
            if aborted:
                return False
            if (deadline - self._clock()).total_seconds() <= 0:
                await _stop_for_budget()
                return False
            try:
                await asyncio.wait_for(
                    self._repository.commit_transition(
                        identity, expected_state=state, next_state=target, data=data
                    ),
                    timeout=max((deadline - self._clock()).total_seconds(), 0.001),
                )
            except asyncio.TimeoutError:
                return await reconcile_unknown(
                    target=target, event_type=None, marker=RESULT_DEADLINE
                )
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.ANSWER_SEALED:
                    # the atomic boundary proved a seal landed concurrently:
                    # the answer is out, so this run may only end COMPLETED
                    aborted = True
                    await self._finalize_answered(
                        identity, time.monotonic() + self._terminalization_grace_s
                    )
                    return False
                if exc.fault is RunRepositoryFault.UNAVAILABLE:
                    # a Redis/Lua write may STILL have landed: unknown result
                    return await reconcile_unknown(
                        target=target, event_type=None, marker=RESULT_NO_ANSWER
                    )
                if exc.fault in _STAND_DOWN_FAULTS and await self._stand_down(identity):
                    aborted = True
                    return False
                raise RunExecutionFault(FaultCategory.STORAGE_TRANSITION) from None
            state = target
            return True

        async def append(event_type: SSEEventType, data: dict[str, Any]) -> bool:
            """State-preserving append under the same budget."""
            nonlocal aborted
            if aborted:
                return False
            if (deadline - self._clock()).total_seconds() <= 0:
                await _stop_for_budget()
                return False
            try:
                await asyncio.wait_for(
                    self._repository.append_event(
                        identity, event_type=event_type, data=data
                    ),
                    timeout=max((deadline - self._clock()).total_seconds(), 0.001),
                )
            except asyncio.TimeoutError:
                await reconcile_unknown(
                    target=None, event_type=event_type, marker=RESULT_DEADLINE
                )
                return False
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.UNAVAILABLE:
                    await reconcile_unknown(
                        target=None,
                        event_type=event_type,
                        marker=RESULT_NO_ANSWER,
                    )
                    return False
                if exc.fault in _STAND_DOWN_FAULTS and await self._stand_down(identity):
                    aborted = True
                    return False
                raise RunExecutionFault(FaultCategory.STORAGE_APPEND) from None
            return True

        async def on_stage(stage: str) -> None:
            """Map one Manager stage onto its run-state transition(s)."""
            if aborted:
                return
            if stage == "guarding":
                await advance(RunState.GUARDING, _stage_message(stage))
            elif stage == "routing":
                await advance(RunState.ROUTING, _stage_message(stage))
            elif stage == "retrieving":
                await advance(RunState.RETRIEVING, _stage_message(stage))
            elif stage == "verifying":
                # The Manager signals "verification begins"; drafting has
                # necessarily finished by then (the verifier only ever sees a
                # finished draft), so the two bookkeeping transitions are
                # committed together, in order.
                if not await advance(RunState.DRAFTING, _stage_message("drafting")):
                    return
                await advance(RunState.VERIFYING, _stage_message("verifying"))

        try:
            result = await self._manager.execute(
                context, snapshot.text, on_stage=on_stage
            )
        except ManagerAgentError as exc:
            # a budget/deadline/routing refusal is a normal terminal outcome:
            # no answer, exactly one terminal event, no late deltas
            if aborted:
                return
            marker = (
                RESULT_DEADLINE
                if exc.code is ErrorCode.TIMEOUT_AGENT
                else RESULT_NO_ANSWER
            )
            await self._close_without_answer(identity, RunState.FAILED, marker, advance)
            return

        if aborted:
            return

        # the run may have been cancelled while the turn ran: re-read once and
        # continue from the DURABLE state, never from the local mirror
        try:
            durable = await self._repository.state(identity)
        except RunRepositoryError:
            raise RunExecutionFault(FaultCategory.STORAGE_TRANSITION) from None
        if is_terminal_state(durable):
            return
        state = durable

        safety = result.safety_status
        if safety == "verified" and result.answer_candidate:
            await self._stream_answer(identity, result, advance, append)
            return
        target, marker = _refusal_outcome(safety)
        await self._close_without_answer(identity, target, marker, advance)

    # -- delivery -----------------------------------------------------------------

    async def _stream_answer(
        self,
        identity: Any,
        result: Any,
        advance: Callable[[RunState, dict[str, Any] | None], Awaitable[bool]],
        append: Callable[[SSEEventType, dict[str, Any]], Awaitable[bool]],
    ) -> None:
        """VERIFYING → STREAMING → deltas → answer.completed → COMPLETED."""
        if not await advance(RunState.STREAMING, _stage_message("streaming")):
            return

        answer = result.answer_candidate
        for start in range(0, len(answer), self._delta_chunk_chars):
            chunk = answer[start : start + self._delta_chunk_chars]
            if not await append(SSEEventType.ANSWER_DELTA, {"delta": chunk}):
                return

        completed = {
            "citations": [
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "knowledge_version": item.knowledge_version,
                    "content_hash": item.content_hash,
                    "source_uri": item.source_uri,
                }
                for item in result.evidence
            ],
            "actions": list(result.actions),
            "content_origin": ContentOrigin.APPROVED_FAQ.value,
        }
        if not await append(SSEEventType.ANSWER_COMPLETED, completed):
            return

        await advance(
            RunState.COMPLETED,
            {"result": RESULT_ANSWERED},
        )

    async def _close_without_answer(
        self,
        identity: Any,
        target: RunState,
        marker: str,
        advance: Callable[[RunState, dict[str, Any] | None], Awaitable[bool]],
    ) -> None:
        """Refusal / escalation / timeout: no answer is delivered, by design."""
        await advance(target, {"result": marker})

    # -- terminalization ------------------------------------------------------------

    async def _terminalize(
        self, identity: Any, *, marker: str, deadline: float | None = None
    ) -> None:
        """THE one bounded terminalization entry point.

        Every "make this run terminal" need goes through here: budget stop,
        fault epilogue, reconcile fallback. Guarantees:

        * ONE monotonic deadline bounds the WHOLE pass — every read and write
          inside uses only the time that is left, so attempts can never stack
          into grace × attempts, and a hanging repository call cannot keep
          this task alive. A caller that already holds a pass deadline (e.g.
          :meth:`_run_turn`'s reconcile) MUST pass it in — a nested
          terminalization never opens a fresh grace;
        * **a SEALED ANSWER is invariant here**: before any ``FAILED`` is
          written, the true durable event tail is checked, and a persisted
          ``answer.completed`` routes the pass to the RETRYING
          ``COMPLETED/answered`` finalize instead — a sealed answer can never
          be followed by a FAILED terminal, whatever fault triggered this pass;
        * a CAS conflict is retried a bounded number of times after a RE-READ
          (the run may have moved to terminal in between);
        * ``NOT_FOUND`` and an already-terminal run are a NORMAL exit — the
          run record is gone or sealed elsewhere, nothing to do;
        * UNAVAILABLE / INVARIANT / a time-up / exhausted retries are reported
          as the fixed ``terminalization_fault`` — this method NEVER silently
          claims a run was sealed when it was not.
        """
        if deadline is None:
            deadline = time.monotonic() + self._terminalization_grace_s
        try:
            durable = await self._bounded(
                lambda: self._repository.state(identity), deadline
            )
        except _TimeUp:
            self._emit_fault(identity, FaultCategory.TERMINALIZATION)
            return
        except RunRepositoryError as exc:
            if exc.fault is RunRepositoryFault.NOT_FOUND:
                return  # the record is gone: normal exit
            self._emit_fault(identity, FaultCategory.TERMINALIZATION)
            return
        if is_terminal_state(durable):
            return  # sealed elsewhere: normal exit
        # THE sealed-answer invariant: check the TRUE durable tail before any
        # FAILED. If the answer is out, this run may only end COMPLETED.
        try:
            page = await self._bounded(
                lambda: self._repository.snapshot(identity, 0, 0.0), deadline
            )
            latest = await self._durable_latest_event(identity, page, deadline)
        except _TimeUp:
            self._emit_fault(identity, FaultCategory.TERMINALIZATION)
            return
        except RunRepositoryError as exc:
            if exc.fault is RunRepositoryFault.NOT_FOUND:
                return
            self._emit_fault(identity, FaultCategory.TERMINALIZATION)
            return
        if latest is not None and latest.event is SSEEventType.ANSWER_COMPLETED:
            await self._finalize_answered(identity, deadline)
            return
        for _ in range(_TERMINALIZE_ATTEMPTS):
            if is_terminal_state(durable):
                return  # sealed elsewhere (e.g. a cancel won): normal exit
            try:
                await self._bounded(
                    lambda d=durable: self._repository.commit_transition(
                        identity,
                        expected_state=d,
                        next_state=RunState.FAILED,
                        data={"result": marker},
                    ),
                    deadline,
                )
                return
            except _TimeUp:
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.NOT_FOUND:
                    return  # the record is gone: normal exit
                if exc.fault is RunRepositoryFault.ANSWER_SEALED:
                    # the repository's ATOMIC boundary just proved a concurrent
                    # seal landed after our tail check: finish COMPLETED within
                    # the SAME remaining grace — never FAILED
                    await self._finalize_answered(identity, deadline)
                    return
                if exc.fault is not RunRepositoryFault.CAS_CONFLICT:
                    # UNAVAILABLE / INVARIANT / ILLEGAL_TRANSITION: never
                    # pretend the run was sealed
                    self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                    return
                # CAS conflict: bounded re-read, then retry
                try:
                    durable = await self._bounded(
                        lambda: self._repository.state(identity), deadline
                    )
                except _TimeUp:
                    self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                    return
                except RunRepositoryError as exc2:
                    if exc2.fault is RunRepositoryFault.NOT_FOUND:
                        return
                    self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                    return
        # retries exhausted: the run may still be live — say so, loudly
        self._emit_fault(identity, FaultCategory.TERMINALIZATION)

    async def _durable_latest_event(
        self, identity: Any, first_page: Any, deadline: float
    ) -> Any | None:
        """The TRUE latest event of the run — never page one of a paginated
        read. Takes ``latest_seq`` from the record, fetches the page that must
        contain it when page one stops short, and verifies the event's seq AND
        run/tenant/device/session identity before trusting it."""
        latest = first_page.latest_seq
        if latest <= 0:
            return None
        events = first_page.events
        if not events or events[-1].seq != latest:
            tail = await self._bounded(
                lambda: self._repository.snapshot(identity, latest - 1, 0.0),
                deadline,
            )
            events = tail.events
        for event in reversed(events):
            if event.seq != latest:
                continue
            if (
                event.run_id == identity.run_id
                and event.tenant_id == identity.tenant_id
                and event.device_id == identity.device_id
                and event.session_id == identity.session_id
            ):
                return event
            return None  # identity mismatch: trust nothing
        return None

    async def _finalize_answered(self, identity: Any, deadline: float) -> None:
        """A sealed answer must end ``COMPLETED/answered`` — RETRYING.

        One transient CAS conflict or UNAVAILABLE must never leave a sealed
        answer FAILED — and the retry covers READS as well as writes: a first
        ``state()`` that reports UNAVAILABLE (the store recovers a moment
        later) is retried, because giving up here would leave a sealed answer
        with NO terminal at all. Every attempt is preceded by a durable
        re-read, so a commit that landed despite the error is ADOPTED and a
        run that ended elsewhere (e.g. a cancel) is RESPECTED. Bounded
        attempts; exhaustion is reported, never silently swallowed."""
        for _ in range(_TERMINALIZE_ATTEMPTS):
            try:
                durable = await self._bounded(
                    lambda: self._repository.state(identity), deadline
                )
            except _TimeUp:
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.NOT_FOUND:
                    return
                if exc.fault in (
                    RunRepositoryFault.CAS_CONFLICT,
                    RunRepositoryFault.UNAVAILABLE,
                ):
                    continue  # transient READ fault: retry the read
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return
            if is_terminal_state(durable):
                return  # ended elsewhere — respect that terminal
            try:
                await self._bounded(
                    lambda d=durable: self._repository.commit_transition(
                        identity,
                        expected_state=d,
                        next_state=RunState.COMPLETED,
                        data={"result": RESULT_ANSWERED},
                    ),
                    deadline,
                )
                return
            except _TimeUp:
                # unknown whether the COMPLETED commit landed, and the pass is
                # out of time: report — never claim success
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.NOT_FOUND:
                    return
                if exc.fault in (
                    RunRepositoryFault.CAS_CONFLICT,
                    RunRepositoryFault.UNAVAILABLE,
                ):
                    continue  # transient: re-read and retry
                self._emit_fault(identity, FaultCategory.TERMINALIZATION)
                return
        self._emit_fault(identity, FaultCategory.TERMINALIZATION)

    # -- fault machinery ------------------------------------------------------------

    async def _stand_down(self, identity: Any) -> bool:
        """True when it is SAFE to abandon the run.

        Strictly bounded by ONE grace deadline: a hanging repository read can
        never keep the caller waiting. ``NOT_FOUND`` ends the argument (the
        record is gone — nothing to fight over); a CAS conflict is resolved
        ONLY by this bounded read, never assumed."""
        deadline = time.monotonic() + self._terminalization_grace_s
        try:
            state = await self._bounded(
                lambda: self._repository.state(identity), deadline
            )
        except _TimeUp:
            return False  # cannot confirm within the grace: a fault, not a yield
        except RunRepositoryError as exc:
            return exc.fault is RunRepositoryFault.NOT_FOUND
        return is_terminal_state(state)

    def _emit_fault(self, identity: Any, category: FaultCategory) -> None:
        """Report a fault through the sink with the FIXED category only."""
        try:
            self._fault_sink(category, identity.run_id)
        except Exception:  # noqa: BLE001 - a broken sink must not mask the run
            _LOGGER.warning(
                "run executor fault category=%s run_id=%s (sink failed)",
                FaultCategory.UNEXPECTED.value,
                identity.run_id,
            )

    @staticmethod
    def _default_fault_sink(category: FaultCategory, run_id: str) -> None:
        _LOGGER.warning(
            "run executor fault category=%s run_id=%s", category.value, run_id
        )


def _refusal_outcome(safety: str) -> tuple[RunState, str]:
    """Map the Manager's non-delivering outcome onto a terminal state."""
    if safety == "escalated":
        return RunState.HANDOFF, RESULT_ESCALATED
    if safety == "blocked":
        return RunState.FAILED, RESULT_BLOCKED
    if safety == "revised":
        return RunState.FAILED, RESULT_REVISED
    return RunState.FAILED, RESULT_NO_ANSWER
