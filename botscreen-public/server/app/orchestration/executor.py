"""Run executor (#55A) — drives ONE admitted run through the state machine.

This is the piece that was missing: ``RunAdmissionService.create_run`` admits a
run (``ACCEPTED`` + ``run.accepted``) and returns, and until now nothing ever
moved it further. The executor is the ONLY non-test code that connects

    API admission → ManagerAgent → (MedicalQAAgent RAG → Verifier) →
    RunRepository → dual-layer SSE

and it makes no safety decision of its own:

* every state change goes through the repository's ``commit_transition`` with
  the EXPECTED state (CAS). A lost race means "the run moved elsewhere" and the
  executor stands down — but ONLY after confirming the durable run really is
  terminal. A CAS conflict against a live run, or an invariant / illegal
  transition, is an EXPLICIT fault: the executor surfaces it and then commits
  exactly one terminal ``FAILED`` (with its ``run.completed``) when the
  repository is still writable, so no admitted run is ever left as a live
  orphan;
* staged transitions are driven by the Manager's optional progress hook
  (``ManagerAgent.execute(on_stage=...)``), whose fixed stage vocabulary maps
  one-to-one onto run states. The hook carries no result data, so agent
  internals cannot leak into the stream through it;
* what reaches the client is decided by the MANAGER (only a verifier PASS
  delivers an answer) — the executor merely packages the already-verified
  result into the SSE contract's allowlisted keys;
* the whole turn runs under ``AgentContext.deadline`` (``run_timeout_ms``), so
  the configured run budget actually bounds a slow model.

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
"task exception was never retrieved" log with its raw text. The executor logs a
FIXED fault category (never the original message) through ``fault_sink`` and
always leaves the run terminal when the repository allows it.
"""

from __future__ import annotations

import asyncio
import logging
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


#: fixed fault categories for the fault sink. NEVER an exception's own text.
class FaultCategory(str, Enum):
    STORAGE_TRANSITION = "storage_transition_fault"
    STORAGE_APPEND = "storage_append_fault"
    DEADLINE = "deadline_exceeded"
    CANCELLED = "cancelled"
    UNEXPECTED = "unexpected_fault"


class RunExecutionFault(RuntimeError):
    """An executor fault carrying ONLY its fixed category.

    The original repository exception is deliberately not chained or embedded:
    this error crosses the fault sink boundary, and its text must never reach a
    log or a client."""

    def __init__(self, category: FaultCategory) -> None:
        super().__init__(category.value)
        self.category = category


#: repository faults that mean "this run moved / ended elsewhere". The executor
#: stands down instead of fighting them — but only after confirming the run is
#: terminal (see ``_stand_down``). NOT_FOUND means the run record is gone.
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
        fault_sink: Callable[[FaultCategory, str], None] | None = None,
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._delta_chunk_chars = max(1, int(delta_chunk_chars))
        self._run_timeout_ms = max(1, int(run_timeout_ms))
        self._fault_sink = fault_sink or self._default_fault_sink
        self._tasks: set[asyncio.Task[None]] = set()

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
            await self._fail_from(identity)
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
        context = AgentContext(
            tenant_id=identity.tenant_id,
            device_id=identity.device_id,
            session_id=identity.session_id,
            run_id=identity.run_id,
            channel=Channel(snapshot.channel),
            normalized_input=snapshot.text,
            # the configured RUN budget actually applies: the Manager reads the
            # deadline for every engagement (runner / verifier) and the
            # executor maps an exceeded budget onto a terminal FAILED
            deadline=self._clock() + timedelta(milliseconds=self._run_timeout_ms),
        )

        # the executor's OWN view of the run state: every write CASes from it,
        # and a lost race flips `aborted`, ending the whole turn
        state = RunState.ACCEPTED
        aborted = False
        # the SAME budget bounds the whole turn — the Manager's engagements AND
        # every state write / SSE append this executor performs afterwards
        deadline = self._clock() + timedelta(milliseconds=self._run_timeout_ms)

        async def _deadline_stop() -> None:
            """Commit the ONE terminal FAILED for an exhausted budget.

            After this point no ``answer.completed`` (and no further write) is
            produced by this executor; the terminal stays unique because it is
            CASed from the durable state."""
            nonlocal aborted
            aborted = True
            try:
                durable = await self._repository.state(identity)
            except RunRepositoryError:
                return
            if is_terminal_state(durable):
                return
            try:
                await self._repository.commit_transition(
                    identity,
                    expected_state=durable,
                    next_state=RunState.FAILED,
                    data={"result": RESULT_DEADLINE},
                )
            except RunRepositoryError:
                return
            self._emit_fault(identity, FaultCategory.DEADLINE)

        def _remaining_s() -> float:
            return (deadline - self._clock()).total_seconds()

        async def advance(target: RunState, data: dict[str, Any] | None) -> bool:
            """One CAS transition from the executor's current state.

            A CAS loss stands down only when the durable run is confirmed
            terminal; anything else (including a lost race against a LIVE run)
            is an explicit fault."""
            nonlocal state, aborted
            if aborted:
                return False
            if _remaining_s() <= 0:
                await _deadline_stop()
                return False
            try:
                await asyncio.wait_for(
                    self._repository.commit_transition(
                        identity, expected_state=state, next_state=target, data=data
                    ),
                    timeout=max(_remaining_s(), 0.001),
                )
            except asyncio.TimeoutError:
                await _deadline_stop()
                return False
            except RunRepositoryError as exc:
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
            if _remaining_s() <= 0:
                await _deadline_stop()
                return False
            try:
                await asyncio.wait_for(
                    self._repository.append_event(
                        identity, event_type=event_type, data=data
                    ),
                    timeout=max(_remaining_s(), 0.001),
                )
            except asyncio.TimeoutError:
                await _deadline_stop()
                return False
            except RunRepositoryError as exc:
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

    # -- fault machinery ------------------------------------------------------------

    async def _stand_down(self, identity: Any) -> bool:
        """True when the durable run is confirmed TERMINAL (safe to abandon)."""
        try:
            state = await self._repository.state(identity)
        except RunRepositoryError:
            return False  # cannot confirm: this is a fault, not a yield
        return is_terminal_state(state)

    async def _fail_from(self, identity: Any) -> bool:
        """Best-effort terminal move; ``True`` when the run IS terminal now.

        Exactly one terminal ``FAILED`` / ``run.completed`` is committed when the
        repository is still writable; an already-terminal run is left alone."""
        try:
            state = await self._repository.state(identity)
        except RunRepositoryError:
            return False
        if is_terminal_state(state):
            return True
        try:
            await self._repository.commit_transition(
                identity,
                expected_state=state,
                next_state=RunState.FAILED,
                data={"result": RESULT_NO_ANSWER},
            )
        except RunRepositoryError:
            return False
        return True

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
