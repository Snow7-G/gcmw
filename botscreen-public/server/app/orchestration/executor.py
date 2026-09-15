"""Run executor (#55A) — drives ONE admitted run through the state machine.

This is the piece that was missing: ``RunAdmissionService.create_run`` admits a
run (``ACCEPTED`` + ``run.accepted``) and returns, and until now nothing ever
moved it further. The executor is the ONLY non-test code that connects

    API admission → ManagerAgent → (MedicalQAAgent RAG → Verifier) →
    RunRepository → dual-layer SSE

and it makes no safety decision of its own:

* every state change goes through the repository's ``commit_transition`` with
  the EXPECTED state (CAS), so a concurrent cancel can never be overwritten —
  a lost race is read as "the run moved elsewhere" and the executor stands
  down for the rest of the turn;
* staged transitions are driven by the Manager's optional progress hook
  (``ManagerAgent.execute(on_stage=...)``), whose fixed stage vocabulary maps
  one-to-one onto run states. The hook carries no result data, so agent
  internals cannot leak into the stream through it;
* what reaches the client is decided by the MANAGER (only a verifier PASS
  delivers an answer) — the executor merely packages the already-verified
  result into the SSE contract's allowlisted keys.

Outcome mapping (fixed, public, no agent prose):

=================  ============  ==============================
Manager outcome    Run state     ``run.completed`` result
=================  ============  ==============================
verified (PASS)    COMPLETED     ``answered``
escalated          HANDOFF       ``escalated_to_human``
blocked            FAILED        ``refused_blocked``
revised (final)    FAILED        ``refused_revised``
unverified/failed  FAILED        ``refused_no_answer``
=================  ============  ==============================

A cancelled run needs no epilogue from here: the CAS cancel itself commits the
terminal transition and therefore the terminal ``run.completed`` frame.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from ..agents.manager import ManagerAgent
from ..contracts.agent import AgentContext, Channel
from ..contracts.events import ContentOrigin, SSEEventType
from ..contracts.run import RunState
from ..storage.run_repository import RunRepositoryError, RunRepositoryFault
from .state_machine import is_terminal_state

#: how large one ``answer.delta`` chunk is. The answer is verified verbatim
#: content, so chunking is presentation only.
_DELTA_CHUNK_CHARS = 24

#: repository faults that mean "this run moved / ended elsewhere": the executor
#: stands down instead of fighting or surfacing them. Anything else is a real
#: storage fault and propagates to the unexpected-fault path.
_STOP_FAULTS = frozenset(
    {
        RunRepositoryFault.CAS_CONFLICT,
        RunRepositoryFault.ILLEGAL_TRANSITION,
        RunRepositoryFault.INVARIANT,
        RunRepositoryFault.NOT_FOUND,
    }
)

#: fixed public result markers (``run.completed`` data key ``result``). A
#: refusal is a FIXED string, never composed from agent output.
RESULT_ANSWERED = "answered"
RESULT_ESCALATED = "escalated_to_human"
RESULT_BLOCKED = "refused_blocked"
RESULT_REVISED = "refused_revised"
RESULT_NO_ANSWER = "refused_no_answer"

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
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._delta_chunk_chars = max(1, int(delta_chunk_chars))
        self._tasks: set[asyncio.Task[None]] = set()

    # -- lifecycle ---------------------------------------------------------------

    def schedule(self, record: Any) -> None:
        """Start driving one admitted run in the background."""
        task = asyncio.create_task(
            self.execute(record), name=f"gcmw-run:{record.run_id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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

        Every failure path ends the run through the repository so the client
        always receives exactly one terminal ``run.completed`` frame."""
        identity = record.identity
        try:
            await self._run_turn(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail_from(identity)
            raise

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
        )

        # the executor's OWN view of the run state: every write CASes from it,
        # and a lost race flips `aborted`, ending the whole turn
        state = RunState.ACCEPTED
        aborted = False

        async def advance(target: RunState, data: dict[str, Any] | None) -> bool:
            """One CAS transition from the executor's current state."""
            nonlocal state, aborted
            try:
                await self._repository.commit_transition(
                    identity, expected_state=state, next_state=target, data=data
                )
            except RunRepositoryError as exc:
                if exc.fault in _STOP_FAULTS:
                    aborted = True
                    return False
                raise
            state = target
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

        result = await self._manager.execute(context, snapshot.text, on_stage=on_stage)
        if aborted:
            return

        # the run may have been cancelled while the turn ran: re-read once and
        # continue from the DURABLE state, never from the local mirror
        try:
            durable = await self._repository.state(identity)
        except RunRepositoryError:
            return
        if is_terminal_state(durable):
            return
        state = durable

        safety = result.safety_status
        if safety == "verified" and result.answer_candidate:
            await self._stream_answer(identity, result, advance)
            return
        await self._close_without_answer(identity, safety, advance)

    # -- delivery -----------------------------------------------------------------

    async def _stream_answer(
        self,
        identity: Any,
        result: Any,
        advance: Callable[[RunState, dict[str, Any] | None], Awaitable[bool]],
    ) -> None:
        """VERIFYING → STREAMING → deltas → answer.completed → COMPLETED."""
        if not await advance(RunState.STREAMING, _stage_message("streaming")):
            return

        answer = result.answer_candidate
        for start in range(0, len(answer), self._delta_chunk_chars):
            chunk = answer[start : start + self._delta_chunk_chars]
            if not await self._append(
                identity, SSEEventType.ANSWER_DELTA, {"delta": chunk}
            ):
                return

        completed = {
            "citations": [
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "knowledge_version": item.knowledge_version,
                    "source_uri": item.source_uri,
                }
                for item in result.evidence
            ],
            "actions": list(result.actions),
            "content_origin": ContentOrigin.APPROVED_FAQ.value,
        }
        if not await self._append(identity, SSEEventType.ANSWER_COMPLETED, completed):
            return

        await advance(
            RunState.COMPLETED,
            {"result": RESULT_ANSWERED},
        )

    async def _close_without_answer(
        self,
        identity: Any,
        safety: str,
        advance: Callable[[RunState, dict[str, Any] | None], Awaitable[bool]],
    ) -> None:
        """Refusal / escalation: no answer is delivered, by design."""
        if safety == "escalated":
            target, result = RunState.HANDOFF, RESULT_ESCALATED
        elif safety == "blocked":
            target, result = RunState.FAILED, RESULT_BLOCKED
        elif safety == "revised":
            target, result = RunState.FAILED, RESULT_REVISED
        else:
            target, result = RunState.FAILED, RESULT_NO_ANSWER
        await advance(target, {"result": result})

    # -- helpers --------------------------------------------------------------------

    async def _append(
        self, identity: Any, event_type: SSEEventType, data: dict[str, Any]
    ) -> bool:
        """State-preserving append; ``False`` means the run moved or sealed."""
        try:
            await self._repository.append_event(
                identity, event_type=event_type, data=data
            )
        except RunRepositoryError as exc:
            if exc.fault in _STOP_FAULTS:
                return False
            raise
        return True

    async def _fail_from(self, identity: Any) -> None:
        """Best-effort terminal move after an unexpected executor fault."""
        try:
            state = await self._repository.state(identity)
        except RunRepositoryError:
            return
        if is_terminal_state(state):
            return
        try:
            await self._repository.commit_transition(
                identity,
                expected_state=state,
                next_state=RunState.FAILED,
                data={"result": RESULT_NO_ANSWER},
            )
        except RunRepositoryError:
            return
