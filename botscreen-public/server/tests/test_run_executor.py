"""Tests for the #55A run executor: the vertical slice wiring.

These tests drive the REAL stack end to end — synthetic approved knowledge →
MockProvider → MedicalQAAgent (RAG) → SafetyEvidenceVerifier → ManagerAgent →
RunExecutor → MemoryRunRepository — and assert on the durable event stream the
SSE layer would replay. No HTTP, no cloud provider, no real clinical content.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pytest import mark

from app.contracts.common import Channel
from app.contracts.events import SSEEventType
from app.contracts.run import RunState
from app.orchestration.executor import (
    RESULT_ANSWERED,
    RESULT_DEADLINE,
    RESULT_ESCALATED,
    RESULT_NO_ANSWER,
    FaultCategory,
    RunExecutor,
)
from app.orchestration.state_machine import is_terminal_state
from app.storage.run_repository import (
    MemoryRunRepository,
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
)

_APPROVED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
TENANT = "t1"
DEVICE = "d1"
SESSION = "s1"

FEVER_SENTENCE = "体温超过38.5建议门诊就诊。"
FEVER_REPLY = "体温超过38.5建议门诊就诊（资料[1]）。"


# -- harness -------------------------------------------------------------------


@dataclass
class Stack:
    repository: MemoryRunRepository
    executor: RunExecutor
    tasks: list = field(default_factory=list)

    async def admit_and_drive(
        self, text: str, *, run_id: str, tenant: str = TENANT
    ) -> RunIdentity:
        """Admit a run the way the API does, then hand it to the executor."""
        identity = RunIdentity(
            run_id=run_id, tenant_id=tenant, device_id=DEVICE, session_id=SESSION
        )
        await self.repository.create(identity)
        record = _FakeRecord(identity=identity, text=text)
        self.executor.schedule(record)
        return identity

    async def events(self, identity: RunIdentity):
        snapshot = await self.repository.snapshot(identity, cursor=0, timeout_s=0.0)
        return snapshot.events


@dataclass
class _FakeRecord:
    """The slice of ``RunRecord`` the executor reads."""

    identity: RunIdentity
    text: str

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def snapshot(self):
        return _FakeSnapshot(text=self.text)


@dataclass
class _FakeSnapshot:
    text: str
    channel: Channel = Channel.TOUCH
    locale: str = "zh-CN"


def _stack(
    delay_ms: int = 0,
    *,
    red_flags: tuple[str, ...] = ("自杀",),
    repository: MemoryRunRepository | None = None,
    run_timeout_ms: int = 15_000,
    fault_sink=None,
) -> Stack:
    from app.agents.manager import ManagerAgent, RedFlagRules, RiskRules
    from app.agents.medical_qa import MedicalQAAgent
    from app.agents.registry import AgentManifest, AgentRegistry
    from app.agents.verifier import SafetyEvidenceVerifier
    from app.contracts.common import TenantContext
    from app.contracts.knowledge import (
        ApprovalDecision,
        CandidateInput,
        KnowledgeSourceType,
    )
    from app.knowledge.store import KnowledgeStore
    from app.providers.mock import MockProvider
    from app.providers.model_gateway import ModelGateway
    from app.tools.builtins import build_gateway

    store = KnowledgeStore()
    canned: dict[str, str] = {}
    context = TenantContext(tenant_id=TENANT)
    for source_id, title, content, question in (
        ("faq-fever", "发热护理须知", FEVER_SENTENCE, "发热"),
        ("faq-eye", "用眼卫生须知", "眼部不适需及时就诊。", "眼睛"),
    ):
        store.add_candidate(
            context,
            CandidateInput(
                source_id=source_id,
                source_type=KnowledgeSourceType.FAQ,
                title=title,
                content=content,
                source_uri=f"kbase://{source_id}",
            ),
            actor="demo",
        )
        store.mark_in_review(context, source_id, actor="demo")
        store.approve(
            context,
            source_id,
            ApprovalDecision(reviewer="dr-demo", valid_from=_APPROVED_AT),
        )
        canned[question] = f"{content.rstrip('。')}（资料[1]）。"

    tools = build_gateway(knowledge_store=store, audit_sink=lambda _r: None)
    models = ModelGateway(active_provider_id="mock")
    models.register(MockProvider(canned=canned, delay_ms=delay_ms))
    qa = MedicalQAAgent(models=models, tools=tools)

    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            agent_id="qa",
            version="1.0.0",
            supported_intents=["knowledge"],
            risk_level="medium",
        )
    )
    rules = RedFlagRules(
        patterns=red_flags, approved_by="demo-attestation", approved_at=_APPROVED_AT
    )
    risk_rules = RiskRules(
        patterns=("剧烈",), approved_by="demo-attestation", approved_at=_APPROVED_AT
    )

    async def run_qa(ctx):
        return await qa.run(ctx)

    manager = ManagerAgent(
        registry=registry,
        agent_runners={"qa": run_qa},
        verifier=SafetyEvidenceVerifier(red_flag_rules=rules),
        red_flag_rules=rules,
        risk_rules=risk_rules,
    )
    repository = repository if repository is not None else MemoryRunRepository()
    executor = RunExecutor(
        repository=repository,
        manager=manager,
        run_timeout_ms=run_timeout_ms,
        fault_sink=fault_sink,
    )
    return Stack(repository=repository, executor=executor)


async def _drain(stack: Stack, identity: RunIdentity, timeout_s: float = 5.0):
    """Wait until the run reaches a terminal state (or the timeout expires)."""

    async def _wait():
        while True:
            state = await stack.repository.state(identity)
            if is_terminal_state(state):
                return state
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_wait(), timeout=timeout_s)


def _by_type(events, event_type):
    return [e for e in events if e.event is event_type]


# -- scenarios -------------------------------------------------------------------


class TestAnsweredRun:
    @mark.asyncio
    async def test_a_verified_answer_completes_with_citations(self):
        stack = _stack()
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-answer")
        state = await _drain(stack, identity)
        assert state is RunState.COMPLETED

        events = await stack.events(identity)
        deltas = _by_type(events, SSEEventType.ANSWER_DELTA)
        completed = _by_type(events, SSEEventType.ANSWER_COMPLETED)
        assert deltas, "the verified answer must be streamed as deltas"
        assert len(completed) == 1
        citation = completed[0].data["citations"][0]
        assert citation["source_id"] == "faq-fever"
        assert completed[0].data["content_origin"] == "approved_faq"

        # the streamed deltas reassemble into the answer the Manager delivered
        streamed = "".join(e.data["delta"] for e in deltas)
        assert streamed == FEVER_REPLY

        # process.status stages appear in execution order
        stages = [
            e.data["stage"] for e in _by_type(events, SSEEventType.PROCESS_STATUS)
        ]
        assert stages == [
            "guarding",
            "routing",
            "retrieving",
            "drafting",
            "verifying",
            "streaming",
        ]

        # exactly one terminal event, and it closes the run
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)
        assert len(terminal) == 1
        assert terminal[0].data["result"] == RESULT_ANSWERED

    @mark.asyncio
    async def test_streaming_completes_only_after_the_answer_is_sealed(self):
        stack = _stack()
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-seal")
        await _drain(stack, identity)
        events = await stack.events(identity)
        seqs = {e.event: e.seq for e in events}
        # STREAMING -> COMPLETED is refused without answer.completed, so the
        # sealing event must precede the terminal one
        assert seqs[SSEEventType.ANSWER_COMPLETED] < seqs[SSEEventType.RUN_COMPLETED]


class TestRefusals:
    @mark.asyncio
    async def test_a_question_without_evidence_is_refused(self):
        stack = _stack()
        identity = await stack.admit_and_drive("近视激光手术多少钱", run_id="r-noev")
        state = await _drain(stack, identity)
        assert state is RunState.FAILED
        events = await stack.events(identity)
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)[0]
        assert terminal.data["result"] == RESULT_NO_ANSWER
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)

    @mark.asyncio
    async def test_another_tenant_finds_no_evidence(self):
        """Isolation is real: knowledge published for t1 is invisible to t2."""
        stack = _stack()
        identity = await stack.admit_and_drive(
            "发热怎么办", run_id="r-tenant", tenant="other-tenant"
        )
        state = await _drain(stack, identity)
        assert state is RunState.FAILED
        events = await stack.events(identity)
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)[0]
        assert terminal.data["result"] == RESULT_NO_ANSWER


class TestEscalation:
    @mark.asyncio
    async def test_a_red_flag_question_goes_to_a_human(self):
        stack = _stack()
        identity = await stack.admit_and_drive("我想自杀", run_id="r-escalate")
        state = await _drain(stack, identity)
        assert state is RunState.HANDOFF
        events = await stack.events(identity)
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)[0]
        assert terminal.data["result"] == RESULT_ESCALATED
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)
        assert not _by_type(events, SSEEventType.ANSWER_DELTA)


class TestCancellation:
    @mark.asyncio
    async def test_a_cancelled_run_stops_without_an_answer(self):
        stack = _stack(delay_ms=1500)
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-cancel")
        # give the executor time to leave ACCEPTED, then cancel like the API does
        await asyncio.sleep(0.05)
        state = await stack.repository.state(identity)
        assert state is not RunState.ACCEPTED
        await stack.repository.commit_transition(
            identity, expected_state=state, next_state=RunState.CANCELLED
        )
        final = await _drain(stack, identity)
        assert final is RunState.CANCELLED

        # the executor stands down: no answer is streamed for a cancelled run
        await asyncio.sleep(0.05)
        events = await stack.events(identity)
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)
        terminal = [e for e in events if e.event is SSEEventType.RUN_COMPLETED]
        assert len(terminal) == 1


class TestLifecycle:
    @mark.asyncio
    async def test_shutdown_cancels_in_flight_runs(self):
        stack = _stack(delay_ms=1500)
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-shutdown")
        await asyncio.sleep(0.05)
        await stack.executor.shutdown()
        state = await stack.repository.state(identity)
        assert not state.value in {"completed"}  # never finished
        assert stack.executor._tasks == set()


# -- fault injection (review P1: the executor must never orphan a run) ---------


class _FaultyRepository(MemoryRunRepository):
    """Injects ONE repository fault, then behaves normally.

    ``transition_to`` arms a fault on the NEXT transition TO that state;
    ``append_event_type`` arms a fault on the next append of that type."""

    def __init__(
        self,
        *,
        transition_to: RunState | None = None,
        append_event_type: SSEEventType | None = None,
        fault: RunRepositoryFault = RunRepositoryFault.INVARIANT,
    ) -> None:
        super().__init__()
        self.transition_to = transition_to
        self.append_event_type = append_event_type
        self.fault = fault
        self.spent = False

    async def commit_transition(
        self, identity, *, expected_state, next_state, data=None
    ):
        if (
            self.transition_to is not None
            and next_state is self.transition_to
            and not self.spent
        ):
            self.spent = True
            raise RunRepositoryError(self.fault)
        return await super().commit_transition(
            identity, expected_state=expected_state, next_state=next_state, data=data
        )

    async def append_event(self, identity, *, event_type, data=None):
        if (
            self.append_event_type is not None
            and event_type is self.append_event_type
            and not self.spent
        ):
            self.spent = True
            raise RunRepositoryError(self.fault)
        return await super().append_event(identity, event_type=event_type, data=data)


class TestFaultPaths:
    """A storage fault must be SURFACED and the run driven terminal — the
    executor may neither swallow it into silence nor dump it into the event
    loop's unhandled-task log."""

    async def _drive(self, stack: Stack, run_id: str, text: str = "发热怎么办"):
        identity = RunIdentity(
            run_id=run_id, tenant_id=TENANT, device_id=DEVICE, session_id=SESSION
        )
        await stack.repository.create(identity)
        record = _FakeRecord(identity=identity, text=text)
        task = asyncio.create_task(stack.executor.execute(record))
        await asyncio.wait_for(task, timeout=10)
        # the executor CONSUMES its faults: a fire-and-forget task must never
        # carry an exception into the loop's "never retrieved" log
        assert task.exception() is None
        return identity

    @mark.asyncio
    async def test_an_invariant_fault_is_surfaced_and_the_run_still_fails(self):
        faults: list = []
        repository = _FaultyRepository(transition_to=RunState.VERIFYING)
        stack = _stack(
            repository=repository, fault_sink=lambda c, r: faults.append((c, r))
        )
        identity = await self._drive(stack, "r-inv")
        state = await _drain(stack, identity)
        assert state is RunState.FAILED  # no live orphan, no stuck STREAMING
        events = await stack.events(identity)
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)
        assert len(terminal) == 1
        assert terminal[0].data["result"] == RESULT_NO_ANSWER
        assert FaultCategory.STORAGE_TRANSITION in [c for c, _ in faults]

    @mark.asyncio
    async def test_an_illegal_transition_fault_is_surfaced_too(self):
        faults: list = []
        repository = _FaultyRepository(
            transition_to=RunState.STREAMING,
            fault=RunRepositoryFault.ILLEGAL_TRANSITION,
        )
        stack = _stack(
            repository=repository, fault_sink=lambda c, r: faults.append((c, r))
        )
        identity = await self._drive(stack, "r-illegal")
        assert await _drain(stack, identity) is RunState.FAILED
        assert FaultCategory.STORAGE_TRANSITION in [c for c, _ in faults]

    @mark.asyncio
    async def test_an_append_fault_mid_stream_still_seals_the_run(self):
        faults: list = []
        repository = _FaultyRepository(append_event_type=SSEEventType.ANSWER_DELTA)
        stack = _stack(
            repository=repository, fault_sink=lambda c, r: faults.append((c, r))
        )
        identity = await self._drive(stack, "r-append")
        state = await _drain(stack, identity)
        assert state is RunState.FAILED
        events = await stack.events(identity)
        assert len(_by_type(events, SSEEventType.RUN_COMPLETED)) == 1
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)
        assert FaultCategory.STORAGE_APPEND in [c for c, _ in faults]

    @mark.asyncio
    async def test_a_cas_conflict_against_a_live_run_is_not_ignored(self):
        """The OLD behaviour swallowed every CAS loss. Losing against a run
        that is still LIVE is a divergence, not a yield."""
        faults: list = []
        repository = _FaultyRepository(
            transition_to=RunState.GUARDING,
            fault=RunRepositoryFault.CAS_CONFLICT,
        )
        stack = _stack(
            repository=repository, fault_sink=lambda c, r: faults.append((c, r))
        )
        identity = await self._drive(stack, "r-cas-live")
        state = await _drain(stack, identity)
        assert state is RunState.FAILED  # not left in ACCEPTED
        assert FaultCategory.STORAGE_TRANSITION in [c for c, _ in faults]

    @mark.asyncio
    async def test_a_cas_conflict_with_a_terminal_run_stands_down_silently(self):
        """Cancel wins the race: the executor yields WITHOUT logging a fault and
        WITHOUT writing a second terminal event."""
        faults: list = []
        repository = _FaultyRepository(
            transition_to=RunState.STREAMING,
            fault=RunRepositoryFault.CAS_CONFLICT,
        )
        stack = _stack(
            delay_ms=400,
            repository=repository,
            fault_sink=lambda c, r: faults.append((c, r)),
        )
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-cas-cancel")
        await asyncio.sleep(0.1)  # the model is still slow at this point
        state = await stack.repository.state(identity)
        assert state is RunState.RETRIEVING
        await stack.repository.commit_transition(
            identity, expected_state=state, next_state=RunState.CANCELLED
        )
        final = await _drain(stack, identity)
        assert final is RunState.CANCELLED
        await asyncio.sleep(0.05)
        events = await stack.events(identity)
        assert len(_by_type(events, SSEEventType.RUN_COMPLETED)) == 1
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)
        assert faults == []  # a lost race against a TERMINAL run is a yield


class TestRunBudget:
    @mark.asyncio
    async def test_the_run_deadline_actually_bounds_a_slow_model(self):
        faults: list = []
        stack = _stack(
            delay_ms=5_000,
            run_timeout_ms=300,
            fault_sink=lambda c, r: faults.append((c, r)),
        )
        started = time.monotonic()
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-timeout")
        state = await _drain(stack, identity, timeout_s=10)
        elapsed = time.monotonic() - started
        assert state is RunState.FAILED
        assert elapsed < 3.0  # the 15 s default would have blown this
        events = await stack.events(identity)
        terminal = _by_type(events, SSEEventType.RUN_COMPLETED)
        assert len(terminal) == 1
        assert terminal[0].data["result"] == RESULT_DEADLINE
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)

    @mark.asyncio
    async def test_an_explicit_cancel_produces_no_late_answer(self):
        stack = _stack(delay_ms=2_000)
        identity = await stack.admit_and_drive("发热怎么办", run_id="r-del")
        await asyncio.sleep(0.1)
        state = await stack.repository.state(identity)
        await stack.repository.commit_transition(
            identity, expected_state=state, next_state=RunState.CANCELLED
        )
        stack.executor.cancel(identity.run_id)  # what the API does after the CAS
        final = await _drain(stack, identity)
        assert final is RunState.CANCELLED
        await asyncio.sleep(0.2)  # the slow model would have finished by now
        events = await stack.events(identity)
        assert not _by_type(events, SSEEventType.ANSWER_COMPLETED)
        assert not _by_type(events, SSEEventType.ANSWER_DELTA)
        assert len(_by_type(events, SSEEventType.RUN_COMPLETED)) == 1
        seqs = sorted(e.seq for e in events)
        assert seqs == list(range(1, len(seqs) + 1))  # no gap, terminal once
