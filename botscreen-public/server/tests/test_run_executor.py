"""Tests for the #55A run executor: the vertical slice wiring.

These tests drive the REAL stack end to end — synthetic approved knowledge →
MockProvider → MedicalQAAgent (RAG) → SafetyEvidenceVerifier → ManagerAgent →
RunExecutor → MemoryRunRepository — and assert on the durable event stream the
SSE layer would replay. No HTTP, no cloud provider, no real clinical content.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pytest import mark

from app.contracts.common import Channel
from app.contracts.events import SSEEventType
from app.contracts.run import RunState
from app.orchestration.executor import (
    RESULT_ANSWERED,
    RESULT_ESCALATED,
    RESULT_NO_ANSWER,
    RunExecutor,
)
from app.orchestration.state_machine import is_terminal_state
from app.storage.run_repository import MemoryRunRepository, RunIdentity

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


def _stack(delay_ms: int = 0, *, red_flags: tuple[str, ...] = ("自杀",)) -> Stack:
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
    repository = MemoryRunRepository()
    executor = RunExecutor(repository=repository, manager=manager)
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
