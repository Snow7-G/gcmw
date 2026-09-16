"""Tests for #55A-B: call-time tool quota + server-side trusted provenance.

Coverage (per the review instructions):
- quota 0: a sub-agent granted zero calls never touches the gateway;
- call-time check: the gateway call does not even START once the granted
  ceiling is reached (never a post-hoc count alone);
- cross-revision accumulation: a revision is granted only the LEFTOVER of the
  run-cumulative budget — it can never start a fresh quota;
- forged provenance: whatever the sub-agent (or the model output) claims about
  provider/model identity is DISCARDED and replaced with the server-side
  ModelGateway/Provider configuration;
- SSE publishes ONLY that trusted provenance (answer.completed.model);
- an expired deadline produces no late engagement and no late tool calls.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pytest import mark, raises

from app.agents.manager import (
    AgentExecution,
    ManagerAgent,
    ManagerAgentError,
    RedFlagRules,
    RiskRules,
    Verdict,
    VerifierOutcome,
)
from app.agents.medical_qa import MedicalQAAgent
from app.agents.registry import AgentManifest, AgentRegistry
from app.contracts.agent import AgentContext, AgentResult, AgentStatus, Evidence
from app.contracts.common import Channel
from app.contracts.events import SSEEventType
from app.orchestration.executor import RunExecutor
from app.storage.run_repository import MemoryRunRepository, RunIdentity

APPROVED_AT = datetime(2026, 1, 5, tzinfo=timezone.utc)
TRUSTED = ("mock", "mock-model", "1.0.0")
FORGED = ("fake-cloud", "gpt-infinity", "9.9.9")


def _ctx(**overrides):
    fields = {
        "tenant_id": "t1",
        "device_id": "d1",
        "session_id": "s1",
        "run_id": "r1",
        "channel": Channel.TOUCH,
        "normalized_input": "发热怎么办",
        "deadline": datetime.now(timezone.utc) + timedelta(seconds=30),
    }
    fields.update(overrides)
    return AgentContext(**fields)


def _rules():
    return (
        RedFlagRules(patterns=("自杀",), approved_by="t", approved_at=APPROVED_AT),
        RiskRules(patterns=("剧烈",), approved_by="t", approved_at=APPROVED_AT),
    )


class _CountingGateway:
    """Counts gateway round-trips; the search always finds nothing."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, ctx, request, *, allowed_tools=None, agent_id=None):
        self.calls += 1
        return SimpleNamespace(ok=True, data={"items": [], "total": 0})


class _FakeModels:
    async def chat(self, request):  # pragma: no cover - never reached here
        raise AssertionError("the model must not be called without evidence")


def _qa(gateway: _CountingGateway) -> MedicalQAAgent:
    return MedicalQAAgent(models=_FakeModels(), tools=gateway)


# -- call-time quota (QA agent level) ------------------------------------------


class TestCallTimeQuota:
    @mark.asyncio
    async def test_quota_zero_never_touches_the_gateway(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx(tool_budget_granted=0))
        assert gateway.calls == 0  # the call never even started
        assert execution.status is AgentStatus.FAILED
        assert execution.safety_status == "no_evidence"
        assert execution.tool_calls == 0

    @mark.asyncio
    async def test_a_grant_of_one_stops_after_one_call(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx(tool_budget_granted=1))
        assert gateway.calls == 1  # the second call never started
        assert execution.tool_calls == 1
        assert execution.status is AgentStatus.FAILED

    @mark.asyncio
    async def test_no_grant_uses_the_agent_own_limit(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx())  # tool_budget_granted=None
        # legacy callers stay unrestricted by the agent's own limit; the search
        # finds nothing, so exactly one call happens
        assert gateway.calls == 1
        assert execution.safety_status == "no_evidence"


# -- cross-revision accumulation (Manager level) --------------------------------


class _RecordingRunner:
    """Records the quota granted per engagement and reports the tool count of
    a WELL-BEHAVED sub-agent: min(demand, granted) — the call-time quota makes
    exceeding the grant impossible for it."""

    def __init__(self, demand: int) -> None:
        self.demand = demand
        self.granted: list[int] = []
        self.reported: list[int] = []

    async def __call__(self, ctx: AgentContext) -> AgentExecution:
        granted = ctx.tool_budget_granted
        if granted is None:
            granted = self.demand
        self.granted.append(granted)
        used = min(self.demand, granted)
        self.reported.append(used)
        return AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="发热建议门诊就诊。",
            evidence=(),
            tool_calls=used,
            provider_id=FORGED[0],
            model_id=FORGED[1],
            model_version=FORGED[2],
            safety_status="unknown",
        )


class _ScriptedVerifier:
    def __init__(self, verdicts) -> None:
        self.verdicts = list(verdicts)

    async def __call__(self, ctx, execution) -> Verdict:
        return self.verdicts.pop(0)


def _manager(runner, verifier, **overrides) -> ManagerAgent:
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            agent_id="qa",
            version="1.0.0",
            supported_intents=["knowledge"],
            risk_level="medium",
        )
    )
    red, risk = _rules()
    return ManagerAgent(
        registry=registry,
        agent_runners={"qa": runner},
        verifier=verifier,
        red_flag_rules=red,
        risk_rules=risk,
        trusted_model_provenance={"qa": TRUSTED},
        **overrides,
    )


class TestCrossRevisionAccumulation:
    @mark.asyncio
    async def test_a_revision_is_granted_only_the_leftover(self):
        runner = _RecordingRunner(demand=2)  # each engagement demands 2
        verifier = _ScriptedVerifier(
            [
                Verdict(VerifierOutcome.REVISE, "ungrounded"),
                Verdict(VerifierOutcome.PASS, ""),
            ]
        )
        manager = _manager(runner, verifier)
        result = await manager.execute(_ctx(), "发热怎么办")
        assert runner.granted == [4, 2]  # the revision sees the LEFTOVER only
        assert result.safety_status == "verified"

    @mark.asyncio
    async def test_an_exhausted_budget_grants_zero(self):
        runner = _RecordingRunner(demand=4)  # burns the whole budget
        verifier = _ScriptedVerifier(
            [
                Verdict(VerifierOutcome.REVISE, "ungrounded"),
                Verdict(VerifierOutcome.PASS, ""),
            ]
        )
        manager = _manager(runner, verifier)
        result = await manager.execute(_ctx(), "发热怎么办")
        assert runner.granted == [4, 0]  # zero left: zero calls possible
        assert runner.reported == [4, 0]  # a well-behaved agent made ZERO calls
        assert result.safety_status == "verified"


# -- trusted provenance ----------------------------------------------------------


class TestTrustedProvenance:
    @mark.asyncio
    async def test_forged_model_identity_is_discarded(self):
        runner = _RecordingRunner(demand=1)  # claims FORGED identity
        verifier = _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        manager = _manager(runner, verifier)
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.model == {
            "provider_id": TRUSTED[0],
            "model_id": TRUSTED[1],
            "model_version": TRUSTED[2],
        }  # the FORGED claim never reaches the public result

    @mark.asyncio
    async def test_a_refusal_publishes_no_provenance(self):
        class _EscalatingVerifier:
            async def __call__(self, ctx, execution) -> Verdict:
                return Verdict(VerifierOutcome.ESCALATE, "")

        manager = _manager(_RecordingRunner(demand=1), _EscalatingVerifier())
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.model is None  # no answer, no provenance on the wire

    @mark.asyncio
    async def test_an_expired_deadline_produces_no_late_engagement(self):
        class _NeverCalled(_RecordingRunner):
            async def __call__(self, ctx) -> AgentExecution:  # pragma: no cover
                raise AssertionError("a timed-out run must not engage the agent")

        runner = _NeverCalled(demand=0)
        manager = _manager(runner, _ScriptedVerifier([]))
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        with raises(ManagerAgentError):
            await manager.execute(_ctx(deadline=past), "发热怎么办")


# -- SSE publishes only the trusted provenance -----------------------------------


class _VerifiedManager:
    """Minimal manager stub delivering a verified answer with provenance."""

    def __init__(self, model: dict[str, str] | None) -> None:
        self._model = model

    async def execute(self, ctx, text, on_stage=None) -> AgentResult:
        for stage in ("guarding", "routing", "retrieving", "drafting", "verifying"):
            if on_stage is not None:
                await on_stage(stage)
        return AgentResult(
            agent_id="manager",
            status=AgentStatus.COMPLETED,
            answer_candidate="体温超过38.5建议门诊就诊。",
            evidence=[
                Evidence(
                    source_id="faq-fever",
                    source_type="faq",
                    title="发热护理须知",
                    content="体温超过38.5建议门诊就诊。",
                    source_uri="kbase://faq-fever",
                    content_hash="h",
                    knowledge_version="v1",
                )
            ],
            actions=[],
            safety_status="verified",
            model=self._model,
        )


class TestSsePublishesOnlyTrustedProvenance:
    async def _drive(self, model):
        repository = MemoryRunRepository()
        executor = RunExecutor(
            repository=repository,
            manager=_VerifiedManager(model),  # type: ignore[arg-type]
        )
        identity = RunIdentity(
            run_id="r-sse-prov", tenant_id="t1", device_id="d1", session_id="s1"
        )
        await repository.create(identity)

        class _Record:
            pass

        record = _Record()
        record.identity = identity

        class _Snap:
            channel = "text"
            text = "发热怎么办"

        record.snapshot = _Snap()
        await asyncio.wait_for(executor.execute(record), timeout=5)
        page = await repository.snapshot(identity, 0, 0.0)
        completed = [e for e in page.events if e.event is SSEEventType.ANSWER_COMPLETED]
        assert len(completed) == 1
        return completed[0].data

    @mark.asyncio
    async def test_the_wire_carries_the_trusted_triple(self):
        data = await self._drive(
            {"provider_id": "mock", "model_id": "mock-model", "model_version": "1.0.0"}
        )
        assert data["model"] == {
            "provider_id": "mock",
            "model_id": "mock-model",
            "model_version": "1.0.0",
        }

    @mark.asyncio
    async def test_no_provenance_attached_means_no_provenance_on_the_wire(self):
        data = await self._drive(None)
        assert "model" not in data  # a Manager without provenance publishes none
