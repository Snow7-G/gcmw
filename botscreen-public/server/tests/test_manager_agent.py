"""Tests for ManagerAgent (issue #52).

Acceptance coverage (V2.3 §6.1 subset):
- guard: incomplete context / empty / oversized input are rejected with
  structured codes; PII-ish patterns are desensitized before any runner;
- pre-model red flags escalate without invoking any model/tool/runner;
- intent classification + registry routing are deterministic; unknown intent
  raises NOT_FOUND_AGENT;
- hard budgets: ≤2 handoffs / ≤4 tool calls / ≤1 revision are enforced
  (RUN_BUDGET_EXCEEDED / TOOL_OVER_LIMIT);
- evidence from the executed agent lands on the final AgentResult;
- verifier handoff happens at most after completion; a reject triggers at
  most one controlled revision; provider/model ids are recorded for runs;
- the final AgentResult carries safe markers only (no chain-of-thought).
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from pytest import mark

from app.agents.manager import (
    AgentExecution,
    ManagerAgent,
    ManagerAgentError,
    ManagerLimits,
    Verdict,
    desensitize,
)
from app.agents.registry import AgentManifest, AgentRegistry
from app.contracts.agent import AgentContext, AgentStatus, Evidence
from app.contracts.common import Channel
from app.contracts.errors import ErrorCode


def _agent_execution(
    answer="发热咳嗽请挂呼吸内科",
    evidence=(),
    tool_calls=0,
    provider_id="mock",
    model_id="mock-model",
    model_version="1.0.0",
    status=AgentStatus.COMPLETED,
):
    return AgentExecution(
        agent_id="qa",
        status=status,
        answer_candidate=answer,
        evidence=tuple(evidence),
        tool_calls=tool_calls,
        provider_id=provider_id,
        model_id=model_id,
        model_version=model_version,
    )


def _context(**overrides):
    fields = {
        "tenant_id": "t1",
        "device_id": "d1",
        "session_id": "s1",
        "run_id": "r1",
        "channel": Channel.TOUCH,
        "deadline": datetime.now(timezone.utc) + timedelta(seconds=30),
    }
    fields.update(overrides)
    return AgentContext(**fields)


def _qa_manifest(**overrides):
    fields = {
        "agent_id": "qa",
        "version": "1.0.0",
        "supported_intents": ["knowledge"],
        "risk_level": "medium",
    }
    fields.update(overrides)
    return AgentManifest(**fields)


class _Runner:
    def __init__(self, result=None, sleep_ms=0):
        self._result = result if result is not None else _agent_execution()
        self.calls = []
        self.sleep_ms = sleep_ms

    async def __call__(self, ctx: AgentContext) -> AgentExecution:
        self.calls.append(ctx.normalized_input)
        if self.sleep_ms:
            await asyncio.sleep(self.sleep_ms / 1000)
        return self._result


class _Verifier:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen = []

    async def __call__(self, ctx, execution: AgentExecution) -> Verdict:
        self.seen.append(execution.answer_candidate)
        return self.verdicts.pop(0)


@pytest.fixture
def registry():
    reg = AgentRegistry()
    reg.register(_qa_manifest())
    return reg


class TestDesensitize:
    def test_api_key_pattern_removed(self):
        cleaned = desensitize("sk-abcdef1234567890abcdef1234 帮我挂号")
        assert "abcdef1234567890" not in cleaned
        assert "帮我挂号" in cleaned

    def test_mobile_and_id_removed(self):
        cleaned = desensitize("电话 13800138000 身份证 11010119900307777X 问诊")
        assert "13800138000" not in cleaned
        assert "11010119900307777X" not in cleaned
        assert "问诊" in cleaned

    def test_plain_text_untouched(self):
        assert desensitize("发烧两天了") == "发烧两天了"


class TestGuard:
    @mark.asyncio
    async def test_expired_deadline_rejected(self, registry):
        m = ManagerAgent(registry=registry, agent_runners={"qa": _Runner()})
        past = _context(deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(past, "问点什么")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT

    @mark.asyncio
    async def test_empty_input_rejected(self, registry):
        m = ManagerAgent(registry=registry, agent_runners={"qa": _Runner()})
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "   ")
        assert exc.value.code is ErrorCode.VALIDATION_INVALID_INPUT

    @mark.asyncio
    async def test_oversized_input_rejected(self, registry):
        m = ManagerAgent(registry=registry, agent_runners={"qa": _Runner()})
        limit = ManagerLimits().max_input_chars
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "字" * (limit + 1))
        assert exc.value.code is ErrorCode.VALIDATION_INVALID_INPUT

    @mark.asyncio
    async def test_input_is_desensitized_before_runner(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        m = ManagerAgent(registry=reg, agent_runners={"qa": runner})
        await m.execute(_context(), "发烧 电话 13800138000")
        assert runner.calls == ["发烧 电话 [已脱敏]"]


class TestRedFlagGate:
    @mark.asyncio
    async def test_escalation_skips_model_tools_and_runner(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": runner},
            red_flags=("自杀",),
        )
        result = await m.execute(_context(), "我想自杀怎么办")
        assert result.safety_status == "escalated"
        assert result.answer_candidate == ""
        assert runner.calls == []  # nothing executed
        marker_types = [a["type"] for a in result.actions]
        assert "safety.escalate" in marker_types
        assert "route.handoff" not in marker_types


class TestRoutingAndBudgets:
    @mark.asyncio
    async def test_unknown_intent_raises_not_found(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": _Runner()},
            default_intent="nonsense",
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "你好")
        assert exc.value.code is ErrorCode.NOT_FOUND_AGENT

    @mark.asyncio
    async def test_keyword_route_beats_default(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        reg.register(
            AgentManifest(
                agent_id="booker",
                version="1.0.0",
                supported_intents=["booking"],
            )
        )
        qa_runner = _Runner()
        booker_runner = _Runner()
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": qa_runner, "booker": booker_runner},
            intent_routes={"booking": ("预约", "挂号")},
        )
        result = await m.execute(_context(), "帮我预约明天")
        handoff = next(a for a in result.actions if a["type"] == "route.handoff")
        assert handoff["agent_id"] == "booker"
        assert qa_runner.calls == []
        assert booker_runner.calls == ["帮我预约明天"]

    @mark.asyncio
    async def test_tool_budget_violation_raises_tool_over_limit(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": _Runner(result=_agent_execution(tool_calls=5))},
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT

    @mark.asyncio
    async def test_handoff_budget_violation_raises_budget_exceeded(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": _Runner()},
            verifier=_Verifier([Verdict(True)]),
            limits=ManagerLimits(max_handoffs=1),  # qa + verifier = 2 > 1
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.RUN_BUDGET_EXCEEDED

    @mark.asyncio
    async def test_missing_runner_raises_not_found(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        m = ManagerAgent(registry=reg, agent_runners={})
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.NOT_FOUND_AGENT


class TestPipeline:
    @mark.asyncio
    async def test_happy_path_routes_and_collects_evidence(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        evidence = (
            Evidence(
                source_id="faq-1",
                source_type="faq",
                title="发热指南",
                content="片段",
                content_hash="h1",
            ),
        )
        runner = _Runner(result=_agent_execution(evidence=evidence))
        m = ManagerAgent(registry=reg, agent_runners={"qa": runner})
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.answer_candidate == "发热咳嗽请挂呼吸内科"
        assert result.evidence[0].source_id == "faq-1"
        assert result.safety_status == "passed"
        markers = [a["type"] for a in result.actions]
        assert "route.handoff" in markers
        assert "agent.done" in markers
        model_call = next(a for a in result.actions if a["type"] == "model.call")
        assert model_call["provider_id"] == "mock"
        assert model_call["model_id"] == "mock-model"
        assert model_call["model_version"] == "1.0.0"
        # safe markers only: no chain-of-thought or raw content in actions
        allowed_keys = {
            "type",
            "intent",
            "agent_id",
            "handoffs",
            "status",
            "approved",
            "revision",
            "count",
            "provider_id",
            "model_id",
            "model_version",
        }
        for action in result.actions:
            assert set(action).issubset(allowed_keys)

    @mark.asyncio
    async def test_verifier_approves_first_round(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        verifier = _Verifier([Verdict(True, "ok")])
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.safety_status == "verified"
        assert result.confidence_band == "high"
        assert verifier.seen == ["发热咳嗽请挂呼吸内科"]
        assert [a for a in result.actions if a["type"] == "verify.verdict"]

    @mark.asyncio
    async def test_reject_then_controlled_revision_once(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        verifier = _Verifier([Verdict(False, "revision"), Verdict(True, "ok")])
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.safety_status == "verified"
        assert len(runner.calls) == 2  # initial + one controlled revision
        revise_markers = [a for a in result.actions if a["type"] == "verify.revise"]
        assert len(revise_markers) == 1

    @mark.asyncio
    async def test_no_more_than_one_revision_even_if_still_rejected(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        verifier = _Verifier([Verdict(False)] * 3)
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        # at most one revision: initial run + one rerun, two verdicts consumed
        assert len(runner.calls) == 2
        assert len(verifier.seen) == 2
        assert result.safety_status == "revised"

    @mark.asyncio
    async def test_failed_agent_run_skips_verifier(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner(result=_agent_execution(status=AgentStatus.FAILED))
        verifier = _Verifier([Verdict(True)])
        m = ManagerAgent(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert verifier.seen == []  # no verification of failed runs


class TestDeadline:
    @mark.asyncio
    async def test_runner_deadline_maps_to_timeout(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner(sleep_ms=3000)
        m = ManagerAgent(registry=reg, agent_runners={"qa": runner})
        ctx = _context(deadline=datetime.now(timezone.utc) + timedelta(milliseconds=80))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(ctx, "发烧怎么办")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT
