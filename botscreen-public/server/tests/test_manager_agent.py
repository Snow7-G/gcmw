"""Tests for ManagerAgent (issue #52).

Acceptance coverage (V2.3 §6.1 subset):
- guard: incomplete context / empty / oversized input are rejected with
  structured codes; PII-ish patterns are desensitized before any runner;
- pre-model red flags escalate without invoking any model/tool/runner;
- intent classification + registry routing are deterministic; unknown intent
  raises NOT_FOUND_AGENT;
- budgets: the ≤2 handoff cap is checked BEFORE each engagement
  (RUN_BUDGET_EXCEEDED); the ≤4 TOOL cap is a POST-HOC, RUN-CUMULATIVE check
  over the sub-agent's own report — the first execution plus every revision are
  summed (TOOL_OVER_LIMIT) — and the ≤1 revision cap is a hard ceiling. A
  call-time tool quota is NOT implemented here; it arrives with the ToolGateway
  quota path (#55A);
- the sub-agent's return value is untrusted input: identity must match the
  trusted routing decision, only COMPLETED/FAILED/CANCELLED are reportable,
  tool_calls must be a non-bool non-negative int, answer/evidence are strictly
  typed, and every violation maps to MODEL_OUTPUT_UNPARSEABLE with a fixed
  message (no echoed value, no original exception text);
- validated results are deep-copied snapshots: the verifier sees an isolated
  copy and delivery takes a fresh snapshot, so neither the runner nor the
  verifier can mutate what is delivered;
- evidence from the executed agent lands on the final AgentResult ONLY on the
  COMPLETED + PASS path; every refusal publishes no answer and no evidence;
- verifier handoff happens at most after completion; a reject triggers at most
  one controlled revision; model provenance is NOT published (no trusted source
  yet — the run record gets it from the ModelGateway wiring in #55A);
- the final AgentResult carries safe markers only (no chain-of-thought, no
  free text from any source).
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from pytest import mark

from app.agents.manager import (
    SAFE_MARKER_KEYS,
    SAFE_MARKER_TYPES,
    AgentExecution,
    ManagerAgent,
    ManagerAgentError,
    ManagerLimits,
    RedFlagRules,
    RiskRules,
    Verdict,
    VerifierOutcome,
    desensitize,
    highest_risk,
    is_safe_marker_value,
)
from app.agents.registry import AgentManifest, AgentRegistry
from app.contracts.agent import AgentContext, AgentStatus, Evidence, RiskLevel
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
    agent_id="qa",
    safety_status="unknown",
):
    return AgentExecution(
        agent_id=agent_id,
        status=status,
        answer_candidate=answer,
        evidence=tuple(evidence),
        tool_calls=tool_calls,
        provider_id=provider_id,
        model_id=model_id,
        model_version=model_version,
        safety_status=safety_status,
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
        self.risks = []
        self.sleep_ms = sleep_ms

    async def __call__(self, ctx: AgentContext) -> AgentExecution:
        self.calls.append(ctx.normalized_input)
        self.risks.append(ctx.risk_level)
        if self.sleep_ms:
            await asyncio.sleep(self.sleep_ms / 1000)
        return self._result


class _Verifier:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen = []
        self.risks = []

    async def __call__(self, ctx, execution: AgentExecution) -> Verdict:
        self.seen.append(execution.answer_candidate)
        self.risks.append(ctx.risk_level)
        return self.verdicts.pop(0)


APPROVED_AT = datetime(2026, 1, 5, tzinfo=timezone.utc)


def approved_red_flags(*patterns: str) -> RedFlagRules:
    return RedFlagRules(
        patterns=patterns or ("自杀", "胸痛"),
        approved_by="clinical-board",
        approved_at=APPROVED_AT,
    )


def approved_risk_rules(*patterns: str) -> RiskRules:
    return RiskRules(
        patterns=patterns or ("急诊", "剧烈"),
        approved_by="clinical-board",
        approved_at=APPROVED_AT,
    )


def build_manager(**overrides):
    """Manager with the APPROVED rule sets the constructor now requires."""
    overrides.setdefault("red_flag_rules", approved_red_flags())
    overrides.setdefault("risk_rules", approved_risk_rules())
    return ManagerAgent(**overrides)


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
        m = build_manager(registry=registry, agent_runners={"qa": _Runner()})
        past = _context(deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(past, "问点什么")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT

    @mark.asyncio
    async def test_empty_input_rejected(self, registry):
        m = build_manager(registry=registry, agent_runners={"qa": _Runner()})
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "   ")
        assert exc.value.code is ErrorCode.VALIDATION_INVALID_INPUT

    @mark.asyncio
    async def test_oversized_input_rejected(self, registry):
        m = build_manager(registry=registry, agent_runners={"qa": _Runner()})
        limit = ManagerLimits().max_input_chars
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "字" * (limit + 1))
        assert exc.value.code is ErrorCode.VALIDATION_INVALID_INPUT

    @mark.asyncio
    async def test_input_is_desensitized_before_runner(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        m = build_manager(registry=reg, agent_runners={"qa": runner})
        await m.execute(_context(), "发烧 电话 13800138000")
        assert runner.calls == ["发烧 电话 [已脱敏]"]


class TestRedFlagGate:
    @mark.asyncio
    async def test_escalation_skips_model_tools_and_runner(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        m = build_manager(
            registry=reg,
            agent_runners={"qa": runner},
            red_flag_rules=approved_red_flags(
                "自杀",
            ),
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
        m = build_manager(
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
        # the routed agent must report the identity routing gave it: the booker
        # runner used to answer as "qa", which the trust boundary now refuses
        booker_runner = _Runner(result=_agent_execution(agent_id="booker"))
        m = build_manager(
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
        m = build_manager(
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
        m = build_manager(
            registry=reg,
            agent_runners={"qa": _Runner()},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            limits=ManagerLimits(max_handoffs=1),  # qa + verifier = 2 > 1
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.RUN_BUDGET_EXCEEDED

    @mark.asyncio
    async def test_missing_runner_raises_not_found(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        m = build_manager(registry=reg, agent_runners={})
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
        m = build_manager(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.answer_candidate == "发热咳嗽请挂呼吸内科"
        assert result.evidence[0].source_id == "faq-1"
        assert result.safety_status == "verified"
        markers = [a["type"] for a in result.actions]
        assert "route.handoff" in markers
        assert "agent.done" in markers
        assert "manager.risk" in markers  # risk is decided before routing
        # model provenance is NOT published from a sub-agent self-report: the
        # run record takes it from the trusted ModelGateway (#55A)
        assert not any(a["type"] == "model.call" for a in result.actions)
        # safe markers only: every marker and every key must be inside the
        # module's own allowlist (enforced by ManagerAgent._mark)
        for action in result.actions:
            assert action["type"] in SAFE_MARKER_TYPES
            assert set(action).issubset(SAFE_MARKER_KEYS)

    @mark.asyncio
    async def test_verifier_approves_first_round(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        verifier = _Verifier([Verdict(VerifierOutcome.PASS, "ok")])
        m = build_manager(
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
        verifier = _Verifier(
            [
                Verdict(VerifierOutcome.REVISE, "revision"),
                Verdict(VerifierOutcome.PASS, "ok"),
            ]
        )
        m = build_manager(
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
        verifier = _Verifier([Verdict(VerifierOutcome.REVISE)] * 3)
        m = build_manager(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        # at most one revision: initial run + one rerun, two verdicts consumed
        assert len(runner.calls) == 2
        assert len(verifier.seen) == 2
        # an unverified medical answer is never delivered: the run fails clean
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        marker_types = [a["type"] for a in result.actions]
        assert "verify.reject_final" in marker_types
        assert "manager.revised" in marker_types

    @mark.asyncio
    async def test_rejected_twice_marks_reject_final_only_after_revision(self):
        # reject -> one revision -> reject again: still no answer delivered
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner()
        verifier = _Verifier(
            [Verdict(VerifierOutcome.REVISE), Verdict(VerifierOutcome.REVISE)]
        )
        m = build_manager(
            registry=reg,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert [a["type"] for a in result.actions].count("verify.reject_final") == 1
        assert len(runner.calls) == 2

    @mark.asyncio
    async def test_failed_agent_run_skips_verifier(self):
        reg = AgentRegistry()
        reg.register(_qa_manifest())
        runner = _Runner(result=_agent_execution(status=AgentStatus.FAILED))
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
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
        m = build_manager(registry=reg, agent_runners={"qa": runner})
        ctx = _context(deadline=datetime.now(timezone.utc) + timedelta(milliseconds=80))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(ctx, "发烧怎么办")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT


class TestMissingVerifierNeverAnswers:
    """Review P1: no verifier = no medical answer (was marked "passed")."""

    @mark.asyncio
    async def test_missing_verifier_fails_closed(self, registry):
        runner = _Runner(
            result=_agent_execution(
                evidence=(
                    Evidence(
                        source_id="faq-1",
                        source_type="faq",
                        title="t",
                        content="c",
                        content_hash="h",
                    ),
                )
            )
        )
        m = build_manager(registry=registry, agent_runners={"qa": runner})
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert result.safety_status == "unverified"
        assert any(a["type"] == "verify.missing" for a in result.actions)
        assert len(runner.calls) == 1  # the agent ran, but nothing was delivered

    @mark.asyncio
    async def test_no_verifier_means_no_verifier_handoff_marker(self, registry):
        m = build_manager(
            registry=registry,
            agent_runners={"qa": _Runner(result=_agent_execution())},
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert not any(a["type"] == "verify.handoff" for a in result.actions)


class TestRuleSetsAreRequired:
    """Review P1: unapproved/absent rules must never default to "allow"."""

    def test_manager_requires_red_flag_rules(self, registry):
        with pytest.raises(ValueError) as exc:
            ManagerAgent(
                registry=registry,
                agent_runners={},
                risk_rules=approved_risk_rules(),
            )
        assert "red_flag_rules" in str(exc.value)

    def test_manager_requires_risk_rules(self, registry):
        with pytest.raises(ValueError) as exc:
            ManagerAgent(
                registry=registry,
                agent_runners={},
                red_flag_rules=approved_red_flags(),
            )
        assert "risk_rules" in str(exc.value)

    @mark.parametrize(
        "factory",
        [
            lambda: RedFlagRules(patterns=(), approved_by="x", approved_at=APPROVED_AT),
            lambda: RedFlagRules(
                patterns=("  ",), approved_by="x", approved_at=APPROVED_AT
            ),
            lambda: RedFlagRules(
                patterns=("胸痛",), approved_by=" ", approved_at=APPROVED_AT
            ),
            lambda: RiskRules(patterns=(), approved_by="x", approved_at=APPROVED_AT),
            lambda: RiskRules(
                patterns=("急诊",), approved_by="", approved_at=APPROVED_AT
            ),
        ],
    )
    def test_empty_or_unapproved_rule_sets_are_rejected(self, factory):
        with pytest.raises(ValueError):
            factory()


class TestVerdictIsNotABoolean:
    """Review P1: PASS/REVISE retry rules; BLOCK and ESCALATE never retry."""

    @staticmethod
    def _manager(registry, outcomes, runner=None):
        return build_manager(
            registry=registry,
            agent_runners={"qa": runner or _Runner(result=_agent_execution())},
            verifier=_Verifier(outcomes),
        )

    @mark.asyncio
    async def test_pass_delivers_the_answer(self, registry):
        m = self._manager(registry, [Verdict(VerifierOutcome.PASS)])
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.answer_candidate != ""
        assert result.safety_status == "verified"

    @mark.asyncio
    async def test_block_stops_immediately_without_retry(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.BLOCK)])
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert result.safety_status == "blocked"
        assert len(verifier.seen) == 1  # BLOCK is not retried
        assert len(runner.calls) == 1  # and the agent is not re-run
        assert any(a["type"] == "verify.blocked" for a in result.actions)

    @mark.asyncio
    async def test_escalate_hands_off_without_an_ai_answer(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.ESCALATE)])
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.answer_candidate == ""
        assert result.safety_status == "escalated"
        assert len(verifier.seen) == 1 and len(runner.calls) == 1
        assert any(a["type"] == "verify.escalated" for a in result.actions)

    @mark.asyncio
    async def test_only_revise_retries_and_only_once(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier(
            [
                Verdict(VerifierOutcome.REVISE),
                Verdict(VerifierOutcome.REVISE),  # still not satisfied
            ]
        )
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert len(runner.calls) == 2  # original + exactly one revision
        assert len(verifier.seen) == 2
        assert any(a["type"] == "verify.reject_final" for a in result.actions)

    @mark.asyncio
    async def test_revise_then_pass_delivers_the_revised_answer(self, registry):
        m = self._manager(
            registry,
            [Verdict(VerifierOutcome.REVISE), Verdict(VerifierOutcome.PASS)],
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.safety_status == "verified"


class TestRedFlagIsPreModel:
    """A red-flag hit must not reach any model or tool."""

    @mark.asyncio
    async def test_red_flag_hit_makes_zero_model_and_tool_calls(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            red_flag_rules=approved_red_flags("自杀"),
        )
        result = await m.execute(_context(), "我想自杀")
        assert result.answer_candidate == ""
        assert result.safety_status == "escalated"
        assert len(runner.calls) == 0
        assert len(verifier.seen) == 0
        assert [a["type"] for a in result.actions].count("safety.escalate") == 1


class TestRiskGrading:
    @mark.asyncio
    async def test_high_risk_question_is_classified_before_routing(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        result = await m.execute(_context(), "剧烈头痛需要急诊吗")
        risk_marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert risk_marker["level"] == "high"
        # high risk still goes through the FULL path (no fast-path shortcut)
        assert any(a["type"] == "verify.handoff" for a in result.actions)

    @mark.asyncio
    async def test_low_risk_question_is_classified_low(self, registry):
        m = build_manager(
            registry=registry,
            agent_runners={"qa": _Runner(result=_agent_execution())},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        result = await m.execute(_context(), "孩子近视后需要复查吗")
        risk_marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert risk_marker["level"] == "low"


class TestBudgetScopeIsHonest:
    """The tool budget is a POST-HOC check today (documented, not enforced)."""

    @mark.asyncio
    async def test_over_reported_tool_calls_fail_the_run(self, registry):
        runner = _Runner(result=_agent_execution(tool_calls=99))
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT

    @mark.asyncio
    async def test_under_reported_tool_calls_are_not_detected(self, registry):
        """CHARACTERISATION of the documented limitation: a run that really
        exceeded the tool budget but under-reports is NOT stopped here. The
        call-time hard cap arrives with #55A (ToolGateway quota), so this PR
        must not claim the tool budget is preemptively enforced."""
        runner = _Runner(result=_agent_execution(tool_calls=1))  # claims 1, did N
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED  # no call-time counter here

    def test_limits_docstring_states_the_scope(self):
        assert "POST-HOC" in (ManagerLimits.__doc__ or "")


class TestRiskIsNeverDowngraded:
    """Review P1: a rule MISS must not lower a declared MEDIUM/CRITICAL risk.

    ``classify`` only ever returns LOW/HIGH from *text* rules, so using it alone
    reported LOW for a context that arrived as MEDIUM or CRITICAL. The effective
    level is the higher of the two, across all four contract levels.
    """

    @mark.parametrize(
        ("declared", "expected"),
        [
            (RiskLevel.LOW, RiskLevel.LOW),
            (RiskLevel.MEDIUM, RiskLevel.MEDIUM),
            (RiskLevel.HIGH, RiskLevel.HIGH),
            (RiskLevel.CRITICAL, RiskLevel.CRITICAL),
        ],
    )
    @mark.asyncio
    async def test_a_rule_miss_keeps_the_declared_level(
        self, registry, declared, expected
    ):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            risk_rules=approved_risk_rules("剧烈"),  # matches NOTHING here
        )
        result = await m.execute(_context(risk_level=declared), "孩子近视后需要复查吗")
        marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert marker["level"] == expected.value
        assert marker["declared"] == declared.value
        assert runner.risks == [expected]  # what the routed agent receives
        assert verifier.risks == [expected]  # and the verifier

    @mark.asyncio
    async def test_a_rule_hit_raises_a_lower_declared_level(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        result = await m.execute(
            _context(risk_level=RiskLevel.LOW), "剧烈头痛需要急诊吗"
        )
        marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert marker["level"] == "high"
        assert marker["declared"] == "low"
        assert runner.risks == [RiskLevel.HIGH]

    @mark.asyncio
    async def test_critical_stays_critical_when_the_rule_also_hits(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        await m.execute(_context(risk_level=RiskLevel.CRITICAL), "剧烈头痛需要急诊吗")
        assert runner.risks == [RiskLevel.CRITICAL]  # never downgraded to high

    def test_the_ordering_covers_every_contract_level(self):
        assert highest_risk(RiskLevel.LOW, RiskLevel.CRITICAL) is RiskLevel.CRITICAL
        assert highest_risk(RiskLevel.MEDIUM, RiskLevel.HIGH) is RiskLevel.HIGH
        assert highest_risk(RiskLevel.HIGH, RiskLevel.MEDIUM) is RiskLevel.HIGH
        assert highest_risk(RiskLevel.LOW, RiskLevel.MEDIUM) is RiskLevel.MEDIUM
        assert highest_risk(RiskLevel.LOW) is RiskLevel.LOW
        with pytest.raises(KeyError):
            highest_risk(RiskLevel.LOW, "urgent")  # type: ignore[arg-type]


class TestRiskReachesTheExecutionChain:
    """Review P1: the Manager's risk decision must reach the runner/verifier.

    The marker said ``high`` while ``AgentContext.risk_level`` stayed ``low``, so
    a downstream fast path that reads only the context saw a low-risk question.
    """

    @mark.asyncio
    async def test_high_risk_is_written_into_the_context(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            risk_rules=approved_risk_rules("剧烈"),
        )
        result = await m.execute(_context(), "剧烈头痛需要急诊吗")
        risk_marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert risk_marker["level"] == "high"
        assert runner.risks == [RiskLevel.HIGH]  # what the routed agent saw
        assert verifier.risks == [RiskLevel.HIGH]  # and what the verifier saw
        assert result.status is AgentStatus.COMPLETED

    @mark.asyncio
    async def test_the_context_risk_is_the_contract_enum(self, registry):
        """The value must be the shared contract type, not a look-alike enum."""
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        await m.execute(_context(), "剧烈头痛需要急诊吗")
        assert isinstance(runner.risks[0], RiskLevel)
        assert runner.risks[0] is RiskLevel.HIGH

    @mark.asyncio
    async def test_low_risk_stays_low_and_is_not_upgraded(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            risk_rules=approved_risk_rules("剧烈"),
        )
        result = await m.execute(_context(), "孩子近视后需要复查吗")
        risk_marker = next(a for a in result.actions if a["type"] == "manager.risk")
        assert risk_marker["level"] == "low"
        assert runner.risks == [RiskLevel.LOW]

    @mark.asyncio
    async def test_a_revision_also_carries_the_risk(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier(
                [Verdict(VerifierOutcome.REVISE), Verdict(VerifierOutcome.PASS)]
            ),
            risk_rules=approved_risk_rules("剧烈"),
        )
        await m.execute(_context(), "剧烈头痛需要急诊吗")
        assert runner.risks == [RiskLevel.HIGH, RiskLevel.HIGH]


def _evidence(content="证据正文：疑似脑膜炎，请立即服用抗生素") -> Evidence:
    return Evidence(
        source_id="faq-1",
        source_type="faq",
        title="t",
        content=content,
        content_hash="h",
    )


class TestUnverifiedEvidenceIsNeverPublished:
    """Review P1: the draft was cleared on FAILED, but ``Evidence.content`` was not.

    Evidence content is model-generated material that no verifier has passed, so
    it is exactly as unpublishable as the draft. Every non-PASS exit must publish
    an empty evidence list.
    """

    MARKER = "疑似脑膜炎"

    @mark.asyncio
    async def test_failed_run_publishes_no_evidence(self, registry):
        runner = _Runner(
            result=_agent_execution(
                answer="草稿",
                evidence=(_evidence(),),
                status=AgentStatus.FAILED,
            )
        )
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.evidence == []
        assert self.MARKER not in json.dumps(result.model_dump(), ensure_ascii=False)

    @mark.parametrize(
        ("verdict", "expected_safety"),
        [
            (Verdict(VerifierOutcome.BLOCK), "blocked"),
            (Verdict(VerifierOutcome.ESCALATE), "escalated"),
            (Verdict(VerifierOutcome.REVISE), "revised"),
        ],
    )
    @mark.asyncio
    async def test_no_pass_verdict_publishes_no_evidence(
        self, registry, verdict, expected_safety
    ):
        runner = _Runner(result=_agent_execution(evidence=(_evidence(),)))
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            # two verdicts: REVISE consumes the one controlled revision
            verifier=_Verifier([verdict, verdict]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.safety_status == expected_safety
        assert result.answer_candidate == ""
        assert result.evidence == []
        assert self.MARKER not in json.dumps(result.model_dump(), ensure_ascii=False)

    @mark.asyncio
    async def test_missing_verifier_publishes_no_evidence(self, registry):
        runner = _Runner(result=_agent_execution(evidence=(_evidence(),)))
        m = build_manager(registry=registry, agent_runners={"qa": runner})
        result = await m.execute(_context(), "发烧怎么办")
        assert result.safety_status == "unverified"
        assert result.evidence == []

    @mark.asyncio
    async def test_a_pass_run_still_publishes_its_evidence(self, registry):
        """The refusal must not be over-broad: verified evidence is delivered."""
        runner = _Runner(
            result=_agent_execution(
                answer="发热咳嗽请挂呼吸内科", evidence=(_evidence(),)
            )
        )
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.safety_status == "verified"
        assert [item.source_id for item in result.evidence] == ["faq-1"]

    @mark.asyncio
    async def test_red_flag_path_publishes_no_evidence(self, registry):
        runner = _Runner(result=_agent_execution(evidence=(_evidence(),)))
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            red_flag_rules=approved_red_flags("自杀"),
        )
        result = await m.execute(_context(), "我想自杀")
        assert result.safety_status == "escalated"
        assert result.evidence == []
        assert len(runner.calls) == 0


class TestHandoffBudgetIsCheckedBeforeTheCall:
    """Review P1: a tightened handoff budget must forbid the call, not report it."""

    @mark.asyncio
    async def test_max_handoffs_zero_never_runs_the_agent(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            limits=ManagerLimits(max_handoffs=0),
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.RUN_BUDGET_EXCEEDED
        assert len(runner.calls) == 0  # zero engagements: refused up front
        assert len(verifier.seen) == 0

    @mark.asyncio
    async def test_the_first_handoff_is_still_allowed_at_the_default(self, registry):
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert len(runner.calls) == 1

    @mark.asyncio
    async def test_the_verifier_handoff_is_also_checked_before_it_runs(self, registry):
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            limits=ManagerLimits(max_handoffs=1),  # qa + verifier = 2 > 1
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.RUN_BUDGET_EXCEEDED
        assert len(runner.calls) == 1  # the routed agent ran...
        assert verifier.seen == []  # ...but the verifier was never engaged


class TestClinicalRuleSetsAreValidated:
    """Review: a rule that can never fire is worse than no rule at all."""

    @mark.parametrize(
        "patterns",
        [
            (" 自杀 ",),  # un-normalized: would never match "我想自杀"
            ("自杀 ",),
            (" 自杀",),
            ("自杀\n",),
            ("自杀\t胸痛",),
            ("",),
            ("   ",),
        ],
    )
    def test_unusable_patterns_are_rejected(self, patterns):
        with pytest.raises(ValueError):
            RedFlagRules(
                patterns=patterns, approved_by="clinical-board", approved_at=APPROVED_AT
            )
        with pytest.raises(ValueError):
            RiskRules(
                patterns=patterns, approved_by="clinical-board", approved_at=APPROVED_AT
            )

    def test_a_normalized_pattern_actually_matches(self):
        rules = RedFlagRules(
            patterns=("自杀",), approved_by="clinical-board", approved_at=APPROVED_AT
        )
        assert rules.matches("我想自杀")

    def test_naive_approval_timestamps_are_rejected(self):
        naive = datetime(2026, 1, 5)  # noqa: DTZ001 - naive on purpose: must be refused
        with pytest.raises(ValueError):
            RedFlagRules(patterns=("自杀",), approved_by="board", approved_at=naive)
        with pytest.raises(ValueError):
            RiskRules(patterns=("剧烈",), approved_by="board", approved_at=naive)

    @mark.parametrize("approved_by", ["", "   ", " board", "board "])
    def test_the_approver_must_be_a_normalized_name(self, approved_by):
        with pytest.raises(ValueError):
            RedFlagRules(
                patterns=("自杀",), approved_by=approved_by, approved_at=APPROVED_AT
            )

    def test_the_manager_still_refuses_to_start_without_rules(self, registry):
        """Startup failure, not a silent default — the whole point of the check."""
        with pytest.raises(ValueError):
            ManagerAgent(
                registry=registry,
                agent_runners={},
                risk_rules=approved_risk_rules(),
            )
        with pytest.raises(ValueError):
            ManagerAgent(
                registry=registry,
                agent_runners={},
                red_flag_rules=approved_red_flags(),
            )

    def test_the_docstring_does_not_claim_a_real_signature(self):
        text = (RedFlagRules.__doc__ or "") + (RiskRules.__doc__ or "")
        assert "attestation" in text.lower()


class TestNonCompletedRunNeverDelivers:
    """Review P1: a failed run's draft must not leak out as ``passed``."""

    DRAFT = "疑似脑膜炎，请立即服用抗生素"  # a draft nobody verified

    @mark.asyncio
    async def test_failed_run_with_a_draft_delivers_nothing(self, registry):
        runner = _Runner(
            result=_agent_execution(answer=self.DRAFT, status=AgentStatus.FAILED)
        )
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""  # the draft never leaves
        assert result.safety_status == "failed"  # and is never called "passed"
        assert verifier.seen == []  # a failed run is not verified
        assert any(a["type"] == "verify.not_run" for a in result.actions)
        assert self.DRAFT not in json.dumps(result.actions, ensure_ascii=False)
        assert self.DRAFT not in json.dumps(result.public_trace, ensure_ascii=False)

    @mark.asyncio
    async def test_cancelled_run_delivers_nothing(self, registry):
        runner = _Runner(
            result=_agent_execution(answer=self.DRAFT, status=AgentStatus.CANCELLED)
        )
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.CANCELLED
        assert result.answer_candidate == ""
        assert result.safety_status == "failed"

    @mark.asyncio
    async def test_a_failed_revision_delivers_nothing(self, registry):
        drafts = [
            _agent_execution(answer="第一版草稿"),
            _agent_execution(answer=self.DRAFT, status=AgentStatus.FAILED),
        ]

        class _TwoStep:
            def __init__(self):
                self.calls = []
                self.risks = []

            async def __call__(self, ctx):
                self.calls.append(ctx.normalized_input)
                self.risks.append(ctx.risk_level)
                return drafts[len(self.calls) - 1]

        runner = _TwoStep()
        verifier = _Verifier([Verdict(VerifierOutcome.REVISE)])
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert result.safety_status == "failed"
        assert len(runner.calls) == 2  # original + the one revision
        assert len(verifier.seen) == 1  # the failed revision is not re-verified
        assert any(a["type"] == "verify.not_run" for a in result.actions)
        assert self.DRAFT not in json.dumps(result.actions, ensure_ascii=False)


class TestMarkerValuesAreConstrained:
    """Review P1: key allowlisting alone still let text out under a good key."""

    PRIVATE = "患者自述：三天前开始发热咳嗽，住址朝阳区"

    @mark.parametrize(
        "values",
        [
            {"provider_id": "SYNTHETIC_PRIVATE_TEXT"},
            {"model_id": "SYNTHETIC_PRIVATE_TEXT"},
            {"model_version": "SYNTHETIC_PRIVATE_TEXT"},
            {"provider_id": "患者自述：三天前开始发热咳嗽"},
            {"model_version": ""},
        ],
    )
    @mark.asyncio
    async def test_model_identity_is_never_published_from_a_self_report(
        self, registry, values
    ):
        """A *short ASCII* string is not a *trusted* string.

        ``SYNTHETIC_PRIVATE_TEXT`` passes the token rule, so value checks alone
        cannot make this field safe — the Manager simply cannot attest provider
        or model identity from a sub-agent's own report. Until the trusted
        ModelGateway wiring (#55A) supplies provenance, nothing about the model
        is published.
        """
        fields = {
            "provider_id": "mock",
            "model_id": "mock-model",
            "model_version": "1.0.0",
        }
        fields.update(values)
        runner = _Runner(result=_agent_execution(**fields))
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        # the run itself still works and its answer is still verified...
        assert result.status is AgentStatus.COMPLETED
        assert result.safety_status == "verified"
        assert result.answer_candidate != ""
        # ...and no model metadata is published under any name
        dumped = json.dumps(result.actions, ensure_ascii=False)
        assert "SYNTHETIC_PRIVATE_TEXT" not in dumped
        assert "患者自述" not in dumped
        assert not any(a["type"] == "model.call" for a in result.actions)
        for action in result.actions:
            assert not ({"provider_id", "model_id", "model_version"} & set(action))

    @mark.asyncio
    async def test_model_metadata_keys_are_not_even_allowed(self, registry):
        """The allowlist itself no longer sanctions model-identity markers."""
        m = build_manager(registry=registry, agent_runners={})
        with pytest.raises(ValueError):
            m._mark([], "model.call", provider_id="mock")
        with pytest.raises(ValueError):
            m._mark([], "route.handoff", provider_id="SYNTHETIC_PRIVATE_TEXT")
        assert "provider_id" not in SAFE_MARKER_KEYS
        assert "model.call" not in SAFE_MARKER_TYPES

    @mark.asyncio
    async def test_internal_markers_refuse_a_private_value(self, registry):
        m = build_manager(registry=registry, agent_runners={})
        with pytest.raises(ValueError):
            m._mark([], "manager.risk", level=self.PRIVATE)
        with pytest.raises(ValueError):
            m._mark([], "manager.route", intent=self.PRIVATE)
        with pytest.raises(ValueError):
            m._mark([], "agent.done", status="patient said fever")
        with pytest.raises(ValueError):
            m._mark([], "route.handoff", agent_id=self.PRIVATE, handoffs=1)
        with pytest.raises(ValueError):
            m._mark([], "route.handoff", handoffs=10_000_000)
        with pytest.raises(ValueError):
            m._mark([], "route.handoff", handoffs=True)

    def test_value_guard_accepts_only_tokens_and_bounded_ints(self):
        assert is_safe_marker_value("agent_id", "qa")
        assert is_safe_marker_value("model_id", "mock-model-1.0")
        assert is_safe_marker_value("handoffs", 2)
        assert not is_safe_marker_value("agent_id", self.PRIVATE)
        assert not is_safe_marker_value("agent_id", "line\nbreak")
        assert not is_safe_marker_value("agent_id", "a" * 65)
        assert not is_safe_marker_value("handoffs", -1)
        assert not is_safe_marker_value("handoffs", True)
        assert is_safe_marker_value("level", "critical")  # all four levels
        assert is_safe_marker_value("declared", "medium")
        assert not is_safe_marker_value("level", "unknown")


class TestUntrustedExecutionIsValidated:
    """Review P1: the runner's return value is untrusted input.

    Forged identity, impossible statuses and bad types used to reach the
    verifier, the markers or the delivered result. Everything now maps to the
    SAME code with a FIXED message, so no value and no exception text escapes.
    """

    @staticmethod
    def _manager(registry, raw):
        async def runner(ctx):
            return raw

        return build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )

    @mark.asyncio
    async def test_a_forged_agent_identity_is_refused(self, registry):
        """Routing said ``qa``; the runner claims to be someone else."""
        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry,
            agent_runners={
                "qa": _Runner(result=_agent_execution(agent_id="different-agent"))
            },
            verifier=verifier,
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE
        assert str(exc.value) == "routed agent returned an invalid execution result"
        assert "different-agent" not in str(exc.value)
        assert verifier.seen == []  # a forged result never reaches the verifier
        assert exc.value.__cause__ is None and exc.value.__context__ is None

    @mark.parametrize("status", [AgentStatus.PENDING, AgentStatus.RUNNING])
    @mark.asyncio
    async def test_in_flight_statuses_are_not_results(self, registry, status):
        m = self._manager(registry, _agent_execution(status=status))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE

    @mark.parametrize("tool_calls", [-1, True, 1.5, float("nan"), "3", None])
    @mark.asyncio
    async def test_tool_call_counts_must_be_strict_non_negative_ints(
        self, registry, tool_calls
    ):
        m = self._manager(registry, _agent_execution(tool_calls=tool_calls))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE

    @mark.parametrize(
        "values",
        [
            {"answer": None},
            {"answer": 42},
            {"status": "completed"},  # a string status, not the enum
            {"evidence": ("not-an-evidence",)},
            {"safety_status": None},
        ],
    )
    @mark.asyncio
    async def test_wrong_field_types_map_to_one_public_code(self, registry, values):
        m = self._manager(registry, _agent_execution(**values))
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE
        assert exc.value.__cause__ is None and exc.value.__context__ is None

    @mark.asyncio
    async def test_a_non_tuple_evidence_container_is_refused(self, registry):
        """Built directly: the test helper would coerce a list to a tuple."""
        raw = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="答案",
            evidence=[_evidence()],  # type: ignore[arg-type] - list, not tuple
        )
        m = self._manager(registry, raw)
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE

    @mark.asyncio
    async def test_a_non_execution_return_value_is_refused(self, registry):
        class _Duck:
            agent_id = "qa"
            status = AgentStatus.COMPLETED
            answer_candidate = "伪装的答案"
            evidence = ()
            tool_calls = 0
            provider_id = ""
            model_id = ""
            model_version = ""
            safety_status = "unknown"

        m = self._manager(registry, _Duck())
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.MODEL_OUTPUT_UNPARSEABLE
        assert "伪装的答案" not in str(exc.value)


class TestTrustedSnapshotIsolation:
    """Review P1: holding a reference must not be able to change the result."""

    @mark.asyncio
    async def test_runner_mutation_after_return_does_not_change_the_result(
        self, registry
    ):
        kept = _evidence(content="原始证据")
        original = _agent_execution(answer="原始答案", evidence=(kept,))

        async def runner(ctx):
            return original

        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.answer_candidate == "原始答案"
        assert result.evidence[0].content == "原始证据"

        # the runner keeps a reference to the very Evidence object it returned
        # (AgentExecution is frozen; Evidence.content is the mutable channel)
        # and writes through it AFTER delivery
        kept.content = "篡改后的证据"
        assert original.evidence[0].content == "篡改后的证据"
        assert result.evidence[0].content == "原始证据"

    @mark.asyncio
    async def test_a_verifier_cannot_mutate_what_is_delivered(self, registry):
        class _MutatingVerifier:
            def __init__(self):
                self.seen = []
                self.risks = []
                self.mutated = False

            async def __call__(self, ctx, execution):
                self.seen.append(execution.answer_candidate)
                self.risks.append(ctx.risk_level)
                # the verifier only holds an isolated copy: this write must not
                # reach the delivered result
                if execution.evidence:
                    execution.evidence[0].content = "验证器篡改的证据"
                    self.mutated = True
                return Verdict(VerifierOutcome.PASS)

        runner = _Runner(
            result=_agent_execution(
                answer="原始答案", evidence=(_evidence(content="原始证据"),)
            )
        )
        verifier = _MutatingVerifier()
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert verifier.mutated is True  # the write really happened...
        assert result.answer_candidate == "原始答案"
        assert result.evidence[0].content == "原始证据"  # ...and changed nothing
        dumped = json.dumps(result.model_dump(), ensure_ascii=False)
        assert "篡改" not in dumped

    @mark.asyncio
    async def test_the_delivered_evidence_is_a_copy_of_the_validated_snapshot(
        self, registry
    ):
        runner = _Runner(
            result=_agent_execution(evidence=(_evidence(content="原始证据"),))
        )
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        result = await m.execute(_context(), "发烧怎么办")
        result.evidence[0].content = "调用方篡改"  # the caller's copy is its own
        assert runner._result.evidence[0].content == "原始证据"


class TestToolBudgetIsRunCumulative:
    """Review P1: the ≤4 tool cap is summed over the whole run."""

    @staticmethod
    def _revision_manager(registry, first_calls, revision_calls, verifier=None):
        drafts = [
            _agent_execution(tool_calls=first_calls),
            _agent_execution(tool_calls=revision_calls),
        ]

        class _TwoStep:
            def __init__(self):
                self.calls = []
                self.risks = []

            async def __call__(self, ctx):
                self.calls.append(ctx.normalized_input)
                self.risks.append(ctx.risk_level)
                return drafts[len(self.calls) - 1]

        runner = _TwoStep()
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier
            or _Verifier(
                [Verdict(VerifierOutcome.REVISE), Verdict(VerifierOutcome.PASS)]
            ),
        )
        return m, runner

    @mark.asyncio
    async def test_two_plus_two_is_within_the_budget(self, registry):
        m, runner = self._revision_manager(registry, 2, 2)
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.safety_status == "verified"
        assert len(runner.calls) == 2

    @mark.parametrize(("first", "revision"), [(3, 2), (3, 3), (4, 1), (0, 5), (5, 0)])
    @mark.asyncio
    async def test_over_budget_totals_are_refused(self, registry, first, revision):
        """3 + 2 = 5 > 4: comparing per engagement would have allowed this."""
        m, runner = self._revision_manager(registry, first, revision)
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT
        # an over-budget FIRST execution is refused before any revision
        assert len(runner.calls) == (1 if first > 4 else 2)

    @mark.asyncio
    async def test_a_single_over_budget_execution_is_still_refused(self, registry):
        m = build_manager(
            registry=registry,
            agent_runners={"qa": _Runner(result=_agent_execution(tool_calls=5))},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT

    @mark.asyncio
    async def test_the_counter_spans_the_verifier_handoff_too(self, registry):
        """4 tools in the first execution + 1 in the revision = 5, refused."""
        m, runner = self._revision_manager(
            registry, 4, 1, verifier=_Verifier([Verdict(VerifierOutcome.REVISE)])
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(), "发烧怎么办")
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT
        assert len(runner.calls) == 2


class TestLimitsRejectNonIntegers:
    """Review P1: NaN defeats every ceiling comparison (NaN > ceiling is False)."""

    @mark.parametrize(
        "value",
        [float("nan"), float("inf"), 4.0, "4", True, False, None, [4]],
    )
    def test_non_integer_budgets_are_refused(self, value):
        with pytest.raises(TypeError):
            ManagerLimits(max_tool_calls=value)
        with pytest.raises(TypeError):
            ManagerLimits(max_handoffs=value)
        with pytest.raises(TypeError):
            ManagerLimits(max_revisions=value)
        with pytest.raises(TypeError):
            ManagerLimits(max_input_chars=value)

    def test_integer_budgets_still_work(self):
        limits = ManagerLimits(
            max_input_chars=100, max_handoffs=2, max_tool_calls=4, max_revisions=1
        )
        assert limits.max_tool_calls == 4

    def test_out_of_range_integers_are_still_refused(self):
        with pytest.raises(ValueError):
            ManagerLimits(max_tool_calls=5)
        with pytest.raises(ValueError):
            ManagerLimits(max_revisions=2)


class TestDeadlineCreatesNoOrphanCoroutine:
    """Review P2: the budget is checked BEFORE the coroutine is created."""

    @mark.asyncio
    async def test_an_expired_budget_never_creates_the_coroutine(self, registry):
        start = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        deadline = start + timedelta(seconds=5)

        class _LateClock:
            """guard + first budget read see ``start``; then time is up."""

            def __init__(self):
                self.reads = 0

            def __call__(self):
                self.reads += 1
                if self.reads <= 2:
                    return start
                return deadline + timedelta(seconds=1)

        class _RecordingVerifier:
            def __init__(self):
                self.invocations = 0  # counts CALLS to the callable itself

            def __call__(self, ctx, execution):
                self.invocations += 1

                async def _verdict():
                    return Verdict(VerifierOutcome.PASS)

                return _verdict()

        verifier = _RecordingVerifier()
        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            clock=_LateClock(),
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(deadline=deadline), "发烧怎么办")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT
        assert len(runner.calls) == 1  # the routed agent did run...
        assert verifier.invocations == 0  # ...but no verifier coroutine was made

    @mark.asyncio
    async def test_an_expired_budget_before_the_first_engagement_skips_the_runner(
        self, registry
    ):
        start = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        deadline = start + timedelta(seconds=5)

        class _LateFromTheStart:
            def __init__(self):
                self.reads = 0

            def __call__(self):
                self.reads += 1
                return start if self.reads == 1 else deadline + timedelta(seconds=1)

        runner = _Runner(result=_agent_execution())
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=_Verifier([Verdict(VerifierOutcome.PASS)]),
            clock=_LateFromTheStart(),
        )
        with pytest.raises(ManagerAgentError) as exc:
            await m.execute(_context(deadline=deadline), "发烧怎么办")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT
        assert len(runner.calls) == 0  # no coroutine, no call, no leak


class TestRevisionCeilingIsLocked:
    """Review: a hard revision budget must not be an amplifiable config."""

    def test_revisions_cannot_be_raised(self):
        with pytest.raises(ValueError) as exc:
            ManagerLimits(max_revisions=3)
        assert "max_revisions" in str(exc.value)

    @mark.parametrize(
        "kwargs",
        [
            {"max_handoffs": 3},
            {"max_tool_calls": 5},
            {"max_revisions": 2},
            {"max_handoffs": -1},
            {"max_input_chars": 0},
        ],
    )
    def test_every_ceiling_is_enforced(self, kwargs):
        with pytest.raises(ValueError):
            ManagerLimits(**kwargs)

    def test_defaults_are_the_v23_numbers(self):
        limits = ManagerLimits()
        assert (limits.max_handoffs, limits.max_tool_calls, limits.max_revisions) == (
            2,
            4,
            1,
        )

    @mark.asyncio
    async def test_revisions_can_be_tightened_to_zero(self, registry):
        """Lowering the budget is allowed (stricter); it just forbids retries."""
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.REVISE)])
        m = build_manager(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=verifier,
            limits=ManagerLimits(max_revisions=0),
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert result.status is AgentStatus.FAILED
        assert result.answer_candidate == ""
        assert len(runner.calls) == 1  # no retry at all
        assert len(verifier.seen) == 1
        assert any(a["type"] == "verify.reject_final" for a in result.actions)

    @mark.asyncio
    async def test_the_one_revision_budget_is_not_exceeded(self, registry):
        """Even with a verifier that always says REVISE: 1 revision, 2 runs."""
        runner = _Runner(result=_agent_execution())
        verifier = _Verifier([Verdict(VerifierOutcome.REVISE)] * 5)
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        result = await m.execute(_context(), "发烧怎么办")
        assert len(runner.calls) == 2
        assert len(verifier.seen) == 2
        assert result.answer_candidate == ""


class TestCancellation:
    """Review counterexample: a cancelled run must never turn into an answer."""

    @mark.asyncio
    async def test_cancel_during_agent_run_propagates_and_delivers_nothing(
        self, registry
    ):
        state = {"inner_cancelled": False}

        class _Hanging:
            async def __call__(self, ctx):
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    state["inner_cancelled"] = True
                    raise
                return _agent_execution()  # pragma: no cover

        verifier = _Verifier([Verdict(VerifierOutcome.PASS)])
        m = build_manager(
            registry=registry, agent_runners={"qa": _Hanging()}, verifier=verifier
        )
        task = asyncio.ensure_future(m.execute(_context(), "发烧怎么办"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()  # no AgentResult was produced at all
        assert state["inner_cancelled"] is True  # the routed run really stopped
        assert len(verifier.seen) == 0  # and nothing reached the verifier

    @mark.asyncio
    async def test_cancel_during_verification_does_not_deliver_the_candidate(
        self, registry
    ):
        """A candidate answer exists at this point — it still must not leave."""

        class _HangingVerifier:
            def __init__(self):
                self.seen = []

            async def __call__(self, ctx, execution):
                self.seen.append(execution.answer_candidate)
                await asyncio.sleep(30)
                return Verdict(VerifierOutcome.PASS)  # pragma: no cover

        runner = _Runner(result=_agent_execution())
        verifier = _HangingVerifier()
        m = build_manager(
            registry=registry, agent_runners={"qa": runner}, verifier=verifier
        )
        delivered: list = []

        async def _run():
            delivered.append(await m.execute(_context(), "发烧怎么办"))

        task = asyncio.ensure_future(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert len(runner.calls) == 1  # the agent did run and draft an answer
        assert len(verifier.seen) == 1  # the verifier was engaged
        # nothing was ever handed back to the caller: the unverified candidate
        # answer has no route out of the Manager
        assert delivered == []


class TestMarkersAreAllowlisted:
    """``AgentResult.actions`` is public: it may only carry typed decisions."""

    def test_unknown_marker_type_is_refused(self, registry):
        m = build_manager(registry=registry, agent_runners={})
        with pytest.raises(ValueError):
            m._mark([], "manager.thought", level="low")

    def test_unknown_key_cannot_carry_raw_text(self, registry):
        m = build_manager(registry=registry, agent_runners={})
        with pytest.raises(ValueError):
            m._mark([], "manager.route", text="患者自述：发热三天")

    def test_sanctioned_marker_is_recorded(self, registry):
        m = build_manager(registry=registry, agent_runners={})
        actions: list[dict] = []
        m._mark(actions, "manager.risk", level="high")
        assert actions == [{"type": "manager.risk", "level": "high"}]
