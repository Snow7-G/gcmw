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
        booker_runner = _Runner()
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
        model_call = next(a for a in result.actions if a["type"] == "model.call")
        assert model_call["provider_id"] == "mock"
        assert model_call["model_id"] == "mock-model"
        assert model_call["model_version"] == "1.0.0"
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
