"""Tests for SafetyEvidenceVerifier (issue #54).

Coverage:
- FOUR-STATE outcome (PASS / REVISE / BLOCK / ESCALATE) returned to the Manager
  as Verdict(VerifierOutcome) — never a boolean;
- only PASS may deliver: BLOCK/ESCALATE/REVISE carry no answer and no evidence;
- the candidate output is untrusted: duck objects, wrong container types, bad
  statuses, mutated/invalid Evidence and bad field types are refused (BLOCK)
  instead of being trusted, because model_copy does not re-validate;
- citation consistency: unknown/out-of-range ids, missing citations, uncited
  evidence and incomplete source_id/knowledge_version/content_hash all fail;
- safety boundary: red flags escalate to a human, privacy leaks block, non-PASS
  never returns candidate content;
- pure and side-effect free: repeated decisions are identical, the input is not
  mutated, and a cancellation/exception from an injected scanner propagates;
- no chain-of-thought: decisions carry fixed flags and a fixed marker only.
"""

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

import pytest
from pytest import mark

from app.agents.manager import (
    AgentExecution,
    ManagerAgent,
    RedFlagRules,
    RiskRules,
    VerifierOutcome,
)
from app.agents.manager import (
    Verdict as ManagerVerdict,
)
from app.agents.registry import AgentManifest, AgentRegistry
from app.agents.verifier import (
    REVISION_INSTRUCTIONS,
    SafetyEvidenceVerifier,
    VerifierDecision,
    faq_fast_decision,
)
from app.contracts.agent import AgentContext, AgentStatus, Evidence, RiskLevel
from app.contracts.common import Channel

APPROVED_AT = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _context(risk=RiskLevel.LOW, **overrides):
    fields = {
        "tenant_id": "t1",
        "device_id": "d1",
        "session_id": "s1",
        "run_id": "r1",
        "channel": Channel.TOUCH,
        "risk_level": risk,
    }
    fields.update(overrides)
    return AgentContext(**fields)


def _evidence(source_id="faq-fever", content="体温超过38.5建议门诊就诊。", **overrides):
    fields = {
        "source_id": source_id,
        "source_type": "faq",
        "title": source_id,
        "content": content,
        "source_uri": f"kbase://{source_id}",
        "content_hash": f"hash-{source_id}",
        "knowledge_version": f"{source_id}-v1",
    }
    fields.update(overrides)
    return Evidence(**fields)


def _execution(answer=None, evidence=(), status=AgentStatus.COMPLETED, tool_calls=0):
    return AgentExecution(
        agent_id="qa",
        status=status,
        answer_candidate=(
            "体温超过38.5建议门诊就诊（来源 faq-fever）。" if answer is None else answer
        ),
        evidence=tuple(evidence),
        tool_calls=tool_calls,
    )


def _faq_answer(source_id="faq-fever", answer=None):
    evidence = [_evidence(source_id=source_id)]
    if answer is None:
        answer = f"体温超过38.5建议门诊就诊（来源 {source_id}）。"
    return _execution(answer=answer, evidence=evidence)


class TestFourStateOutcome:
    def test_every_decision_carries_the_full_flag_set_and_a_fixed_marker(self):
        verifier = SafetyEvidenceVerifier(red_flags=("自杀",))
        cases = [
            _faq_answer(),  # PASS
            _execution(answer="无来源答案", evidence=[]),
            _execution(answer="处方剂量为每日10mg。", evidence=[_evidence()]),
            _execution(answer="电话 13800138000", evidence=[_evidence()]),
            "not-an-execution",  # refused outright
        ]
        for case in cases:
            decision = verifier.decide(_context(), case)
            assert isinstance(decision, VerifierDecision)
            assert isinstance(decision.outcome, VerifierOutcome)
            for flag in (
                "grounded",
                "citation_coverage",
                "medical_scope_ok",
                "privacy_ok",
                "unsupported_claims",
                "contradictions",
                "red_flags",
                "fast_path",
            ):
                assert isinstance(getattr(decision, flag), bool)
            assert (
                decision.revision_instructions in REVISION_INSTRUCTIONS
                or decision.outcome is VerifierOutcome.PASS
            )

    def test_only_pass_carries_no_marker(self):
        assert (
            SafetyEvidenceVerifier()
            .decide(_context(), _faq_answer())
            .revision_instructions
            == ""
        )
        for case in (
            _execution(answer="无引用。", evidence=[_evidence()]),
            _execution(answer="电话 13800138000", evidence=[_evidence()]),
        ):
            decision = SafetyEvidenceVerifier().decide(_context(), case)
            assert decision.revision_instructions != ""


class TestPassAndFastPath:
    def test_faq_fast_path_passes_with_citation(self):
        decision = SafetyEvidenceVerifier().decide(_context(), _faq_answer())
        assert decision.outcome is VerifierOutcome.PASS
        assert decision.fast_path is True
        assert decision.grounded and decision.citation_coverage

    def test_fast_path_shortcut_helper(self):
        decision = faq_fast_decision(_context(), _faq_answer())
        assert decision is not None
        assert decision.outcome is VerifierOutcome.PASS

    def test_full_path_passes_multisource_with_citations(self):
        execution = _execution(
            answer="建议就诊（来源 a、来源 b）。",
            evidence=[
                _evidence("a", "内容甲"),
                _evidence("b", "内容乙", source_type="document"),
            ],
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.PASS
        assert decision.fast_path is False

    def test_faq_marker_style_citation_parsed(self):
        execution = _execution(answer="建议就诊（资料[1]）。", evidence=[_evidence()])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.PASS

    def test_high_risk_runs_full_path_not_fast(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(risk=RiskLevel.HIGH), _faq_answer()
        )
        assert decision.fast_path is False
        assert decision.outcome is VerifierOutcome.PASS  # still passes with citations

    @mark.parametrize("risk", [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL])
    def test_no_fast_path_above_low_risk(self, risk):
        assert (
            SafetyEvidenceVerifier()
            .decide(_context(risk=risk), _faq_answer())
            .fast_path
            is False
        )


class TestCitationConsistency:
    """Review scope: unknown ids, missing citations, count mismatch, incomplete
    provenance — none of them may PASS."""

    def test_missing_citation_fails_coverage(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _faq_answer(answer="体温高建议就诊。")
        )
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_missing"
        assert decision.citation_coverage is False

    def test_unknown_citation_is_unsupported(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _faq_answer(answer="建议就诊（来源 ghost）。")
        )
        assert decision.unsupported_claims is True
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_unknown"

    def test_out_of_range_index_is_unknown(self):
        execution = _execution(answer="建议就诊（资料[9]）。", evidence=[_evidence()])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_unknown"

    def test_uncited_evidence_is_a_mismatch(self):
        """Two items delivered, only one cited: the citation set is inconsistent."""
        execution = _execution(
            answer="建议就诊（来源 a）。",
            evidence=[_evidence("a", "内容甲"), _evidence("b", "内容乙")],
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_mismatch"

    @mark.parametrize("field", ["knowledge_version", "content_hash"])
    def test_incomplete_provenance_never_passes(self, field):
        """An empty version/hash satisfies the contract but is not verifiable."""
        item = _evidence().model_copy(update={field: ""})
        execution = _execution(answer="建议就诊（来源 faq-fever）。", evidence=[item])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_incomplete"

    def test_a_structurally_invalid_source_id_is_blocked(self):
        """An empty source_id is refused by strict re-validation, not "revised"."""
        item = _evidence().model_copy(update={"source_id": ""})
        execution = _execution(answer="建议就诊（来源 faq-fever）。", evidence=[item])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.BLOCK
        assert decision.revision_instructions == "invalid_execution"

    def test_ungrounded_answer_revises_on_full_path(self):
        execution = _execution(
            answer="发烧应服用阿司匹林。",
            evidence=[_evidence("doc-1", source_type="document")],
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "citation_missing"

    def test_contradiction_scan_hit_revises(self):
        def scan(answer, evidence):
            return "矛盾" in answer

        execution = _execution(
            answer="内容矛盾（来源 a）。", evidence=[_evidence("a", "正文")]
        )
        decision = SafetyEvidenceVerifier(contradiction_scan=scan).decide(
            _context(), execution
        )
        assert decision.contradictions is True
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "contradiction"


class TestBlockAndEscalate:
    @pytest.mark.parametrize(
        "payload",
        [
            "体温 13800138000 电话（来源 faq-fever）",
            "身份证 11010119900307777X",
            "sk-abcdef1234567890abcdef1234",
        ],
    )
    def test_privacy_leak_blocks(self, payload):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _faq_answer(answer=payload)
        )
        assert decision.privacy_ok is False
        assert decision.outcome is VerifierOutcome.BLOCK
        assert decision.revision_instructions == "privacy_leak"

    @mark.parametrize("source_id", ["faq-fever", "faq-fever-1", "faq.fever_2"])
    def test_red_flag_escalates_to_a_human(self, source_id):
        """Review scope: red flags go to a HUMAN, they are not merely blocked."""
        execution = _execution(
            answer=f"建议尽快处理（来源 {source_id}）。",
            evidence=[_evidence(source_id, "内容含 自杀 风险提示")],
        )
        decision = SafetyEvidenceVerifier(red_flags=("自杀",)).decide(
            _context(), execution
        )
        assert decision.red_flags is True
        assert decision.outcome is VerifierOutcome.ESCALATE
        assert decision.revision_instructions == "red_flag_escalate"

    def test_privacy_outranks_red_flag(self):
        """A privacy leak is blocked even when a red flag is also present."""
        execution = _execution(
            answer="自杀 电话 13800138000（来源 faq-fever）",
            evidence=[_evidence()],
        )
        decision = SafetyEvidenceVerifier(red_flags=("自杀",)).decide(
            _context(), execution
        )
        assert decision.red_flags is True and decision.privacy_ok is False
        assert decision.outcome is VerifierOutcome.BLOCK

    @pytest.mark.parametrize(
        "text",
        [
            "建议开具处方剂量每日10mg（来源 faq-fever）",
            "诊断为肺炎（来源 faq-fever）",
        ],
    )
    def test_medical_scope_violation_escalates(self, text):
        decision = SafetyEvidenceVerifier().decide(_context(), _faq_answer(answer=text))
        assert decision.medical_scope_ok is False
        assert decision.outcome is VerifierOutcome.ESCALATE
        assert decision.revision_instructions == "medical_scope_out_of_ai_boundary"

    def test_scope_violation_in_evidence_escalates(self):
        execution = _execution(
            answer="建议就诊（来源 f1）。",
            evidence=[_evidence("f1", "开具处方需线下完成。")],
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.medical_scope_ok is False
        assert decision.outcome is VerifierOutcome.ESCALATE


class TestUntrustedCandidateOutput:
    """Review scope: never trust a duck object, a mutable Evidence or model_copy."""

    def test_a_duck_typed_execution_is_blocked(self):
        class _Duck:
            agent_id = "qa"
            status = AgentStatus.COMPLETED
            answer_candidate = "体温超过38.5建议门诊就诊（来源 faq-fever）。"
            evidence = (_evidence(),)
            tool_calls = 0
            provider_id = ""
            model_id = ""
            model_version = ""
            safety_status = "grounded"

        decision = SafetyEvidenceVerifier().decide(_context(), _Duck())
        assert decision.outcome is VerifierOutcome.BLOCK
        assert decision.revision_instructions == "invalid_execution"

    @mark.parametrize("status", [AgentStatus.PENDING, AgentStatus.RUNNING, "completed"])
    def test_unreportable_or_non_enum_status_is_blocked(self, status):
        decision = SafetyEvidenceVerifier().decide(
            _context(),
            _execution(
                answer="答案（来源 faq-fever）。", evidence=[_evidence()]
            ).__class__(
                agent_id="qa",
                status=status,
                answer_candidate="答案（来源 faq-fever）。",
                evidence=(_evidence(),),
            ),
        )
        assert decision.outcome is VerifierOutcome.BLOCK
        assert decision.revision_instructions == "invalid_execution"

    def test_list_evidence_is_blocked(self):
        raw = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="答案（来源 faq-fever）。",
            evidence=[_evidence()],  # type: ignore[arg-type] - list, not tuple
        )
        decision = SafetyEvidenceVerifier().decide(_context(), raw)
        assert decision.outcome is VerifierOutcome.BLOCK

    @mark.parametrize(
        "patch",
        [
            {"source_id": ""},
            {"content": {"patient": "secret"}},
            {"knowledge_version": 42},
            {"source_type": None},
        ],
    )
    def test_invalid_evidence_is_blocked_not_trusted(self, patch):
        """A legal-then-mutated item (or model_copy(update=...)) fails validation."""
        raw = _evidence().model_copy(update=patch)
        execution = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="答案（来源 faq-fever）。",
            evidence=(raw,),
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.outcome is VerifierOutcome.BLOCK
        assert decision.revision_instructions == "invalid_execution"

    @mark.parametrize("answer", [None, 42])
    def test_non_string_answer_is_blocked(self, answer):
        raw = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate=answer,  # type: ignore[arg-type]
            evidence=(_evidence(),),
        )
        assert SafetyEvidenceVerifier().decide(_context(), raw).outcome is (
            VerifierOutcome.BLOCK
        )

    @mark.parametrize("tool_calls", [-1, True, 1.5, "2", None])
    def test_bad_tool_calls_are_blocked(self, tool_calls):
        raw = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="答案（来源 faq-fever）。",
            evidence=(_evidence(),),
            tool_calls=tool_calls,  # type: ignore[arg-type]
        )
        assert SafetyEvidenceVerifier().decide(_context(), raw).outcome is (
            VerifierOutcome.BLOCK
        )

    def test_the_snapshot_is_isolated_from_later_mutation(self):
        """The verified snapshot is its own object: mutations cannot matter."""
        item = _evidence()
        execution = AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="答案（来源 faq-fever）。",
            evidence=(item,),
        )
        verifier = SafetyEvidenceVerifier()
        first = verifier.decide(_context(), execution)
        assert first.outcome is VerifierOutcome.PASS
        item.content = "篡改后的内容"  # mutate after the decision
        second = verifier.decide(_context(), execution)
        assert second.outcome is first.outcome
        assert second.citation_coverage is True


class TestEmptyAnswerGuard:
    def test_empty_completed_answer_never_passes(self):
        # e.g. the #53 no_evidence shape: COMPLETED with evidence but no answer
        execution = _execution(answer="", evidence=[_evidence()])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.grounded is True
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "empty_answer"

    def test_empty_answer_without_evidence_revises_ungrounded(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _execution(answer="", evidence=[])
        )
        assert decision.outcome is VerifierOutcome.REVISE
        assert decision.revision_instructions == "ungrounded"


class TestManagerSlot:
    @mark.asyncio
    async def test_the_slot_returns_a_four_state_verdict(self):
        verifier = SafetyEvidenceVerifier()
        verdict = await verifier(_context(), _faq_answer())
        assert isinstance(verdict, ManagerVerdict)
        assert verdict.outcome is VerifierOutcome.PASS
        assert verdict.approved is True

        revised = await verifier(_context(), _faq_answer(answer="无引用的答案。"))
        assert revised.outcome is VerifierOutcome.REVISE
        assert revised.approved is False

        blocked = await verifier(_context(), "not-an-execution")
        assert blocked.outcome is VerifierOutcome.BLOCK

        escalating = SafetyEvidenceVerifier(red_flags=("自杀",))
        escalated = await escalating(
            _context(),
            _execution(
                answer="建议就诊（来源 f1）。",
                evidence=[_evidence("f1", "内容含 自杀 风险提示")],
            ),
        )
        assert escalated.outcome is VerifierOutcome.ESCALATE

    @mark.parametrize("status", [AgentStatus.FAILED, AgentStatus.CANCELLED])
    @mark.asyncio
    async def test_a_non_completed_execution_never_passes(self, status):
        execution = _execution(
            answer="未完成内容", evidence=[_evidence()], status=status
        )
        verdict = await SafetyEvidenceVerifier()(_context(), execution)
        assert verdict.outcome is VerifierOutcome.BLOCK
        assert verdict.reason == "not_completed"

    @mark.asyncio
    async def test_an_in_flight_execution_is_not_a_candidate_at_all(self):
        execution = _execution(
            answer="未完成内容", evidence=[_evidence()], status=AgentStatus.RUNNING
        )
        verdict = await SafetyEvidenceVerifier()(_context(), execution)
        assert verdict.outcome is VerifierOutcome.BLOCK
        assert verdict.reason == "invalid_execution"

    @mark.asyncio
    async def test_the_manager_drives_the_verified_answer_end_to_end(self):
        """The real four-state slot plugged into the real Manager."""
        execution = _faq_answer()

        async def runner(ctx):
            return execution

        registry = AgentRegistry()
        registry.register(
            AgentManifest(
                agent_id="qa",
                version="1.0.0",
                supported_intents=["knowledge"],
                risk_level="medium",
            )
        )
        manager = ManagerAgent(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=SafetyEvidenceVerifier(red_flags=("自杀",)),
            red_flag_rules=RedFlagRules(
                patterns=("自杀",), approved_by="board", approved_at=APPROVED_AT
            ),
            risk_rules=RiskRules(
                patterns=("剧烈",), approved_by="board", approved_at=APPROVED_AT
            ),
        )
        result = await manager.execute(_context(), "发热怎么办")
        assert result.status is AgentStatus.COMPLETED
        assert result.safety_status == "verified"
        assert result.evidence[0].source_id == "faq-fever"

        # an unverifiable answer is refused by the Manager (REVISE → no delivery)
        execution_unverified = _execution(
            answer="无引用的答案。", evidence=[_evidence()]
        )

        async def runner_unverified(ctx):
            return execution_unverified

        manager2 = ManagerAgent(
            registry=registry,
            agent_runners={"qa": runner_unverified},
            verifier=SafetyEvidenceVerifier(),
            red_flag_rules=RedFlagRules(
                patterns=("自杀",), approved_by="board", approved_at=APPROVED_AT
            ),
            risk_rules=RiskRules(
                patterns=("剧烈",), approved_by="board", approved_at=APPROVED_AT
            ),
        )
        refused = await manager2.execute(_context(), "发热怎么办")
        assert refused.answer_candidate == ""
        assert refused.evidence == []
        assert refused.status is AgentStatus.FAILED

    @mark.asyncio
    async def test_red_flag_escalation_delivers_no_candidate_answer(self):
        execution = _execution(
            answer="建议就诊（来源 f1）。",
            evidence=[_evidence("f1", "内容含 自杀 风险提示")],
        )

        async def runner(ctx):
            return execution

        registry = AgentRegistry()
        registry.register(
            AgentManifest(
                agent_id="qa", version="1.0.0", supported_intents=["knowledge"]
            )
        )
        manager = ManagerAgent(
            registry=registry,
            agent_runners={"qa": runner},
            verifier=SafetyEvidenceVerifier(red_flags=("自杀",)),
            red_flag_rules=RedFlagRules(
                patterns=("需要人工介入",), approved_by="b", approved_at=APPROVED_AT
            ),
            risk_rules=RiskRules(
                patterns=("剧烈",), approved_by="b", approved_at=APPROVED_AT
            ),
        )
        result = await manager.execute(_context(), "发热怎么办")
        assert result.safety_status == "escalated"  # to a human
        assert result.answer_candidate == ""
        assert result.evidence == []
        assert any(a["type"] == "verify.escalated" for a in result.actions)


class TestPurityAndNoChainOfThought:
    def test_decisions_are_deterministic_and_do_not_mutate_the_input(self):
        execution = _faq_answer()
        before = asdict(execution)
        verifier = SafetyEvidenceVerifier()
        first = verifier.decide(_context(), execution)
        second = verifier.decide(_context(), execution)
        assert first == second
        assert asdict(execution) == before  # nothing written back

    def test_no_chain_of_thought_or_content_in_the_decision(self):
        secret = "患者自述：三天前发热，住址朝阳区某小区"
        decision = SafetyEvidenceVerifier().decide(
            _context(),
            _execution(
                answer="体温超过38.5建议门诊就诊（来源 faq-fever）。",
                evidence=[_evidence(content=secret)],
            ),
        )
        dumped = repr(decision) + decision.revision_instructions
        assert secret not in dumped
        assert "体温超过38.5" not in dumped  # flags and markers only

    def test_a_non_pass_verdict_carries_no_candidate_content(self):
        secret = "患者自述：住址朝阳区"
        verdict = asyncio.run(
            SafetyEvidenceVerifier()(
                _context(),
                _execution(
                    answer="无引用的答案。",
                    evidence=[_evidence(content=secret)],
                ),
            )
        )
        dumped = repr(verdict) + str(verdict) + repr(vars(verdict))
        assert secret not in dumped
        assert "无引用的答案" not in dumped

    def test_every_emitted_marker_is_in_the_fixed_vocabulary(self):
        cases = [
            _execution(answer="普通答案", evidence=[_evidence()]),
            _execution(answer="无引用。", evidence=[_evidence()]),
            _execution(answer="电话 13800138000", evidence=[_evidence()]),
            _execution(answer="建议就诊（来源 ghost）。", evidence=[_evidence()]),
            _execution(answer="", evidence=[]),
            "not-an-execution",
        ]
        verifier = SafetyEvidenceVerifier(red_flags=("自杀",))
        for case in cases:
            decision = verifier.decide(_context(), case)
            assert decision.revision_instructions in REVISION_INSTRUCTIONS

    @mark.asyncio
    async def test_cancellation_propagates_unchanged(self):
        """A cancellation inside an injected scanner is never swallowed."""

        def scan(answer, evidence):
            raise asyncio.CancelledError

        verifier = SafetyEvidenceVerifier(contradiction_scan=scan)
        with pytest.raises(asyncio.CancelledError):
            await verifier(_context(), _faq_answer())

    @mark.asyncio
    async def test_cancellation_after_a_decision_changes_nothing(self):
        """The verifier keeps no state: a cancelled caller cannot leave a PASS
        behind, and a second call re-derives the same decision."""
        verifier = SafetyEvidenceVerifier()
        execution = _faq_answer()
        task = asyncio.ensure_future(verifier(_context(), execution))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        # nothing was recorded and the next call is independent
        assert (await verifier(_context(), execution)).outcome is VerifierOutcome.PASS
