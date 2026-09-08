"""Tests for SafetyEvidenceVerifier (issue #54).

Coverage:
- fixed decision structure with all eight flags present on every decision;
- PASS for grounded, fully cited answers (fast FAQ path and full path);
- REVISE for missing/unknown citations, ungrounded answers, contradiction
  scan hits;
- ESCALATE for clinician-only scope content in answer or evidence;
- BLOCK for red-flag or privacy-leaking content (answer or evidence);
- non-completed executions are rejected without further checks;
- no chain-of-thought: decisions carry structured flags only.
"""

import pytest
from pytest import mark

from app.agents.verifier import (
    SafetyEvidenceVerifier,
    Verdict,
    VerifierDecision,
    faq_fast_decision,
)
from app.contracts.agent import AgentContext, AgentStatus, RiskLevel
from app.contracts.common import Channel


class FakeEvidence:
    def __init__(self, source_id, content, source_type="faq") -> None:
        self.source_id = source_id
        self.content = content
        self.source_type = source_type
        self.knowledge_version = f"{source_id}-v1"
        self.content_hash = f"hash-{source_id}"
        self.title = source_id
        self.source_uri = f"kbase://{source_id}"


class FakeExecution:
    def __init__(
        self,
        answer="",
        evidence=(),
        status=AgentStatus.COMPLETED,
    ) -> None:
        self.answer_candidate = answer
        self.evidence = list(evidence)
        self.status = status


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


def _faq_answer(source_id="faq-fever", answer=None):
    evidence = [FakeEvidence(source_id=source_id, content="体温超过38.5建议门诊就诊。")]
    if answer is None:
        answer = f"体温超过38.5建议门诊就诊（来源 {source_id}）。"
    return FakeExecution(answer=answer, evidence=evidence)


class TestDecisionShape:
    def test_every_decision_carries_the_full_flag_set(self):
        verifier = SafetyEvidenceVerifier(red_flags=("自杀",))
        cases = [
            _faq_answer(),  # PASS
            FakeExecution(answer="无来源答案", evidence=[]),
            FakeExecution(answer="处方剂量为每日10mg。", evidence=[]),
            FakeExecution(answer="电话 13800138000", evidence=[]),
        ]
        for case in cases:
            decision = verifier.decide(_context(), case)
            assert isinstance(decision, VerifierDecision)
            assert decision.verdict in {"PASS", "REVISE", "BLOCK", "ESCALATE"}
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
                assert getattr(decision, flag) in (True, False)
            assert decision.revision_instructions != "" or decision.verdict == "PASS"


class TestPassAndFastPath:
    def test_faq_fast_path_passes_with_citation(self):
        decision = SafetyEvidenceVerifier().decide(_context(), _faq_answer())
        assert decision.verdict == "PASS"
        assert decision.fast_path is True
        assert decision.grounded and decision.citation_coverage

    def test_fast_path_shortcut_helper(self):
        decision = faq_fast_decision(_context(), _faq_answer())
        assert decision is not None
        assert decision.verdict == "PASS"

    def test_full_path_passes_multisource_with_citations(self):
        execution = FakeExecution(
            answer="建议就诊（来源 a、来源 b）。",
            evidence=[
                FakeEvidence("a", "内容甲"),
                FakeEvidence("b", "内容乙", source_type="document"),
            ],
        )
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.verdict == "PASS"
        assert decision.fast_path is False

    def test_faq_marker_style_citation_parsed(self):
        evidence = FakeEvidence("faq-fever", "内容")
        execution = FakeExecution(answer="建议就诊（资料[1]）。", evidence=[evidence])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.verdict == "PASS"

    def test_high_risk_runs_full_path_not_fast(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(risk=RiskLevel.HIGH), _faq_answer()
        )
        assert decision.fast_path is False
        assert decision.verdict == "PASS"  # still passes with citations


class TestRevise:
    def test_missing_citation_fails_coverage(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _faq_answer(answer="体温高建议就诊。")
        )
        assert decision.verdict == "REVISE"
        assert decision.revision_instructions == "citation_missing"
        assert decision.citation_coverage is False

    def test_unknown_citation_is_unsupported(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(), _faq_answer(answer="建议就诊（来源 ghost）。")
        )
        assert decision.unsupported_claims is True
        assert decision.verdict == "REVISE"

    def test_ungrounded_answer_revises_on_full_path(self):
        execution = FakeExecution(
            answer="发烧应服用阿司匹林。",
            evidence=[FakeEvidence("doc-1", "内容", source_type="document")],
        )
        # citation mentions nothing -> fast is False (document) -> REVISE
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.verdict == "REVISE"
        assert decision.revision_instructions == "citation_missing"

    def test_contradiction_scan_hit_revises(self):
        def scan(answer, evidence):
            return "矛盾" in answer

        execution = FakeExecution(
            answer="内容矛盾（来源 a）。",
            evidence=[FakeEvidence("a", "正文")],
        )
        decision = SafetyEvidenceVerifier(contradiction_scan=scan).decide(
            _context(), execution
        )
        assert decision.contradictions is True
        assert decision.verdict == "REVISE"


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
        execution = _faq_answer(answer=payload)
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.privacy_ok is False
        assert decision.verdict == "BLOCK"

    def test_red_flag_in_evidence_blocks(self):
        evidence = FakeEvidence("f1", "内容含 自杀 风险提示")
        execution = FakeExecution(answer="建议就诊（来源 f1）。", evidence=[evidence])
        decision = SafetyEvidenceVerifier(red_flags=("自杀",)).decide(
            _context(), execution
        )
        assert decision.red_flags is True
        assert decision.verdict == "BLOCK"

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
        assert decision.verdict == "ESCALATE"
        assert decision.revision_instructions == "medical_scope_out_of_ai_boundary"

    def test_scope_violation_in_evidence_escalates(self):
        evidence = FakeEvidence("f1", "开具处方需线下完成。")
        execution = FakeExecution(answer="建议就诊（来源 f1）。", evidence=[evidence])
        decision = SafetyEvidenceVerifier().decide(_context(), execution)
        assert decision.medical_scope_ok is False
        assert decision.verdict == "ESCALATE"


class TestRunnerSlot:
    @mark.asyncio
    async def test_pending_execution_rejected_without_checks(self):
        execution = FakeExecution(answer="未完成内容", status=AgentStatus.RUNNING)
        verdict = await SafetyEvidenceVerifier()(_context(), execution)
        assert isinstance(verdict, Verdict)
        assert verdict.approved is False
        assert verdict.reason == "not_completed"

    @mark.asyncio
    async def test_runner_verdict_maps_decision(self):
        verifier = SafetyEvidenceVerifier()
        approved = await verifier(_context(), _faq_answer())
        assert approved.approved is True
        assert approved.reason == "PASS"
        rejected = await verifier(_context(), _faq_answer(answer="无引用的答案。"))
        assert rejected.approved is False
        assert rejected.reason == "REVISE"

    def test_no_chain_of_thought_in_decision(self):
        decision = SafetyEvidenceVerifier().decide(
            _context(),
            _faq_answer(answer="体温超过38.5建议门诊就诊（来源 faq-fever）。"),
        )
        dumped = repr(decision)
        assert "体温超过38.5" not in dumped  # flags only, never content
