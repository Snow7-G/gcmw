"""SafetyEvidenceVerifier (issue #54) — deterministic evidence verification.

FOUR-STATE DECISION (V2.3 §6.3), returned to the Manager as
:class:`app.agents.manager.Verdict` wrapping
:class:`app.agents.manager.VerifierOutcome` — never collapsed into a boolean:

* ``PASS``     — the only outcome that may deliver an answer;
* ``REVISE``   — a fixable defect (missing/unknown/incomplete citations,
                 ungrounded or empty answer, contradiction scan hit): the Manager
                 runs its ONE controlled revision;
* ``BLOCK``    — stop and deliver nothing (privacy leak, invalid candidate
                 output, non-completed run);
* ``ESCALATE`` — hand the case to a human (red flags, clinician-only scope):
                 no candidate answer is released.

Checks are deterministic (no model call, no chain-of-thought) and cover:
grounding, citation COVERAGE *and* CONSISTENCY (every marker resolves to a
delivered evidence item, every cited id exists, every delivered item is cited,
``source_id``/``knowledge_version``/``content_hash`` present), privacy patterns,
red-flag rules, and a clinician-only action vocabulary.

TRUST: the candidate output is untrusted input. It is accepted only as the
#52 contract type — a fresh, strictly re-validated ``AgentExecution`` whose
evidence items are REBUILT from a whitelisted payload with
``Evidence.model_validate(..., strict=True)``. A duck-typed object, a list where
a tuple is declared, a mutated/invalid Evidence or a bad status is refused with
``BLOCK`` — ``model_copy`` is never trusted, because it does not re-validate.
The Manager already hands over an isolated snapshot; this is an independent
check so the verifier is safe when called on its own.

Every emitted reason/instruction comes from the FIXED vocabulary in
:data:`REVISION_INSTRUCTIONS`: no answer text, prompt, tool argument, model
output or chain-of-thought can reach the decision, the Manager's actions, an
audit record or an error object.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agents.manager import (
    AgentExecution,
    VerifierOutcome,
)
from app.agents.manager import (
    Verdict as ManagerVerdict,
)
from app.contracts.agent import AgentContext, AgentStatus, Evidence, RiskLevel

_CITATION_RE = re.compile(r"来源\s+([A-Za-z0-9][A-Za-z0-9._-]*)|\[#?(\d+)\]")
_PRIVACY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)
# clinician-only action vocabulary — the AI must not issue these
_SCOPE_MARKERS: tuple[str, ...] = (
    "处方剂量",
    "开具处方",
    "调整剂量",
    "诊断为",
    "确诊为",
)

#: FIXED audit vocabulary. Every reason a decision can carry is one of these
#: markers: nothing derived from the answer, the prompt, tool arguments or model
#: output is ever emitted.
REVISION_INSTRUCTIONS: frozenset[str] = frozenset(
    {
        "pass",
        "invalid_execution",
        "not_completed",
        "privacy_leak",
        "red_flag_escalate",
        "medical_scope_out_of_ai_boundary",
        "contradiction",
        "ungrounded",
        "empty_answer",
        "citation_missing",
        "citation_unknown",
        "citation_incomplete",
        "citation_mismatch",
    }
)

#: the ONLY Evidence fields carried across this boundary; anything else attached
#: to an object is ignored, never forwarded
_EVIDENCE_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_type",
    "title",
    "content",
    "source_uri",
    "content_hash",
    "knowledge_version",
)

#: statuses a routed agent may report; PENDING/RUNNING are not results
_REPORTABLE_STATUSES = frozenset(
    {AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED}
)


@dataclass(frozen=True)
class VerifierDecision:
    """Structured verification outcome (fixed flags and markers only)."""

    outcome: VerifierOutcome
    revision_instructions: str = ""
    grounded: bool = False
    citation_coverage: bool = False
    medical_scope_ok: bool = True
    privacy_ok: bool = True
    unsupported_claims: bool = False
    contradictions: bool = False
    red_flags: bool = False
    fast_path: bool = False


def _validated_evidence(item: Any) -> Evidence | None:
    """Rebuild one Evidence item by RE-VALIDATING a whitelisted payload."""
    payload: dict[str, Any] = {
        name: getattr(item, name, None) for name in _EVIDENCE_FIELDS
    }
    try:
        # pydantic's ValidationError is a ValueError subclass
        return Evidence.model_validate(payload, strict=True)
    except (TypeError, ValueError):
        return None


def _validated_execution(execution: Any) -> AgentExecution | None:
    """Strictly re-validate the candidate output; ``None`` means refuse.

    Duck-typed objects are rejected outright: only the #52 contract type is
    accepted, its status must be a reportable :class:`AgentStatus` (a plain
    string is not), ``answer_candidate`` must be a string, ``tool_calls`` a
    non-bool non-negative integer, ``evidence`` a tuple of Evidence items that
    each survive strict re-validation. The returned snapshot is a NEW object with
    fresh evidence, so later mutation of the original cannot change what was
    verified.
    """
    if not isinstance(execution, AgentExecution):
        return None
    if not isinstance(execution.status, AgentStatus):
        return None
    if execution.status not in _REPORTABLE_STATUSES:
        return None
    if not isinstance(execution.answer_candidate, str):
        return None
    if not isinstance(execution.evidence, tuple):
        return None
    tool_calls = execution.tool_calls
    if isinstance(tool_calls, bool) or not isinstance(tool_calls, int):
        return None
    if tool_calls < 0:
        return None
    if not all(
        isinstance(value, str)
        for value in (
            execution.agent_id,
            execution.safety_status,
            execution.provider_id,
            execution.model_id,
            execution.model_version,
        )
    ):
        return None
    rebuilt: list[Evidence] = []
    for item in execution.evidence:
        if not isinstance(item, Evidence):
            return None
        copy = _validated_evidence(item)
        if copy is None:
            return None
        rebuilt.append(copy)
    return AgentExecution(
        agent_id=execution.agent_id,
        status=execution.status,
        answer_candidate=execution.answer_candidate,
        evidence=tuple(rebuilt),
        tool_calls=tool_calls,
        provider_id=execution.provider_id,
        model_id=execution.model_id,
        model_version=execution.model_version,
        safety_status=execution.safety_status,
    )


def _citations_in(answer: str) -> list[tuple[str | None, int | None]]:
    """Parse citation markers: ``来源 <source_id>`` or ``[N]`` (1-based index
    into the evidence list). Returns (source_id, index) pairs."""
    found: list[tuple[str | None, int | None]] = []
    for match in _CITATION_RE.finditer(answer or ""):
        if match.group(1):
            found.append((match.group(1), None))
        else:
            found.append((None, int(match.group(2))))
    return found


def _resolve_citations(answer: str, evidence: list[Evidence]) -> tuple[list[str], bool]:
    """Resolve citation markers onto evidence source ids.

    Returns (resolved_ids, all_known): an out-of-range ``[N]`` marker is an
    unknown citation, never a silent pass.
    """
    resolved: list[str] = []
    all_known = True
    for source_id, index in _citations_in(answer):
        if source_id is not None:
            resolved.append(source_id)
            continue
        if index is not None and 1 <= index <= len(evidence):
            resolved.append(evidence[index - 1].source_id)
        else:
            all_known = False
            resolved.append(f"#{index}")
    return resolved, all_known


def _contains_red_flag(text: str, rules: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(rule.lower() in lowered for rule in rules)


def _contains_scope_violation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in _SCOPE_MARKERS)


def _leaks_privacy(text: str) -> bool:
    for pattern in _PRIVACY_PATTERNS:
        if pattern.search(text or ""):
            return True
    return False


class SafetyEvidenceVerifier:
    """Deterministic PASS/REVISE/BLOCK/ESCALATE over an agent execution."""

    def __init__(
        self,
        *,
        red_flags: tuple[str, ...] = (),
        contradiction_scan: Callable[[str, list[Evidence]], bool] | None = None,
    ) -> None:
        self._red_flags = tuple(red_flags)
        self._contradiction_scan = contradiction_scan

    # -- core decision --------------------------------------------------------

    def decide(self, ctx: AgentContext, execution: Any) -> VerifierDecision:
        """Run the deterministic check set over one candidate execution.

        Pure and side-effect free: no model call, no audit write, no mutation of
        the input — so a late or duplicated invocation can never append a second
        terminal record, and cancellation/timeout semantics stay the Manager's.
        """
        snapshot = _validated_execution(execution)
        if snapshot is None:
            return _refused("invalid_execution")

        answer = snapshot.answer_candidate.strip()
        evidence = list(snapshot.evidence)
        fast = _fast_path(ctx, snapshot)

        evidence_text = " ".join(item.content for item in evidence)
        red = _contains_red_flag(answer, self._red_flags) or _contains_red_flag(
            evidence_text, self._red_flags
        )
        privacy_ok = not (_leaks_privacy(answer) or _leaks_privacy(evidence_text))
        scope_ok = not (
            _contains_scope_violation(answer)
            or _contains_scope_violation(evidence_text)
        )

        grounded = bool(evidence)
        resolved, citations_known = _resolve_citations(answer, evidence)
        known_ids = {item.source_id for item in evidence}
        cited_ids = set(resolved)
        provenance_complete = all(
            item.source_id and item.knowledge_version and item.content_hash
            for item in evidence
        )

        citation_coverage = True
        citation_reason = ""
        unsupported = bool(answer) and not grounded
        if answer and grounded:
            if not citations_known or not cited_ids <= known_ids:
                citation_coverage = False
                citation_reason = "citation_unknown"
                unsupported = True
            elif not cited_ids:
                citation_coverage = False
                citation_reason = "citation_missing"
            elif not provenance_complete:
                citation_coverage = False
                citation_reason = "citation_incomplete"
            elif cited_ids != known_ids:
                # every delivered item must be cited: a citation set that does
                # not match the evidence set is an inconsistent answer
                citation_coverage = False
                citation_reason = "citation_mismatch"

        contradictions = False
        if self._contradiction_scan is not None:
            contradictions = bool(self._contradiction_scan(answer, evidence))

        outcome, instructions = _classify(
            privacy_ok=privacy_ok,
            red=red,
            contradictions=contradictions,
            grounded=grounded,
            answer_present=bool(answer),
            scope_ok=scope_ok,
            citation_coverage=citation_coverage,
            citation_reason=citation_reason,
        )
        return VerifierDecision(
            outcome=outcome,
            revision_instructions=instructions,
            grounded=grounded,
            citation_coverage=citation_coverage,
            medical_scope_ok=scope_ok,
            privacy_ok=privacy_ok,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            red_flags=red,
            fast_path=fast,
        )

    # -- #52 runner slot --------------------------------------------------------

    async def __call__(self, context: AgentContext, execution: Any) -> ManagerVerdict:
        """Verifier slot for the Manager: always a FOUR-STATE verdict.

        The Manager decides what to do with it (only ``PASS`` delivers, only
        ``REVISE`` retries once, ``BLOCK`` stops, ``ESCALATE`` goes to a human);
        this method never returns the candidate answer or its evidence, so a
        non-PASS outcome cannot leak unverified content through the verdict.
        """
        snapshot = _validated_execution(execution)
        if snapshot is None:
            return ManagerVerdict(VerifierOutcome.BLOCK, "invalid_execution")
        if snapshot.status is not AgentStatus.COMPLETED:
            # a run that did not complete has nothing to verify and nothing to
            # deliver; the Manager never reaches here, this is the standalone guard
            return ManagerVerdict(VerifierOutcome.BLOCK, "not_completed")
        decision = self.decide(context, snapshot)
        reason = decision.revision_instructions or "pass"
        return ManagerVerdict(decision.outcome, reason)


def _refused(instructions: str) -> VerifierDecision:
    """A BLOCK for input that could not be validated.

    No check ran, so the flags report "no violation observed" for the two
    negative checks (privacy/scope) and False for everything that would claim a
    positive property (grounded/coverage) — an unvalidated candidate may never
    look half-verified.
    """
    return VerifierDecision(
        outcome=VerifierOutcome.BLOCK,
        revision_instructions=instructions,
        grounded=False,
        citation_coverage=False,
        medical_scope_ok=True,
        privacy_ok=True,
        unsupported_claims=False,
        contradictions=False,
        red_flags=False,
        fast_path=False,
    )


def _classify(
    *,
    privacy_ok: bool,
    red: bool,
    contradictions: bool,
    grounded: bool,
    answer_present: bool,
    scope_ok: bool,
    citation_coverage: bool,
    citation_reason: str,
) -> tuple[VerifierOutcome, str]:
    """Precedence: BLOCK (never deliver) before ESCALATE (human) before REVISE."""
    if not privacy_ok:
        return VerifierOutcome.BLOCK, "privacy_leak"
    if red:
        return VerifierOutcome.ESCALATE, "red_flag_escalate"
    if contradictions:
        return VerifierOutcome.REVISE, "contradiction"
    if not grounded:
        return VerifierOutcome.REVISE, "ungrounded"
    if not answer_present:
        return VerifierOutcome.REVISE, "empty_answer"
    if not scope_ok:
        return VerifierOutcome.ESCALATE, "medical_scope_out_of_ai_boundary"
    if not citation_coverage:
        return VerifierOutcome.REVISE, citation_reason or "citation_missing"
    return VerifierOutcome.PASS, ""


def _fast_path(ctx: AgentContext, execution: AgentExecution) -> bool:
    """Single FAQ evidence at LOW risk → deterministic fast verification.

    The Manager writes the EFFECTIVE risk (never downgraded) into the context, so
    a high-risk question can never take this shortcut.
    """
    if ctx.risk_level is not RiskLevel.LOW:
        return False
    items = list(execution.evidence)
    return len(items) == 1 and items[0].source_type == "faq"


def faq_fast_decision(ctx: AgentContext, execution: Any) -> VerifierDecision | None:
    """Fast-path shortcut used by the E2E assembly to pre-classify FAQ runs."""
    snapshot = _validated_execution(execution)
    if snapshot is None:
        return None
    if _fast_path(ctx, snapshot):
        return SafetyEvidenceVerifier().decide(ctx, snapshot)
    return None
