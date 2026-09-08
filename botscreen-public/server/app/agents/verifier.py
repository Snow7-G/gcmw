"""SafetyEvidenceVerifier (issue #54) — deterministic evidence verification.

V2.3 §6.3 surface:
- fixed decision structure PASS / REVISE / BLOCK / ESCALATE with the flag set
  (grounded / citation_coverage / medical_scope_ok / privacy_ok /
  unsupported_claims / contradictions / red_flags) plus safe
  revision_instructions;
- tiered verification: single FAQ evidence at LOW risk runs the deterministic
  fast path; anything else (multi-source, other source types, higher risk)
  runs the full check set;
- the Verifier never issues a medical-correctness proof and its outputs carry
  no chain-of-thought — decisions are structured flags only.

The agent consumes any execution object structurally shaped like #52/#53's
runner results (``status / answer_candidate / evidence`` with evidence items
carrying ``source_id / content / source_type / knowledge_version``) and
implements the Manager (#52) verifier slot (``approved/reason`` verdict).
No module from #52/#53 is imported — this branch stays standalone until the
#55 E2E assembly wires the real ones together.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.contracts.agent import AgentContext, AgentStatus, RiskLevel

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


@dataclass(frozen=True)
class VerifierDecision:
    """Structured verification outcome (no free text beyond instructions)."""

    verdict: str  # PASS | REVISE | BLOCK | ESCALATE
    grounded: bool = False
    citation_coverage: bool = False
    medical_scope_ok: bool = True
    privacy_ok: bool = True
    unsupported_claims: bool = False
    contradictions: bool = False
    red_flags: bool = False
    revision_instructions: str = ""
    fast_path: bool = False


@dataclass(frozen=True)
class Verdict:
    """Runner-shaped verdict for the #52 Manager slot (approved/reason)."""

    approved: bool
    reason: str
    decision: VerifierDecision | None = field(default=None)


def _evidence_list(execution: Any) -> list[Any]:
    return list(getattr(execution, "evidence", None) or ())


def _citations_in(answer: str) -> list[tuple[str | None, int | None]]:
    """Parse citation markers: ``来源 <source_id>`` or ``[N]`` (1-based
    index into the evidence list). Returns (source_id, index) pairs."""
    found: list[tuple[str | None, int | None]] = []
    for match in _CITATION_RE.finditer(answer or ""):
        if match.group(1):
            found.append((match.group(1), None))
        else:
            found.append((None, int(match.group(2))))
    return found


def _resolve_citations(answer: str, evidence: list[Any]) -> tuple[list[str], bool]:
    """Resolve citation markers onto evidence source ids.

    Returns (resolved_ids, all_known): an out-of-range ``[N]`` marker is an
    unknown citation (unsupported claim), never a silent pass.
    """
    resolved: list[str] = []
    all_known = True
    for source_id, index in _citations_in(answer):
        if source_id is not None:
            resolved.append(source_id)
            continue
        if index is not None and 1 <= index <= len(evidence):
            resolved.append(getattr(evidence[index - 1], "source_id", ""))
        else:
            all_known = False
            resolved.append(f"#{index}")
    return resolved, all_known


def _source_ids(evidence: list[Any]) -> set[str]:
    return {getattr(e, "source_id", "") for e in evidence}


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
        contradiction_scan: Callable[[str, list[Any]], bool] | None = None,
    ) -> None:
        self._red_flags = tuple(red_flags)
        self._contradiction_scan = contradiction_scan

    # -- core decision --------------------------------------------------------

    def decide(self, ctx: AgentContext, execution: Any) -> VerifierDecision:
        """Run the tiered check set over one agent execution (no model call,
        no chain-of-thought, deterministic flags only)."""
        answer = (getattr(execution, "answer_candidate", "") or "").strip()
        evidence = _evidence_list(execution)
        fast = _fast_path(ctx, execution)

        evidence_text = " ".join(getattr(e, "content", "") for e in evidence)
        red = _contains_red_flag(answer, self._red_flags) or _contains_red_flag(
            evidence_text, self._red_flags
        )
        privacy_ok = not (_leaks_privacy(answer) or _leaks_privacy(evidence_text))
        scope_ok = not (
            _contains_scope_violation(answer)
            or _contains_scope_violation(evidence_text)
        )

        grounded = bool(evidence)
        resolved_citations, citations_known = _resolve_citations(answer, evidence)
        known_ids = _source_ids(evidence)
        citation_coverage = True
        if answer and grounded:
            # an answer that cites nothing (or cites unknown/out-of-range ids)
            # is fixable by revision
            citation_coverage = (
                citations_known
                and bool(resolved_citations)
                and all(c in known_ids for c in resolved_citations)
            )
        unsupported = bool(answer) and not grounded
        if answer and grounded:
            unknown = [c for c in resolved_citations if c not in known_ids]
            unsupported = bool(unknown)

        contradictions = False
        if self._contradiction_scan is not None:
            contradictions = self._contradiction_scan(answer, evidence)

        if red or not privacy_ok:
            decision = "BLOCK"
            instructions = "red_flag_or_privacy" if red else "privacy_leak"
        elif contradictions:
            decision = "REVISE"
            instructions = "contradiction"
        else:
            decision, instructions = self._classify(
                grounded=grounded,
                scope_ok=scope_ok,
                citation_coverage=citation_coverage,
                unsupported=unsupported,
                full=not fast,
            )

        return VerifierDecision(
            verdict=decision,
            grounded=grounded,
            citation_coverage=citation_coverage,
            medical_scope_ok=scope_ok,
            privacy_ok=privacy_ok,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            red_flags=red,
            revision_instructions=instructions,
            fast_path=fast,
        )

    @staticmethod
    def _classify(
        *,
        grounded: bool,
        scope_ok: bool,
        citation_coverage: bool,
        unsupported: bool,
        full: bool = False,
    ) -> tuple[str, str]:
        if not grounded:
            return "REVISE", "ungrounded"
        if full and unsupported:
            return "REVISE", "unsupported_claims"
        if not scope_ok:
            return "ESCALATE", "medical_scope_out_of_ai_boundary"
        if not citation_coverage:
            return "REVISE", "citation_missing"
        return "PASS", ""

    # -- #52 runner slot --------------------------------------------------------

    async def __call__(self, context: AgentContext, execution: Any) -> Verdict:
        """Runner-compatible verification for the Manager's verifier slot.

        PASS → approved; anything else is a rejection (the Manager drives the
        at-most-one revision loop and fails the run on a final rejection).
        """
        if getattr(execution, "status", None) is not AgentStatus.COMPLETED:
            return Verdict(False, "not_completed")
        decision = self.decide(context, execution)
        approved = decision.verdict == "PASS"
        return Verdict(approved=approved, reason=decision.verdict, decision=decision)


def _fast_path(ctx: AgentContext, execution: Any) -> bool:
    """Single FAQ evidence at low risk → deterministic fast verification."""
    if ctx.risk_level is not RiskLevel.LOW:
        return False
    items = _evidence_list(execution)
    return len(items) == 1 and getattr(items[0], "source_type", "") == "faq"


def faq_fast_decision(ctx: AgentContext, execution: Any) -> VerifierDecision | None:
    """Fast-path shortcut used by #55 E2E to pre-classify FAQ runs."""
    if _fast_path(ctx, execution):
        return SafetyEvidenceVerifier().decide(ctx, execution)
    return None
