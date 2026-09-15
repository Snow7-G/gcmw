"""SafetyEvidenceVerifier (issue #54) — deterministic evidence verification.

ONE DECISION ENTRY: :meth:`SafetyEvidenceVerifier.decide` is the only place a
judgement is produced. :meth:`SafetyEvidenceVerifier.__call__` (the #52 Manager
slot) is a thin wrapper over it, so there is no second route that could skip a
check. ``decide`` requires the candidate to be ``COMPLETED``: a
``FAILED``/``CANCELLED`` run is refused with ``BLOCK`` and the fixed reason
``not_completed`` no matter how well-cited its draft looks.

FOUR-STATE DECISION (V2.3 §6.3), returned to the Manager as
:class:`app.agents.manager.Verdict` wrapping
:class:`app.agents.manager.VerifierOutcome` — never collapsed into a boolean:

* ``PASS``     — the only outcome that may deliver an answer;
* ``REVISE``   — a fixable defect (missing/unknown/incomplete citations, a
                 duplicated source, an answer with no substantive claim, an
                 unsupported answer): the Manager runs its ONE controlled
                 revision;
* ``BLOCK``    — stop and deliver nothing (privacy leak, invalid candidate
                 output, non-completed run);
* ``ESCALATE`` — hand the case to a human (red flags, clinician-only scope):
                 no candidate answer is released.

Checks are deterministic (no model call, no chain-of-thought) and cover:
grounding, citation COVERAGE *and* CONSISTENCY (every marker resolves to a
delivered evidence item, every cited id exists, every delivered item is cited,
``source_id``/``knowledge_version``/``content_hash`` present, ``source_id``
unique), an EXACT-MATCH support gate (below), privacy patterns over the WHOLE
delivered payload, approved red-flag rules, and a clinician-only action
vocabulary.

EXACT-MATCH SUPPORT GATE (honest scope): with the citation markers removed the
answer is split into claims. A claim counts only if it actually asserts
something (a markers-only body does not). Every claim must carry a citation and
its normalized body must be EQUAL to a complete unit of the evidence it cites —
the WHOLE normalized ``content`` of a short FAQ entry, or one of that content's
COMPLETE sentences. Never a substring: substring matching let 「使用激素」 be
lifted out of 「不建议患者使用激素」 / 「仅在医生指导下使用激素」 and pass as
verified, silently deleting a contraindication or a precondition. Consequences
of equality, stated plainly: a paraphrase is refused, and so is a claim that is
merely part of a longer evidence sentence. This gate performs NO entailment, NO
polarity inference and NO medical semantic verification, and must never be
described as such.

The citation protocol is ONE form, matching the #53 prompt: ``资料[N]``
(1-based index into the delivered evidence) and ``来源 <source_id>``. A bare
``[N]``, ``参考[N]``, ``依据[N]`` or ``出处[N]`` is not a marker, so
「参考[2024]版指南」 is not read as citing item 2024.

TRUST: the candidate output is untrusted input. It is accepted only as the
#52 contract type — a fresh, strictly re-validated ``AgentExecution`` whose
evidence items are REBUILT from a whitelisted payload with
``Evidence.model_validate(..., strict=True)``. A duck-typed object, a list where
a tuple is declared, a mutated/invalid Evidence or a bad status is refused with
``BLOCK`` — ``model_copy`` is never trusted, because it does not re-validate.
The Manager already hands over an isolated snapshot; this is an independent
check so the verifier is safe when called on its own.

FAIL-CLOSED RULES: the verifier requires an APPROVED
:class:`app.agents.manager.RedFlagRules` instance (the same object the Manager
uses) — an empty or unapproved rule set is rejected at ASSEMBLY time, so
"nothing configured" can never mean "no red flag found".

Every emitted reason/instruction comes from the FIXED vocabulary in
:data:`REVISION_INSTRUCTIONS`: no answer text, prompt, tool argument, model
output or chain-of-thought can reach the decision, the Manager's actions, an
audit record or an error object.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.agents.manager import (
    AgentExecution,
    RedFlagRules,
    VerifierOutcome,
)
from app.agents.manager import (
    Verdict as ManagerVerdict,
)
from app.contracts.agent import AgentContext, AgentStatus, Evidence, RiskLevel

#: The ONE citation protocol, matching the #53 prompt (``资料[1]``) and the
#: ``来源 <source_id>`` form the medical QA agent renders:
#:
#:   ``资料[N]``  — 1-based index into the delivered evidence list
#:   ``来源 <source_id>``
#:
#: Nothing else is a citation. A bare ``[N]``, ``参考[N]``, ``依据[N]``,
#: ``出处[N]`` are NOT markers: an answer containing 「参考[2024]版指南」 must not
#: be read as citing item 2024 (that produced a spurious ``citation_unknown``
#: REVISE on an otherwise supported answer).
_CITATION_RE = re.compile(
    r"来源\s+([A-Za-z0-9][A-Za-z0-9._-]*)"
    r"|资料\[#?(\d+)\]"
)
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

#: sentence terminators used by the extractive support gate. ASCII ``.`` is
#: deliberately NOT a splitter (it is the decimal separator in ``38.5``); the
#: gate targets the Chinese medical FAQ surface, not general English prose.
_CLAIM_SPLIT_RE = re.compile(r"[。！？；!?;\n]+")
#: punctuation removed when NORMALIZING a claim and its evidence content, so a
#: claim still matches across 「，」/「（）」 style differences. ``.`` is kept for
#: the same reason as above.
_TRIVIAL_PUNCT = frozenset("，、：（）()【】[]「」『』“”‘’\"'·—…《》~～")

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
        "ungrounded",
        "empty_answer",
        "citation_missing",
        "citation_unknown",
        "citation_incomplete",
        "citation_mismatch",
        "duplicate_source",
        "evidence_unsupported",
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
    red_flags: bool = False
    fast_path: bool = False
    evidence_supported: bool = False
    duplicate_source: bool = False


def _delivered_text(answer: str, evidence: list[Evidence]) -> str:
    """Every string field that will actually be handed to the caller.

    The scan surface is the DELIVERY surface: ``answer_candidate`` plus all
    string fields of every Evidence item the Manager will put into
    ``AgentResult.evidence``. Scanning ``content`` alone left ``title`` and
    ``source_uri`` — both delivered verbatim — unchecked, so a phone number in a
    title used to reach the caller under a PASS.
    """
    parts = [answer]
    for item in evidence:
        for name in _EVIDENCE_FIELDS:
            value = getattr(item, name, "")
            parts.append(value if isinstance(value, str) else "")
    return " ".join(parts)


def _has_duplicate_source_ids(evidence: list[Evidence]) -> bool:
    """True when one execution delivers the same ``source_id`` twice.

    Two versions/hashes behind one id collapse in a set comparison, so a
    citation that "matches" could silently refer to either of them. The first
    simple safety rule is flat refusal (the Manager may retry once).
    """
    return len({item.source_id for item in evidence}) != len(evidence)


def _normalize(text: str) -> str:
    """NFKC + drop whitespace and trivial punctuation (claim/evidence both)."""
    folded = unicodedata.normalize("NFKC", text or "")
    return "".join(
        char for char in folded if not char.isspace() and char not in _TRIVIAL_PUNCT
    ).lower()


def _strip_citation_markers(text: str) -> str:
    """The claim body: the sentence with its citation markers removed."""
    return _CITATION_RE.sub(" ", text or "")


def _split_claims(answer: str) -> list[str]:
    """Split the raw answer into sentences, keeping each sentence's markers."""
    return [part for part in _CLAIM_SPLIT_RE.split(answer or "") if part.strip()]


def _is_substantive(body: str) -> bool:
    """True when a claim body actually ASSERTS something.

    A body that is empty, or that is nothing but the residue of markers and
    digits (``[1]`` normalizes to ``"1"``), asserts nothing: it can neither be
    verified nor count as an answer. Without this check a markers-only answer
    such as ``[1]`` walked through the gate and came out PASS.
    """
    return any(char.isalpha() for char in body)


def _claim_bodies(answer: str) -> list[tuple[str, list[tuple[str | None, int | None]]]]:
    """Split the answer into SUBSTANTIVE claims: (normalized body, its markers).

    Sentences that assert nothing — empty, or nothing but the residue of markers
    and digits (``[1]`` normalizes to ``"1"``) — are dropped here, in ONE place,
    so that "does a deliverable answer exist at all?" and "is every claim
    supported?" can never disagree. A markers-only answer used to walk through
    the gate and come out PASS.
    """
    claims: list[tuple[str, list[tuple[str | None, int | None]]]] = []
    for sentence in _split_claims(answer):
        body = _normalize(_strip_citation_markers(sentence))
        if _is_substantive(body):
            claims.append((body, _citations_in(sentence)))
    return claims


def _supported_units(content: str) -> frozenset[str]:
    """The normalized strings a claim is allowed to EQUAL.

    EQUALITY, never arbitrary substring containment. A substring rule let
    「使用激素」 be extracted from 「不建议患者使用激素」, 「没有必要使用激素」,
    「严禁儿童自行服用阿司匹林」 or 「仅在医生指导下使用激素」 and be reported as
    supported — i.e. a contraindication or a precondition was silently deleted
    and the answer still shipped as verified. The allowed units are the WHOLE
    normalized content (a short FAQ entry) and each of its COMPLETE sentences;
    anything narrower is not support.
    """
    units: set[str] = set()
    whole = _normalize(_strip_citation_markers(content))
    if whole:
        units.add(whole)
    for sentence in _split_claims(content):
        unit = _normalize(_strip_citation_markers(sentence))
        if unit:
            units.add(unit)
    return frozenset(units)


def _check_extractive_support(
    claims: list[tuple[str, list[tuple[str | None, int | None]]]],
    evidence: list[Evidence],
) -> tuple[bool, str]:
    """The deterministic exact-match gate. Returns ``(supported, reason)``.

    ``empty_answer``: there is no substantive claim at all, so there is no
    deliverable answer to PASS. ``citation_missing``: a substantive claim carries
    no marker. ``citation_unknown``: a marker that does not resolve.
    ``evidence_unsupported``: a cited claim whose normalized body is not EQUAL to
    one of its evidence units.
    """
    if not claims:
        return False, "empty_answer"
    by_id = {item.source_id: item for item in evidence}
    units = {id(item): _supported_units(item.content) for item in evidence}
    for body, markers in claims:
        if not markers:
            return False, "citation_missing"
        targets: list[Evidence] = []
        for source_id, index in markers:
            if source_id is not None:
                item = by_id.get(source_id)
                if item is None:
                    return False, "citation_unknown"
                targets.append(item)
            elif index is not None and 1 <= index <= len(evidence):
                targets.append(evidence[index - 1])
            else:
                return False, "citation_unknown"
        if not any(body in units[id(target)] for target in targets):
            return False, "evidence_unsupported"
    return True, ""


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
    """Parse citation markers: ``来源 <source_id>`` or ``资料[N]`` (1-based index
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

    Returns (resolved_ids, all_known): an out-of-range ``资料[N]`` marker is an
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


def _contains_red_flag(text: str, rules: RedFlagRules) -> bool:
    return rules.matches(text)


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
        red_flag_rules: RedFlagRules,
    ) -> None:
        """``red_flag_rules`` is REQUIRED and must be the Manager's own approved
        :class:`RedFlagRules` instance.

        There is deliberately no default: an empty tuple used to mean "no rule
        fired", so a verifier built without configuration would happily PASS a
        model answer containing a red flag. Passing a non-``RedFlagRules``
        object, an empty rule set or an unapproved one fails HERE, at assembly
        time, because ``RedFlagRules.__post_init__`` refuses to represent such a
        set at all.

        There is also deliberately NO injectable scan hook. A synchronous
        callback ran inside the event loop, so a slow one blocked the loop and
        made the Manager's deadline unenforceable (``asyncio.wait_for(..., 10ms)``
        could still return PASS after ~155ms). Nothing in the repository used it
        outside tests, so the extension point is gone rather than wrapped in a
        thread pool: real medical semantic/contradiction analysis is a separate
        task, not a hook on this deterministic gate.
        """
        if not isinstance(red_flag_rules, RedFlagRules):
            raise TypeError(
                "red_flag_rules must be an approved RedFlagRules instance "
                "(no empty default: nothing configured must never mean "
                "no red flag found)"
            )
        self._red_flag_rules = red_flag_rules

    # -- core decision --------------------------------------------------------

    def decide(self, ctx: AgentContext, execution: Any) -> VerifierDecision:
        """THE decision entry point. Run the deterministic check set.

        Only path to a judgement in this module: :meth:`__call__` delegates
        here, so no caller can reach PASS through a shortcut that skips a check.
        A ``FAILED``/``CANCELLED`` candidate is refused with the fixed
        ``not_completed`` reason before any content check runs — a well-cited
        draft from a run that did not complete is not a result.

        Pure and side-effect free: no model call, no audit write, no mutation of
        the input — so a late or duplicated invocation can never append a second
        terminal record, and cancellation/timeout semantics stay the Manager's.
        """
        snapshot = _validated_execution(execution)
        if snapshot is None:
            return _refused("invalid_execution")
        if snapshot.status is not AgentStatus.COMPLETED:
            # FAILED / CANCELLED: nothing was produced, so nothing can pass.
            # This guard lives HERE, in the single decision entry, not only in
            # the Manager slot, so a direct call cannot deliver a failed draft.
            return _refused("not_completed")

        answer = snapshot.answer_candidate.strip()
        evidence = list(snapshot.evidence)
        fast = _fast_path(ctx, snapshot)

        # the scan surface is the DELIVERY surface: answer + every Evidence
        # string field the Manager will hand to the caller
        delivered = _delivered_text(answer, evidence)
        red = _contains_red_flag(delivered, self._red_flag_rules)
        privacy_ok = not _leaks_privacy(delivered)
        scope_ok = not _contains_scope_violation(delivered)

        grounded = bool(evidence)
        duplicate_source = _has_duplicate_source_ids(evidence)
        # ONE split of the answer: the same claim list answers both "does a
        # deliverable answer exist?" (answer_present) and "is it supported?"
        claims = _claim_bodies(answer)
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

        # the exact-match gate only runs on an otherwise clean answer: a
        # structural citation defect is reported as such
        evidence_supported = False
        support_reason = ""
        if grounded and citation_coverage:
            evidence_supported, support_reason = _check_extractive_support(
                claims, evidence
            )
        if not evidence_supported:
            unsupported = True

        outcome, instructions = _classify(
            privacy_ok=privacy_ok,
            red=red,
            scope_ok=scope_ok,
            duplicate_source=duplicate_source,
            grounded=grounded,
            answer_present=bool(claims),
            citation_coverage=citation_coverage,
            citation_reason=citation_reason,
            evidence_supported=evidence_supported,
            support_reason=support_reason,
        )
        return VerifierDecision(
            outcome=outcome,
            revision_instructions=instructions,
            grounded=grounded,
            citation_coverage=citation_coverage,
            medical_scope_ok=scope_ok,
            privacy_ok=privacy_ok,
            unsupported_claims=unsupported,
            red_flags=red,
            fast_path=fast,
            evidence_supported=evidence_supported,
            duplicate_source=duplicate_source,
        )

    # -- #52 runner slot --------------------------------------------------------

    async def __call__(self, context: AgentContext, execution: Any) -> ManagerVerdict:
        """Verifier slot for the Manager: always a FOUR-STATE verdict.

        A THIN wrapper over :meth:`decide` — it adds no judgement of its own, so
        the Manager slot and a direct call cannot diverge (they used to: the
        status check lived only here, and ``faq_fast_decision`` bypassed it).

        The Manager decides what to do with the verdict (only ``PASS`` delivers,
        only ``REVISE`` retries once, ``BLOCK`` stops, ``ESCALATE`` goes to a
        human); this method never returns the candidate answer or its evidence,
        so a non-PASS outcome cannot leak unverified content through the verdict.
        """
        decision = self.decide(context, execution)
        return ManagerVerdict(
            decision.outcome, decision.revision_instructions or "pass"
        )


def _refused(instructions: str) -> VerifierDecision:
    """A BLOCK for a candidate that could not be verified at all.

    No content check ran, so the flags report "no violation observed" for the
    two negative checks (privacy/scope) and False for everything that would
    claim a positive property (grounded/coverage/support) — an unvalidated or
    non-completed candidate may never look half-verified.
    """
    return VerifierDecision(
        outcome=VerifierOutcome.BLOCK,
        revision_instructions=instructions,
        grounded=False,
        citation_coverage=False,
        medical_scope_ok=True,
        privacy_ok=True,
        unsupported_claims=False,
        red_flags=False,
        fast_path=False,
        evidence_supported=False,
        duplicate_source=False,
    )


def _classify(
    *,
    privacy_ok: bool,
    red: bool,
    scope_ok: bool,
    duplicate_source: bool,
    grounded: bool,
    answer_present: bool,
    citation_coverage: bool,
    citation_reason: str,
    evidence_supported: bool,
    support_reason: str,
) -> tuple[VerifierOutcome, str]:
    """Precedence: BLOCK (never deliver) before ESCALATE (human) before REVISE.

    The two ESCALATE branches are therefore evaluated before ANY ``REVISE``
    branch: a clinician-only scope violation must reach a human rather than be
    demoted to a mechanical retry because the draft was also uncited.
    """
    if not privacy_ok:
        return VerifierOutcome.BLOCK, "privacy_leak"
    if red:
        return VerifierOutcome.ESCALATE, "red_flag_escalate"
    if not scope_ok:
        return VerifierOutcome.ESCALATE, "medical_scope_out_of_ai_boundary"
    if duplicate_source:
        return VerifierOutcome.REVISE, "duplicate_source"
    if not grounded:
        return VerifierOutcome.REVISE, "ungrounded"
    if not answer_present:
        return VerifierOutcome.REVISE, "empty_answer"
    if not citation_coverage:
        return VerifierOutcome.REVISE, citation_reason or "citation_missing"
    if not evidence_supported:
        return VerifierOutcome.REVISE, support_reason or "evidence_unsupported"
    return VerifierOutcome.PASS, ""


def _fast_path(ctx: AgentContext, execution: AgentExecution) -> bool:
    """Single FAQ evidence at LOW risk → OBSERVATION ONLY.

    This flag is recorded on the decision and nothing else: it selects no
    branch in :func:`_classify` and skips no check. A low-risk single-FAQ run is
    verified by exactly the same gates as any other run. (The removed
    ``faq_fast_decision`` helper was the only thing that ever turned this flag
    into different behaviour, by silently swapping in an unconfigured verifier.)

    The Manager writes the EFFECTIVE risk (never downgraded) into the context, so
    a high-risk question can never take this shortcut.
    """
    if ctx.risk_level is not RiskLevel.LOW:
        return False
    items = list(execution.evidence)
    return len(items) == 1 and items[0].source_type == "faq"
