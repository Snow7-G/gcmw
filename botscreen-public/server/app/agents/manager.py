"""ManagerAgent (issue #52): deterministic safety pipeline + lightweight routing.

Implements the V2.3 §6.1 run shape inside one in-process controller — the
guarding/route/retrieve/draft/verify/stream vocabulary mirrors the #10/#21
state machine so a later runner can map each phase onto SSE ``process.status``
events without re-deriving semantics:

1. guard      — context/deadline validation, input length cap, PII
                desensitization, pre-model red-flag gate (escalate: no model,
                no tools, no runner);
2. route      — risk decided BEFORE routing and written into the context the
                routed agent and the verifier receive (``AgentContext.risk_level``,
                the shared contract enum), then deterministic intent
                classification and registry routing; hard budgets: ≤ max_handoffs
                engagements, ≤ max_tool_calls tool calls (reported by the
                executed agent), ≤ max_revisions verifier-driven revisions —
                the V2.3 numbers are CEILINGS that cannot be configured upward;
3. execute    — the routed agent runs (its own ModelGateway/ToolGateway usage
                arrives with #53) under the remaining deadline;
4. verify     — optional Verifier handoff (runner arrives with #54) with
                at-most-one controlled revision when the verdict is reject;
5. finalize   — AgentResult with evidence, safe public trace markers and the
                actual provider/model ids for run records. ONE delivery
                predicate decides whether an answer may leave this module
                (``_may_deliver``: the run COMPLETED *and* the verifier returned
                PASS); every other combination — failed or cancelled run, missing
                verdict, blocked/escalated/rejected verdict, revision that never
                completed — delivers an empty answer.

No free multi-agent chat: agents only reach models through ModelGateway and
tools through ToolGateway; the Manager itself never calls the model for
"thinking" — routing is rule-based and lightweight, and no chain-of-thought,
prompt or raw input ever leaves this module except as the desensitized text
that the routed agent is allowed to see. Public trace markers are allowlisted on
THREE axes (type, key, value — see :func:`is_safe_marker_value`), so
sub-agent-supplied metadata that is not a safe token is simply not published.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import AwareDatetime

from app.contracts.agent import (
    AgentContext,
    AgentResult,
    AgentStatus,
    Evidence,
    RiskLevel,
)
from app.contracts.errors import ErrorCode

# PII-ish patterns removed before any model/runner sees the text. The
# replacement marker never echoes the matched value.
_REDACTED = "[已脱敏]"
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),  # vendor API key style
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),  # CN mobile number
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),  # CN id card number
)


# The ONLY marker types this module may publish, and the ONLY keys those
# markers may carry. Both axes are allowlisted in :meth:`ManagerAgent._mark`, so
# free text (raw input, prompt, model output, chain-of-thought) has no path into
# ``AgentResult.actions`` even by mistake: a marker is a typed decision, not a
# place to attach prose.
SAFE_MARKER_TYPES: frozenset[str] = frozenset(
    {
        "manager.guard",
        "manager.risk",
        "manager.route",
        "manager.revised",
        "route.handoff",
        "agent.done",
        "model.call",
        "safety.escalate",
        "verify.handoff",
        "verify.verdict",
        "verify.revise",
        "verify.missing",
        "verify.not_run",
        "verify.blocked",
        "verify.escalated",
        "verify.reject_final",
    }
)

SAFE_MARKER_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "intent",
        "level",
        "agent_id",
        "handoffs",
        "status",
        "outcome",
        "revision",
        "count",
        "provider_id",
        "model_id",
        "model_version",
    }
)

# VALUES are constrained too — key allowlisting alone would still let arbitrary
# text ride out under an allowed key. A marker value is a short ASCII token or a
# bounded integer; enumerated markers carry only their known value sets (defined
# below, once the enums they pin exist).
_MARKER_TOKEN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:@+-]{0,63}\Z")
_MARKER_INT_MAX = 10_000


def desensitize(text: str) -> str:
    """Remove PII-ish credential/identifier patterns (deterministic).

    Only the marker is substituted — matches are never kept or logged.
    """
    cleaned = text
    for pattern in _PII_PATTERNS:
        cleaned = pattern.sub(_REDACTED, cleaned)
    return cleaned


class ManagerAgentError(RuntimeError):
    """Manager failure carrying a stable ErrorCode (mapped by the #36 boundary)."""

    def __init__(self, code: ErrorCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


@dataclass(frozen=True)
class RedFlagRules:
    """Clinically APPROVED red-flag rule set (V2.3 §5.3).

    An empty or unapproved rule set cannot be represented, and the Manager
    refuses to run without one: "no rules configured" must never mean "no red
    flags found". ``approved_by``/``approved_at`` record the clinical sign-off.
    """

    patterns: tuple[str, ...]
    approved_by: str
    approved_at: AwareDatetime

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("red-flag rules must not be empty")
        if not all(isinstance(item, str) and item.strip() for item in self.patterns):
            raise ValueError("red-flag rules must be non-empty strings")
        if not self.approved_by or not self.approved_by.strip():
            raise ValueError("red-flag rules need an approving clinician")

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(rule.lower() in lowered for rule in self.patterns)


@dataclass(frozen=True)
class RiskRules:
    """Clinically APPROVED high-risk question markers.

    Same fail-closed contract as :class:`RedFlagRules`: without an approved
    marker set the Manager cannot claim a question is low risk. The verdict uses
    the CONTRACT enum :class:`app.contracts.agent.RiskLevel` — the same type the
    routed agent and the verifier read from ``AgentContext.risk_level``, so a
    high-risk decision cannot be lost in translation between two look-alike
    enums.
    """

    patterns: tuple[str, ...]
    approved_by: str
    approved_at: AwareDatetime

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("risk markers must not be empty")
        if not self.approved_by or not self.approved_by.strip():
            raise ValueError("risk markers need an approving clinician")

    def classify(self, text: str) -> RiskLevel:
        lowered = text.lower()
        for pattern in self.patterns:
            if pattern.lower() in lowered:
                return RiskLevel.HIGH
        return RiskLevel.LOW


@dataclass(frozen=True)
class ManagerLimits:
    """Per-run budgets (V2.3 §6.1: ≤2 handoffs, ≤4 tools, ≤1 revision).

    The V2.3 numbers are CEILINGS, not tunable defaults: this class refuses to be
    configured above them, so a caller cannot amplify a hard safety budget
    (``max_revisions=3`` would permit three revisions / four sub-agent executions
    and break the "≤1 controlled revision" rule). A budget may be tightened
    (disabled) but never raised.

    IMPORTANT (honest scope): the handoff budget is enforced by this module at
    call time, but ``max_tool_calls`` is a **POST-HOC check** — tool calls are
    read from the sub-agent's own report after it returns. Call-time hard
    counting requires the ToolGateway quota path and lands with #55A; until
    then this PR must not claim the tool budget is preemptively enforced.
    """

    max_input_chars: int = 2000
    max_handoffs: int = 2
    max_tool_calls: int = 4
    max_revisions: int = 1

    def __post_init__(self) -> None:
        if self.max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        ceilings = {"max_handoffs": 2, "max_tool_calls": 4, "max_revisions": 1}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if value < 0 or value > ceiling:
                raise ValueError(
                    f"{name} must be within 0..{ceiling} (V2.3 hard ceiling)"
                )


@dataclass(frozen=True)
class AgentExecution:
    """Structured outcome of one routed agent run (safe subset only)."""

    agent_id: str
    status: AgentStatus
    answer_candidate: str = ""
    evidence: tuple[Evidence, ...] = ()
    tool_calls: int = 0
    provider_id: str = ""
    model_id: str = ""
    model_version: str = ""
    safety_status: str = "unknown"


class VerifierOutcome(str, Enum):
    """The verifier's decision. A boolean cannot express it (V2.3 §6.4):

    * ``PASS``     — the answer may be delivered;
    * ``REVISE``   — the ONE controlled retry (and only this outcome retries);
    * ``BLOCK``    — stop immediately: no retry, no answer;
    * ``ESCALATE`` — hand the case to a human: no AI answer is delivered.
    """

    PASS = "pass"
    REVISE = "revise"
    BLOCK = "block"
    ESCALATE = "escalate"


#: Enumerated marker keys may only carry their known values (value-level guard
#: companion to :data:`SAFE_MARKER_KEYS`); all other keys accept bounded ints or
#: short ASCII tokens (see :func:`is_safe_marker_value`).
SAFE_MARKER_VALUE_SETS: dict[str, frozenset[str]] = {
    "level": frozenset({RiskLevel.LOW.value, RiskLevel.HIGH.value}),
    "outcome": frozenset(item.value for item in VerifierOutcome),
    "status": frozenset(item.value for item in AgentStatus),
}


def is_safe_marker_value(key: str, value: Any) -> bool:
    """True when ``value`` may be published under ``key`` (VALUE-level guard).

    Allowlisting keys alone would still let arbitrary text ride out under an
    allowed key (a synthetic private string in ``provider_id``, say). So values
    are constrained too: enumerated keys are pinned to their known value sets,
    and every other key accepts only a bounded integer or a short ASCII token.
    Free text — CJK, whitespace, quotes, newlines, prompts, patient statements —
    never qualifies, so a marker can carry a decision but never prose.
    """
    allowed = SAFE_MARKER_VALUE_SETS.get(key)
    if allowed is not None:
        return isinstance(value, str) and value in allowed
    if isinstance(value, bool):  # bool is an int subclass: never a marker value
        return False
    if isinstance(value, int):
        return 0 <= value <= _MARKER_INT_MAX
    return isinstance(value, str) and _MARKER_TOKEN.match(value) is not None


@dataclass(frozen=True)
class Verdict:
    outcome: VerifierOutcome
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.outcome is VerifierOutcome.PASS


class AgentRunner(Protocol):
    """Executes the routed agent. Arrives with #53 (MedicalQA) / #54
    (Verifier); the Manager only ever sees the safe AgentExecution result."""

    def __call__(self, context: AgentContext) -> Awaitable[AgentExecution]: ...


class VerifierRunner(Protocol):
    def __call__(
        self, context: AgentContext, execution: AgentExecution
    ) -> Awaitable[Verdict]: ...


class ManagerAgent:
    """Deterministic run controller (one run = one call to ``execute``)."""

    def __init__(
        self,
        *,
        registry: Any,
        agent_runners: Mapping[str, AgentRunner] | None = None,
        verifier: VerifierRunner | None = None,
        intent_routes: Mapping[str, tuple[str, ...]] | None = None,
        default_intent: str = "knowledge",
        red_flag_rules: RedFlagRules | None = None,
        risk_rules: RiskRules | None = None,
        clock: Callable[[], datetime] | None = None,
        limits: ManagerLimits | None = None,
    ) -> None:
        self._registry = registry
        self._runners: dict[str, AgentRunner] = dict(agent_runners or {})
        self._verifier = verifier
        # intent -> keyword tuple; first keyword hit wins (registration order)
        self._intent_routes: dict[str, tuple[str, ...]] = dict(intent_routes or {})
        self._default_intent = default_intent
        # FAIL CLOSED at construction: a Manager without clinically approved
        # red-flag and risk rules could silently "find no red flags" and route a
        # high-risk question down the normal path.
        if red_flag_rules is None:
            raise ValueError(
                "red_flag_rules are required: an unconfigured red-flag set must "
                "never default to 'no red flags found'"
            )
        if risk_rules is None:
            raise ValueError(
                "risk_rules are required: without approved markers the Manager "
                "cannot classify a question as low risk"
            )
        self._red_flag_rules = red_flag_rules
        self._risk_rules = risk_rules
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._limits = limits or ManagerLimits()

    # -- guard ----------------------------------------------------------------

    def _guard(self, ctx: AgentContext, text: str) -> str:
        if not (
            ctx.tenant_id
            and ctx.device_id
            and ctx.session_id
            and ctx.run_id
            and ctx.channel
        ):
            raise ManagerAgentError(
                ErrorCode.VALIDATION_INVALID_INPUT, "run context is incomplete"
            )
        if ctx.deadline is not None and self._clock() >= ctx.deadline:
            raise ManagerAgentError(ErrorCode.TIMEOUT_AGENT, "deadline already passed")
        cleaned = (text or "").strip()
        if not cleaned:
            raise ManagerAgentError(
                ErrorCode.VALIDATION_INVALID_INPUT, "input is empty"
            )
        if len(cleaned) > self._limits.max_input_chars:
            raise ManagerAgentError(
                ErrorCode.VALIDATION_INVALID_INPUT,
                f"input exceeds {self._limits.max_input_chars} characters",
            )
        return desensitize(cleaned)

    # -- red-flag pre-model gate ------------------------------------------------

    def _red_flag_escalated(self, cleaned: str) -> bool:
        """True when an APPROVED clinical red-flag rule matches.

        The rule set is mandatory (see ``__init__``): an unconfigured or
        unapproved set can never reach this point, so "no match" always means
        "the approved rules were actually evaluated".
        """
        return self._red_flag_rules.matches(cleaned)

    # -- lightweight routing ----------------------------------------------------

    def _classify_intent(self, cleaned: str) -> str:
        lowered = cleaned.lower()
        for intent, keywords in self._intent_routes.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                return intent
        return self._default_intent

    def _resolve_agent(self, intent: str, ctx: AgentContext) -> Any:
        manifest = self._registry.resolve(intent)
        if manifest is None or not getattr(manifest, "enabled", True):
            raise ManagerAgentError(
                ErrorCode.NOT_FOUND_AGENT,
                f"no enabled agent handles intent {intent!r}",
            )
        return manifest

    # -- budget helpers ---------------------------------------------------------

    def _raise_if_over_budget(self, handoffs: int, tool_calls: int) -> None:
        if handoffs > self._limits.max_handoffs:
            raise ManagerAgentError(
                ErrorCode.RUN_BUDGET_EXCEEDED,
                f"handoff budget exceeded ({handoffs} > {self._limits.max_handoffs})",
            )
        if tool_calls > self._limits.max_tool_calls:
            raise ManagerAgentError(
                ErrorCode.TOOL_OVER_LIMIT,
                f"tool budget exceeded ({tool_calls} > {self._limits.max_tool_calls})",
            )

    # -- execute ----------------------------------------------------------------

    @staticmethod
    def _may_deliver(execution: AgentExecution, verdict: Verdict | None) -> bool:
        """THE single delivery predicate: a completed run AND an explicit PASS.

        Everything else — a failed or cancelled run, a missing verdict, a
        revision that never completed — delivers an empty answer. Keeping this
        in one place is what makes "非 COMPLETED 或未获 PASS 时答案为空" hold for
        the first execution and for every revision alike.
        """
        return (
            execution.status is AgentStatus.COMPLETED
            and verdict is not None
            and verdict.outcome is VerifierOutcome.PASS
        )

    async def execute(self, ctx: AgentContext, text: str) -> AgentResult:
        """Run one guarded, routed, verified turn. Returns the Manager's final
        AgentResult (safe markers only — no chain-of-thought)."""
        actions: list[dict[str, Any]] = []
        evidence: list[Evidence] = []
        self._mark(actions, "manager.guard")
        cleaned = self._guard(ctx, text)

        if self._red_flag_escalated(cleaned):
            self._mark(actions, "safety.escalate")
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.COMPLETED,
                answer="",
                safety="escalated",
            )

        # risk is decided BEFORE routing: a high-risk question may never take a
        # low-risk shortcut, and it always requires verification
        risk = self._risk_rules.classify(cleaned)
        self._mark(actions, "manager.risk", level=risk.value)

        intent = self._classify_intent(cleaned)
        self._mark(actions, "manager.route", intent=intent)
        manifest = self._resolve_agent(intent, ctx)
        agent_id = manifest.agent_id
        handoffs = 1  # manager -> routed agent (distinct engagements only)
        if agent_id not in self._runners:
            raise ManagerAgentError(
                ErrorCode.NOT_FOUND_AGENT,
                f"no runner registered for agent {agent_id!r}",
            )
        self._mark(actions, "route.handoff", agent_id=agent_id, handoffs=handoffs)

        # The declared risk travels WITH the run: the routed agent and the
        # verifier both read it from ``AgentContext.risk_level`` (the same
        # contract enum), so a high-risk decision cannot be downgraded silently
        # by a downstream fast path that only inspects the context.
        sub_ctx = ctx.model_copy(
            deep=True,
            update={"normalized_input": cleaned, "risk_level": risk},
        )
        execution = await self._run_with_deadline(
            self._runners[agent_id](sub_ctx), self._remaining_ms(ctx)
        )
        self._raise_if_over_budget(handoffs, execution.tool_calls)
        evidence.extend(execution.evidence)
        self._record_model(actions, execution)
        self._mark(
            actions,
            "agent.done",
            agent_id=agent_id,
            status=execution.status.value,
        )

        verdict: Verdict | None = None
        revisions = 0
        if self._verifier is None:
            # NO VERIFIER = NO MEDICAL ANSWER. The previous behaviour marked the
            # run "passed"; an unverified answer must never leave this module.
            self._mark(actions, "verify.missing")
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.FAILED,
                answer="",
                safety="unverified",
            )
        if execution.status is AgentStatus.COMPLETED:
            # verification is a second distinct engagement; revisions re-run the
            # same routed agent in-loop and do not consume new handoffs
            handoffs += 1
            self._raise_if_over_budget(handoffs, execution.tool_calls)
            self._mark(
                actions, "verify.handoff", agent_id="verifier", handoffs=handoffs
            )
            verdict = await self._run_with_deadline(
                self._verifier(sub_ctx, execution), self._remaining_ms(ctx)
            )
            self._mark(actions, "verify.verdict", outcome=verdict.outcome.value)
            # ONLY "revise" retries, exactly once. BLOCK stops immediately and
            # ESCALATE goes to a human — neither may be retried or re-answered.
            while (
                verdict is not None
                and verdict.outcome is VerifierOutcome.REVISE
                and revisions < self._limits.max_revisions
            ):
                revisions += 1
                self._mark(actions, "verify.revise", revision=revisions)
                execution = await self._run_with_deadline(
                    self._runners[agent_id](sub_ctx), self._remaining_ms(ctx)
                )
                self._raise_if_over_budget(handoffs, execution.tool_calls)
                evidence = list(execution.evidence)
                self._record_model(actions, execution)
                if execution.status is not AgentStatus.COMPLETED:
                    # a revision that did not complete is not verifiable: stop
                    # here and let the single delivery decision below refuse it
                    break
                verdict = await self._run_with_deadline(
                    self._verifier(sub_ctx, execution), self._remaining_ms(ctx)
                )
                self._mark(actions, "verify.verdict", outcome=verdict.outcome.value)

        # ---- NOT COMPLETED: nothing was verified, nothing may be delivered ---
        # This covers the first execution AND every revision, so a draft that
        # never completed can never be handed out (it used to be returned with
        # safety="passed").
        if execution.status is not AgentStatus.COMPLETED:
            self._mark(actions, "verify.not_run")
            if revisions:
                self._mark(actions, "manager.revised", count=revisions)
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=execution.status,
                answer="",
                safety="failed",
            )

        if verdict is not None and verdict.outcome is VerifierOutcome.BLOCK:
            # BLOCK: stop now, no retry, no answer
            self._mark(actions, "verify.blocked")
            if revisions:
                self._mark(actions, "manager.revised", count=revisions)
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.FAILED,
                answer="",
                safety="blocked",
            )
        if verdict is not None and verdict.outcome is VerifierOutcome.ESCALATE:
            # ESCALATE: a human handles the case; this run delivers no AI answer
            self._mark(actions, "verify.escalated")
            if revisions:
                self._mark(actions, "manager.revised", count=revisions)
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.COMPLETED,
                answer="",
                safety="escalated",
            )
        if verdict is not None and verdict.outcome is VerifierOutcome.REVISE:
            # the single revision did not satisfy the verifier: refuse to answer
            self._mark(actions, "verify.reject_final")
            if revisions:
                self._mark(actions, "manager.revised", count=revisions)
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.FAILED,
                answer="",
                safety="revised",
            )

        # ---- THE single delivery decision ----------------------------------
        # The execution COMPLETED, so the only question left is the verdict: a
        # missing or non-PASS verdict delivers NOTHING. `_may_deliver` is the one
        # predicate that decides this, for the first execution and revisions.
        deliverable = self._may_deliver(execution, verdict)
        if revisions:
            self._mark(actions, "manager.revised", count=revisions)
        if not deliverable:
            self._mark(actions, "verify.not_run")
            return self._finalize(
                ctx,
                actions,
                evidence,
                status=AgentStatus.FAILED,
                answer="",
                safety="unverified",
            )

        # only COMPLETED + PASS reach this point (see _may_deliver)
        return self._finalize(
            ctx,
            actions,
            evidence,
            status=execution.status,
            answer=execution.answer_candidate,
            safety="verified",
            confidence="high",
        )

    # -- helpers ---------------------------------------------------------------

    async def _run_with_deadline(self, awaitable: Awaitable, remaining_ms: int) -> Any:
        if remaining_ms <= 0:
            raise ManagerAgentError(ErrorCode.TIMEOUT_AGENT, "deadline expired")
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining_ms / 1000)
        except asyncio.TimeoutError as exc:
            raise ManagerAgentError(ErrorCode.TIMEOUT_AGENT, "agent timed out") from exc

    def _remaining_ms(self, ctx: AgentContext) -> int:
        if ctx.deadline is None:
            return 60_000
        return max(int((ctx.deadline - self._clock()).total_seconds() * 1000), 0)

    def _record_model(
        self, actions: list[dict[str, Any]], execution: AgentExecution
    ) -> None:
        """Publish provider/model provenance ONLY when it is a safe token.

        These three fields come from the sub-agent's own report, i.e. from
        outside this module's trusted core. They are therefore treated as
        UNTRUSTED: a value that is not a short ASCII token is simply not
        published — never truncated, never hashed, never echoed — so the public
        ``actions`` list cannot be used as a text channel.

        * provider/model id unsafe (or empty) → no ``model.call`` marker at all;
        * only the version unsafe (or empty) → provenance is still recorded,
          the version key is dropped.
        """
        provider_id = execution.provider_id
        model_id = execution.model_id
        if not provider_id or not model_id:
            return
        if not (
            is_safe_marker_value("provider_id", provider_id)
            and is_safe_marker_value("model_id", model_id)
        ):
            return
        details: dict[str, Any] = {"provider_id": provider_id, "model_id": model_id}
        version = execution.model_version
        if version and is_safe_marker_value("model_version", version):
            details["model_version"] = version
        self._mark(actions, "model.call", **details)

    def _mark(
        self,
        actions: list[dict[str, Any]],
        marker: str,
        **details: Any,
    ) -> None:
        """Append one SAFE public trace marker (allowlisted on all three axes).

        Marker TYPE, KEY and VALUE are each constrained: an unknown marker name,
        an unknown key, or a value that is not a bounded integer / short ASCII
        token (or not in an enumerated key's value set) is a programming error
        and fails the run loudly rather than publishing something the allowlist
        did not sanction.
        """
        if marker not in SAFE_MARKER_TYPES:
            raise ValueError(f"unknown marker type {marker!r}")
        unknown = set(details) - SAFE_MARKER_KEYS
        if unknown:
            raise ValueError(f"unsafe marker keys for {marker!r}: {sorted(unknown)}")
        unsafe = sorted(
            key
            for key, value in details.items()
            if not is_safe_marker_value(key, value)
        )
        if unsafe:
            raise ValueError(f"unsafe marker values for {marker!r}: {unsafe}")
        entry: dict[str, Any] = {"type": marker}
        entry.update(details)
        actions.append(entry)

    def _finalize(
        self,
        ctx: AgentContext,
        actions: list[dict[str, Any]],
        evidence: list[Evidence],
        *,
        status: AgentStatus,
        answer: str,
        safety: str,
        confidence: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            agent_id="manager",
            status=status,
            answer_candidate=answer,
            evidence=evidence,
            actions=actions,
            confidence_band=confidence,
            safety_status=safety,
        )
