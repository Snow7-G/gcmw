"""ManagerAgent (issue #52): deterministic safety pipeline + lightweight routing.

Implements the V2.3 §6.1 run shape inside one in-process controller — the
guarding/route/retrieve/draft/verify/stream vocabulary mirrors the #10/#21
state machine so a later runner can map each phase onto SSE ``process.status``
events without re-deriving semantics:

1. guard      — context/deadline validation, input length cap, PII
                desensitization, pre-model red-flag gate (escalate: no model,
                no tools, no runner);
2. route      — deterministic intent classification, then registry routing;
                hard budgets: ≤ max_handoffs engagements, ≤ max_tool_calls
                tool calls (reported by the executed agent), ≤ max_revisions
                verifier-driven revisions;
3. execute    — the routed agent runs (its own ModelGateway/ToolGateway usage
                arrives with #53) under the remaining deadline;
4. verify     — optional Verifier handoff (runner arrives with #54) with
                at-most-one controlled revision when the verdict is reject;
5. finalize   — AgentResult with evidence, safe public trace markers and the
                actual provider/model ids for run records.

No free multi-agent chat: agents only reach models through ModelGateway and
tools through ToolGateway; the Manager itself never calls the model for
"thinking" — routing is rule-based and lightweight, and no chain-of-thought,
prompt or raw input ever leaves this module except as the desensitized text
that the routed agent is allowed to see.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from app.contracts.agent import AgentContext, AgentResult, AgentStatus, Evidence
from app.contracts.errors import ErrorCode

# PII-ish patterns removed before any model/runner sees the text. The
# replacement marker never echoes the matched value.
_REDACTED = "[已脱敏]"
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),  # vendor API key style
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),  # CN mobile number
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),  # CN id card number
)


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
class ManagerLimits:
    """Hard per-run budgets (V2.3 §6.1: ≤2 handoffs, ≤4 tools, ≤1 revision)."""

    max_input_chars: int = 2000
    max_handoffs: int = 2
    max_tool_calls: int = 4
    max_revisions: int = 1


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


@dataclass(frozen=True)
class Verdict:
    approved: bool
    reason: str = ""


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
        red_flags: tuple[str, ...] = (),
        clock: Callable[[], datetime] | None = None,
        limits: ManagerLimits | None = None,
    ) -> None:
        self._registry = registry
        self._runners: dict[str, AgentRunner] = dict(agent_runners or {})
        self._verifier = verifier
        # intent -> keyword tuple; first keyword hit wins (registration order)
        self._intent_routes: dict[str, tuple[str, ...]] = dict(intent_routes or {})
        self._default_intent = default_intent
        self._red_flags = tuple(red_flags)
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
        lowered = cleaned.lower()
        return any(rule.lower() in lowered for rule in self._red_flags)

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

        sub_ctx = ctx.model_copy(deep=True, update={"normalized_input": cleaned})
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
        if self._verifier is not None and execution.status is AgentStatus.COMPLETED:
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
            self._mark(actions, "verify.verdict", approved=verdict.approved)
            while not verdict.approved and revisions < self._limits.max_revisions:
                revisions += 1
                self._mark(actions, "verify.revise", revision=revisions)
                execution = await self._run_with_deadline(
                    self._runners[agent_id](sub_ctx), self._remaining_ms(ctx)
                )
                self._raise_if_over_budget(handoffs, execution.tool_calls)
                evidence = list(execution.evidence)
                self._record_model(actions, execution)
                verdict = await self._run_with_deadline(
                    self._verifier(sub_ctx, execution), self._remaining_ms(ctx)
                )
                self._mark(actions, "verify.verdict", approved=verdict.approved)

        if verdict is not None and not verdict.approved:
            # final rejection after ≤1 controlled revision: an unverified
            # medical answer must never be delivered — the run fails cleanly
            # with audit markers only (no answer leaves this module)
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

        safety = (
            "passed"
            if execution.safety_status in ("", "unknown")
            else execution.safety_status
        )
        if verdict is not None:
            safety = "verified" if verdict.approved else safety
        if revisions:
            self._mark(actions, "manager.revised", count=revisions)
        return self._finalize(
            ctx,
            actions,
            evidence,
            status=execution.status,
            answer=execution.answer_candidate,
            safety=safety,
            confidence=("high" if (verdict and verdict.approved) else None),
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
        if execution.provider_id and execution.model_id:
            actions.append(
                {
                    "type": "model.call",
                    "provider_id": execution.provider_id,
                    "model_id": execution.model_id,
                    "model_version": execution.model_version,
                }
            )

    def _mark(
        self,
        actions: list[dict[str, Any]],
        marker: str,
        **details: Any,
    ) -> None:
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
