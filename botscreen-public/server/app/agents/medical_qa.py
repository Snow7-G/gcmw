"""MedicalQAAgent (issue #53): approved-knowledge question answering.

Hard rules (V2.3 §6.2/§7):
- knowledge is reachable ONLY through the whitelisted read-only tools
  (``knowledge.search`` / ``knowledge.get_fragment``) which in turn only see
  the production view of the governance store — unreviewed/expired content is
  unreachable by construction and the agent holds no direct store reference;
- the answer model is reachable ONLY through the ModelGateway duck — the
  agent never calls a provider SDK directly; every call is recorded with the
  actual provider/model ids for the run record;
- answers are evidence-grounded: when retrieval finds nothing the agent FAILS
  the run (``status=FAILED``, ``safety="no_evidence"``, empty answer) — it never
  invents content from a model call without approved evidence, and the model is
  not even asked;
- citations are carried on the evidence itself: every delivered
  :class:`~app.contracts.agent.Evidence` carries the ``source_id`` /
  ``knowledge_version`` / ``content_hash`` of the approved production record it
  came from, so a citation can be verified against the store;
- tenancy comes from the trusted ToolGateway context only: the agent never sends
  a tenant/session/reviewer argument (those are not in the tool schema at all —
  ``additionalProperties: false``), so it cannot ask for another tenant's data;
- tool usage is bounded (≤ ``max_tool_calls``) and the EXACT number of gateway
  round-trips — including failed ones — is reported as ``tool_calls`` for the
  Manager's run budget; marker actions carry no chain-of-thought or raw text.

The agent implements the runner contract ManagerAgent (#52) drives: it consumes
an AgentContext and returns :class:`app.agents.manager.AgentExecution` (evidence
as a tuple, identity bound to the routed agent id), so the Manager's trust
boundary accepts it without an adapter.

Model provenance note: provider/model ids are still filled in on the execution,
but the Manager does NOT publish them (no trusted source yet, see #52) — the run
record gets provenance from the trusted ModelGateway wiring in #55A.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.agents.manager import AgentExecution
from app.contracts.agent import AgentContext, AgentStatus, Evidence, ToolRequest
from app.contracts.model import ModelRequest
from app.rag.retrieval import RetrievalHit

_GROUNDED_HEADER = (
    "请仅依据下方已审核资料作答，不得引用资料外信息；"
    "回答中必须标注所用资料的编号（如 资料[1]），"
    "不得引用未给出的编号。\n\n已审核资料：\n"
)


class ToolGatewayDuck(Protocol):
    """The #57 ToolGateway surface the agent is allowed to touch.

    Identity travels ONLY as the trusted context (first positional argument) —
    there is no tenant/run/request kwarg to pass, and the gateway rejects any
    identity key that appears inside ``request.arguments`` anyway.
    """

    async def ainvoke(
        self,
        context: Any,
        request: ToolRequest,
        *,
        allowed_tools: Any = (),
        agent_id: str = "",
    ) -> Any: ...


class ModelGatewayDuck(Protocol):
    """The #37 ModelGateway surface (chat only for this agent)."""

    async def chat(self, request: ModelRequest) -> Any: ...


def _evidence_from(hit: RetrievalHit, content: str) -> Evidence:
    return Evidence(
        source_id=hit.source_id,
        source_type=hit.source_type,
        title=hit.title,
        content=content,
        source_uri=hit.source_uri,
        content_hash=hit.content_hash,
        knowledge_version=hit.knowledge_version,
    )


class MedicalQAAgent:
    """Evidence-grounded QA over approved knowledge via gatewayed tools."""

    def __init__(
        self,
        *,
        models: ModelGatewayDuck,
        tools: ToolGatewayDuck,
        allowed_tools: tuple[str, ...] = (
            "knowledge.search",
            "knowledge.get_fragment",
        ),
        max_tool_calls: int = 4,
        max_fragments: int = 2,
        provider_hint: str | None = None,
        agent_id: str = "qa",
    ) -> None:
        self._models = models
        self._tools = tools
        self._allowed_tools = list(allowed_tools)
        self._max_tool_calls = max_tool_calls
        self._max_fragments = max_fragments
        self._provider_hint = provider_hint
        # the id this agent reports back: the Manager's trust boundary requires
        # it to equal the routed manifest id, so wiring sets it explicitly
        self._agent_id = agent_id

    # -- internal helpers -----------------------------------------------------

    async def _search(
        self, ctx: AgentContext, budget: dict[str, int]
    ) -> list[RetrievalHit]:
        result = await self._tools.ainvoke(
            ctx,  # trusted identity: the gateway derives tenant/session/run here
            ToolRequest(
                tool_name="knowledge.search",
                # NO identity arguments: tenant/session come from the context the
                # gateway injects, and the schema forbids extra properties
                arguments={
                    "query": ctx.normalized_input,
                    "top_k": self._max_fragments + 2,
                },
            ),
            allowed_tools=self._allowed_tools,
            agent_id=self._agent_id,
        )
        budget["calls"] += 1
        if not result.ok:
            return []
        data = result.data or {}
        hits: list[RetrievalHit] = []
        for index, item in enumerate(data.get("items", [])[: self._max_fragments]):
            hits.append(
                RetrievalHit(
                    source_id=item.get("source_id", ""),
                    score=data.get("total", 0) - index,
                    source_type=item.get("source_type", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    content_hash=item.get("content_hash", ""),
                    source_uri=item.get("source_uri", ""),
                    medical_domain=item.get("medical_domain", ""),
                    audience=item.get("audience", ""),
                    knowledge_version=item.get("knowledge_version", ""),
                )
            )
        return hits

    async def _fragment(
        self, ctx: AgentContext, hit: RetrievalHit, budget: dict[str, int]
    ) -> str:
        result = await self._tools.ainvoke(
            ctx,
            ToolRequest(
                tool_name="knowledge.get_fragment",
                arguments={"source_id": hit.source_id},
            ),
            allowed_tools=self._allowed_tools,
            agent_id=self._agent_id,
        )
        budget["calls"] += 1
        if not result.ok:
            return hit.snippet
        data = result.data or {}
        return data.get("text") or hit.snippet

    # -- main entry -------------------------------------------------------------

    def _execution(
        self,
        *,
        status: AgentStatus,
        safety: str,
        tool_calls: int,
        answer: str = "",
        evidence: list[Evidence] | None = None,
        response: Any | None = None,
    ) -> AgentExecution:
        """Build the immutable runner result the Manager's boundary expects."""
        return AgentExecution(
            agent_id=self._agent_id,
            status=status,
            answer_candidate=answer,
            evidence=tuple(evidence or ()),
            tool_calls=tool_calls,
            provider_id=getattr(response, "provider_id", ""),
            model_id=getattr(response, "model_id", ""),
            model_version=getattr(response, "model_version", ""),
            safety_status=safety,
        )

    async def run(self, context: AgentContext) -> AgentExecution:
        """Answer one grounded turn. Raises nothing by design except gateway
        violations — failures surface as structured results for the Manager.

        All mutable run state (tool budget) lives in a per-run dict so
        concurrent runs on one shared agent instance can never interleave
        counters (the budget is a security control, not shared state)."""
        budget = {"calls": 0}
        if not (context.normalized_input or "").strip():
            return self._execution(
                status=AgentStatus.FAILED, safety="invalid_input", tool_calls=0
            )

        hits = await self._search(context, budget)

        evidence: list[Evidence] = []
        grounded: list[str] = []
        for hit in hits[: self._max_fragments]:
            if budget["calls"] >= self._max_tool_calls:
                break
            text = await self._fragment(context, hit, budget)
            evidence.append(_evidence_from(hit, text))
            grounded.append(
                f"[{len(grounded) + 1}] {hit.title}"
                f"（来源 {hit.source_id} / {hit.knowledge_version}）\n{text}"
            )

        tool_calls = budget["calls"]
        if not evidence:
            # no approved evidence: FAIL the run (never an empty success) and do
            # not even ask the model — an answer may not come from "common sense"
            return self._execution(
                status=AgentStatus.FAILED, safety="no_evidence", tool_calls=tool_calls
            )

        user_text = (
            _GROUNDED_HEADER
            + "\n\n".join(grounded)
            + f"\n\n问题：{context.normalized_input}"
        )
        response = await self._models.chat(
            ModelRequest(
                messages=[{"role": "user", "content": user_text}],
                trace_id=context.run_id,
                deadline_ms=10_000,
                token_budget=400,
                provider_hint=self._provider_hint,
            )
        )
        answer = (response.content or "").strip()
        if not answer:
            # grounded but empty model output: do not deliver an empty success
            return self._execution(
                status=AgentStatus.FAILED,
                safety="empty_reply",
                tool_calls=tool_calls,
                evidence=evidence,
                response=response,
            )
        return self._execution(
            status=AgentStatus.COMPLETED,
            safety="grounded",
            tool_calls=tool_calls,
            answer=answer,
            evidence=evidence,
            response=response,
        )
