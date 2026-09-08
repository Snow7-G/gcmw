"""MedicalQAAgent (issue #53): approved-knowledge question answering.

Hard rules (V2.3 §6.2/§7):
- knowledge is reachable ONLY through the whitelisted read-only tools
  (``knowledge.search`` / ``knowledge.get_fragment``) which in turn only see
  the production view of the governance store — unreviewed/expired content is
  unreachable by construction and the agent holds no direct store reference;
- the answer model is reachable ONLY through the ModelGateway duck — the
  agent never calls a provider SDK directly; every call is recorded with the
  actual provider/model ids for the run record;
- answers are evidence-grounded: when retrieval finds nothing the agent
  returns an empty answer with ``safety="no_evidence"`` — it never invents
  content from a model call without approved evidence;
- tool usage is bounded (≤ ``max_tool_calls``) and counted for the Manager's
  run budget; marker actions carry no chain-of-thought or raw text.

The agent implements the runner shape that ManagerAgent (#52) drives: it
consumes an AgentContext and returns an execution result whose attributes
match #52's ``AgentExecution`` structurally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.contracts.agent import AgentContext, AgentStatus, Evidence, ToolRequest
from app.contracts.model import ModelRequest
from app.rag.retrieval import RetrievalHit

_GROUNDED_HEADER = "请仅依据下方已审核资料作答，不得引用资料外信息。\n\n已审核资料：\n"


@dataclass
class QAExecution:
    """Runner result structurally compatible with ManagerAgent.AgentExecution."""

    agent_id: str = "qa"
    status: AgentStatus = AgentStatus.COMPLETED
    answer_candidate: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    tool_calls: int = 0
    provider_id: str = ""
    model_id: str = ""
    model_version: str = ""
    safety_status: str = "unknown"


class ToolGatewayDuck(Protocol):
    """The #57 ToolGateway surface the agent is allowed to touch."""

    async def invoke(
        self,
        request: ToolRequest,
        *,
        allowed_tools: list[str],
        agent_id: str,
        tenant_id: str,
        run_id: str,
        request_id: str,
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
    ) -> None:
        self._models = models
        self._tools = tools
        self._allowed_tools = list(allowed_tools)
        self._max_tool_calls = max_tool_calls
        self._max_fragments = max_fragments
        self._provider_hint = provider_hint

    # -- internal helpers -----------------------------------------------------

    async def _search(self, ctx: AgentContext) -> list[RetrievalHit]:
        result = await self._tools.invoke(
            ToolRequest(
                tool_name="knowledge.search",
                arguments={
                    "query": ctx.normalized_input,
                    "tenant_id": ctx.tenant_id,
                    "top_k": self._max_fragments + 2,
                },
            ),
            allowed_tools=self._allowed_tools,
            agent_id="qa",
            tenant_id=ctx.tenant_id,
            run_id=ctx.run_id,
            request_id=ctx.run_id,
        )
        self._tool_calls += 1
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

    async def _fragment(self, ctx: AgentContext, hit: RetrievalHit) -> str:
        result = await self._tools.invoke(
            ToolRequest(
                tool_name="knowledge.get_fragment",
                arguments={
                    "source_id": hit.source_id,
                    "tenant_id": ctx.tenant_id,
                },
            ),
            allowed_tools=self._allowed_tools,
            agent_id="qa",
            tenant_id=ctx.tenant_id,
            run_id=ctx.run_id,
            request_id=ctx.run_id,
        )
        self._tool_calls += 1
        if not result.ok:
            return hit.snippet
        data = result.data or {}
        return data.get("text") or hit.snippet

    # -- main entry -------------------------------------------------------------

    async def run(self, context: AgentContext) -> QAExecution:
        """Answer one grounded turn. Raises nothing by design except gateway
        violations — failures surface as structured results for the Manager."""
        self._tool_calls = 0
        execution = QAExecution(agent_id="qa", status=AgentStatus.COMPLETED)

        hits = await self._search(context)

        evidence: list[Evidence] = []
        grounded: list[str] = []
        for hit in hits[: self._max_fragments]:
            if self._tool_calls >= self._max_tool_calls:
                break
            text = await self._fragment(context, hit)
            evidence.append(_evidence_from(hit, text))
            grounded.append(
                f"[{len(grounded) + 1}] {hit.title}"
                f"（来源 {hit.source_id} / {hit.knowledge_version}）\n{text}"
            )

        execution.evidence = evidence
        execution.tool_calls = self._tool_calls
        if not evidence:
            # no approved evidence: never invent an answer
            execution.safety_status = "no_evidence"
            return execution

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
        execution.provider_id = response.provider_id
        execution.model_id = response.model_id
        execution.model_version = response.model_version
        execution.answer_candidate = (response.content or "").strip()
        execution.safety_status = "grounded"
        return execution
