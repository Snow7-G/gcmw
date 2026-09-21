"""Demo assembly for the #55A vertical slice (development / test ONLY).

This is the composition root for the demo MVP: synthetic, APPROVED knowledge +
MockProvider + MedicalQAAgent (RAG) + SafetyEvidenceVerifier + ManagerAgent,
wired into one ``RunExecutor`` that the run admission service schedules.

Honest scope, stated plainly:

* the knowledge is SYNTHETIC text pushed through the real #56 governance
  lifecycle (candidate → in review → approved) — nothing here is clinical
  content and nothing bypasses review;
* the model is the #53 MockProvider: canned replies only, no cloud provider,
  no real medical capability;
* the red-flag / risk rule sets carry a RECORDED ATTESTATION (``approved_by``),
  which the code cannot verify — a real clinical sign-off stays a process gate;
* the slice exists ONLY for ``GCMW_ENV`` development / test. Staging and
  production get ``None`` and the API keeps admitting runs that no executor
  drives — the same fail-closed posture the memory repository already forces.

Tenant note: the synthetic knowledge is published for
:data:`DEMO_TENANT_ID` only, so tenant isolation is demonstrable for real —
a run from any other tenant finds no evidence and is refused.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..agents.manager import ManagerAgent, RedFlagRules, RiskRules
from ..agents.medical_qa import MedicalQAAgent
from ..agents.registry import AgentManifest, AgentRegistry
from ..agents.verifier import SafetyEvidenceVerifier
from ..config import Settings
from ..knowledge.demo_seed import (
    DEMO_TENANT_ID,
    canned_replies,
    seed_demo_knowledge,
)
from ..knowledge.store import KnowledgeStore
from ..providers.mock import MockProvider
from ..providers.model_gateway import ModelGateway
from ..tools.builtins import build_gateway
from .executor import RunExecutor

# 演示语料与「可重复执行」的装载流程都在 app.knowledge.demo_seed：语料文本、
# 单句约束、审核发布路径与幂等语义集中一处，便于审计与测试。DEMO_TENANT_ID
# 依旧从这个模块可见（``__all__`` 保留），历史引用不受影响。

#: red-flag / risk markers. RECORDED ATTESTATION ONLY — see module docstring.
_DEMO_RED_FLAGS: tuple[str, ...] = ("自杀",)
_DEMO_RISK_MARKERS: tuple[str, ...] = ("剧烈", "出血")

#: the demo slice's OWN tool-audit logger — the same "gcmw.audit" channel the
#: server's rate limiter and top-level gateway write to. A silent ``lambda
#: _: None`` would make tool calls LOOK audited while nothing was recorded.
_AUDIT_LOGGER = logging.getLogger("gcmw.audit")


def _tool_audit_sink(record: Any) -> None:
    """One structured JSON line per tool call (routable to file/SIEM)."""
    _AUDIT_LOGGER.warning(record.model_dump_json())


def build_agent_executor(
    *,
    repository: Any,
    settings: Settings,
    audit_sink: Callable[[Any], None] | None = None,
) -> RunExecutor | None:
    """Assemble the demo agent stack, or ``None`` where it must not run.

    The returned executor is the ONLY thing that moves an admitted run forward;
    the admission service schedules it right after a run is created. The
    internal ToolGateway's audit sink defaults to the structured ``gcmw.audit``
    logger (an injected sink wins) and the gateway itself is attached to the
    executor as ``tool_gateway`` so the application lifecycle can release its
    worker pool on shutdown."""
    if settings.environment not in {"development", "test"}:
        return None

    store = KnowledgeStore()
    # 演示语料的装载走 app.knowledge.demo_seed 的**可重复执行**流程：内容未变
    # 时零写入（重复启动不重复造版本），内容变化时才 revoke → 重新发布。
    seed_demo_knowledge(store)
    # the reply quotes the approved sentence VERBATIM plus its marker, so
    # the exact-match support gate passes honestly (no paraphrase)
    canned = canned_replies()

    tools = build_gateway(
        knowledge_store=store, audit_sink=audit_sink or _tool_audit_sink
    )
    # #37 ModelGateway: the draft model follows GCMW_ACTIVE_PROVIDER.
    # "mock" (default) keeps the offline demo honest; "cloud" wires the
    # DashScope OpenAI-compatible chat adapter and FAILS CLOSED when the API
    # key env is missing (the config validator already checks it - this is the
    # composition-layer backstop, because the key VALUE must never be stored
    # anywhere). "local" is not part of the demo slice and stays mock here.
    use_cloud = settings.active_provider == "cloud"
    if use_cloud:
        import os

        api_key = os.getenv(settings.cloud.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                "GCMW_ACTIVE_PROVIDER=cloud requires a non-empty "
                + settings.cloud.api_key_env
                + ": refusing to assemble a demo executor that cannot reach "
                + "the configured cloud model"
            )
    models = ModelGateway(active_provider_id="cloud" if use_cloud else "mock")
    models.register(MockProvider(canned=canned))
    if use_cloud:
        from ..providers.dashscope_cloud import DashScopeCloudProvider

        models.register(
            DashScopeCloudProvider(
                api_base=settings.cloud.chat_base,
                model=settings.cloud.chat_model,
                api_key=os.getenv(settings.cloud.api_key_env, ""),
                timeout_ms=settings.cloud.timeout_ms,
            )
        )

    qa = MedicalQAAgent(models=models, tools=tools)

    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            agent_id="qa",
            version="1.0.0",
            supported_intents=["knowledge"],
            risk_level="medium",
        )
    )

    approved_at = datetime.now(timezone.utc)
    red_flags = RedFlagRules(
        patterns=_DEMO_RED_FLAGS,
        approved_by="demo-attestation",
        approved_at=approved_at,
    )
    risk_rules = RiskRules(
        patterns=_DEMO_RISK_MARKERS,
        approved_by="demo-attestation",
        approved_at=approved_at,
    )

    async def run_qa(context: Any) -> Any:
        return await qa.run(context)

    manager = ManagerAgent(
        registry=registry,
        agent_runners={"qa": run_qa},
        verifier=SafetyEvidenceVerifier(red_flag_rules=red_flags),
        red_flag_rules=red_flags,
        risk_rules=risk_rules,
        # #55A-B: provenance comes from the SERVER-SIDE ModelGateway/Provider
        # configuration only — model-reported identities are discarded
        trusted_model_provenance={"qa": models.provenance()},
    )
    executor = RunExecutor(
        repository=repository,
        manager=manager,
        # the configured RUN budget: the executor writes it into the
        # AgentContext deadline, so the Manager enforces it per engagement and
        # an exceeded budget lands as a terminal FAILED — never a stuck run
        run_timeout_ms=settings.run_timeout_ms,
    )
    # the app lifecycle releases this gateway's worker pool on shutdown
    executor.tool_gateway = tools
    # 同 tool_gateway 的 duck-attach：网关随 executor 可检视（测试/可观测性）
    executor.models = models
    return executor


__all__ = ["DEMO_TENANT_ID", "build_agent_executor"]
