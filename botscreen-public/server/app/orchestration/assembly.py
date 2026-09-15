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

from datetime import datetime, timezone
from typing import Any

from ..agents.manager import ManagerAgent, RedFlagRules, RiskRules
from ..agents.medical_qa import MedicalQAAgent
from ..agents.registry import AgentManifest, AgentRegistry
from ..agents.verifier import SafetyEvidenceVerifier
from ..config import Settings
from ..contracts.common import TenantContext
from ..contracts.knowledge import ApprovalDecision, CandidateInput, KnowledgeSourceType
from ..knowledge.store import KnowledgeStore
from ..providers.mock import MockProvider
from ..providers.model_gateway import ModelGateway
from ..tools.builtins import build_gateway
from .executor import RunExecutor

#: the tenant the synthetic demo knowledge is published for. A run from any
#: other tenant must find no evidence and be refused (isolation is real).
DEMO_TENANT_ID = "t1"

#: knowledge validity window for the synthetic corpus (fixed, aware UTC).
_KNOWLEDGE_VALID_FROM = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: the demo corpus: SYNTHETIC sentences whose full text the MockProvider echoes
#: back, so every demo answer passes the exact-match support gate honestly.
_DEMO_KNOWLEDGE: tuple[dict[str, str], ...] = (
    {
        "source_id": "faq-fever",
        "title": "发热护理须知",
        "content": "体温超过38.5建议门诊就诊。",
        "question": "发热",
    },
    {
        "source_id": "faq-eye",
        "title": "用眼卫生须知",
        "content": "眼部不适需及时就诊。",
        "question": "眼睛",
    },
)

#: red-flag / risk markers. RECORDED ATTESTATION ONLY — see module docstring.
_DEMO_RED_FLAGS: tuple[str, ...] = ("自杀",)
_DEMO_RISK_MARKERS: tuple[str, ...] = ("剧烈", "出血")


def _cite(sentence: str) -> str:
    """Compose a model reply whose citation sits INSIDE the sentence.

    The support gate splits claims on sentence terminators, so a marker placed
    after the closing 。 would land in a bodyless fragment and be read as a
    dangling citation. A citation always goes BEFORE the sentence-ending
    punctuation — which is also what the #53 prompt asks for."""
    body = sentence.rstrip("。")
    return f"{body}（资料[1]）。"


def _publish(store: KnowledgeStore, source_id: str, title: str, content: str) -> None:
    """Drive the #56 governance lifecycle to APPROVED for the demo tenant."""
    context = TenantContext(tenant_id=DEMO_TENANT_ID)
    store.add_candidate(
        context,
        CandidateInput(
            source_id=source_id,
            source_type=KnowledgeSourceType.FAQ,
            title=title,
            content=content,
            source_uri=f"kbase://{source_id}",
        ),
        actor="demo-content-owner",
    )
    store.mark_in_review(context, source_id, actor="demo-content-owner")
    store.approve(
        context,
        source_id,
        ApprovalDecision(reviewer="demo-reviewer", valid_from=_KNOWLEDGE_VALID_FROM),
    )


def build_agent_executor(*, repository: Any, settings: Settings) -> RunExecutor | None:
    """Assemble the demo agent stack, or ``None`` where it must not run.

    The returned executor is the ONLY thing that moves an admitted run forward;
    the admission service schedules it right after a run is created."""
    if settings.environment not in {"development", "test"}:
        return None

    store = KnowledgeStore()
    canned: dict[str, str] = {}
    for entry in _DEMO_KNOWLEDGE:
        _publish(store, entry["source_id"], entry["title"], entry["content"])
        # the reply quotes the approved sentence VERBATIM plus its marker, so
        # the exact-match support gate passes honestly (no paraphrase)
        canned[entry["question"]] = _cite(entry["content"])

    tools = build_gateway(knowledge_store=store, audit_sink=lambda _record: None)
    models = ModelGateway(active_provider_id="mock")
    models.register(MockProvider(canned=canned))

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
    )
    return RunExecutor(repository=repository, manager=manager)


__all__ = ["DEMO_TENANT_ID", "build_agent_executor"]
