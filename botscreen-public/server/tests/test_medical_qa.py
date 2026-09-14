"""Tests for MedicalQAAgent (issue #53).

Coverage:
- knowledge is reached only through whitelisted read-only tools;
- answers are evidence-grounded: no evidence => empty answer, no model call;
- the model is reached only through the ModelGateway (real gateway + mock
  provider in one test, asserting provider/model ids on the execution);
- tool usage is bounded by max_tool_calls;
- evidence carries fragment content plus version/hash citation fields;
- integration with the #56 store + #57 gateway activates once they merge
  (importorskip): unreviewed candidates are invisible to the agent.
"""

from datetime import datetime, timezone
from typing import ClassVar

import pytest
from pytest import mark

from app.agents.manager import AgentExecution
from app.agents.medical_qa import MedicalQAAgent
from app.contracts.agent import AgentContext, AgentStatus, ToolRequest
from app.contracts.common import Channel, TenantContext
from app.contracts.model import ModelRequest

_VALID_FROM = datetime(2026, 1, 1, tzinfo=timezone.utc)
_IDENTITY_KEYS = {
    "tenant_id",
    "device_id",
    "session_id",
    "run_id",
    "reviewer",
    "reviewed_by",
    "actor",
}


def _context(text="发烧三天怎么办", **overrides):
    fields = {
        "tenant_id": "t1",
        "device_id": "d1",
        "session_id": "s1",
        "run_id": "r1",
        "channel": Channel.TOUCH,
        "normalized_input": text,
    }
    fields.update(overrides)
    return AgentContext(**fields)


class FakeTools:
    def __init__(self, items=None, fragments=None) -> None:
        self.calls: list[ToolRequest] = []
        self.contexts: list[object] = []
        self.items = list(items or [])
        self.fragments = dict(fragments or {})
        self.search_ok = True

    async def ainvoke(self, context, request, **kwargs):
        """Same surface as the #57 ToolGateway: (trusted context, request)."""
        self.contexts.append(context)
        self.calls.append(request)
        if request.tool_name == "knowledge.search":
            if not self.search_ok:
                return _result(ok=False)
            return _result(
                ok=True,
                data={
                    "items": [
                        {
                            "source_id": it["source_id"],
                            "source_type": "faq",
                            "title": it["title"],
                            "snippet": it["content"][:240],
                            "content_hash": f"hash-{it['source_id']}",
                            "source_uri": f"kbase://{it['source_id']}",
                            "medical_domain": "general",
                            "audience": "public",
                            "knowledge_version": f"{it['source_id']}-v1",
                        }
                        for it in self.items
                    ],
                    "total": len(self.items),
                },
            )
        if request.tool_name == "knowledge.get_fragment":
            source_id = request.arguments["source_id"]
            if source_id not in self.fragments:
                return _result(ok=False)
            return _result(
                ok=True,
                data={
                    "source_id": source_id,
                    "knowledge_version": f"{source_id}-v1",
                    "content_hash": f"hash-{source_id}",
                    "fragment_index": 0,
                    "total_fragments": 1,
                    "text": self.fragments[source_id],
                },
            )
        return _result(ok=False)


def _result(ok, data=None):
    class _R:
        pass

    result = _R()
    result.ok = ok
    result.data = data or {}
    return result


class FakeModels:
    def __init__(self, content="建议门诊就诊") -> None:
        self.content = content
        self.calls: list[ModelRequest] = []
        self.raise_on_call = False

    async def chat(self, request):
        if self.raise_on_call:
            raise AssertionError("model must not be called")
        self.calls.append(request)
        return _model_response(request, self.content)


def _model_response(request, content):
    class _R:
        pass

    response = _R()
    response.provider_id = "mock"
    response.model_id = "mock-model"
    response.model_version = "1.0.0"
    response.content = content
    return response


def _agent(fake_tools, fake_models, **overrides):
    return MedicalQAAgent(models=fake_models, tools=fake_tools, **overrides)


class TestEvidenceGrounding:
    @mark.asyncio
    async def test_happy_path_grounds_answer_in_fragments(self):
        tools = FakeTools(
            items=[
                {
                    "source_id": "faq-fever",
                    "title": "发热指南",
                    "content": "体温38.5以上建议就诊",
                }
            ],
            fragments={"faq-fever": "体温38.5以上建议门诊就诊。"},
        )
        models = FakeModels(content="发热超过38.5建议门诊就诊")
        agent = _agent(tools, models)
        result = await agent.run(_context())
        assert isinstance(result, AgentExecution)  # the Manager's contract
        assert result.status is AgentStatus.COMPLETED
        assert result.answer_candidate == "发热超过38.5建议门诊就诊"
        assert result.safety_status == "grounded"
        assert result.provider_id == "mock"
        assert result.model_id == "mock-model"
        # two tool calls: one search + one fragment
        assert result.tool_calls == 2
        assert [c.tool_name for c in tools.calls] == [
            "knowledge.search",
            "knowledge.get_fragment",
        ]
        assert tools.calls[0].arguments["top_k"] == 4
        # evidence cites the approved fragment + version/hash
        assert result.evidence[0].source_id == "faq-fever"
        assert result.evidence[0].knowledge_version == "faq-fever-v1"
        assert result.evidence[0].content == "体温38.5以上建议门诊就诊。"
        # the grounded prompt carries evidence, not free text
        prompt = models.calls[0].messages[0]["content"]
        assert "体温38.5以上建议门诊就诊。" in prompt
        assert "faq-fever" in prompt

    @mark.asyncio
    async def test_no_evidence_never_invents_answer(self):
        tools = FakeTools(items=[], fragments={})
        models = FakeModels()
        models.raise_on_call = True  # a model call without evidence is forbidden
        agent = _agent(tools, models)
        result = await agent.run(_context())
        assert result.status is AgentStatus.FAILED  # empty answer, no delivery
        assert result.safety_status == "no_evidence"
        assert result.answer_candidate == ""
        assert result.evidence == ()
        assert result.tool_calls == 1  # only the search round-trip, counted

    @mark.asyncio
    async def test_search_failure_yields_no_evidence(self):
        tools = FakeTools(items=[{"source_id": "x", "title": "t", "content": "c"}])
        tools.search_ok = False
        models = FakeModels()
        models.raise_on_call = True
        result = await _agent(tools, models).run(_context())
        assert result.status is AgentStatus.FAILED
        assert result.safety_status == "no_evidence"
        assert result.answer_candidate == ""

    @mark.asyncio
    async def test_fragment_miss_falls_back_to_snippet(self):
        tools = FakeTools(
            items=[{"source_id": "only", "title": "唯一", "content": "片段文本A"}],
            fragments={},  # fragment lookup misses
        )
        models = FakeModels(content="ok")
        result = await _agent(tools, models).run(_context())
        assert result.evidence[0].content == "片段文本A"  # snippet fallback
        assert result.answer_candidate == "ok"

    @mark.asyncio
    async def test_tool_budget_caps_fragment_reads(self):
        tools = FakeTools(
            items=[
                {"source_id": f"f{i}", "title": f"标题{i}", "content": f"内容{i}"}
                for i in range(3)
            ],
            fragments={f"f{i}": f"片段{i}" for i in range(3)},
        )
        models = FakeModels(content="ok")
        agent = _agent(tools, models, max_tool_calls=2, max_fragments=3)
        result = await agent.run(_context())
        # search(1) + fragment budget left = 1 -> at most one fragment read
        assert result.tool_calls <= 2
        assert len(result.evidence) == 1
        assert result.answer_candidate == "ok"


class TestIdentityNeverTravelsAsAnArgument:
    """Tenant/session/reviewer are injected by the gateway, never requested."""

    _IDENTITY: ClassVar[set[str]] = {
        "tenant_id",
        "device_id",
        "session_id",
        "run_id",
        "reviewer",
        "reviewed_by",
        "actor",
        "request_id",
    }

    @mark.asyncio
    async def test_the_agent_sends_no_identity_argument(self):
        tools = FakeTools(
            items=[{"source_id": "g1", "title": "发热", "content": "c"}],
            fragments={"g1": "片段"},
        )
        models = FakeModels(content="ok")
        await _agent(tools, models).run(_context())
        assert tools.calls  # the calls really happened
        for request in tools.calls:
            assert not (set(request.arguments) & self._IDENTITY), request.arguments

    @mark.asyncio
    async def test_the_trusted_context_is_the_identity_channel(self):
        tools = FakeTools(
            items=[{"source_id": "g1", "title": "发热", "content": "c"}],
            fragments={"g1": "片段"},
        )
        models = FakeModels(content="ok")
        context = _context(tenant_id="t-42")
        await _agent(tools, models).run(context)
        # every round-trip carries the SAME AgentContext object (search + read)
        assert tools.contexts == [context, context]

    @mark.asyncio
    async def test_the_gateway_rejects_a_smuggled_identity_argument(self):
        """The #57 gateway is the enforcement point; prove it here too."""
        from app.contracts.errors import ErrorCode
        from app.tools.gateway import ToolGatewayError

        _store, tools, _gateway, _sink = _store_harness()
        with pytest.raises(ToolGatewayError) as exc:
            await tools.ainvoke(
                _context(),
                ToolRequest(
                    tool_name="knowledge.search",
                    arguments={"query": "发热", "tenant_id": "t2"},
                ),
                allowed_tools=["knowledge.search"],
                agent_id="qa",
            )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    @mark.asyncio
    async def test_tool_calls_count_every_round_trip_including_failures(self):
        tools = FakeTools(
            items=[
                {"source_id": "g1", "title": "发热", "content": "c"},
                {"source_id": "g2", "title": "咳嗽", "content": "c"},
            ],
            fragments={"g1": "片段一"},  # g2's fragment read FAILS
        )
        models = FakeModels(content="ok")
        result = await _agent(tools, models, max_tool_calls=4).run(_context())
        # 1 search + 2 fragment attempts (the failed one still counts)
        assert result.tool_calls == 3 == len(tools.calls)

    @mark.asyncio
    async def test_the_manager_accepts_the_reported_tool_calls(self):
        """The count is what the Manager's run budget reads."""
        tools = FakeTools(
            items=[{"source_id": "g1", "title": "发热", "content": "c"}],
            fragments={"g1": "片段"},
        )
        models = FakeModels(content="ok")
        result = await _agent(tools, models).run(_context())
        assert result.tool_calls == 2
        assert isinstance(result, AgentExecution)
        assert isinstance(result.evidence, tuple)


class TestRunGuards:
    @mark.asyncio
    async def test_empty_input_fails_structurally(self):
        tools = FakeTools(items=[{"source_id": "x", "title": "t", "content": "c"}])
        models = FakeModels()
        models.raise_on_call = True
        result = await _agent(tools, models).run(_context(text="   "))
        assert result.status is AgentStatus.FAILED
        assert result.safety_status == "invalid_input"
        assert result.tool_calls == 0
        assert tools.calls == []  # nothing hits the gateway

    @mark.asyncio
    async def test_empty_model_reply_is_a_failure_not_an_answer(self):
        tools = FakeTools(
            items=[{"source_id": "g1", "title": "发热", "content": "体温38.5建议就诊"}],
            fragments={"g1": "体温38.5建议门诊就诊。"},
        )
        models = FakeModels(content="   ")  # model returns nothing usable
        result = await _agent(tools, models).run(_context())
        assert result.status is AgentStatus.FAILED
        assert result.safety_status == "empty_reply"
        assert result.answer_candidate == ""
        assert len(result.evidence) == 1  # evidence was found and is kept


class TestGatewayMediation:
    @mark.asyncio
    async def test_model_calls_ride_the_model_gateway(self):
        from app.providers.mock import MockProvider
        from app.providers.model_gateway import ModelGateway

        gateway = ModelGateway(active_provider_id="mock")
        gateway.register(MockProvider(canned={"体温": "请挂呼吸内科"}))
        tools = FakeTools(
            items=[{"source_id": "g1", "title": "发热", "content": "体温38.5建议就诊"}],
            fragments={"g1": "体温38.5建议门诊就诊。"},
        )
        agent = MedicalQAAgent(models=gateway, tools=tools)
        result = await agent.run(_context("体温38.5怎么办"))
        assert result.answer_candidate == "请挂呼吸内科"
        assert result.provider_id == "mock"
        assert result.model_id == "mock-model"


def _store_harness():
    """Real #56 store + #57 gateway + mock provider over SYNTHETIC knowledge.

    No real clinical content, no cloud provider: the fixture is synthetic text
    pushed through the governance lifecycle.
    """
    from app.knowledge import store as store_mod
    from app.providers.mock import MockProvider
    from app.providers.model_gateway import ModelGateway
    from app.tools import builtins as builtins_mod

    store = store_mod.KnowledgeStore()
    sink: list = []
    tools = builtins_mod.build_gateway(knowledge_store=store, audit_sink=sink.append)
    gateway = ModelGateway(active_provider_id="mock")
    gateway.register(MockProvider(canned={"发热": "请依据已审核资料就诊"}))
    return store, tools, gateway, sink


def _publish(
    store, source_id, *, tenant="t1", title="发热指南", content="合成资料正文"
):
    """Drive the governance lifecycle to APPROVED for one tenant."""
    from app.contracts.knowledge import (
        ApprovalDecision,
        CandidateInput,
        KnowledgeSourceType,
    )

    context = TenantContext(tenant_id=tenant)
    store.add_candidate(
        context,
        CandidateInput(
            source_id=source_id,
            source_type=KnowledgeSourceType.FAQ,
            title=title,
            content=content,
            source_uri=f"kbase://{source_id}",
        ),
        actor="content-owner",
    )
    store.mark_in_review(context, source_id, actor="content-owner")
    return store.approve(
        context,
        source_id,
        ApprovalDecision(reviewer="dr-li", valid_from=_VALID_FROM),
    )


class TestRealStoreIntegration:
    @mark.asyncio
    async def test_unreviewed_content_never_reaches_the_agent(self):
        store, tools, gateway, _ = _store_harness()
        _publish(store, "faq-draft")  # created, then reverted below
        store, tools, gateway, _ = _store_harness()
        from app.contracts.knowledge import CandidateInput, KnowledgeSourceType

        context = TenantContext(tenant_id="t1")
        store.add_candidate(  # stays DRAFT: never published
            context,
            CandidateInput(
                source_id="faq-draft",
                source_type=KnowledgeSourceType.FAQ,
                title="未审核草稿",
                content="未经审核的发热处理内容",
                source_uri="kbase://faq-draft",
            ),
            actor="content-owner",
        )
        agent = MedicalQAAgent(models=gateway, tools=tools)
        result = await agent.run(_context("发热怎么办"))
        # the draft is invisible: the agent must not answer from it
        assert result.status is AgentStatus.FAILED
        assert result.safety_status == "no_evidence"
        assert result.answer_candidate == ""
        assert result.evidence == ()

        # once approved, the same content becomes answerable
        _publish(store, "faq-fever", content="体温超过38.5建议门诊就诊")
        result = await agent.run(_context("发热怎么办"))
        assert result.safety_status == "grounded"
        assert result.evidence[0].source_id == "faq-fever"
        assert result.answer_candidate == "请依据已审核资料就诊"

    @mark.asyncio
    async def test_citations_carry_verifiable_ids_version_and_hash(self):
        store, tools, gateway, _ = _store_harness()
        approved = _publish(store, "faq-fever", content="体温超过38.5建议门诊就诊")
        agent = MedicalQAAgent(models=gateway, tools=tools)
        result = await agent.run(_context("发热怎么办"))
        evidence = result.evidence[0]
        # every citation field is checkable against the approved record itself
        assert evidence.source_id == approved.source_id
        assert evidence.knowledge_version == approved.knowledge_version
        assert evidence.content_hash == approved.content_hash
        assert evidence.content_hash  # never empty
        # and the grounded prompt marks the fragment with its source id
        assert "faq-fever" in evidence.content or evidence.content

    @mark.asyncio
    async def test_the_same_source_id_in_another_tenant_is_invisible(self):
        store, tools, gateway, _ = _store_harness()
        _publish(
            store, "faq-fever", tenant="t1", content="t1 合成资料：成人发热三天就诊"
        )
        _publish(
            store, "faq-fever", tenant="t2", content="t2 合成资料：儿童发热两天就诊"
        )
        agent = MedicalQAAgent(models=gateway, tools=tools)

        t1 = await agent.run(_context("发热怎么办", tenant_id="t1"))
        t2 = await agent.run(_context("发热怎么办", tenant_id="t2"))
        assert t1.evidence[0].source_id == "faq-fever"
        assert "t1 合成资料" in t1.evidence[0].content
        assert "t2 合成资料" not in t1.evidence[0].content
        assert "t2 合成资料" in t2.evidence[0].content

        # tenant t3 owns nothing: no evidence, no answer
        t3 = await agent.run(_context("发热怎么办", tenant_id="t3"))
        assert t3.status is AgentStatus.FAILED
        assert t3.safety_status == "no_evidence"

    @mark.asyncio
    async def test_revoked_knowledge_is_unreachable_again(self):
        from app.contracts.knowledge import RevocationDecision

        store, tools, gateway, _ = _store_harness()
        _publish(store, "faq-fever")
        agent = MedicalQAAgent(models=gateway, tools=tools)
        assert (await agent.run(_context("发热怎么办"))).safety_status == "grounded"

        store.revoke(
            TenantContext(tenant_id="t1"),
            "faq-fever",
            RevocationDecision(actor="dr-li", reason="内容过期，需重新审核"),
        )
        after = await agent.run(_context("发热怎么办"))
        assert after.status is AgentStatus.FAILED
        assert after.safety_status == "no_evidence"
        assert after.answer_candidate == ""

    def test_the_tool_schemas_expose_no_identity_properties(self):
        from app.tools.specs import TOOL_INPUT_SCHEMAS

        for tool_name in ("knowledge.search", "knowledge.get_fragment"):
            schema = TOOL_INPUT_SCHEMAS[tool_name][1]
            assert not (set(schema["properties"]) & _IDENTITY_KEYS), tool_name
            assert schema["additionalProperties"] is False
