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

import pytest
from pytest import mark

from app.agents.medical_qa import MedicalQAAgent, QAExecution
from app.contracts.agent import AgentContext, AgentStatus, ToolRequest
from app.contracts.common import Channel
from app.contracts.model import ModelRequest


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
        self.items = list(items or [])
        self.fragments = dict(fragments or {})
        self.search_ok = True

    async def invoke(self, request, **kwargs):
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
        assert isinstance(result, QAExecution)
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
        assert result.safety_status == "no_evidence"
        assert result.answer_candidate == ""
        assert result.evidence == []
        assert result.tool_calls == 1  # only the failed search round-trip

    @mark.asyncio
    async def test_search_failure_yields_no_evidence(self):
        tools = FakeTools(items=[{"source_id": "x", "title": "t", "content": "c"}])
        tools.search_ok = False
        models = FakeModels()
        models.raise_on_call = True
        result = await _agent(tools, models).run(_context())
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


class TestRealStoreIntegration:
    @mark.asyncio
    async def test_unreviewed_content_never_reaches_the_agent(self):
        """Active once #56+#57 land; skipped while they are off-branch."""
        store_mod = pytest.importorskip("app.knowledge.store")
        contracts_mod = pytest.importorskip("app.contracts.knowledge")
        builtins_mod = pytest.importorskip("app.tools.builtins")
        pytest.importorskip("app.tools.gateway")
        from app.providers.mock import MockProvider
        from app.providers.model_gateway import ModelGateway

        store = store_mod.KnowledgeStore()
        draft = contracts_mod.KnowledgeItem(
            source_id="faq-draft",
            tenant_id="t1",
            source_type=contracts_mod.KnowledgeSourceType.FAQ,
            title="未审核草稿",
            content="未经审核的发热处理内容",
            source_uri="kbase://faq-draft",
        )
        store.add_candidate(draft)  # stays DRAFT — never published
        sink = []
        tools = builtins_mod.build_gateway(
            knowledge_store=store, audit_sink=sink.append
        )
        gateway = ModelGateway(active_provider_id="mock")
        gateway.register(MockProvider(canned={"发热": "仅当有已审核资料才作答"}))

        agent = MedicalQAAgent(models=gateway, tools=tools)
        result = await agent.run(_context("发热怎么办"))
        # the draft is invisible: the agent must not answer from it
        assert result.safety_status == "no_evidence"
        assert result.answer_candidate == ""
        assert result.evidence == []

        # once approved, the same content becomes answerable
        store.approve("faq-draft", reviewer="dr-li")
        result = await agent.run(_context("发热怎么办"))
        assert result.safety_status == "grounded"
        assert result.evidence[0].source_id == "faq-draft"
        assert result.answer_candidate == "仅当有已审核资料才作答"
