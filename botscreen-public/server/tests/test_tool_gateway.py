"""Tests for the ToolGateway (issue #57).

Acceptance coverage:
- non-whitelisted calls are rejected (structured error + audit) and the
  executor never runs — including a hidden-set of dangerous tool names;
- schema/size/timeout violations raise structured errors with audit records;
- unauthorized calls (not in the caller's declared capability set) never
  succeed;
- arguments and results never leak into audit records verbatim.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from app.contracts.agent import ToolRequest
from app.contracts.errors import ErrorCode
from app.tools.builtins import build_gateway
from app.tools.gateway import ToolGateway, ToolGatewayError
from app.tools.specs import READONLY_TOOL_NAMES, WhitelistError, canonical_spec


class _Item:
    """Structurally satisfies what the knowledge builtins read (the #56
    KnowledgeStore carries the same attributes)."""

    def __init__(
        self,
        source_id: str = "faq-1",
        content: str = "内容",
        tenant_id: str = "t1",
        title: str | None = None,
    ) -> None:
        self.source_id = source_id
        self.tenant_id = tenant_id
        self.source_type = "faq"
        self.title = title or f"标题 {source_id}"
        self.content = content
        self.source_uri = f"kbase://faq/{source_id}"
        self.medical_domain = "general"
        self.knowledge_version = f"{source_id}-v1"
        self.content_hash = f"hash-{source_id}"


class _KnowledgeSource:
    """Duck-typed production view with the store's tenant filter semantics."""

    def __init__(self, items) -> None:
        self._items = list(items)

    def production_items(self, tenant_id: str | None = None):
        if tenant_id is None:
            return list(self._items)
        return [it for it in self._items if it.tenant_id == tenant_id]


def _store_with_one_approved(content: str = "内容"):
    return _KnowledgeSource([_Item(content=content)])


class _Sink:
    def __init__(self) -> None:
        self.records = []

    def __call__(self, record) -> None:
        self.records.append(record)


def _request(name: str, arguments: dict | None = None, **overrides):
    fields = {"tool_name": name, "arguments": arguments or {}}
    fields.update(overrides)
    return ToolRequest(**fields)


@pytest.fixture
def gateway():
    sink = _Sink()
    gw = build_gateway(
        knowledge_store=_store_with_one_approved(content="发热咳嗽挂呼吸内科"),
        audit_sink=sink,
        clock=lambda: datetime.now(timezone.utc),
    )
    gw._sink = sink  # type: ignore[attr-defined]
    return gw


def _call(gw, name, arguments=None, allowed=READONLY_TOOL_NAMES, **overrides):
    return gw.invoke(
        _request(name, arguments),
        allowed_tools=list(allowed),
        agent_id="agent-x",
        tenant_id="t1",
        run_id="run-1",
        request_id=overrides.pop("request_id", "req-1"),
        **overrides,
    )


class TestWhitelistIsSealed:
    def test_dangerous_tools_cannot_even_be_declared(self):
        for name in ("shell.exec", "fs.read", "db.query", "web.fetch"):
            with pytest.raises(WhitelistError):
                canonical_spec(name)

    def test_register_requires_executor(self):
        gw = ToolGateway()
        with pytest.raises(ValueError):
            gw.register(canonical_spec("knowledge.search"))

    def test_duplicate_registration_rejected(self, gateway):
        spec = canonical_spec("memory.read_short", executor=lambda args: {})
        with pytest.raises(ToolGatewayError) as exc:
            gateway.register(spec)
        assert exc.value.code is ErrorCode.CONFLICT_IDEMPOTENCY

    def test_whitelist_is_the_canonical_six(self, gateway):
        assert gateway.tools() == list(READONLY_TOOL_NAMES)

    def test_tool_disable_then_call_rejected(self, gateway):
        gateway.disable("knowledge.search")
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {"query": "发热"})
        assert exc.value.code is ErrorCode.TOOL_DISABLED
        gateway.enable("knowledge.search")
        assert _call(gateway, "knowledge.search", {"query": "发热"}).ok is True


class TestWhitelistOutsideCallsRejected:
    @pytest.mark.parametrize(
        "name",
        [
            "fs.read",
            "shell.exec",
            "db.query",
            "web.fetch",
            "file.read",
            "network.http_get",
            "agent.handoff",
            "memory.write",
            "knowledge.write",
            "sql.select",
            "os.system",
        ],
    )
    def test_dangerous_calls_never_succeed(self, gateway, name):
        # even when the caller *declares* the capability, a non-whitelisted
        # name must be rejected before any executor exists
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, name, {"q": "x"}, allowed=[name, "knowledge.search"])
        assert exc.value.code is ErrorCode.TOOL_DISABLED
        assert gateway._sink.records[-1].error_code is ErrorCode.TOOL_DISABLED

    def test_unregistered_canonical_tool_is_disabled(self):
        gw = ToolGateway()  # nothing registered
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "发热"})
        assert exc.value.code is ErrorCode.TOOL_DISABLED


class TestPermissionGate:
    def test_call_outside_allowed_tools_denied(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "memory.read_short", allowed=["knowledge.search"])
        assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN
        assert gateway._sink.records[-1].error_code is ErrorCode.AUTHZ_FORBIDDEN

    def test_empty_declaration_denies_everything(self, gateway):
        for name in READONLY_TOOL_NAMES:
            with pytest.raises(ToolGatewayError) as exc:
                _call(gateway, name, {}, allowed=[])
            assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN

    def test_none_declaration_denies_everything(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            gateway.invoke(
                _request("knowledge.search", {"query": "x"}),
                allowed_tools=None,
                agent_id="agent-x",
                request_id="req-1",
            )
        assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN

    def test_declared_read_tools_work(self, gateway):
        result = _call(
            gateway, "knowledge.search", {"query": "发热"}, allowed=["knowledge.search"]
        )
        assert result.ok is True
        assert result.data["total"] >= 1


class TestSchemaGate:
    def test_missing_required_field_rejected(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        assert "$.query: required" in str(exc.value)

    def test_wrong_type_rejected(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {"query": 123})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_extra_property_rejected(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {"query": "发热", "inject": "x"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_out_of_range_integer_rejected(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {"query": "x", "top_k": 0})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_fragment_index_must_be_non_negative(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(
                gateway,
                "knowledge.get_fragment",
                {"source_id": "faq-1", "fragment_index": -1},
            )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_output_schema_mismatch_rejected(self):
        gw = ToolGateway()
        gw.register(
            canonical_spec(
                "knowledge.search",
                executor=lambda args: ["not", "an", "object"],
            )
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "x"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        assert "$: type" in str(exc.value)

    def test_schema_violation_audited_without_value_echo(self, gateway):
        secret = "s3cr3t-value-never-logged"
        with pytest.raises(ToolGatewayError):
            _call(gateway, "knowledge.search", {"query": secret, "extra": 1})
        record = gateway._sink.records[-1]
        assert record.error_code is ErrorCode.TOOL_SCHEMA_REJECTED
        dumped = record.model_dump_json()
        assert secret not in dumped


class TestSizeGate:
    def test_result_over_limit_rejected_with_audit(self):
        sink = _Sink()
        gw = build_gateway(
            audit_sink=sink,
            default_max_result_bytes=64,
            knowledge_store=_store_with_one_approved(content="x" * 5000),
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "x"})
        assert exc.value.code is ErrorCode.TOOL_OVER_LIMIT
        assert "result_bytes=" in sink.records[-1].result
        assert "exceeded" in str(exc.value)

    def test_oversized_payload_never_reaches_audit_text(self):
        sink = _Sink()
        gw = build_gateway(
            audit_sink=sink,
            default_max_result_bytes=64,
            knowledge_store=_store_with_one_approved(content="超长" * 500),
        )
        with pytest.raises(ToolGatewayError):
            _call(gw, "knowledge.search", {"query": "超长"})
        dumped = "".join(r.model_dump_json() for r in sink.records)
        assert "超长" not in dumped

    def test_under_limit_result_passes(self, gateway):
        result = _call(gateway, "knowledge.search", {"query": "发热"})
        assert result.ok is True
        assert result.data["total"] >= 1


class TestTimeoutGate:
    def test_runaway_executor_times_out(self):
        sink = _Sink()
        release = threading.Event()
        ran = {"done": False}

        def slow(args):
            release.wait(timeout=10)
            ran["done"] = True
            return {"summary": "late"}

        gw = ToolGateway(audit_sink=sink, default_timeout_ms=30)
        gw.register(canonical_spec("memory.read_short", executor=slow))
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short")
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        assert sink.records[-1].error_code is ErrorCode.TOOL_TIMEOUT
        assert ran["done"] is False
        release.set()  # let the daemon thread finish so tests stay tidy
        deadline = time.monotonic() + 5
        while not ran["done"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ran["done"] is True

    def test_deadline_honored_without_explicit_clock(self):
        # gateway defaults to a real UTC clock, so deadlines always apply
        gw = ToolGateway()  # no clock injected
        gw.register(
            canonical_spec("memory.read_short", executor=lambda args: {"summary": "s"})
        )
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                _request("memory.read_short", deadline=past),
                allowed_tools=["memory.read_short"],
                agent_id="a",
                request_id="r",
            )
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT

    def test_injected_runner_timeout_is_structured(self):
        sink = _Sink()

        def always_times_out(fn, seconds):
            raise TimeoutError("faked")

        gw = ToolGateway(audit_sink=sink, runner=always_times_out)
        gw.register(
            canonical_spec("memory.read_short", executor=lambda args: {"summary": "s"})
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short")
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        assert sink.records[-1].error_code is ErrorCode.TOOL_TIMEOUT

    def test_expired_deadline_rejected_before_executor(self, gateway):
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        with pytest.raises(ToolGatewayError) as exc:
            gateway.invoke(
                _request("knowledge.search", {"query": "x"}, deadline=past),
                allowed_tools=["knowledge.search"],
                agent_id="agent-x",
                request_id="req-1",
            )
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        assert gateway._sink.records[-1].error_code is ErrorCode.TOOL_TIMEOUT


class TestDomainAndSuccessPaths:
    def test_knowledge_search_reads_only_production_view(self, gateway):
        result = _call(gateway, "knowledge.search", {"query": "发热"})
        assert result.ok is True
        assert result.data["items"][0]["source_id"] == "faq-1"
        assert result.data["items"][0]["snippet"]
        assert gateway._sink.records[-1].action == "tool.invoke"
        assert gateway._sink.records[-1].result.startswith("ok:result_bytes=")

    def test_search_misses_return_empty(self, gateway):
        result = _call(gateway, "knowledge.search", {"query": "不存在的词"})
        assert result.ok is True
        assert result.data == {"items": [], "total": 0}

    def test_get_fragment_returns_requested_chunk(self):
        store = _KnowledgeSource([_Item(content="片" * 300)])
        sink = _Sink()
        gw = build_gateway(knowledge_store=store, audit_sink=sink)
        result = _call(
            gw,
            "knowledge.get_fragment",
            {"source_id": "faq-1", "fragment_chars": 128, "fragment_index": 1},
        )
        assert result.ok is True
        assert result.data["total_fragments"] == 3
        assert 0 < len(result.data["text"]) <= 128

    def test_get_fragment_unknown_source_raises_registry_code(self, gateway):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.get_fragment", {"source_id": "ghost"})
        assert exc.value.code is ErrorCode.NOT_FOUND_KNOWLEDGE
        assert gateway._sink.records[-1].error_code is ErrorCode.NOT_FOUND_KNOWLEDGE

    def test_directory_search_over_injected_index(self):
        sink = _Sink()
        gw = build_gateway(
            audit_sink=sink,
            directories={
                "department": [
                    {"tenant_id": "t1", "name": "呼吸内科", "location": "1号楼"},
                    {"tenant_id": "t2", "name": "心内科", "location": "2号楼"},
                ]
            },
        )
        result = _call(gw, "department.search", {"query": "呼吸"})
        assert result.data["items"] == [
            {"tenant_id": "t1", "name": "呼吸内科", "location": "1号楼"}
        ]
        # tenant isolation applies when requested
        result = _call(gw, "department.search", {"query": "内科", "tenant_id": "t2"})
        assert [i["name"] for i in result.data["items"]] == ["心内科"]

    def test_memory_read_short_with_reader(self):
        sink = _Sink()
        gw = build_gateway(audit_sink=sink, memory_reader=lambda session: "摘要A")
        result = _call(gw, "memory.read_short", {"session_id": "s1"})
        assert result.data == {"summary": "摘要A", "available": True}

    def test_memory_read_short_default_empty(self, gateway):
        result = _call(gateway, "memory.read_short")
        assert result.data == {"summary": "", "available": False}

    def test_success_audit_never_echoes_results(self, gateway):
        result = _call(gateway, "knowledge.search", {"query": "发热"})
        assert result.ok
        record = gateway._sink.records[-1]
        dumped = record.model_dump_json()
        assert "发热" not in dumped
        assert record.actor_id_hash  # agent id is hashed, never raw
        assert record.tool_names == ["knowledge.search"]
        assert record.error_code is None

    def test_every_failure_writes_exactly_one_audit_record(self, gateway):
        failures = [
            lambda: _call(gateway, "fs.read"),
            lambda: _call(gateway, "knowledge.search", {"query": 1}),
            lambda: _call(gateway, "memory.read_short", allowed=[]),
            lambda: _call(gateway, "knowledge.get_fragment", {"source_id": "ghost"}),
        ]
        before = len(gateway._sink.records)
        for attempt in failures:
            with pytest.raises(ToolGatewayError):
                attempt()
        assert len(gateway._sink.records) == before + len(failures)


class TestRealStoreIntegration:
    def test_builtins_work_with_the_governance_store_when_available(self):
        """Active once #56 lands; skipped while the store is off-branch."""
        ks = pytest.importorskip("app.knowledge.store")
        kc = pytest.importorskip("app.contracts.knowledge")
        store = ks.KnowledgeStore()
        store.add_candidate(
            kc.KnowledgeItem(
                source_id="faq-1",
                tenant_id="t1",
                source_type=kc.KnowledgeSourceType.FAQ,
                title="发热指南",
                content="发热咳嗽请挂呼吸内科门诊",
                source_uri="kbase://faq/1",
            )
        )
        store.approve("faq-1", reviewer="dr-li")
        sink = _Sink()
        gw = build_gateway(knowledge_store=store, audit_sink=sink)
        result = _call(gw, "knowledge.search", {"query": "发热"})
        assert result.ok is True
        assert result.data["items"][0]["source_id"] == "faq-1"
        assert result.data["items"][0]["knowledge_version"] == "faq-1-v1"
        fragment = _call(gw, "knowledge.get_fragment", {"source_id": "faq-1"})
        assert fragment.data["total_fragments"] == 1


class TestAliasSafety:
    def test_canonical_specs_own_their_schemas(self):
        a = canonical_spec("knowledge.search", executor=lambda args: {})
        b = canonical_spec("knowledge.search", executor=lambda args: {})
        a.input_schema["required"] = []
        assert b.input_schema["required"] == ["query"]  # decl uncorrupted
        from app.tools.specs import TOOL_INPUT_SCHEMAS

        assert TOOL_INPUT_SCHEMAS["knowledge.search"][1]["required"] == ["query"]

    def test_registered_spec_is_isolated_from_caller_mutation(self):
        gw = ToolGateway()
        spec = canonical_spec("knowledge.search", executor=lambda args: {})
        gw.register(spec)
        spec.input_schema["required"] = []  # caller mutates its own copy
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED  # still required

    def test_spec_returns_deep_copy(self):
        gw = ToolGateway()
        gw.register(canonical_spec("knowledge.search", executor=lambda args: {}))
        got = gw.spec("knowledge.search")
        got.input_schema["required"] = []
        assert gw.spec("knowledge.search").input_schema["required"] == ["query"]
        assert gw.spec("ghost") is None


class TestDomainExecutorSpy:
    def test_executor_never_runs_on_rejected_or_unauthorized_calls(self):
        calls = []

        def spy(args):
            calls.append(args)
            return {"summary": "x"}

        sink = _Sink()
        gw = ToolGateway(audit_sink=sink)
        gw.register(canonical_spec("memory.read_short", executor=spy))
        # schema violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                _request("memory.read_short", {"session_id": 9}),
                allowed_tools=["memory.read_short"],
                agent_id="a",
                request_id="r",
            )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        # permission violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                _request("memory.read_short"),
                allowed_tools=[],
                agent_id="a",
                request_id="r",
            )
        assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN
        # whitelist violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                _request("shell.exec"),
                allowed_tools=["shell.exec"],
                agent_id="a",
                request_id="r",
            )
        assert exc.value.code is ErrorCode.TOOL_DISABLED
        assert calls == []  # unauthorized tool invocations succeeded 0 times
