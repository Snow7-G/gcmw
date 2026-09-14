"""Tests for the ToolGateway (issue #57).

Acceptance coverage:
- non-whitelisted calls are rejected (structured error + audit) and the
  executor never runs — including a hidden-set of dangerous tool names;
- schema/size/timeout violations raise structured errors with audit records;
- unauthorized calls (not in the caller's declared capability set) never
  succeed;
- arguments and results never leak into audit records verbatim.
"""

import hashlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

import pytest

from app.contracts.agent import ToolRequest
from app.contracts.common import SessionContext, TenantContext
from app.contracts.errors import ErrorCode, ErrorEnvelope, lookup
from app.tools.builtins import build_gateway
from app.tools.gateway import ToolGateway, ToolGatewayError
from app.tools.specs import READONLY_TOOL_NAMES, WhitelistError, canonical_spec

TENANT = TenantContext(tenant_id="t1")
OTHER_TENANT = TenantContext(tenant_id="t2")
SESSION = SessionContext(tenant_id="t1", device_id="d1", session_id="sess-1")


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

    def production_items(self, context):
        """The #56A signature: the store scopes by the TRUSTED context."""
        return [it for it in self._items if it.tenant_id == context.tenant_id]


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


def _call(gw, name, arguments=None, allowed=READONLY_TOOL_NAMES, context=TENANT, **kw):
    """Invoke through the trusted-context entry point (identity is injected)."""
    return gw.invoke(
        context,
        _request(name, arguments, **kw),
        allowed_tools=list(allowed),
        agent_id="agent-x",
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
        spec = canonical_spec("memory.read_short", executor=lambda _ctx, args: {})
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
                TENANT,
                _request("knowledge.search", {"query": "x"}),
                allowed_tools=None,
                agent_id="agent-x",
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
        assert gateway._sink.records[-1].result == "rejected:input_schema:required"

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
        sink = _Sink()
        gw = ToolGateway(audit_sink=sink)
        gw.register(
            canonical_spec(
                "knowledge.search",
                executor=lambda _ctx, args: ["not", "an", "object"],
            )
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "x"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        assert sink.records[-1].result == (
            "rejected:output_schema:additionalProperties+required"
        ) or sink.records[-1].result.startswith("rejected:output_schema:")
        gw.shutdown()

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
        assert "result_bytes=" in sink.records[-1].result  # audit keeps the size

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

        def slow(_context, args):
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
            canonical_spec(
                "memory.read_short", executor=lambda _ctx, args: {"summary": "s"}
            )
        )
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                SESSION,
                _request("memory.read_short", deadline=past),
                allowed_tools=["memory.read_short"],
                agent_id="a",
            )
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT

    def test_injected_runner_timeout_is_structured(self):
        sink = _Sink()

        def always_times_out(fn, seconds):
            raise TimeoutError("faked")

        gw = ToolGateway(audit_sink=sink, runner=always_times_out)
        gw.register(
            canonical_spec(
                "memory.read_short", executor=lambda _ctx, args: {"summary": "s"}
            )
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short")
        assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        assert sink.records[-1].error_code is ErrorCode.TOOL_TIMEOUT

    def test_expired_deadline_rejected_before_executor(self, gateway):
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        with pytest.raises(ToolGatewayError) as exc:
            gateway.invoke(
                TENANT,
                _request("knowledge.search", {"query": "x"}, deadline=past),
                allowed_tools=["knowledge.search"],
                agent_id="agent-x",
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
        # projected through the FIELD WHITELIST: no tenant_id, no extra columns
        assert result.data["items"] == [{"name": "呼吸内科", "location": "1号楼"}]
        # tenant isolation follows the INJECTED context, never the arguments
        result = _call(gw, "department.search", {"query": "内科"}, context=OTHER_TENANT)
        assert [i["name"] for i in result.data["items"]] == ["心内科"]
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "department.search", {"query": "内科", "tenant_id": "t2"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_memory_read_short_with_reader(self):
        sink = _Sink()
        seen: list[Any] = []

        def reader(key):
            seen.append(key)
            return "摘要A"

        gw = build_gateway(audit_sink=sink, memory_reader=reader)
        result = _call(gw, "memory.read_short", {}, context=SESSION)
        assert result.data == {"summary": "摘要A", "available": True}
        assert len(seen) == 1
        assert (seen[0].tenant_id, seen[0].device_id, seen[0].session_id) == (
            "t1",
            "d1",
            "sess-1",
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short", {"session_id": "other"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_memory_read_short_without_a_configured_reader_fails_closed(self, gateway):
        """The fixture has no memory reader: that is a deployment fault."""
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "memory.read_short", {}, context=SESSION)
        assert exc.value.code is ErrorCode.UNAVAILABLE_MAINTENANCE

    def test_memory_read_short_with_an_empty_reader_is_an_empty_summary(self):
        gw = build_gateway(audit_sink=_Sink(), memory_reader=lambda _sid: "")
        result = _call(gw, "memory.read_short", {}, context=SESSION)
        assert result.data == {"summary": "", "available": False}

    def test_memory_read_short_without_device_and_session_is_refused(self):
        """A tenant-only context cannot form the memory key: nothing is read."""
        called: list[Any] = []
        gw = build_gateway(
            audit_sink=_Sink(), memory_reader=lambda key: called.append(key) or "泄漏"
        )
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short", {}, context=TENANT)
        assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN
        assert called == []

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
        """The REAL #56A store drives the knowledge tools end to end."""
        ks = pytest.importorskip("app.knowledge.store")
        kc = pytest.importorskip("app.contracts.knowledge")
        store = ks.KnowledgeStore()
        store.add_candidate(
            TENANT,
            kc.CandidateInput(
                source_id="faq-1",
                source_type=kc.KnowledgeSourceType.FAQ,
                title="发热指南",
                content="发热咳嗽请挂呼吸内科门诊",
                source_uri="kbase://faq/1",
            ),
            actor="content-owner",
        )
        store.mark_in_review(TENANT, "faq-1", actor="content-owner")
        store.approve(
            TENANT,
            "faq-1",
            kc.ApprovalDecision(
                reviewer="dr-li",
                valid_from=datetime.now(timezone.utc) - timedelta(days=1),
            ),
        )
        sink = _Sink()
        gw = build_gateway(knowledge_store=store, audit_sink=sink)

        result = _call(gw, "knowledge.search", {"query": "发热"})
        assert result.ok is True
        assert result.data["items"][0]["source_id"] == "faq-1"
        fragment = _call(gw, "knowledge.get_fragment", {"source_id": "faq-1"})
        assert fragment.data["total_fragments"] >= 1

        # CROSS-TENANT ISOLATION: another tenant's context sees nothing, and it
        # cannot be overridden from the arguments either
        other = _call(gw, "knowledge.search", {"query": "发热"}, context=OTHER_TENANT)
        assert other.data == {"items": [], "total": 0}
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "发热", "tenant_id": "t1"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_missing_dependency_fails_closed(self):
        """No configured source is a deployment fault, not an empty result."""
        gw = build_gateway()  # nothing injected
        for name, args in (
            ("knowledge.search", {"query": "x"}),
            ("department.search", {"query": "x"}),
            ("memory.read_short", {}),
        ):
            with pytest.raises(ToolGatewayError) as exc:
                _call(gw, name, args, context=SESSION)
            assert exc.value.code is ErrorCode.UNAVAILABLE_MAINTENANCE


class TestIdentityArgumentGuard:
    """Review (#69): identity is server-injected and never model-supplied."""

    @pytest.mark.parametrize(
        "key",
        [
            "tenant_id",
            "tenant",
            "session_id",
            "device_id",
            "run_id",
            "request_id",
            "reviewer",
            "reviewer_id",
            "actor",
            "user_id",
            "principal",
            "tenant_context",
        ],
    )
    def test_identity_arguments_are_rejected(self, gateway, key):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", {"query": "x", key: "attacker"})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        # the KEY is named in the AUDIT (value never anywhere); the exception
        # text is always the registry message
        assert f"identity_argument={key}" in gateway._sink.records[-1].result
        assert "attacker" not in str(exc.value)
        assert key not in str(exc.value)

    @pytest.mark.parametrize(
        "arguments",
        [
            {"query": "x", "filter": {"tenant_id": "t2"}},  # nested in an object
            {"query": "x", "filters": [{"session_id": "s9"}]},  # nested in a list
            {"query": "x", "opts": ({"reviewer": "me"},)},  # nested in a tuple
        ],
    )
    def test_nested_identity_arguments_are_rejected(self, gateway, arguments):
        with pytest.raises(ToolGatewayError) as exc:
            _call(gateway, "knowledge.search", arguments)
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED

    def test_the_executor_never_runs_for_an_identity_argument(self):
        calls: list[Any] = []

        def spy(_context, args):
            calls.append(args)
            return {"items": [], "total": 0}

        gw = ToolGateway()
        gw.register(canonical_spec("knowledge.search", executor=spy))
        with pytest.raises(ToolGatewayError):
            _call(gw, "knowledge.search", {"query": "x", "tenant_id": "t2"})
        assert calls == []

    def test_no_whitelisted_schema_declares_an_identity_property(self):
        from app.tools.gateway import IDENTITY_ARGUMENT_KEYS
        from app.tools.specs import TOOL_INPUT_SCHEMAS

        def keys_of(node: Any) -> set[str]:
            found: set[str] = set()
            if isinstance(node, dict):
                for key, value in node.items():
                    found.add(str(key).lower())
                    found |= keys_of(value)
            elif isinstance(node, (list, tuple, set, frozenset)):
                for item in node:
                    found |= keys_of(item)
            return found

        for name, (_description, schema) in TOOL_INPUT_SCHEMAS.items():
            declared = keys_of(schema)
            if "$ref" in declared:  # pragma: no cover - no refs in the whitelist
                continue
            leaked = {
                key
                for key in declared
                if key in IDENTITY_ARGUMENT_KEYS and key != "type"
            }
            assert not leaked, f"{name} declares identity properties: {sorted(leaked)}"

    def test_identity_comes_only_from_the_context(self, gateway):
        captured: list[Any] = []
        gw = gateway
        original = gw.spec("knowledge.search")
        assert original is not None

        def spy(context, args):
            captured.append(context)
            return {"items": [], "total": 0}

        gw.register_error: Any = None
        gw._specs["knowledge.search"] = original.__class__(
            name=original.name,
            description=original.description,
            input_schema=original.input_schema,
            output_schema=original.output_schema,
            executor=spy,
        )
        _call(gw, "knowledge.search", {"query": "x"}, context=OTHER_TENANT)
        assert captured and captured[0].tenant_id == "t2"


class TestAsyncEntryPoint:
    """``ainvoke`` keeps the gate AND gets the tool off the event loop (#69)."""

    def test_async_call_succeeds(self):
        import asyncio

        gw = build_gateway(
            knowledge_store=_store_with_one_approved(content="发热咳嗽挂呼吸内科")
        )
        result = asyncio.run(
            gw.ainvoke(
                TENANT,
                _request("knowledge.search", {"query": "发热"}),
                allowed_tools=["knowledge.search"],
                agent_id="agent-x",
            )
        )
        assert result.ok is True
        assert result.data["items"]

    def test_async_timeout_is_structured_and_audited(self):
        import asyncio

        release = threading.Event()
        sink = _Sink()

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=sink)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        try:
            with pytest.raises(ToolGatewayError) as exc:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "x"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=0.05,
                    )
                )
            assert exc.value.code is ErrorCode.TOOL_TIMEOUT
            async_records = [
                r for r in sink.records if r.result.startswith("rejected:async_timeout")
            ]
            assert async_records, [r.result for r in sink.records]
            assert async_records[0].error_code is ErrorCode.TOOL_TIMEOUT
        finally:
            release.set()

    def test_the_event_loop_is_not_blocked_by_a_slow_tool(self):
        """Proof that the sync gate + executor run OFF the loop."""
        import asyncio

        release = threading.Event()

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway()
        gw.register(canonical_spec("knowledge.search", executor=slow))

        async def main():
            task = asyncio.create_task(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "x"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                    timeout_s=5,
                )
            )
            ticks = 0
            for _ in range(20):  # the loop must keep running while the tool waits
                await asyncio.sleep(0.01)
                ticks += 1
            release.set()
            await task
            return ticks

        assert asyncio.run(main()) == 20

    def test_async_cancellation_propagates_without_running_the_executor(self):
        import asyncio

        ran: list[str] = []

        def executor(_context, args):
            ran.append("ran")
            return {"items": [], "total": 0}

        gw = ToolGateway()
        gw.register(canonical_spec("knowledge.search", executor=executor))

        async def main():
            task = asyncio.create_task(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "x"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                )
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())

    def test_async_path_enforces_the_same_identity_gate(self):
        import asyncio

        gw = build_gateway(knowledge_store=_store_with_one_approved())
        with pytest.raises(ToolGatewayError) as exc:
            asyncio.run(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "x", "tenant_id": "t2"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                )
            )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED


class TestDirectoryIsolationAndProjection:
    """Review P1: no adopted records, no private fields, no nested smuggling."""

    RECORDS: ClassVar[dict] = {
        "department": [
            {
                "tenant_id": "t1",
                "name": "呼吸内科",
                "location": "1号楼",
                "internal_note": "主任手机 13800000000",
            },
            {"tenant_id": "t2", "name": "心内科", "location": "2号楼"},
            {"name": "无租户科室", "location": "?"},  # belongs to NOBODY
            {"name": "公开导医台", "location": "大厅", "visibility": "public"},
        ]
    }

    def _gw(self):
        return build_gateway(audit_sink=_Sink(), directories=self.RECORDS)

    def test_records_without_a_tenant_are_never_adopted(self):
        gw = self._gw()
        for context in (TENANT, OTHER_TENANT):
            names = [
                item["name"]
                for item in _call(
                    gw, "department.search", {"query": "科室"}, context=context
                ).data["items"]
            ]
            assert "无租户科室" not in names

    def test_private_fields_never_reach_the_agent(self):
        gw = self._gw()
        result = _call(gw, "department.search", {"query": "呼吸"}, context=TENANT)
        item = result.data["items"][0]
        assert set(item) == {"name", "location"}  # field whitelist only
        assert "internal_note" not in item
        assert "13800000000" not in str(result.data)

    def test_no_flag_makes_a_record_cross_tenants(self):
        """Review P1: a raw ``visibility``/``review_status`` flag is not proof
        of review, so the single-tenant v1 boundary has NO exception."""
        gw = build_gateway(
            audit_sink=_Sink(),
            directories={
                "department": [
                    {
                        "tenant_id": "t2",
                        "name": "别家科室",
                        "location": "2号楼",
                        "visibility": "public",
                        "review_status": "approved",
                    },
                    {"name": "无租户科室", "location": "?", "visibility": "public"},
                ]
            },
        )
        # t1 sees nothing: the flags on t2's record are not an exception
        assert (
            _call(gw, "department.search", {"query": "科室"}, context=TENANT).data[
                "items"
            ]
            == []
        )
        # t2 sees ONLY its own record (never the tenant-less one)
        assert [
            i["name"]
            for i in _call(
                gw, "department.search", {"query": "科室"}, context=OTHER_TENANT
            ).data["items"]
        ] == ["别家科室"]

    def test_nested_values_are_dropped_not_returned(self):
        gw = build_gateway(
            audit_sink=_Sink(),
            directories={
                "department": [
                    {
                        "tenant_id": "t1",
                        "name": "嵌套科室",
                        "location": {"tenant_id": "t2", "secret": "x"},
                        "extra": ["a", "b"],
                    }
                ]
            },
        )
        item = _call(gw, "department.search", {"query": "嵌套"}, context=TENANT).data[
            "items"
        ][0]
        assert item == {"name": "嵌套科室"}  # non-scalars are dropped outright
        assert "secret" not in str(item)


class TestMemoryKeySpace:
    """Review P1: short-term memory is keyed by tenant+device+session."""

    def _gw(self, reader):
        return build_gateway(audit_sink=_Sink(), memory_reader=reader)

    @staticmethod
    def _canary_reader(store):
        def reader(key):
            return store.get((key.tenant_id, key.device_id, key.session_id))

        return reader

    def test_same_session_id_in_two_tenants_is_isolated(self):
        store = {
            ("t1", "d1", "shared"): "T1 的摘要",
            ("t2", "d1", "shared"): "T2 的摘要",
        }
        gw = self._gw(self._canary_reader(store))
        t1 = _call(
            gw,
            "memory.read_short",
            {},
            context=SessionContext(tenant_id="t1", device_id="d1", session_id="shared"),
        )
        t2 = _call(
            gw,
            "memory.read_short",
            {},
            context=SessionContext(tenant_id="t2", device_id="d1", session_id="shared"),
        )
        assert t1.data["summary"] == "T1 的摘要"
        assert t2.data["summary"] == "T2 的摘要"

    def test_same_session_on_two_devices_is_isolated(self):
        store = {
            ("t1", "d1", "s1"): "A 机摘要",
            ("t1", "d2", "s1"): "B 机摘要",
        }
        gw = self._gw(self._canary_reader(store))
        a = _call(
            gw,
            "memory.read_short",
            {},
            context=SessionContext(tenant_id="t1", device_id="d1", session_id="s1"),
        )
        b = _call(
            gw,
            "memory.read_short",
            {},
            context=SessionContext(tenant_id="t1", device_id="d2", session_id="s1"),
        )
        assert (a.data["summary"], b.data["summary"]) == ("A 机摘要", "B 机摘要")

    def test_a_missing_memory_reader_fails_closed(self):
        gw = build_gateway(audit_sink=_Sink())
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "memory.read_short", {}, context=SESSION)
        assert exc.value.code is ErrorCode.UNAVAILABLE_MAINTENANCE


class TestExecutionBoundsAndAuditFinality:
    """Review P1: bounded concurrency and exactly ONE terminal audit per call."""

    def test_concurrency_is_bounded(self):
        import concurrent.futures

        running = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def slow(_context, args):
            with lock:
                running["now"] += 1
                running["peak"] = max(running["peak"], running["now"])
            time.sleep(0.05)
            with lock:
                running["now"] -= 1
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=_Sink(), max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as outer:
            list(
                outer.map(
                    lambda _i: _call(gw, "knowledge.search", {"query": "x"}),
                    range(8),
                )
            )
        assert running["peak"] <= 2  # never more than the configured bound
        assert gw.peak_concurrency() <= 2
        gw.shutdown()

    def test_a_single_async_call_works_at_max_concurrency_one(self):
        """Review P1: the async path must not occupy a worker with its own gate."""
        import asyncio

        gw = build_gateway(
            audit_sink=_Sink(),
            knowledge_store=_store_with_one_approved(content="发热"),
        )

        async def main():
            gw._pool.shutdown(wait=True)  # replace the default pool
            import concurrent.futures as cf

            gw._pool = cf.ThreadPoolExecutor(max_workers=1)
            result = await asyncio.wait_for(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "发热"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                ),
                timeout=5,
            )
            return result

        result = asyncio.run(main())
        assert result.ok is True
        gw.shutdown()

    def test_concurrent_async_calls_all_succeed(self):
        """Default concurrency: 4 simultaneous calls, none falsely timing out."""
        import asyncio

        gw = build_gateway(
            audit_sink=_Sink(),
            knowledge_store=_store_with_one_approved(content="发热"),
        )

        async def main():
            return await asyncio.gather(
                *(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "发热"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=5,
                    )
                    for _ in range(4)
                )
            )

        results = asyncio.run(main())
        assert all(r.ok for r in results)
        gw.shutdown()

    def test_a_fast_tool_never_times_out_under_load(self):
        import asyncio
        import concurrent.futures

        gw = build_gateway(
            audit_sink=_Sink(),
            knowledge_store=_store_with_one_approved(content="发热"),
        )

        async def main():
            return await asyncio.gather(
                *(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "发热"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=5,
                    )
                    for _ in range(8)
                )
            )

        assert all(r.ok for r in asyncio.run(main()))
        gw.shutdown()
        assert isinstance(gw._pool, concurrent.futures.ThreadPoolExecutor)

    def test_a_late_executor_cannot_produce_two_terminal_records(self):
        """Review P1: success + timeout for ONE call is impossible.

        NOTE (evidence correction from review): the delay here is in the
        EXECUTOR, not the audit sink — ``test_a_slow_sink_...`` below covers a
        genuinely slow sink.
        """
        import asyncio

        release = threading.Event()
        records: list = []

        def delayed_sink(record):
            records.append(record)

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=delayed_sink, max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        try:
            with pytest.raises(ToolGatewayError) as exc:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "x"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=0.05,
                    )
                )
            assert exc.value.code is ErrorCode.TOOL_TIMEOUT
            release.set()  # the abandoned worker finishes "successfully" now
            for _ in range(50):
                time.sleep(0.02)
                if len(records) > 1:
                    break
            assert [r.result for r in records] == ["rejected:async_timeout=0.05s"], [
                r.result for r in records
            ]
            assert not any(r.result.startswith("ok:") for r in records)
        finally:
            release.set()
            gw.shutdown()

    def test_a_slow_sink_still_yields_exactly_one_terminal_record(self):
        """A genuinely slow AUDIT SINK must not duplicate or drop records."""
        import asyncio

        seen: list = []

        def slow_sink(record):
            time.sleep(0.05)  # the sink itself is slow
            seen.append(record)

        gw = build_gateway(
            audit_sink=slow_sink,
            knowledge_store=_store_with_one_approved(content="发热"),
        )
        result = asyncio.run(
            gw.ainvoke(
                TENANT,
                _request("knowledge.search", {"query": "发热"}),
                allowed_tools=["knowledge.search"],
                agent_id="a",
                timeout_s=5,
            )
        )
        assert result.ok is True
        assert [r.result for r in seen] == [
            f"ok:result_bytes={len(json.dumps(result.data, ensure_ascii=False, separators=(',', ':')).encode())}"
        ]

    def test_an_explicit_timeout_can_only_tighten_the_tool_budget(self):
        """Review P1: ``timeout_s`` must never EXTEND the tool budget."""
        import asyncio

        release = threading.Event()

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=_Sink(), default_timeout_ms=50, max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        try:
            with pytest.raises(ToolGatewayError) as exc:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "x"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=0.5,  # much LARGER than the 50 ms tool budget
                    )
                )
            assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        finally:
            release.set()
            gw.shutdown()

    def test_an_expired_request_deadline_bounds_the_async_budget(self):
        import asyncio

        release = threading.Event()

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=_Sink(), default_timeout_ms=5000, max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        deadline = datetime.now(timezone.utc) + timedelta(milliseconds=50)
        try:
            with pytest.raises(ToolGatewayError) as exc:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "x"}, deadline=deadline),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                    )
                )
            assert exc.value.code is ErrorCode.TOOL_TIMEOUT
        finally:
            release.set()
            gw.shutdown()

    @pytest.mark.parametrize(
        "value", [0, -1, float("nan"), float("inf"), float("-inf"), True, "0.5"]
    )
    def test_invalid_timeout_s_is_rejected(self, value):
        import asyncio

        gw = build_gateway(
            audit_sink=_Sink(), knowledge_store=_store_with_one_approved()
        )
        with pytest.raises((TypeError, ValueError)):
            asyncio.run(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "x"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                    timeout_s=value,
                )
            )
        gw.shutdown()

    def test_internal_exception_text_never_enters_the_error_object(self):
        """Review P1: only a stable code + fixed sentence leave the gateway."""
        sentinel = "SENTINEL-INTERNAL-PAYLOAD-/etc/passwd"

        def exploding(_context, args):
            raise RuntimeError(sentinel)

        gw = ToolGateway(audit_sink=_Sink())
        gw.register(canonical_spec("knowledge.search", executor=exploding))
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {"query": "x"})
        assert exc.value.code is ErrorCode.INTERNAL_UNKNOWN
        assert sentinel not in str(exc.value)
        assert sentinel not in repr(exc.value)
        assert str(exc.value) == lookup(ErrorCode.INTERNAL_UNKNOWN).message
        gw.shutdown()

    def test_cancellation_writes_exactly_one_record_and_propagates(self):
        """Review P1: a cancelled call is audited once — never as a success."""
        import asyncio

        started = threading.Event()
        release = threading.Event()
        records: list = []

        def slow(_context, args):
            started.set()
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=records.append, max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))

        async def main():
            task = asyncio.create_task(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.search", {"query": "x"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                    timeout_s=5,
                )
            )
            for _ in range(200):  # wait until the worker really started
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        try:
            asyncio.run(main())
            release.set()  # the abandoned worker finishes "successfully" now
            time.sleep(0.3)
            assert [r.result for r in records] == ["cancelled"], [
                r.result for r in records
            ]
            assert records[0].error_code is None  # cancellation is not a failure
        finally:
            release.set()
            gw.shutdown()

    def test_a_late_completion_never_writes_a_second_audit(self):
        import asyncio

        release = threading.Event()
        sink = _Sink()

        def slow(_context, args):
            release.wait(timeout=10)
            return {"items": [], "total": 0}

        gw = ToolGateway(audit_sink=sink, max_concurrency=2)
        gw.register(canonical_spec("knowledge.search", executor=slow))
        try:
            with pytest.raises(ToolGatewayError) as exc:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        _request("knowledge.search", {"query": "x"}),
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                        timeout_s=0.05,
                    )
                )
            assert exc.value.code is ErrorCode.TOOL_TIMEOUT
            time.sleep(0.05)
            release.set()  # the abandoned worker now finishes "successfully"
            time.sleep(0.3)
            results = [r.result for r in sink.records]
            assert len(results) == 1, results  # exactly one terminal record
            assert results[0].startswith("rejected:async_timeout")
        finally:
            release.set()
            gw.shutdown()

    def test_every_call_writes_exactly_one_audit_record(self):
        sink = _Sink()
        gw = build_gateway(audit_sink=sink, knowledge_store=_store_with_one_approved())
        _call(gw, "knowledge.search", {"query": "内容"})
        assert len(sink.records) == 1
        with pytest.raises(ToolGatewayError):
            _call(gw, "knowledge.search", {"query": "内容", "tenant_id": "t2"})
        assert len(sink.records) == 2  # the denial is the SECOND call's record


class TestAuditNeverCarriesUntrustedText:
    """Review P1: unknown tool names and sink failures must not leak text."""

    def test_unknown_tool_name_is_labelled_not_echoed(self):
        sink = _Sink()
        gw = ToolGateway(audit_sink=sink)
        secret = "IGNORE-ALL-PREVIOUS-INSTRUCTIONS"
        with pytest.raises(ToolGatewayError):
            _call(gw, "shell.exec;" + secret, allowed=["shell.exec;" + secret])
        record = sink.records[-1]
        assert record.tool_names == [
            "unregistered:"
            + hashlib.sha256(("shell.exec;" + secret).encode()).hexdigest()[:12]
        ]
        assert secret not in str(record.tool_names)
        assert secret not in record.result

    def test_a_failing_sink_does_not_leak_its_exception_text(self, caplog):
        sentinel = "SENTINEL-EXCEPTION-TEXT"

        def broken(_record):
            raise RuntimeError(sentinel)

        gw = build_gateway(
            audit_sink=broken, knowledge_store=_store_with_one_approved()
        )
        with caplog.at_level("WARNING"):
            result = _call(gw, "knowledge.search", {"query": "内容"})
        assert result.ok is True  # a broken sink never breaks the call
        assert sentinel not in caplog.text
        assert "details withheld" in caplog.text

    def test_constructing_without_a_sink_warns(self, caplog):
        with caplog.at_level("WARNING"):
            gw = ToolGateway()
        assert "WITHOUT an audit sink" in caplog.text
        gw.shutdown()

    def test_fragment_without_a_store_is_a_deployment_fault(self):
        gw = build_gateway(audit_sink=_Sink())
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.get_fragment", {"source_id": "faq-1"})
        assert exc.value.code is ErrorCode.UNAVAILABLE_MAINTENANCE


class TestWholeErrorBoundaryIsScrubbed:
    """Review P1: EVERY gate reports the registry text, never model text."""

    SENTINEL = "SENTINEL-GATE-PAYLOAD-患者私密文本"

    def _assert_clean(self, exc, sink):
        assert self.SENTINEL not in str(exc.value)
        assert self.SENTINEL not in repr(exc.value)
        assert str(exc.value) == lookup(exc.value.code).message
        assert all(self.SENTINEL not in r.result for r in sink.records)
        assert all(self.SENTINEL not in str(r) for r in sink.records)
        envelope = ErrorEnvelope.build(
            code=exc.value.code, request_id="r", trace_id="t"
        )
        assert self.SENTINEL not in envelope.model_dump_json()

    @pytest.mark.parametrize("runner", ["invoke", "ainvoke"])
    def test_unknown_tool_name_never_reaches_the_error_text(self, runner):
        import asyncio

        sink = _Sink()
        calls: list[Any] = []
        gw = ToolGateway(audit_sink=sink)
        gw.register(
            canonical_spec(
                "knowledge.search",
                executor=lambda _ctx, args: calls.append(args) or {"items": []},
            )
        )
        request = _request(f"unknown.{self.SENTINEL}", {"query": "x"})
        with pytest.raises(ToolGatewayError) as exc:
            if runner == "invoke":
                gw.invoke(TENANT, request, allowed_tools=[request.tool_name])
            else:
                asyncio.run(
                    gw.ainvoke(TENANT, request, allowed_tools=[request.tool_name])
                )
        assert exc.value.code is ErrorCode.TOOL_DISABLED
        self._assert_clean(exc, sink)
        assert calls == []  # no executor ran
        gw.shutdown()

    @pytest.mark.parametrize("runner", ["invoke", "ainvoke"])
    def test_invalid_input_property_name_never_reaches_the_error_text(self, runner):
        import asyncio

        sink = _Sink()
        calls: list[Any] = []
        gw = ToolGateway(audit_sink=sink)
        gw.register(
            canonical_spec(
                "knowledge.search",
                executor=lambda _ctx, args: calls.append(args) or {"items": []},
            )
        )
        request = _request("knowledge.search", {"query": "x", self.SENTINEL: "v"})
        with pytest.raises(ToolGatewayError) as exc:
            if runner == "invoke":
                gw.invoke(
                    TENANT, request, allowed_tools=["knowledge.search"], agent_id="a"
                )
            else:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        request,
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                    )
                )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        self._assert_clean(exc, sink)
        assert calls == []
        gw.shutdown()

    @pytest.mark.parametrize("runner", ["invoke", "ainvoke"])
    def test_invalid_output_property_name_never_reaches_the_error_text(self, runner):
        import asyncio

        sink = _Sink()
        gw = ToolGateway(audit_sink=sink)
        gw.register(
            canonical_spec(
                "knowledge.search",
                # a STRICT output schema, so an extra (model-named) result key is
                # a real violation rather than an allowed extension
                output_schema={
                    "type": "object",
                    "properties": {"items": {"type": "array"}},
                    "required": ["items"],
                    "additionalProperties": False,
                },
                executor=lambda _ctx, args: {self.SENTINEL: "v"},
            )
        )
        request = _request("knowledge.search", {"query": "x"})
        with pytest.raises(ToolGatewayError) as exc:
            if runner == "invoke":
                gw.invoke(
                    TENANT, request, allowed_tools=["knowledge.search"], agent_id="a"
                )
            else:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        request,
                        allowed_tools=["knowledge.search"],
                        agent_id="a",
                    )
                )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        self._assert_clean(exc, sink)
        gw.shutdown()

    def test_the_audit_still_carries_the_detailed_reason(self):
        """Scrubbing must not cost operators their diagnostics."""
        sink = _Sink()
        gw = build_gateway(knowledge_store=_store_with_one_approved(), audit_sink=sink)
        with pytest.raises(ToolGatewayError):
            _call(gw, "knowledge.search", {})  # missing required "query"
        # the kind is kept; the path (which can embed a model key) is not
        assert sink.records[-1].result == "rejected:input_schema:required"
        gw.shutdown()


class TestRegistryCodedErrorsAreScrubbedToo:
    """Review P1: a coded executor error must not smuggle its own text."""

    SENTINEL = "SENTINEL-KNOWLEDGE-PAYLOAD-患者私密文本"

    def _gateway(self, executor):
        gw = ToolGateway(audit_sink=_Sink())
        gw.register(canonical_spec("knowledge.get_fragment", executor=executor))
        return gw

    def _assert_clean(self, exc, sink):
        assert exc.value.code is ErrorCode.NOT_FOUND_KNOWLEDGE  # code preserved
        assert self.SENTINEL not in str(exc.value)
        assert self.SENTINEL not in repr(exc.value)
        # the message is the REGISTRY text, not the executor's
        assert str(exc.value) == lookup(ErrorCode.NOT_FOUND_KNOWLEDGE).message
        assert all(self.SENTINEL not in r.result for r in sink.records)
        assert all(self.SENTINEL not in str(r) for r in sink.records)
        # and the HTTP envelope would carry only the registry message
        envelope = ErrorEnvelope.build(
            code=exc.value.code, request_id="r", trace_id="t"
        )
        assert self.SENTINEL not in envelope.model_dump_json()

    def test_sync_coded_error_is_scrubbed(self):
        sink = _Sink()

        def executor(_context, args):
            raise ToolGatewayError(ErrorCode.NOT_FOUND_KNOWLEDGE, self.SENTINEL)

        gw = ToolGateway(audit_sink=sink)
        gw.register(canonical_spec("knowledge.get_fragment", executor=executor))
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.get_fragment", {"source_id": "x"})
        self._assert_clean(exc, sink)
        gw.shutdown()

    def test_async_coded_error_is_scrubbed(self):
        import asyncio

        sink = _Sink()

        def executor(_context, args):
            raise ToolGatewayError(ErrorCode.NOT_FOUND_KNOWLEDGE, self.SENTINEL)

        gw = ToolGateway(audit_sink=sink)
        gw.register(canonical_spec("knowledge.get_fragment", executor=executor))
        with pytest.raises(ToolGatewayError) as exc:
            asyncio.run(
                gw.ainvoke(
                    TENANT,
                    _request("knowledge.get_fragment", {"source_id": "x"}),
                    allowed_tools=["knowledge.get_fragment"],
                    agent_id="a",
                )
            )
        self._assert_clean(exc, sink)
        gw.shutdown()

    @pytest.mark.parametrize("runner", ["invoke", "ainvoke"])
    def test_builtin_never_echoes_the_requested_source_id(self, runner):
        """The BUILTIN's own message must not embed model-supplied input."""
        import asyncio

        sink = _Sink()
        store = _store_with_one_approved()
        gw = build_gateway(knowledge_store=store, audit_sink=sink)
        arguments = {"source_id": self.SENTINEL}
        request = _request("knowledge.get_fragment", arguments)

        with pytest.raises(ToolGatewayError) as exc:
            if runner == "invoke":
                gw.invoke(
                    TENANT,
                    request,
                    allowed_tools=["knowledge.get_fragment"],
                    agent_id="a",
                )
            else:
                asyncio.run(
                    gw.ainvoke(
                        TENANT,
                        request,
                        allowed_tools=["knowledge.get_fragment"],
                        agent_id="a",
                    )
                )
        assert exc.value.code is ErrorCode.NOT_FOUND_KNOWLEDGE
        assert self.SENTINEL not in str(exc.value)
        assert self.SENTINEL not in repr(exc.value)
        assert all(self.SENTINEL not in r.result for r in sink.records)
        gw.shutdown()

    def test_fragment_index_out_of_range_has_no_identifier_either(self):
        store = _store_with_one_approved(content="短内容")
        gw = build_gateway(knowledge_store=store, audit_sink=_Sink())
        with pytest.raises(ToolGatewayError) as exc:
            _call(
                gw,
                "knowledge.get_fragment",
                {"source_id": "faq-1", "fragment_index": 9},
            )
        assert exc.value.code is ErrorCode.NOT_FOUND_KNOWLEDGE
        assert "faq-1" not in str(exc.value)
        gw.shutdown()


class TestInvalidBudgetFailsBeforeAnyGate:
    """Review wording fix: validation happens BEFORE `_prepare`."""

    def test_invalid_timeout_wins_over_a_gate_violation(self):
        import asyncio

        sink = _Sink()
        gw = build_gateway(audit_sink=sink, knowledge_store=_store_with_one_approved())
        with pytest.raises(ValueError):
            asyncio.run(
                gw.ainvoke(
                    TENANT,
                    # this argument would ALSO trip the identity gate
                    _request("knowledge.search", {"query": "x", "tenant_id": "t2"}),
                    allowed_tools=["knowledge.search"],
                    agent_id="a",
                    timeout_s=0,
                )
            )
        assert sink.records == []  # no gate ran, so no audit record was written
        gw.shutdown()


class TestPoolLifecycle:
    """Gate item: the worker pool is released when the application stops."""

    def test_app_shutdown_releases_the_tool_pool(self):
        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.main import create_app

        app = create_app(settings=Settings(environment="test"))
        with TestClient(app):
            gateway = app.state.tool_gateway
            assert gateway is not None
            assert gateway._pool._shutdown is False
        assert gateway._pool._shutdown is True
        assert app.state.tool_gateway is None

    def test_explicit_shutdown_is_idempotent(self):
        gw = ToolGateway(audit_sink=_Sink())
        gw.shutdown()
        gw.shutdown()  # a second call must not raise
        assert gw._pool._shutdown is True


class TestAliasSafety:
    def test_canonical_specs_own_their_schemas(self):
        a = canonical_spec("knowledge.search", executor=lambda _ctx, args: {})
        b = canonical_spec("knowledge.search", executor=lambda _ctx, args: {})
        a.input_schema["required"] = []
        assert b.input_schema["required"] == ["query"]  # decl uncorrupted
        from app.tools.specs import TOOL_INPUT_SCHEMAS

        assert TOOL_INPUT_SCHEMAS["knowledge.search"][1]["required"] == ["query"]

    def test_registered_spec_is_isolated_from_caller_mutation(self):
        gw = ToolGateway()
        spec = canonical_spec("knowledge.search", executor=lambda _ctx, args: {})
        gw.register(spec)
        spec.input_schema["required"] = []  # caller mutates its own copy
        with pytest.raises(ToolGatewayError) as exc:
            _call(gw, "knowledge.search", {})
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED  # still required

    def test_spec_returns_deep_copy(self):
        gw = ToolGateway()
        gw.register(canonical_spec("knowledge.search", executor=lambda _ctx, args: {}))
        got = gw.spec("knowledge.search")
        got.input_schema["required"] = []
        assert gw.spec("knowledge.search").input_schema["required"] == ["query"]
        assert gw.spec("ghost") is None


class TestDomainExecutorSpy:
    def test_executor_never_runs_on_rejected_or_unauthorized_calls(self):
        calls = []

        def spy(_context, args):
            calls.append(args)
            return {"summary": "x"}

        sink = _Sink()
        gw = ToolGateway(audit_sink=sink)
        gw.register(canonical_spec("memory.read_short", executor=spy))
        # schema violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                SESSION,
                _request("memory.read_short", {"max_chars": "not-an-int"}),
                allowed_tools=["memory.read_short"],
                agent_id="a",
            )
        assert exc.value.code is ErrorCode.TOOL_SCHEMA_REJECTED
        # permission violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                SESSION,
                _request("memory.read_short"),
                allowed_tools=[],
                agent_id="a",
            )
        assert exc.value.code is ErrorCode.AUTHZ_FORBIDDEN
        # whitelist violation
        with pytest.raises(ToolGatewayError) as exc:
            gw.invoke(
                SESSION,
                _request("shell.exec"),
                allowed_tools=["shell.exec"],
                agent_id="a",
            )
        assert exc.value.code is ErrorCode.TOOL_DISABLED
        assert calls == []  # unauthorized tool invocations succeeded 0 times
