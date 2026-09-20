"""Voice Agent adapter tests (#58 语音接入 Agent 快速切片).

Covers: configuration fail-closed rules, the SSE delivery gate (citations /
UTF-8 split / late frames / terminal mapping / hygiene), and the /chat route
wiring (legacy unchanged, agent no-fallback, health non-sensitive).

All HTTP is stubbed — zero network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import voice_agent_adapter as vaa
from voice_agent_adapter import (
    COPY_CANCELLED,
    COPY_ESCALATED,
    COPY_REFUSED,
    COPY_TIMEOUT,
    COPY_UNAVAILABLE,
    DEMO_CONTRACT_BASE_URL,
    PLACEHOLDER_CREDENTIAL,
    SseFrameParser,
    VoiceAgentClient,
    VoiceAgentConfig,
    VoiceAgentConfigError,
    resolve_voice_agent_config,
)

CRED = "voice-test-credential-000001"
_VALID_CITATION = {
    "source_id": "faq-fever",
    "knowledge_version": "faq-fever-v1",
    "content_hash": "a" * 64,
}


# ============ stub HTTP 层（零网络） ============


class StubResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        raw_json: str | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._raw_json = raw_json
        self._chunks = chunks or []
        self.closed = False

    def json(self) -> Any:
        if self._raw_json is not None:
            return json.loads(self._raw_json)
        return self._payload

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return self._chunks

    def close(self) -> None:
        self.closed = True


class StubHttp:
    """记录全部请求；按 (method, url 子串) 返回预设响应。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], StubResponse] = {}
        self.post_responses: list[StubResponse] = []

    def add_route(self, method: str, url_fragment: str, resp: StubResponse) -> None:
        self.routes[(method, url_fragment)] = resp

    def _record(
        self, method: str, url: str, headers: Any, json_body: Any
    ) -> StubResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "json": json_body,
            }
        )
        for (m, frag), resp in self.routes.items():
            if m == method and frag in url:
                return resp
        return StubResponse(status_code=404)

    def post(self, url: str, **kwargs: Any) -> StubResponse:
        return self._record("POST", url, kwargs.get("headers"), kwargs.get("json"))

    def get(self, url: str, **kwargs: Any) -> StubResponse:
        return self._record("GET", url, kwargs.get("headers"), None)

    def delete(self, url: str, **kwargs: Any) -> StubResponse:
        return self._record("DELETE", url, kwargs.get("headers"), None)


def agent_config() -> VoiceAgentConfig:
    return VoiceAgentConfig(
        mode="agent", base_url=DEMO_CONTRACT_BASE_URL, credential=CRED
    )


def make_client(
    chunks: list[bytes], run_status: int = 200
) -> tuple[VoiceAgentClient, StubHttp]:
    http = StubHttp()
    http.add_route("POST", "/sessions", StubResponse(200, {"session_id": "s-1"}))
    http.add_route("POST", "/agent/runs", StubResponse(run_status, {"run_id": "r-1"}))
    http.add_route("GET", "/events", StubResponse(200, chunks=chunks))
    return VoiceAgentClient(agent_config(), http=http), http


def frame_bytes(seq: int, event: str, business: dict[str, Any]) -> bytes:
    return (
        f"id: {seq}\nevent: {event}\n"
        f"data: {json.dumps({'data': business}, ensure_ascii=False)}\n\n"
    ).encode()


def good_answer_stream() -> list[bytes]:
    return [
        frame_bytes(1, "run.accepted", {}),
        frame_bytes(2, "answer.delta", {"delta": "眼部不适需及时就诊"}),
        frame_bytes(3, "answer.completed", {"citations": [_VALID_CITATION]}),
        frame_bytes(4, "run.completed", {"status": "COMPLETED", "result": "answered"}),
    ]


# ============ 配置（fail-closed） ============


class TestConfig:
    def test_default_is_legacy(self):
        cfg = resolve_voice_agent_config({})
        assert cfg.mode == "legacy"

    def test_agent_mode_with_full_config(self):
        cfg = resolve_voice_agent_config(
            {
                "GCMW_VOICE_ANSWER_BACKEND": "agent",
                "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL,
                "GCMW_VOICE_AGENT_CREDENTIAL": CRED,
            }
        )
        assert cfg.mode == "agent" and cfg.credential == CRED

    def test_invalid_mode_rejected(self):
        with pytest.raises(VoiceAgentConfigError):
            resolve_voice_agent_config({"GCMW_VOICE_ANSWER_BACKEND": "auto"})

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8001/api/v1",
            "http://127.0.0.1:9000/api/v1",
            "http://0.0.0.0:8001/api/v1",
            "https://127.0.0.1:8001/api/v1",
            "not a url",
            "",
        ],
    )
    def test_non_contract_base_rejected(self, url):
        with pytest.raises(VoiceAgentConfigError):
            resolve_voice_agent_config(
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "agent",
                    "GCMW_VOICE_AGENT_API_BASE": url,
                    "GCMW_VOICE_AGENT_CREDENTIAL": CRED,
                }
            )

    def test_trailing_slash_normalized_to_contract(self):
        cfg = resolve_voice_agent_config(
            {
                "GCMW_VOICE_ANSWER_BACKEND": "agent",
                "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL + "/",
                "GCMW_VOICE_AGENT_CREDENTIAL": CRED,
            }
        )
        assert cfg.base_url == DEMO_CONTRACT_BASE_URL

    def test_missing_credential_rejected(self):
        with pytest.raises(VoiceAgentConfigError):
            resolve_voice_agent_config(
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "agent",
                    "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL,
                }
            )

    def test_placeholder_credential_rejected(self):
        with pytest.raises(VoiceAgentConfigError):
            resolve_voice_agent_config(
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "agent",
                    "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL,
                    "GCMW_VOICE_AGENT_CREDENTIAL": PLACEHOLDER_CREDENTIAL,
                }
            )

    def test_error_message_never_contains_credential(self):
        # 非法 mode 报错只回显 mode，绝不回显凭据值
        with pytest.raises(VoiceAgentConfigError) as exc:
            resolve_voice_agent_config(
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "weird",
                    "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL,
                    "GCMW_VOICE_AGENT_CREDENTIAL": "sk-secret-value-xyz",
                }
            )
        assert "weird" in str(exc.value)
        assert "sk-secret-value-xyz" not in str(exc.value)


# ============ Agent 客户端（交付门槛 / 映射 / 卫生） ============


class TestClientDeliveryGate:
    def test_legal_cited_answer_delivered(self):
        client, _ = make_client(good_answer_stream())
        answer, source = client.ask("眼部不适怎么办")
        assert source == "agent"
        assert answer == "眼部不适需及时就诊"

    def test_utf8_multibyte_split_across_chunks(self):
        # '眼部不适需及时就诊' 的 UTF-8 字节每 5 字节切一刀 —— 必然切开
        # 多字节字符；单一增量解码器必须跨 chunk 还原
        frame = frame_bytes(1, "answer.delta", {"delta": "眼部不适需及时就诊"})
        tail = frame_bytes(2, "answer.completed", {"citations": [_VALID_CITATION]})
        tail += (
            frame_bytes(3, "run.completed", {"result": "answered"})
            .decode("utf-8")
            .encode()
        )
        chunks = [frame[i : i + 5] for i in range(0, len(frame), 5)] + [tail]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == "眼部不适需及时就诊"

    def test_multiple_deltas_joined_in_order(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "第一句。"}),
            frame_bytes(2, "answer.delta", {"delta": "第二句。"}),
            frame_bytes(3, "answer.completed", {"citations": [_VALID_CITATION]}),
            frame_bytes(4, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, _ = client.ask("q")
        assert answer == "第一句。第二句。"

    def test_answered_without_citations_rejected(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "答案"}),
            frame_bytes(2, "answer.completed", {"citations": []}),
            frame_bytes(3, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_missing_knowledge_version_rejected(self):
        bad = {"source_id": "s", "content_hash": "a" * 64}
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "答案"}),
            frame_bytes(2, "answer.completed", {"citations": [bad]}),
            frame_bytes(3, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    @pytest.mark.parametrize("hash_value", ["A" * 64, "a" * 63, "z" * 64, ""])
    def test_non_64hex_hash_rejected(self, hash_value):
        bad = {"source_id": "s", "knowledge_version": "v", "content_hash": hash_value}
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "答案"}),
            frame_bytes(2, "answer.completed", {"citations": [bad]}),
            frame_bytes(3, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_refused_no_answer_mapping(self):
        chunks = [frame_bytes(1, "run.completed", {"result": "refused_no_answer"})]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == COPY_REFUSED

    def test_escalated_mapping(self):
        chunks = [frame_bytes(1, "run.completed", {"result": "escalated_to_human"})]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == COPY_ESCALATED

    def test_cancelled_mapping(self):
        chunks = [frame_bytes(1, "run.completed", {"result": "cancelled"})]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == COPY_CANCELLED

    def test_deadline_exceeded_mapping(self):
        chunks = [frame_bytes(1, "run.completed", {"result": "deadline_exceeded"})]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == COPY_TIMEOUT

    def test_stream_error_fails_closed(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "部分答案"}),
            frame_bytes(2, "stream.error", {"reason": "x"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_eof_without_terminal_fails_closed(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "部分答案"}),
            frame_bytes(2, "answer.completed", {"citations": [_VALID_CITATION]}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_malformed_json_fails_closed(self):
        chunks = [b"event: answer.delta\ndata: {not-json\n\n"]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_missing_envelope_data_fails_closed(self):
        # 业务载荷不在内层 data —— 违反 SSEEvent 信封契约 → 失败关闭
        chunks = [b'event: run.completed\ndata: {"result": "answered"}\n\n']
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_late_delta_in_same_chunk_as_terminal_not_delivered(self):
        # 终态与迟到帧位于同一个 chunk：迟到 delta 不得交付
        body = (
            'id: 1\nevent: answer.delta\ndata: {"data":{"delta":"已核验部分"}}\n\n'
            'id: 2\nevent: answer.completed\ndata: {"data":{"citations":['
            + json.dumps(_VALID_CITATION)
            + "]}}\n\n"
            'id: 3\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n'
            'id: 4\nevent: answer.delta\ndata: {"data":{"delta":"迟到泄漏"}}\n\n'
        )
        client, _ = make_client([body.encode("utf-8")])
        answer, source = client.ask("q")
        assert source == "agent"
        assert answer == "已核验部分"
        assert "迟到泄漏" not in answer

    def test_partial_answer_discarded_on_failure(self):
        # 已收到部分答案后流失败：部分答案必须被丢弃
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "未核验部分"}),
            frame_bytes(2, "stream.error", {}),
        ]
        client, _ = make_client(chunks)
        answer, _ = client.ask("q")
        assert answer == COPY_UNAVAILABLE and "未核验部分" not in answer

    def test_crlf_and_keepalive_handled(self):
        body = (
            ": keep-alive\r\n\r\n"
            'id: 1\nevent: answer.delta\ndata: {"data":{"delta":"你好"}}\r\n\r\n'
            'id: 2\nevent: answer.completed\r\ndata: {"data":{"citations":['
            + json.dumps(_VALID_CITATION)
            + "]}}\r\n\r\n"
            'id: 3\nevent: run.completed\r\ndata: {"data":{"result":"answered"}}\r\n\r\n'
        )
        client, _ = make_client([body.encode("utf-8")])
        answer, source = client.ask("q")
        assert source == "agent" and answer == "你好"

    def test_multiline_data_concatenated(self):
        raw = (
            'id: 1\nevent: answer.delta\ndata: {"data":\n'
            'data: {"delta": "拼接"}}\n\n'
            'id: 2\nevent: answer.completed\ndata: {"data":{"citations":['
            + json.dumps(_VALID_CITATION)
            + "]}}\n\n"
            'id: 3\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n'
        )
        client, _ = make_client([raw.encode("utf-8")])
        answer, source = client.ask("q")
        assert source == "agent" and answer == "拼接"

    def test_network_failure_cancels_run(self):
        # sessions/runs 正常创建，events 阶段网络故障 → best-effort 取消 run
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(200, {"session_id": "s-1"}))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))

        class FlakyGet:
            def __call__(self, url, **kwargs):
                raise vaa.requests.ConnectionError("boom")

        http.get = FlakyGet()  # type: ignore[method-assign]
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/agent/runs/r-1" in u for u in deletes)

    def test_session_always_cleaned_up_on_success_and_failure(self):
        client, http = make_client(good_answer_stream())
        client.ask("q")
        ok_deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/sessions/" in u for u in ok_deletes)

        client2, http2 = make_client([frame_bytes(1, "stream.error", {})])
        client2.ask("q")
        fail_deletes = [c["url"] for c in http2.calls if c["method"] == "DELETE"]
        assert any("/sessions/" in u for u in fail_deletes)
        assert any("/agent/runs/" in u for u in fail_deletes)

    def test_client_side_deadline_fails_closed_without_terminal(self):
        import voice_agent_adapter as mod

        chunks = [
            ": keep-alive\n\n",  # 只有心跳，永远到不了终态
        ]
        client, _ = make_client(chunks)
        # 借助 monkeypatch 缩短绝对 deadline（避免真实等待）
        original = mod._TOTAL_DEADLINE_S
        mod._TOTAL_DEADLINE_S = 0.0001
        try:
            answer, source = client.ask("q")
        finally:
            mod._TOTAL_DEADLINE_S = original
        assert source == "error" and answer == COPY_UNAVAILABLE


class TestCredentialHygiene:
    def test_credential_only_in_authorization_header(self):
        client, http = make_client(good_answer_stream())
        client.ask("q")
        for call in http.calls:
            auth = call["headers"].get("Authorization", "")
            assert CRED in auth and auth.startswith("Bearer ")
            for key, value in call["headers"].items():
                if key != "Authorization":
                    assert CRED not in value
            assert CRED not in call["url"]

    def test_credential_absent_from_outputs_and_failures(self):
        client, _http = make_client(good_answer_stream())
        answer, _source = client.ask("q")
        assert CRED not in answer
        # 强制各故障路径，断言返回文案不含凭据
        for chunks in (
            [frame_bytes(1, "stream.error", {})],
            [b"event: run.completed\ndata: garbage\n\n"],
            good_answer_stream()[:1],
        ):
            c, _ = make_client(chunks)
            answer, _ = c.ask("q")
            assert CRED not in answer

    def test_http_error_status_never_leaks_body(self):
        http = StubHttp()
        http.add_route(
            "POST", "/sessions", StubResponse(500, {"detail": "secret-internal"})
        )
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error"
        assert "secret-internal" not in answer and answer == COPY_UNAVAILABLE


# ============ SseFrameParser 单元 ============


class TestSseFrameParser:
    def test_crlf_split_frame_completed_only_after_blank_line(self):
        parser = SseFrameParser()
        assert parser.feed(b'event: a\ndata: {"data":{}}\r') == []
        partial = parser.feed(b"\n")  # \r\n 之后还差空行
        assert partial == []
        frames = parser.feed(b"\n")
        assert frames == [{"event": "a", "data": '{"data":{}}'}]

    def test_utf8_split_keeps_char_intact(self):
        parser = SseFrameParser()
        raw = "眼".encode()
        text = parser.feed(raw[:1])
        text += parser.feed(raw[1:] + b"\n\n")
        assert not any("\ufffd" in json.dumps(f) for f in text)

    def test_flush_returns_trailing_frame(self):
        parser = SseFrameParser()
        assert parser.feed(b'event: run.completed\ndata: {"data":{}}\n\n') != []
        assert parser.feed(b'event: x\ndata: {"data":{}}') == []
        assert parser.flush() == [{"event": "x", "data": '{"data":{}}'}]


# ============ /chat 路由接线 ============


@pytest.fixture()
def qa_server_module():
    import qa_server

    return qa_server


class TestChatRoute:
    def _client(self):
        from fastapi.testclient import TestClient

        import qa_server

        return TestClient(qa_server.app), qa_server

    def test_routes_still_exist(self, qa_server_module):
        paths = {r.path for r in qa_server_module.app.routes}
        expected = {
            "/chat",
            "/suggestions",
            "/mic/wakeup",
            "/mic/hw_wakeup",
            "/mic/stop",
            "/mic/status",
            "/mic/notify_asr",
            "/sse",
            "/health",
        }
        assert expected.issubset(paths)

    def test_health_reports_backend_without_credential(self):
        client, _qa_server = self._client()
        body = client.get("/health").json()
        assert body["answer_backend"] == "legacy"
        assert CRED not in json.dumps(body)

    def test_legacy_mode_uses_kb_and_deepseek_unchanged(
        self, qa_server_module, monkeypatch
    ):
        client, qa_server = self._client()
        # KB 命中路径
        monkeypatch.setattr(
            qa_server.kb_searcher,
            "search",
            lambda q: ({"question": "q", "answer": "KB答案"}, 1.0),
        )
        monkeypatch.setattr(
            qa_server,
            "call_deepseek",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("KB 命中不应调 DeepSeek")
            ),
        )
        resp = client.post("/chat", json={"question": "帮我查一下", "session_id": "s"})
        assert resp.json()["source"] == "kb" and resp.json()["robot_answer"] == "KB答案"

        # DeepSeek 兜底路径
        monkeypatch.setattr(qa_server.kb_searcher, "search", lambda q: (None, 0.1))
        monkeypatch.setattr(
            qa_server, "call_deepseek", lambda q, history=None: ("LLM答案", "deepseek")
        )
        resp = client.post("/chat", json={"question": "帮我查一下", "session_id": "s"})
        assert resp.json()["source"] == "deepseek"

    def test_agent_mode_calls_adapter_not_kb_or_deepseek(
        self, qa_server_module, monkeypatch
    ):
        client, qa_server = self._client()
        monkeypatch.setattr(qa_server, "VOICE_AGENT_CONFIG", agent_config())
        monkeypatch.setattr(
            qa_server.kb_searcher,
            "search",
            lambda q: (_ for _ in ()).throw(AssertionError("agent 模式不得搜 KB")),
        )
        monkeypatch.setattr(
            qa_server,
            "call_deepseek",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("agent 模式不得调 DeepSeek")
            ),
        )
        monkeypatch.setattr(
            qa_server.voice_agent_adapter,
            "ask",
            lambda q, cfg: ("已核验的回答", "agent"),
        )
        resp = client.post(
            "/chat", json={"question": "眼部不适怎么办", "session_id": "s"}
        )
        body = resp.json()
        assert body["source"] == "agent" and body["robot_answer"] == "已核验的回答"

    def test_agent_failure_returns_fixed_copy_no_fallback(
        self, qa_server_module, monkeypatch
    ):
        client, qa_server = self._client()
        monkeypatch.setattr(qa_server, "VOICE_AGENT_CONFIG", agent_config())
        monkeypatch.setattr(
            qa_server.kb_searcher,
            "search",
            lambda q: (_ for _ in ()).throw(AssertionError("故障不得回退 KB")),
        )
        monkeypatch.setattr(
            qa_server,
            "call_deepseek",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("故障不得回退 DeepSeek")
            ),
        )
        monkeypatch.setattr(
            qa_server.voice_agent_adapter,
            "ask",
            lambda q, cfg: (COPY_UNAVAILABLE, "error"),
        )
        resp = client.post("/chat", json={"question": "q", "session_id": "s"})
        body = resp.json()
        assert body["source"] == "error" and body["robot_answer"] == COPY_UNAVAILABLE

    def test_agent_answer_still_broadcast_via_sse_state(
        self, qa_server_module, monkeypatch
    ):
        client, qa_server = self._client()
        monkeypatch.setattr(qa_server, "VOICE_AGENT_CONFIG", agent_config())
        monkeypatch.setattr(
            qa_server.voice_agent_adapter,
            "ask",
            lambda q, cfg: ("广播验证", "agent"),
        )
        client.post("/chat", json={"question": "q", "session_id": "s"})
        assert qa_server._latest_answer is not None
        assert qa_server._latest_answer["robot_answer"] == "广播验证"
        assert qa_server._latest_answer["source"] == "agent"
