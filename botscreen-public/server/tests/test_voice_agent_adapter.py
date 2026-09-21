"""Tests for the voice Agent adapter (zero-network, async transport doubles).

Fourth review round: the adapter is a REAL-cancellation async client. The
transport doubles below are async (httpx-compatible surface): post/get/delete
coroutines plus a ``stream()`` async context manager. Slow endpoints hang on
``asyncio.sleep``/``Event.wait`` so a deadline cancellation is OBSERVED by the
stub (``cancelled`` counter) — proving the request coroutine itself stops,
not merely that the caller stopped waiting.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
from typing import Any

import httpx
import pytest

import voice_agent_adapter as vaa
from voice_agent_adapter import (
    COPY_UNAVAILABLE,
    DEMO_CONTRACT_BASE_URL,
    PLACEHOLDER_CREDENTIAL,
    SseFrameParser,
    VoiceAgentClient,
    VoiceAgentConfig,
    VoiceAgentConfigError,
    resolve_voice_agent_config,
)

CRED = "TEST-CREDENTIAL-SENTINEL-0123456789"
_VALID_CITATION = {
    "source_id": "faq-eye",
    "knowledge_version": "kb-2026",
    "content_hash": "a" * 64,
}


# ============ 异步 stub HTTP 层（零网络，httpx 兼容表面） ============


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

    def json(self) -> Any:
        if self._raw_json is not None:
            return json.loads(self._raw_json)
        return self._payload

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _StubStreamCM:
    """async context manager mirroring ``httpx.AsyncClient.stream``."""

    def __init__(self, resp: StubResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> StubResponse:
        return self._resp

    async def __aexit__(self, *exc: object) -> bool:
        return False


class StubHttp:
    """记录全部请求；按 (method, url 子串) 返回预设响应。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], StubResponse] = {}

    def add_route(self, method: str, url_fragment: str, resp: StubResponse) -> None:
        self.routes[(method, url_fragment)] = resp

    def _record(
        self, method: str, url: str, headers: Any, json_body: Any, timeout: Any
    ) -> None:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "json": json_body,
                "timeout": timeout,
            }
        )

    def _route(self, method: str, url: str) -> StubResponse:
        for (m, frag), resp in self.routes.items():
            if m == method and frag in url:
                return resp
        return StubResponse(status_code=404)

    async def post(self, url: str, **kwargs: Any) -> StubResponse:
        self._record(
            "POST",
            url,
            kwargs.get("headers"),
            kwargs.get("json"),
            kwargs.get("timeout"),
        )
        return self._route("POST", url)

    async def get(self, url: str, **kwargs: Any) -> StubResponse:
        self._record("GET", url, kwargs.get("headers"), None, kwargs.get("timeout"))
        return self._route("GET", url)

    async def delete(self, url: str, **kwargs: Any) -> StubResponse:
        self._record("DELETE", url, kwargs.get("headers"), None, kwargs.get("timeout"))
        return self._route("DELETE", url)

    def stream(self, method: str, url: str, **kwargs: Any) -> _StubStreamCM:
        self._record(method, url, kwargs.get("headers"), None, kwargs.get("timeout"))
        return _StubStreamCM(self._route(method, url))


def agent_config() -> VoiceAgentConfig:
    return VoiceAgentConfig(
        mode="agent", base_url=DEMO_CONTRACT_BASE_URL, credential=CRED
    )


#: 本轮固定身份。测试桩 Session 响应授予它，真实信封必须与之一致才被接受。
TENANT_ID = "t1"
DEVICE_ID = "d1"
SESSION_ID = "s-1"
RUN_ID = "r-1"

#: 事件 → 层级（与服务端 EVENT_LAYER_MAP 一致；信封校验会强制）
LAYER_FOR_EVENT = {
    "run.accepted": "process",
    "process.status": "process",
    "evidence.found": "process",
    "reflection.result": "process",
    "heartbeat": "process",
    "run.completed": "process",
    "mic_status": "process",
    "answer.delta": "answer",
    "answer.completed": "answer",
}


def session_payload(session_id: str = SESSION_ID) -> dict[str, Any]:
    """真实 SessionResponse：身份三元组齐全（缺一项即视为畸形响应）。"""
    return {
        "session_id": session_id,
        "tenant_id": TENANT_ID,
        "device_id": DEVICE_ID,
    }


def envelope(
    seq: int,
    event: str,
    data: dict[str, Any],
    *,
    tenant_id: str = TENANT_ID,
    device_id: str = DEVICE_ID,
    session_id: str = SESSION_ID,
    run_id: str = RUN_ID,
    layer: str | None = None,
    protocol_version: str = "1.0",
) -> dict[str, Any]:
    """完整 #30 ``SSEEvent`` 信封（与服务端 ``frame()`` 输出同形）。

    全部字段都可覆写，反例测试据此构造"别的 run / 外层内层不一致 / 错误版本"
    这类真实协议违规——信封是完整真身，不再有 ``{"data": ...}`` 这种只喂内层的
    假信封（那正是此前把绑定缺口藏起来的写法）。
    """
    return {
        "protocol_version": protocol_version,
        "seq": seq,
        "tenant_id": tenant_id,
        "device_id": device_id,
        "session_id": session_id,
        "run_id": run_id,
        "layer": layer if layer is not None else LAYER_FOR_EVENT[event],
        "event": event,
        "data": data,
        "timestamp": "2026-09-21T00:00:00+00:00",
    }


def frame_bytes(
    seq: int,
    event: str,
    business: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    sse_id: str | None = None,
    outer_event: str | None = None,
) -> bytes:
    """真实 SSE 帧：``id: {seq}`` + 完整信封 + ``event:`` 外层名。

    ``payload`` / ``sse_id`` / ``outer_event`` 仅用于构造反例（信封本身、SSE
    ``id`` 与最外层事件名三者可以各自被"篡改"）。
    """
    body = payload if payload is not None else envelope(seq, event, business)
    id_text = str(seq) if sse_id is None else sse_id
    name = event if outer_event is None else outer_event
    return (
        f"id: {id_text}\nevent: {name}\n"
        f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
    ).encode()


def good_answer_stream() -> list[bytes]:
    return [
        frame_bytes(1, "run.accepted", {}),
        frame_bytes(2, "answer.delta", {"delta": "眼部不适需及时就诊"}),
        frame_bytes(
            3,
            "answer.completed",
            {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
        ),
        frame_bytes(4, "run.completed", {"status": "COMPLETED", "result": "answered"}),
    ]


def stream_error_bytes(code: str = "E_INTERNAL_UNKNOWN") -> bytes:
    """真实传输层故障帧：``ErrorEnvelope`` 且**没有** ``id``。

    服务端 ``stream_error_frame()`` 刻意不给它 ``id``（失败不得看起来像进度），
    所以它绝不能走 ``frame_bytes``——那条路会伪造出一个带 id + SSEEvent 信封的
    帧，既不是真实协议，也会把"缺 id 的故障帧必须失败关闭"这条覆盖掉。
    """
    body = {"code": code, "request_id": "req-1", "trace_id": "trace-1"}
    return f"event: stream.error\ndata: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


def make_client(
    chunks: list[bytes], run_status: int = 200
) -> tuple[VoiceAgentClient, StubHttp]:
    http = StubHttp()
    http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
    http.add_route("POST", "/agent/runs", StubResponse(run_status, {"run_id": RUN_ID}))
    http.add_route("GET", "/events", StubResponse(200, chunks=chunks))
    return VoiceAgentClient(agent_config(), http=http), http


def set_budget(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(vaa, "_TOTAL_DEADLINE_S", seconds)


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

    @pytest.mark.parametrize(
        ("bad_env", "sentinel"),
        [
            (
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "MODE-SENTINEL-xyz",
                    "GCMW_VOICE_AGENT_API_BASE": DEMO_CONTRACT_BASE_URL,
                    "GCMW_VOICE_AGENT_CREDENTIAL": "CRED-SENTINEL-xyz",
                },
                ["MODE-SENTINEL-xyz", "CRED-SENTINEL-xyz"],
            ),
            (
                {
                    "GCMW_VOICE_ANSWER_BACKEND": "agent",
                    "GCMW_VOICE_AGENT_API_BASE": "http://127.0.0.1:8001/api/v1?secret=URL-SENTINEL-xyz",
                    "GCMW_VOICE_AGENT_CREDENTIAL": "CRED-SENTINEL-xyz",
                },
                ["URL-SENTINEL-xyz", "CRED-SENTINEL-xyz"],
            ),
        ],
    )
    def test_rejected_config_never_echoes_values(self, bad_env, sentinel):
        """P1-3：配置错误只报字段名与固定原因——mode/base/credential 的
        原始值绝不进入 str/repr/args。"""
        with pytest.raises(VoiceAgentConfigError) as exc:
            resolve_voice_agent_config(bad_env)
        for leak in sentinel:
            assert leak not in str(exc.value)
            assert leak not in repr(exc.value)
            assert all(leak not in str(a) for a in exc.value.args)

    def test_startup_output_never_contains_sentinel(self, tmp_path, capfd):
        """P1-3：qa_server 启动期配置失败时，stdout/stderr 不含原始值。"""
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if not k.startswith("GCMW_VOICE_")}
        env.update(
            GCMW_VOICE_ANSWER_BACKEND="agent",
            GCMW_VOICE_AGENT_API_BASE="http://127.0.0.1:8001/api/v1?secret=STARTUP-SENTINEL",
            GCMW_VOICE_AGENT_CREDENTIAL="cred",
        )
        proc = subprocess.run(
            [sys.executable, "qa_server.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(pathlib.Path(__file__).resolve().parent.parent),
        )
        assert proc.returncode != 0
        assert "STARTUP-SENTINEL" not in proc.stdout
        assert "STARTUP-SENTINEL" not in proc.stderr


# ============ 交付门槛与终态映射 ============


class TestClientDeliveryGate:
    def test_legal_cited_answer_delivered(self):
        client, _ = make_client(good_answer_stream())
        answer, source = client.ask("眼部不适怎么办")
        assert source == "agent" and answer == "眼部不适需及时就诊"

    def test_utf8_multibyte_split_across_chunks(self):
        stream = b"".join(
            [
                frame_bytes(1, "answer.delta", {"delta": "眼睛"}),
                frame_bytes(
                    2,
                    "answer.completed",
                    {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
                ),
                frame_bytes(3, "run.completed", {"result": "answered"}),
            ]
        )
        cut = stream.index("眼".encode()) + 1  # 眼 的首字节后切断
        client, _ = make_client([stream[:cut], stream[cut:]])
        answer, source = client.ask("q")
        assert source == "agent" and answer == "眼睛"

    def test_multiple_deltas_joined_in_order(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "先"}),
            frame_bytes(2, "answer.delta", {"delta": "后"}),
            frame_bytes(
                3,
                "answer.completed",
                {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
            ),
            frame_bytes(4, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == "先后"

    def test_answered_without_citations_rejected(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "答案"}),
            frame_bytes(
                2,
                "answer.completed",
                {"citations": [], "content_origin": "approved_faq"},
            ),
            frame_bytes(3, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_missing_knowledge_version_rejected(self):
        bad = {
            "source_id": "faq-eye",
            "knowledge_version": "",
            "content_hash": "a" * 64,
        }
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "答案"}),
            frame_bytes(
                2,
                "answer.completed",
                {"citations": [bad], "content_origin": "approved_faq"},
            ),
            frame_bytes(3, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        _answer, source = client.ask("q")
        assert source == "error"

    @pytest.mark.parametrize("hash_value", ["", "z" * 64, "a" * 63, "A" * 64])
    def test_non_64hex_hash_rejected(self, hash_value):
        bad = {
            "source_id": "faq-eye",
            "knowledge_version": "kb-2026",
            "content_hash": hash_value,
        }
        chunks = [
            frame_bytes(
                1,
                "answer.completed",
                {"citations": [bad], "content_origin": "approved_faq"},
            ),
            frame_bytes(2, "run.completed", {"result": "answered"}),
        ]
        client, _ = make_client(chunks)
        _answer, source = client.ask("q")
        assert source == "error"

    def test_refused_no_answer_mapping(self):
        client, _ = make_client(
            [frame_bytes(1, "run.completed", {"result": "refused_no_answer"})]
        )
        answer, source = client.ask("q")
        assert source == "agent"
        assert answer == vaa.COPY_REFUSED

    def test_escalated_mapping(self):
        client, _ = make_client(
            [frame_bytes(1, "run.completed", {"result": "escalated_to_human"})]
        )
        answer, source = client.ask("q")
        assert source == "agent" and answer == vaa.COPY_ESCALATED

    def test_cancelled_mapping(self):
        client, _ = make_client(
            [frame_bytes(1, "run.completed", {"result": "cancelled"})]
        )
        answer, source = client.ask("q")
        assert source == "agent" and answer == vaa.COPY_CANCELLED

    def test_deadline_exceeded_mapping(self):
        client, _ = make_client(
            [frame_bytes(1, "run.completed", {"result": "deadline_exceeded"})]
        )
        answer, source = client.ask("q")
        assert source == "agent" and answer == vaa.COPY_TIMEOUT

    def test_stream_error_fails_closed(self):
        client, _ = make_client([stream_error_bytes()])
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_eof_without_terminal_fails_closed(self):
        client, _ = make_client([frame_bytes(1, "run.accepted", {})])
        _answer, source = client.ask("q")
        assert source == "error"

    def test_malformed_json_fails_closed(self):
        client, _ = make_client([b"event: run.completed\ndata: garbage\n\n"])
        _answer, source = client.ask("q")
        assert source == "error"

    def test_missing_envelope_data_fails_closed(self):
        client, _ = make_client([b'event: answer.delta\ndata: {"delta": "x"}\n\n'])
        _answer, source = client.ask("q")
        assert source == "error"

    def test_late_delta_in_same_chunk_as_terminal_not_delivered(self):
        chunks = [
            frame_bytes(1, "run.completed", {"result": "refused_no_answer"}),
            frame_bytes(2, "answer.delta", {"delta": "迟到内容"}),
            frame_bytes(
                3,
                "answer.completed",
                {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
            ),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "agent" and answer == vaa.COPY_REFUSED

    def test_partial_answer_discarded_on_failure(self):
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "半截"}),
            stream_error_bytes(),
        ]
        client, _ = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_crlf_and_keepalive_handled(self):
        # 真实帧 + CRLF 折行 + keep-alive 注释帧：解析器在拼接后归一化 \r\n
        raw = b"".join(
            [
                b": keep-alive\r\n\r\n",
                frame_bytes(1, "answer.delta", {"delta": "你好"}).replace(
                    b"\n", b"\r\n"
                ),
                frame_bytes(
                    2,
                    "answer.completed",
                    {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
                ),
                frame_bytes(3, "run.completed", {"result": "answered"}),
            ]
        )
        client, _ = make_client([raw])
        answer, source = client.ask("q")
        assert source == "agent" and answer == "你好"

    def test_multiline_data_concatenated(self):
        # 多行 data: 由解析器以 \n 拼接：在 JSON 词法安全处（"data": 之后）切开
        body = json.dumps(
            envelope(1, "answer.delta", {"delta": "拼接"}), ensure_ascii=False
        )
        split_at = body.index('"data":') + len('"data":')
        raw = b"".join(
            [
                (
                    f"id: 1\nevent: answer.delta\n"
                    f"data: {body[:split_at]}\ndata: {body[split_at:]}\n\n"
                ).encode(),
                frame_bytes(
                    2,
                    "answer.completed",
                    {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
                ),
                frame_bytes(3, "run.completed", {"result": "answered"}),
            ]
        )
        client, _ = make_client([raw])
        answer, source = client.ask("q")
        assert source == "agent" and answer == "拼接"

    def test_network_failure_cancels_run(self):
        # sessions/runs 正常创建，events 阶段网络故障 → best-effort 取消 run
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))

        class FlakyStream:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = http.calls

            def __call__(self, method: str, url: str, **kwargs: Any):
                http._record(method, url, kwargs.get("headers"), None, None)
                raise ConnectionError("boom")

        http.stream = FlakyStream()  # type: ignore[method-assign]
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/agent/runs/r-1" in u for u in deletes)

    def test_client_side_deadline_fails_closed_without_terminal(self, monkeypatch):
        set_budget(monkeypatch, 0.0001)
        client, _ = make_client([": keep-alive\n\n"])  # 只有心跳，永远到不了终态
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE


# ============ P1（终审第五轮）：SSE 帧绑定本轮身份与序号 ============


def _good_payloads() -> list[tuple[int, str, dict[str, Any]]]:
    """一轮流内部的 (seq, event, data) 序列（终态为 answered）。"""
    return [
        (1, "run.accepted", {}),
        (2, "answer.delta", {"delta": "答案"}),
        (
            3,
            "answer.completed",
            {"citations": [_VALID_CITATION], "content_origin": "approved_faq"},
        ),
        (4, "run.completed", {"result": "answered"}),
    ]


def stream_with(**over: Any) -> list[bytes]:
    """把覆写注入每一帧的信封（用于构造身份/版本/层级类反例）。"""
    out: list[bytes] = []
    for seq, event, data in _good_payloads():
        out.append(
            frame_bytes(seq, event, data, payload=envelope(seq, event, data, **over))
        )
    return out


class TestStreamBinding:
    """每一帧都必须绑定「本轮身份 + 序号连续性」，任一不吻合立即失败关闭。

    这是终审实测缺口的封堵：此前状态机只读外层 ``event`` 与内层 ``data``，
    一个伪造成 ``other-run`` 的完整信封会被照单全收并交付 ``answered``。
    """

    def fails_closed(self, chunks: list[bytes]) -> StubHttp:
        client, http = make_client(chunks)
        answer, source = client.ask("q")
        assert source == "error", "越权/畸形帧绝不能被交付"
        assert answer == COPY_UNAVAILABLE
        return http

    def test_well_bound_stream_still_delivers(self):
        # 正对照：身份与序号全部吻合时照常交付（严格化不能打死正常链路）
        client, _ = make_client(stream_with())
        answer, source = client.ask("q")
        assert source == "agent" and answer == "答案"

    @pytest.mark.parametrize(
        "over",
        [
            {"run_id": "other-run"},
            {"session_id": "other-session"},
            {"device_id": "other-device"},
            {"tenant_id": "other-tenant"},
        ],
    )
    def test_foreign_identity_envelope_rejected(self, over):
        self.fails_closed(stream_with(**over))

    def test_foreign_run_injected_mid_stream_rejected(self):
        # 中途混入别的 run 的帧：不得被当成本轮进度
        chunks = [
            frame_bytes(1, "answer.delta", {"delta": "自己的"}),
            frame_bytes(
                2,
                "answer.delta",
                {"delta": "别人的"},
                payload=envelope(
                    2, "answer.delta", {"delta": "别人的"}, run_id="other-run"
                ),
            ),
        ]
        self.fails_closed(chunks)

    def test_outer_inner_event_mismatch_rejected(self):
        self.fails_closed(
            [frame_bytes(1, "run.accepted", {}, outer_event="answer.delta")]
        )

    @pytest.mark.parametrize("bad_id", ["3", "-1", "abc", "", "1.0", "+1"])
    def test_id_must_be_positive_decimal_equal_to_seq(self, bad_id):
        self.fails_closed([frame_bytes(1, "run.accepted", {}, sse_id=bad_id)])

    def test_duplicate_seq_rejected(self):
        self.fails_closed(
            [
                frame_bytes(1, "answer.delta", {"delta": "一"}),
                frame_bytes(1, "answer.delta", {"delta": "二"}),
            ]
        )

    def test_seq_rewind_rejected(self):
        self.fails_closed(
            [
                frame_bytes(1, "answer.delta", {"delta": "一"}),
                frame_bytes(2, "answer.delta", {"delta": "二"}),
                frame_bytes(1, "answer.delta", {"delta": "三"}),
            ]
        )

    def test_seq_gap_rejected(self):
        self.fails_closed(
            [
                frame_bytes(1, "answer.delta", {"delta": "一"}),
                frame_bytes(3, "answer.delta", {"delta": "三"}),
            ]
        )

    def test_stream_must_start_at_seq_one(self):
        self.fails_closed([frame_bytes(2, "run.accepted", {})])

    def test_wrong_protocol_version_rejected(self):
        self.fails_closed(stream_with(protocol_version="2.0"))

    def test_wrong_layer_rejected(self):
        # answer.completed 只能出现在 answer 层：层级错配由契约拦下
        self.fails_closed(stream_with(layer="process"))

    def test_envelope_missing_identity_fields_rejected(self):
        payload = envelope(1, "answer.delta", {"delta": "答案"})
        payload.pop("tenant_id")
        self.fails_closed([frame_bytes(1, "answer.delta", {}, payload=payload)])

    def test_session_response_without_identity_triple_rejected(self):
        # 身份不全的 Session 响应必须在建 Run / 订阅流之前被拒绝
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, {"session_id": "s-1"}))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": RUN_ID}))
        http.add_route("GET", "/events", StubResponse(200, chunks=stream_with()))
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE
        # 没有建 Run、也没有订阅事件流（只有 Session 请求与失败清理）
        assert not any("/agent/runs" in c["url"] for c in http.calls)
        assert not any(c["method"] == "GET" for c in http.calls)

    def test_transport_error_frame_without_id_fails_closed(self):
        # stream.error 是 ErrorEnvelope、刻意没有 id：仍然只能失败关闭
        self.fails_closed([stream_error_bytes()])

    def test_binding_violation_still_runs_the_bounded_cleanup(self):
        """绑定违规不只返回固定文案——它必须走进既有的有界取消与级联清理。

        上面每条反例都只断言了返回值，于是「失败关闭 == 也做了清理」这条契约
        其实没有任何测试钉住；而清理正是失败轮次不留痕的唯一保证。
        """
        http = self.fails_closed(stream_with(run_id="other-run"))
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        # 已知 Run → 显式取消；已知 Session → 级联回收一切可能已提交的 Run 与事件
        assert any(f"/agent/runs/{RUN_ID}" in url for url in deletes), deletes
        assert any(f"/sessions/{SESSION_ID}" in url for url in deletes), deletes


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
            [stream_error_bytes()],
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
        assert frames == [{"event": "a", "id": "", "data": '{"data":{}}'}]

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
        assert parser.flush() == [{"event": "x", "id": "", "data": '{"data":{}}'}]


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


# ============ P1-1A：绝对 deadline = 真取消域 ============


class HangingStreamResponse(StubResponse):
    """心跳流：每块之间真实 asyncio 等待，永远到不了终态。
    取消域到点时挂起中的等待被**真实取消**（cancelled 计数可断言）。"""

    def __init__(self, chunks: list[bytes], interval: float) -> None:
        super().__init__(200, chunks=chunks)
        self._interval = interval
        self.cancelled = 0

    async def aiter_bytes(self):
        try:
            for chunk in self._chunks:
                await asyncio.sleep(self._interval)
                yield chunk
            # 流保持打开：EOF 前无终态
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class TestAbsoluteDeadline:
    """deadline 从 ask() 入口起算、贯穿全部阶段；到点是真取消——
    挂起中的请求协程停止执行，而不是调用方单方面停止等待。"""

    def test_cancellation_domain_interrupts_blocking_stream(self, monkeypatch):
        set_budget(monkeypatch, 0.05)
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))
        stream = HangingStreamResponse(
            [b": keep-alive\n\n", b": keep-alive\n\n"], interval=0.2
        )
        http.add_route("GET", "/events", stream)
        client = VoiceAgentClient(agent_config(), http=http)
        t0 = time.monotonic()
        answer, source = client.ask("q")
        elapsed = time.monotonic() - t0
        assert source == "error" and answer == COPY_UNAVAILABLE
        assert elapsed < 0.15, f"deadline 未打断阻塞读：{elapsed:.3f}s"
        assert stream.cancelled == 1, "流读取协程未被真实取消"

    def test_answer_arriving_within_deadline_still_succeeds(self, monkeypatch):
        """近 deadline 回归：预算内完成的回答正常交付（对应 ROS 调用方
        不得先超时的语义 —— 适配器预算内成功 = 调用方窗口内成功）。"""
        set_budget(monkeypatch, 0.5)
        client, _ = make_client(good_answer_stream())
        answer, source = client.ask("q")
        assert source == "agent" and answer == "眼部不适需及时就诊"

    def test_budget_propagates_to_session_and_run_timeouts(self, monkeypatch):
        """Session/Run 请求的 connect/read 超时必须 ≤ 剩余预算
        （剩余预算 < 连接超时 5s 时 connect 不得用满额）。"""
        set_budget(monkeypatch, 0.05)

        class SlowCreateHttp(StubHttp):
            def __init__(self) -> None:
                super().__init__()
                self.cancelled = 0

            async def post(self, url: str, **kwargs: Any) -> StubResponse:
                # 先记录（含 timeout 契约），再模拟慢响应
                self._record(
                    "POST",
                    url,
                    kwargs.get("headers"),
                    kwargs.get("json"),
                    kwargs.get("timeout"),
                )
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
                return StubResponse(201, session_payload())

        http = SlowCreateHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        client = VoiceAgentClient(agent_config(), http=http)
        _answer, source = client.ask("q")
        assert source == "error"
        assert http.cancelled == 1, "慢请求协程未被真实取消"
        session_calls = [c for c in http.calls if c["url"].endswith("/sessions")]
        timeout: httpx.Timeout = session_calls[0]["timeout"]
        # 预算（0.05s）远小于默认 connect 5s：connect 也必须随剩余预算收缩，
        # 不允许出现"剩余预算 < 连接超时"时 connect 仍固定 _CONNECT_TIMEOUT_S 的漏洞。
        assert timeout.connect <= 0.1, (
            f"connect 超时未随剩余预算收缩：{timeout.connect}"
        )
        assert timeout.read <= 0.1, f"读超时未按剩余预算收缩：{timeout.read}"

    def test_remaining_budget_uses_the_running_loop_clock(self, monkeypatch):
        """剩余预算必须按**事件循环时钟**计算。

        反例构造：deadline 由被整体平移的 loop 时钟铸出（非默认 loop 上
        ``loop.time()`` 与 ``time.monotonic()`` 不必同纪元）。若实现用
        ``time.monotonic()`` 去减，剩余量会被算成"平移量 + 预算"这一巨大值，
        于是 connect/read 又回到满额 5s/30s —— 本用例正是钉住这个回归。
        """

        class ShiftedLoop:
            """时间整体前移 10000s 的 loop 时钟。"""

            def time(self) -> float:
                return time.monotonic() + 10_000.0

        shifted = ShiftedLoop()
        monkeypatch.setattr(vaa.asyncio, "get_running_loop", lambda: shifted)
        timeout = VoiceAgentClient._remaining_timeout(shifted.time() + 0.5)
        assert timeout.connect <= 0.5, f"connect 未按 loop 时钟收缩：{timeout.connect}"
        assert timeout.read <= 0.5, f"read 未按 loop 时钟收缩：{timeout.read}"

    def test_total_turn_elapsed_is_bounded_on_failure_paths(self, monkeypatch):
        set_budget(monkeypatch, 0.05)
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))
        http.add_route(
            "GET", "/events", HangingStreamResponse([b": keep-alive\n\n"] * 5, 0.2)
        )
        client = VoiceAgentClient(agent_config(), http=http)
        t0 = time.monotonic()
        client.ask("q")
        # 失败路径总耗时上界：预算 + 少量余量（清理另有独立有界 grace）
        assert time.monotonic() - t0 < 0.5


# ============ P1-2：ROS 调用方超时契约 ============


class TestVoiceTurnTimeoutContract:
    def test_ros_timeout_exceeds_adapter_budget(self):
        assert vaa.VOICE_TURN_TIMEOUT_S >= (
            vaa._TOTAL_DEADLINE_S + vaa._CLEANUP_GRACE_S
        )
        assert vaa.VOICE_TURN_TIMEOUT_S == 69.0

    def test_both_ros_nodes_share_the_same_semantics(self):
        """P1-2：两个 ROS 节点必须从适配器取同一端到端预算，不得各自写死。"""
        server_dir = pathlib.Path(__file__).resolve().parent.parent
        for node in ("voice_transfer_node.py", "voice_transfer_node_ros2.py"):
            src = (server_dir / node).read_text()
            assert "from voice_agent_adapter import VOICE_TURN_TIMEOUT_S" in src
            assert "timeout=15)" not in src, node
        # 适配器预算本身即 ROS 预算的组成部分：预算内完成的回答必然早于
        # ROS 超时窗口（近 deadline 回归见 TestAbsoluteDeadline）
        assert vaa.VOICE_TURN_TIMEOUT_S > vaa._TOTAL_DEADLINE_S


# ============ 清理语义（成功零 DELETE / 失败级联 / 有界可取消） ============


class TestBoundedCleanup:
    def test_success_makes_no_cleanup_requests(self):
        """成功终态：Session 交服务端主动 TTL sweeper 回收，Run/事件保留
        可审计 ——零 DELETE 请求（P1-3 语义锁定）。"""
        client, http = make_client(good_answer_stream())
        client.ask("q")
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert deletes == []

    def test_failure_cancels_run_and_deletes_session(self):
        client, http = make_client([stream_error_bytes()])
        client.ask("q")
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/agent/runs/" in u for u in deletes)
        assert any("/sessions/" in u for u in deletes)

    def test_cleanup_requests_use_bounded_short_grace(self):
        client, http = make_client([stream_error_bytes()])
        client.ask("q")
        for c in http.calls:
            if c["method"] == "DELETE":
                timeout: httpx.Timeout = c["timeout"]
                assert timeout.connect <= vaa._CLEANUP_GRACE_S
                assert timeout.read <= vaa._CLEANUP_GRACE_S

    def test_cleanup_non_2xx_does_not_change_return(self):
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))
        http.add_route(
            "GET",
            "/events",
            StubResponse(200, chunks=[stream_error_bytes()]),
        )
        http.add_route("DELETE", "/sessions", StubResponse(500))
        http.add_route("DELETE", "/agent/runs", StubResponse(500))
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_cleanup_exception_does_not_change_return(self):
        http = StubHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))
        http.add_route(
            "GET",
            "/events",
            StubResponse(200, chunks=[stream_error_bytes()]),
        )

        class ExplodingDelete:
            async def __call__(self, url: str, **kwargs: Any) -> StubResponse:
                raise RuntimeError("delete exploded")

        http.delete = ExplodingDelete()  # type: ignore[method-assign]
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE

    def test_hanging_cleanup_is_really_cancelled_within_total_grace(self, monkeypatch):
        """悬挂清理到点被**真取消**（不是"停止等待"）：总耗时被清理总
        期限截断，返回不变，取消被桩观测。"""
        monkeypatch.setattr(vaa, "_CLEANUP_TOTAL_GRACE_S", 0.4)
        monkeypatch.setattr(vaa, "_CLEANUP_GRACE_S", 0.3)

        class HangingDeleteHttp(StubHttp):
            def __init__(self) -> None:
                super().__init__()
                self.cancelled = 0

            async def delete(self, url: str, **kwargs: Any) -> StubResponse:
                self._record("DELETE", url, kwargs.get("headers"), None, None)
                try:
                    await asyncio.Event().wait()  # 真悬挂：永不返回
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
                return StubResponse(204)  # pragma: no cover

        http = HangingDeleteHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        http.add_route("POST", "/agent/runs", StubResponse(200, {"run_id": "r-1"}))
        http.add_route(
            "GET",
            "/events",
            StubResponse(200, chunks=[stream_error_bytes()]),
        )
        client = VoiceAgentClient(agent_config(), http=http)
        t0 = time.monotonic()
        answer, source = client.ask("q")
        elapsed = time.monotonic() - t0
        assert source == "error" and answer == COPY_UNAVAILABLE
        assert elapsed < 1.2, f"清理总期限未生效：{elapsed:.3f}s"
        assert http.cancelled >= 1, "悬挂清理未被真取消"


# ============ P1-1（第四轮）：迟到副作用与幂等对账 ============


class ReconcileSessionHttp(StubHttp):
    """Session POST 首次"已提交但响应迟于 deadline"：真取消打断等待；
    清理阶段用同一 idempotency_key 重放可拿到同一 Session。"""

    def __init__(self) -> None:
        super().__init__()
        self.session_posts = 0
        self.cancelled = 0

    async def post(self, url: str, **kwargs: Any) -> StubResponse:
        if "/sessions" in url:
            self._record(
                "POST",
                url,
                kwargs.get("headers"),
                kwargs.get("json"),
                kwargs.get("timeout"),
            )
            self.session_posts += 1
            if self.session_posts == 1:
                try:
                    await asyncio.sleep(5.0)  # 响应慢于业务 deadline
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
            # 首次（被取消）与重放都返回同一 Session（服务端幂等契约）
            return StubResponse(201, session_payload())
        return await super().post(url, **kwargs)


class HangingRunHttp(StubHttp):
    """Session 正常；Run POST 已提交但响应迟于 deadline。"""

    def __init__(self) -> None:
        super().__init__()
        self.run_posts = 0
        self.cancelled = 0

    async def post(self, url: str, **kwargs: Any) -> StubResponse:
        if "/agent/runs" in url:
            self._record(
                "POST",
                url,
                kwargs.get("headers"),
                kwargs.get("json"),
                kwargs.get("timeout"),
            )
            self.run_posts += 1
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return StubResponse(200, {"run_id": "r-1"})  # pragma: no cover
        return await super().post(url, **kwargs)


class TestLateSideEffectsAndReconciliation:
    """强制回归矩阵一：迟到副作用与取消。"""

    def test_session_committed_but_late_reconciled_with_same_key(self, monkeypatch):
        """① Session POST 结果未知 → ask() 按 deadline 失败关闭；
        清理用同一幂等键对账拿到同一 Session 并删除；零残留。"""
        set_budget(monkeypatch, 0.15)
        http = ReconcileSessionHttp()
        client = VoiceAgentClient(agent_config(), http=http)
        t0 = time.monotonic()
        answer, source = client.ask("q")
        elapsed = time.monotonic() - t0
        assert source == "error" and answer == COPY_UNAVAILABLE
        assert elapsed < 1.0, f"deadline 未生效：{elapsed:.3f}s"
        assert http.cancelled == 1, "迟到的 Session POST 未被真取消"
        posts = [
            c for c in http.calls if c["method"] == "POST" and "/sessions" in c["url"]
        ]
        assert len(posts) == 2, "应恰为：原始请求 + 同键重放对账"
        key1 = posts[0]["json"]["idempotency_key"]
        key2 = posts[1]["json"]["idempotency_key"]
        assert key1 == key2, "对账禁止换新 key（会创建第二个 Session）"
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/sessions/s-1" in u for u in deletes), "对账后的 Session 必须被清理"
        assert not any("/agent/runs/" in u for u in deletes)

    def test_run_committed_but_late_cascade_deletes_session(self, monkeypatch):
        """② Session 已知；Run POST 结果未知 → 禁止换 key 重试（只发一次），
        删除 Session 级联回收可能已提交的 Run；零残留。"""
        set_budget(monkeypatch, 0.15)
        http = HangingRunHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        client = VoiceAgentClient(agent_config(), http=http)
        answer, source = client.ask("q")
        assert source == "error" and answer == COPY_UNAVAILABLE
        assert http.cancelled == 1, "迟到的 Run POST 未被真取消"
        run_posts = [
            c for c in http.calls if "/agent/runs" in c["url"] and c["method"] == "POST"
        ]
        assert len(run_posts) == 1, "超时后不得生成新 key 重试"
        deletes = [c["url"] for c in http.calls if c["method"] == "DELETE"]
        assert any("/sessions/s-1" in u for u in deletes), "级联删除 Session"
        assert not any("/agent/runs/" in u for u in deletes), (
            "run_id 未知，无需显式取消"
        )

    def test_five_concurrent_slow_then_healthy_succeeds(self, monkeypatch):
        """③ 5 路并发慢请求全部超时后，健康请求必须成功——不再有
        全局线程池被占满导致的连锁失败。"""
        set_budget(monkeypatch, 0.2)

        def one_slow_turn() -> tuple[str, str]:
            http = StubHttp()

            class HangingPost(StubHttp):
                async def post(self, url: str, **kwargs: Any) -> StubResponse:
                    if "/sessions" in url:
                        self._record(
                            "POST",
                            url,
                            kwargs.get("headers"),
                            kwargs.get("json"),
                            kwargs.get("timeout"),
                        )
                        await asyncio.sleep(5.0)  # 直到被 deadline 真取消
                        return StubResponse(
                            201, {"session_id": "x"}
                        )  # pragma: no cover
                    return await super().post(url, **kwargs)  # pragma: no cover

            hanging = HangingPost()
            # 让 HangingPost 的 _record 写进 hanging.calls
            hanging.calls = http.calls  # type: ignore[assignment]
            hanging.routes = http.routes  # type: ignore[assignment]
            client = VoiceAgentClient(agent_config(), http=hanging)
            return client.ask("q")

        results: list[tuple[str, str]] = []
        threads = [
            threading.Thread(target=lambda i=i: results.append(one_slow_turn()))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(results) == 5
        assert all(source == "error" for _answer, source in results)

        # 紧随其后的健康请求必须成功
        healthy, _http = make_client(good_answer_stream())
        answer, source = healthy.ask("q")
        assert source == "agent" and answer == "眼部不适需及时就诊"

    def test_no_late_side_effects_after_return(self, monkeypatch):
        """④ ask() 返回后等待超过底层模拟延迟：不得出现新的未对账
        Session/Run，也不得出现迟到的清理请求。"""
        set_budget(monkeypatch, 0.1)
        http = ReconcileSessionHttp()
        client = VoiceAgentClient(agent_config(), http=http)
        _answer, source = client.ask("q")
        assert source == "error"
        calls_at_return = list(http.calls)
        # 底层模拟延迟（5s 挂起）远超此等待：任何迟到副作用都会在此现形
        time.sleep(0.3)
        assert http.calls == calls_at_return, "ask() 返回后出现了迟到请求"

    def test_shutdown_leaves_no_bounded_threads(self, monkeypatch):
        """⑤ 超时请求发生后：无线程池线程残留（全局 _EXECUTOR 已删除，
        每轮 asyncio.run 的任务在返回前必然结束）。"""
        assert not hasattr(vaa, "_EXECUTOR"), "模块级线程池必须已删除"
        set_budget(monkeypatch, 0.1)
        http = HangingRunHttp()
        http.add_route("POST", "/sessions", StubResponse(201, session_payload()))
        client = VoiceAgentClient(agent_config(), http=http)
        _answer, source = client.ask("q")
        assert source == "error"
        names = [t.name for t in threading.enumerate()]
        assert not any("voice-agent-bounded" in n for n in names), names


# ============ 幂等键前置契约 ============


class TestIdempotencyKeyContract:
    def test_session_and_run_keys_minted_before_requests(self):
        """Session/Run 的幂等键都在请求体里、且请求发出前已生成（对账
        依赖该键；见 TestLateSideEffectsAndReconciliation）。"""
        client, http = make_client(good_answer_stream())
        client.ask("q")
        session_posts = [
            c for c in http.calls if c["method"] == "POST" and "/sessions" in c["url"]
        ]
        run_posts = [
            c for c in http.calls if c["method"] == "POST" and "/agent/runs" in c["url"]
        ]
        assert len(session_posts) == 1 and len(run_posts) == 1
        assert session_posts[0]["json"]["idempotency_key"].startswith("voice-session-")
        assert run_posts[0]["json"]["idempotency_key"].startswith("voice-run-")

    def test_success_preserves_audit_records_contract(self):
        """P1-3 语义锁定：成功路径零 DELETE —— Run/事件随 Session 保留到
        服务端主动 TTL，可审计；失败路径才级联销毁。"""
        client, http = make_client(good_answer_stream())
        client.ask("q")
        assert [c for c in http.calls if c["method"] == "DELETE"] == []
