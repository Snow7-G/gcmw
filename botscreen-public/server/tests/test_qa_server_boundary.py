"""8000 语音服务的网络边界（终审第五轮）：只监听回环 + CORS 精确白名单。

该服务带固定低熵演示凭据，且打包后的 Electron 渲染进程会从 ``Origin: null``
直接访问它，所以两件事必须同时钉住：**监听地址**不能被摊到局域网，**CORS**
不能再用通配符。ROS/ROS2 节点与语音桥是服务端直连、不带 Origin，CORS 不参与
它们的流程——这一点也在这里固定下来，避免"收紧 CORS 把机器链路弄坏"。
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

import qa_server

SOURCE = pathlib.Path("qa_server.py").read_text(encoding="utf-8")

ALLOWED_ORIGINS = [
    "null",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(qa_server.app)


def preflight(
    client: TestClient,
    origin: str,
    *,
    method: str = "POST",
    headers: str | None = "content-type",
):
    request_headers = {
        "Origin": origin,
        "Access-Control-Request-Method": method,
    }
    if headers is not None:
        request_headers["Access-Control-Request-Headers"] = headers
    return client.options("/chat", headers=request_headers)


class TestLoopbackBinding:
    def test_default_host_is_loopback(self):
        assert qa_server.HOST == "127.0.0.1"
        assert qa_server.PORT == 8000

    def test_entrypoint_binds_the_loopback_constant(self):
        assert "host=HOST, port=PORT" in SOURCE

    def test_source_never_binds_all_interfaces(self):
        # 连注释里都不应出现通配监听：任何人顺手改回 0.0.0.0 这条断言就会红
        assert "0.0.0.0" not in SOURCE


class TestCorsAllowlist:
    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_is_echoed(self, client, origin):
        res = preflight(client, origin)
        assert res.status_code == 200
        assert res.headers["access-control-allow-origin"] == origin

    def test_configured_allowlist_is_exact(self):
        assert qa_server.ALLOWED_ORIGINS == ALLOWED_ORIGINS
        assert "*" not in qa_server.ALLOWED_ORIGINS

    @pytest.mark.parametrize(
        "origin",
        ["http://evil.example", "http://127.0.0.1:5000", "http://localhost:3000"],
    )
    def test_unknown_origin_gets_no_allow_header(self, client, origin):
        res = preflight(client, origin)
        assert res.status_code == 400
        assert "access-control-allow-origin" not in res.headers

    def test_credentials_and_wildcards_are_not_enabled(self, client):
        res = preflight(client, "null")
        assert "access-control-allow-credentials" not in res.headers
        assert qa_server.ALLOWED_METHODS == ["GET", "POST"]
        assert qa_server.ALLOWED_HEADERS == ["Content-Type"]

    def test_disallowed_method_rejected(self, client):
        """方法白名单只有 GET/POST：别的动词在预检就被否掉。

        注意 Starlette 在拒绝时仍会带上 allow-origin（它由中间件统一生成），
        真正生效的约束是 allow-methods 里没有该动词——所以这里断言的是
        拒绝状态与**方法白名单本身**。
        """
        res = preflight(client, "null", method="DELETE")
        assert res.status_code == 400
        assert res.text == "Disallowed CORS method"
        assert res.headers["access-control-allow-methods"] == "GET, POST"

    def test_disallowed_header_rejected(self, client):
        res = preflight(client, "null", headers="authorization")
        assert res.status_code == 400
        assert res.text == "Disallowed CORS headers"
        allowed = res.headers["access-control-allow-headers"]
        assert "Content-Type" in allowed
        assert "Authorization" not in allowed

    def test_packaged_renderer_can_read_sse_with_get(self, client):
        # EventSource 是简单请求（无预检、无作者请求头）；被问到时 GET 必须放行
        res = preflight(client, "null", method="GET", headers=None)
        assert res.status_code == 200
        assert res.headers["access-control-allow-origin"] == "null"

    def test_origin_null_alone_is_not_enough_to_pass_preflight(self, client):
        # 打包渲染进程只是"允许的来源"，不代表任何方法/头都被放行
        assert preflight(client, "null", method="DELETE").status_code == 400
        assert preflight(client, "null", headers="x-custom").status_code == 400


class TestActualResponsesCarryTheCorsDecision:
    """预检只是"问一次"；真正决定浏览器能否读响应的，是**实际响应**上的头。

    打包页面读 /sse、/health、/suggestions 走的是简单请求（EventSource 与不带
    自定义头的 fetch），浏览器**不会**先发预检——所以只测预检等于漏测了这条路径。
    """

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_is_echoed_on_the_actual_response(self, client, origin):
        res = client.get("/health", headers={"Origin": origin})
        assert res.status_code == 200
        assert res.headers["access-control-allow-origin"] == origin

    @pytest.mark.parametrize(
        "origin",
        ["http://evil.example", "http://127.0.0.1:5000", "http://localhost:3000"],
    )
    def test_unknown_origin_has_no_allow_header_on_the_actual_response(
        self, client, origin
    ):
        res = client.get("/health", headers={"Origin": origin})
        assert res.status_code == 200  # 请求本身照常处理
        assert "access-control-allow-origin" not in res.headers  # 但浏览器读不到

    def test_mic_status_actual_response_also_follows_the_allowlist(self, client):
        allowed = client.get("/mic/status", headers={"Origin": "null"})
        assert allowed.headers["access-control-allow-origin"] == "null"
        blocked = client.get("/mic/status", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in blocked.headers


class TestNonBrowserCallersUnaffected:
    """ROS1/ROS2 节点与语音桥不带 Origin：CORS 不参与，流程照常。"""

    def test_health_without_origin(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_mic_status_without_origin(self, client):
        assert client.get("/mic/status").status_code == 200

    def test_notify_asr_without_origin(self, client):
        res = client.post("/mic/notify_asr", json={"text": "发热怎么办"})
        assert res.status_code == 200

    def test_chat_without_origin(self, client):
        # 语音桥直连 /chat：不带 Origin 也应正常应答
        res = client.post("/chat", json={"question": "发热怎么办"})
        assert res.status_code == 200
        assert set(res.json()) >= {"user_question", "robot_answer", "source"}
