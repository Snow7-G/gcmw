"""#53 云模型接入：DashScope OpenAI 兼容适配器的契约测试（零网络）。

API key 通过注入的构造参数传递；测试里用的是假值，且断言它只出现在
Authorization 头里、绝不出现在 URL。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from app.contracts.errors import ErrorCode
from app.contracts.model import ModelRequest
from app.providers.dashscope_cloud import DashScopeCloudProvider
from app.providers.model_gateway import ModelGatewayError

# --- 审计修复回归：ProviderAdapter 协议符合性与安全降级 -------------------


def test_adapter_satisfies_provider_protocol():
    """ProviderAdapter 协议要求的全部方法都必须存在（结构化协议无注册期校验，
    缺方法只会在调用点炸 —— 这里显式锁定）。"""
    provider = DashScopeCloudProvider(**OPTS)
    for method in (
        "chat",
        "stream",
        "open_realtime_session",
        "is_available",
        "health",
        "model_info",
    ):
        assert callable(getattr(provider, method, None)), method


def test_stream_raises_synchronously_not_a_coroutine():
    """stream 必须是普通 def：调用即同步抛"不支持"，绝不能返回 coroutine
    ——否则 async for 迭代到它时得到 TypeError 而非稳定错误。"""
    import asyncio

    provider = DashScopeCloudProvider(**OPTS)
    result = None
    with pytest.raises(ModelGatewayError) as exc:
        result = provider.stream(make_request())
    assert exc.value.code == ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED
    assert not asyncio.iscoroutine(result), "stream 不得返回 coroutine"


@pytest.mark.asyncio
async def test_health_and_availability_are_local_only():
    """health/is_available 不触网：配置即 AVAILABLE，零配额消耗。"""
    import asyncio

    provider = DashScopeCloudProvider(**OPTS)
    assert provider.is_available() is True
    health = await asyncio.wait_for(provider.health(), timeout=1)
    assert health.provider_id == "cloud"
    assert health.status.value == "available"


@pytest.mark.asyncio
async def test_open_realtime_session_is_explicitly_unsupported():
    """realtime（WebSocket omni）是 #51 语音通道：显式拒绝而非 AttributeError。"""
    import asyncio

    provider = DashScopeCloudProvider(**OPTS)
    with pytest.raises(ModelGatewayError) as exc:
        await asyncio.wait_for(
            provider.open_realtime_session(make_request()), timeout=1
        )
    assert exc.value.code == ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED


def test_httpx_is_a_runtime_dependency():
    """反回归：httpx 是云适配器的运行时依赖，必须在 requirements.txt ——
    只在 dev 依赖时，生产安装 + GCMW_ACTIVE_PROVIDER=cloud 启动即 ImportError。"""
    from pathlib import Path

    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    assert "httpx" in req.read_text()


OPTS = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-plus",
    "api_key": "fake-cloud-key-000001",
    "timeout_ms": 5_000,
}


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    """记录 post 调用并按序返回预设响应/异常。"""

    def __init__(self, outcomes: list):
        self.calls: list[SimpleNamespace] = []
        self._outcomes = list(outcomes)
        self.closed = False

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            SimpleNamespace(url=url, payload=json, headers=headers, timeout=timeout)
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        self.closed = True


def make_request() -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": "发热怎么办"}],
        trace_id="t-1",
        deadline_ms=10_000,
        token_budget=400,
    )


def ok_payload() -> dict:
    return {
        "choices": [
            {
                "message": {"content": "体温超过38.5建议门诊就诊（资料[1]）。"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 42},
    }


@pytest.mark.asyncio
async def test_chat_success_parses_content_and_keeps_key_out_of_url():
    client = FakeAsyncClient([FakeResponse(200, ok_payload())])
    provider = DashScopeCloudProvider(**OPTS, http_client=client)

    res = await provider.chat(make_request())

    assert res.content == "体温超过38.5建议门诊就诊（资料[1]）。"
    assert res.provider_id == "cloud"
    assert res.model_id == "qwen-plus"
    assert res.usage == {"total_tokens": 42}
    call = client.calls[0]
    assert call.url.endswith("/chat/completions")
    assert "fake-cloud-key-000001" not in call.url  # 凭据不进 URL
    assert call.headers["Authorization"] == "Bearer fake-cloud-key-000001"
    assert call.payload["max_tokens"] == 400  # token_budget 传导
    assert call.payload["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_error_status_maps_to_safe_error_codes():
    for status, code in [
        (401, ErrorCode.AUTH_INVALID_CREDENTIALS),
        (403, ErrorCode.AUTH_INVALID_CREDENTIALS),
        (429, ErrorCode.UNAVAILABLE_OVERLOADED),
        (502, ErrorCode.PROVIDER_UNREACHABLE),
    ]:
        client = FakeAsyncClient(
            [FakeResponse(status, {"detail": "provider internal"})]
        )
        provider = DashScopeCloudProvider(**OPTS, http_client=client)
        with pytest.raises(ModelGatewayError) as exc:
            await provider.chat(make_request())
        assert exc.value.code == code


@pytest.mark.asyncio
async def test_timeout_maps_to_timeout_provider():
    import httpx

    client = FakeAsyncClient([httpx.TimeoutException("timed out")])
    provider = DashScopeCloudProvider(**OPTS, http_client=client)
    with pytest.raises(ModelGatewayError) as exc:
        await provider.chat(make_request())
    assert exc.value.code == ErrorCode.TIMEOUT_PROVIDER


@pytest.mark.asyncio
async def test_malformed_payload_maps_to_unreachable():
    client = FakeAsyncClient([FakeResponse(200, {"unexpected": True})])
    provider = DashScopeCloudProvider(**OPTS, http_client=client)
    with pytest.raises(ModelGatewayError) as exc:
        await provider.chat(make_request())
    assert exc.value.code == ErrorCode.PROVIDER_UNREACHABLE


def test_empty_api_key_fails_at_construction():
    with pytest.raises(ModelGatewayError) as exc:
        DashScopeCloudProvider(api_base="https://x", model="qwen-plus", api_key="")
    assert exc.value.code == ErrorCode.AUTH_MISSING_CREDENTIALS


@pytest.mark.asyncio
async def test_stream_is_explicitly_unsupported():
    provider = DashScopeCloudProvider(**OPTS)
    with pytest.raises(ModelGatewayError) as exc:
        await provider.stream(make_request())
    assert exc.value.code == ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED
