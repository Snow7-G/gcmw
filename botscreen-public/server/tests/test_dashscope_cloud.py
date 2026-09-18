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
