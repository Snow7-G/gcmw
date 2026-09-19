"""DashScope OpenAI-compatible chat adapter (#37 ProviderAdapter surface).

Wire format: ``POST {api_base}/chat/completions`` — the OpenAI-compatible
mode of DashScope (Bailian), so qwen-plus / qwen-max / qwen-turbo all work
through the same code path. The realtime omni model (#51 voice channel) is a
DIFFERENT transport (WebSocket) and is intentionally out of scope here.

Security contract (must hold):

* the API key is read from the environment ONCE by the composition root and
  passed in via the constructor — it is NEVER logged, never echoed in error
  messages, never placed in the URL, and never committed;
* error mapping is by HTTP status only: the response body is not surfaced to
  callers (the error code is all the UI ever sees).

Honest scope: ``chat`` only. The text-QA path (#53) is chat-shaped; streaming
and realtime are separate transports with separate adapters.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..contracts.errors import ErrorCode
from ..contracts.model import (
    ContentType,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ProviderHealth,
    ProviderStatus,
)
from .model_gateway import ModelGatewayError


class DashScopeCloudProvider:
    """OpenAI-compatible chat adapter for DashScope (provider_id "cloud")."""

    provider_id = "cloud"

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        timeout_ms: int = 15_000,
        model_version: str = "",
        http_client: Any = None,
    ) -> None:
        if not api_key:
            # fail fast at composition time: a cloud adapter without a
            # credential must never be registered
            raise ModelGatewayError(ErrorCode.AUTH_MISSING_CREDENTIALS)
        self.api_base = api_base.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_ms = timeout_ms
        self._model_version = model_version
        # injectable for tests; a real httpx.AsyncClient is created lazily
        self._http_client = http_client

    @property
    def model_id(self) -> str:
        return self._model

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_id=self.provider_id,
            model_id=self._model,
            model_version=self._model_version or "compatible-mode",
            supported_modalities=[ContentType.TEXT],
            supports_function_calling=False,
            supports_streaming=False,
            supports_realtime=False,
        )

    async def chat(self, request: ModelRequest) -> ModelResponse:
        """One non-streaming chat completion (OpenAI-compatible wire)."""
        started = time.monotonic()
        payload = {
            "model": self._model,
            "messages": request.messages,
            "max_tokens": request.token_budget,
            "temperature": 0,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        timeout_s = min(request.deadline_ms, self._timeout_ms) / 1000
        client = self._http_client
        owned = client is None
        try:
            if client is None:
                client = httpx.AsyncClient(timeout=timeout_s)
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(ErrorCode.TIMEOUT_PROVIDER) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(ErrorCode.PROVIDER_UNREACHABLE) from exc
        finally:
            if owned and client is not None:
                await client.aclose()

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as exc:
                raise ModelGatewayError(ErrorCode.PROVIDER_UNREACHABLE) from exc
        else:
            # map by status ONLY — the body may contain provider internals and
            # is never surfaced
            raise ModelGatewayError(self._error_for_status(response.status_code))

        try:
            choice = data["choices"][0]["message"]["content"]
            content = choice if isinstance(choice, str) else ""
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelGatewayError(ErrorCode.PROVIDER_UNREACHABLE) from exc

        return ModelResponse(
            provider_id=self.provider_id,
            model_id=self._model,
            model_version=self._model_version or "compatible-mode",
            content=content.strip(),
            finish_reason=str(data.get("choices", [{}])[0].get("finish_reason", "")),
            usage=usage or {},
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def stream(self, request: ModelRequest) -> Any:
        """Streaming is a separate transport for this provider (see module docstring).

        Plain ``def`` (matching the Protocol signature ``def stream(...) ->
        AsyncIterator``): the error raises SYNCHRONOUSLY at call time, so a
        caller doing ``async for ... in adapter.stream(req)`` gets the stable
        "unsupported" error instead of a coroutine TypeError.
        """
        raise ModelGatewayError(ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED)

    def is_available(self) -> bool:
        """No network probe: the adapter is usable iff it holds a credential.

        The constructor already fails closed on an empty key, so this is True
        by construction; kept explicit for the ProviderAdapter contract.
        """
        return bool(self._api_key)

    async def health(self) -> ProviderHealth:
        """Local health only — never a network probe (health fan-out must not
        burn provider quota or leak timing). A configured adapter is
        AVAILABLE; reachability is proven by the next real chat call."""
        return ProviderHealth(
            provider_id=self.provider_id,
            status=ProviderStatus.AVAILABLE
            if self._api_key
            else ProviderStatus.UNAVAILABLE,
            message="configured" if self._api_key else "missing credential",
        )

    async def open_realtime_session(self, request: ModelRequest) -> Any:
        """Realtime (WebSocket omni) is the #51 voice channel, out of scope."""
        raise ModelGatewayError(ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED)

    def _error_for_status(self, status: int) -> ErrorCode:
        if status in (401, 403):
            return ErrorCode.AUTH_INVALID_CREDENTIALS
        if status == 429:
            return ErrorCode.UNAVAILABLE_OVERLOADED
        return ErrorCode.PROVIDER_UNREACHABLE
