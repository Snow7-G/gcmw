"""ModelGateway: the only model entry point for every Agent (issue #37).

Hard constraints (V2.3 §8.4):
- Agents never call provider SDKs or vendor HTTP endpoints directly — they go
  through this gateway;
- ``switch_provider`` is a restricted, RBAC-protected operation (dual approval
  + audit wired in #36/#40); the gateway refuses un-authorized switches by
  default and Agent code has no path to it;
- every call carries deadline/token budget/cancellation semantics; errors map
  to :class:`~app.contracts.errors.ErrorCode` values.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from app.contracts.errors import ErrorCode
from app.contracts.model import (
    ModelEvent,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ProviderHealth,
    RealtimeSessionInfo,
    SwitchRequest,
    SwitchResult,
)


class ModelGatewayError(RuntimeError):
    """Gateway-level failure carrying a stable ErrorCode (mapped to envelopes by #36)."""

    def __init__(self, code: ErrorCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


class ProviderAdapter(Protocol):
    """Contract every concrete provider (mock/cloud/local) must implement."""

    provider_id: str

    async def chat(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def open_realtime_session(
        self, request: ModelRequest
    ) -> RealtimeSessionInfo: ...

    def is_available(self) -> bool: ...

    async def health(self) -> ProviderHealth: ...

    def model_info(self) -> ModelInfo: ...


class ModelGateway:
    """Registry + active-provider facade."""

    def __init__(
        self,
        adapters: dict[str, ProviderAdapter] | None = None,
        active_provider_id: str = "mock",
    ) -> None:
        self._adapters: dict[str, ProviderAdapter] = dict(adapters or {})
        self._active_provider_id = active_provider_id

    # -- registry -----------------------------------------------------------

    def register(self, adapter: ProviderAdapter) -> None:
        if adapter.provider_id in self._adapters:
            raise ModelGatewayError(
                ErrorCode.CONFLICT_IDEMPOTENCY, "provider already registered"
            )
        self._adapters[adapter.provider_id] = adapter

    def unregister(self, provider_id: str) -> None:
        self._adapters.pop(provider_id, None)

    @property
    def active_provider_id(self) -> str:
        return self._active_provider_id

    def registered_providers(self) -> list[str]:
        return sorted(self._adapters)

    def _resolve(self, provider_id: str | None) -> ProviderAdapter:
        pid = provider_id or self._active_provider_id
        adapter = self._adapters.get(pid)
        if adapter is None:
            raise ModelGatewayError(
                ErrorCode.PROVIDER_UNREACHABLE, f"unknown provider {pid!r}"
            )
        return adapter

    # -- Agent-facing calls --------------------------------------------------

    async def chat(self, request: ModelRequest) -> ModelResponse:
        """Non-streaming chat with gateway-enforced deadline."""
        adapter = self._resolve(request.provider_hint)
        try:
            return await asyncio.wait_for(
                adapter.chat(request), timeout=request.deadline_ms / 1000
            )
        except asyncio.TimeoutError as exc:
            raise ModelGatewayError(ErrorCode.TIMEOUT_PROVIDER) from exc

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Streaming events with a gateway-enforced overall deadline.

        Caller cancellation (``aclose``/task cancel) propagates into the
        adapter; adapters MUST react to ``CancelledError`` promptly. Uncaught
        adapter exceptions are intentionally NOT swallowed here — the #36
        boundary maps them to safe envelopes.
        """
        adapter = self._resolve(request.provider_hint)
        try:
            async with asyncio.timeout(request.deadline_ms / 1000):
                async for event in adapter.stream(request):
                    yield event
        except TimeoutError as exc:
            raise ModelGatewayError(ErrorCode.TIMEOUT_PROVIDER) from exc

    async def open_realtime_session(self, request: ModelRequest) -> RealtimeSessionInfo:
        """Backend-controlled realtime session (vendors never reach the UI)."""
        adapter = self._resolve(request.provider_hint)
        try:
            return await asyncio.wait_for(
                adapter.open_realtime_session(request),
                timeout=request.deadline_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise ModelGatewayError(ErrorCode.TIMEOUT_PROVIDER) from exc

    # -- availability / info -------------------------------------------------

    def provenance(self, provider_id: str | None = None) -> tuple[str, str, str]:
        """The SERVER-SIDE identity of a configured provider: (provider_id,
        model_id, model_version) straight from the registered adapter's
        configuration — the only provenance source the Manager trusts. Values
        claimed by model OUTPUT are never consulted (#55A-B)."""
        adapter = self._resolve(provider_id)
        info = adapter.model_info()
        return (adapter.provider_id, info.model_id, info.model_version)

    def is_available(self, provider_id: str | None = None) -> bool:
        adapter = self._adapters.get(provider_id or self._active_provider_id)
        return adapter is not None and adapter.is_available()

    async def health(self, provider_id: str | None = None) -> ProviderHealth:
        return await self._resolve(provider_id).health()

    def model_info(self, provider_id: str | None = None) -> ModelInfo:
        return self._resolve(provider_id).model_info()

    # -- restricted switch (RBAC/audit enforced by the #36/#40 layers) -------

    async def switch_provider(
        self, request: SwitchRequest, authorized: bool = False
    ) -> SwitchResult:
        """Restricted switch. ``authorized`` is granted only by the change
        management layer (dual approval + audit); the default is a refusal."""
        if not authorized:
            raise ModelGatewayError(
                ErrorCode.AUTHZ_FORBIDDEN, "provider switch is not authorized"
            )
        if request.target_provider_id not in self._adapters:
            raise ModelGatewayError(
                ErrorCode.PROVIDER_UNREACHABLE,
                f"unknown provider {request.target_provider_id!r}",
            )
        previous = self._active_provider_id
        self._active_provider_id = request.target_provider_id
        return SwitchResult(
            ok=True,
            previous_provider_id=previous,
            active_provider_id=self._active_provider_id,
            release_id=request.release_id,
        )
