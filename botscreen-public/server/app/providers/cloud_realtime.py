"""CloudRealtimeProvider (issue #51): realtime WebSocket adapter for
Qwen3.5-Omni-Realtime via an injectable wire transport.

Hard constraints (issue review confirmation):
- the session model target is always the pinned snapshot
  ``qwen3.5-omni-plus-realtime-2026-03-15``; the floating alias is rejected —
  controlled upgrades change the pin, never ride the alias;
- ``enable_search`` is always explicitly ``false`` in the session config: the
  model must never bypass the approved knowledge base with built-in web
  search;
- the only accepted audio entry is official PCM 16 kHz / mono / 16-bit;
  unsupported formats are safely rejected (structured error, never
  downgraded-and-sent);
- vendor events never reach the UI/ROS verbatim — everything is normalized to
  internal ``ModelEvent`` values (no thinking chain / prompt / credential
  leaks; session info never carries URL or keys).

The wire transport is injected (``wire_factory``): CI and unit tests drive
the provider with scripted, deterministic wires — no real network, no real
data. The provider fetches the API key from the environment at session open
time and never stores or echoes it.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from app.config import CLOUD_MODEL_FLOATING_ALIAS, CLOUD_MODEL_PINNED_SNAPSHOT
from app.contracts.errors import ErrorCode
from app.contracts.model import (
    ContentPart,
    ContentType,
    ModelEvent,
    ModelEventType,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ProviderHealth,
    ProviderStatus,
    RealtimeSessionInfo,
)
from app.providers.model_gateway import ModelGatewayError

_LOGGER = logging.getLogger(__name__)

# official audio entry: PCM 16 kHz / mono / 16-bit little-endian
AUDIO_RATE_HZ = 16_000
AUDIO_CHANNELS = 1
AUDIO_BITS = 16
AUDIO_BYTES_PER_SAMPLE = AUDIO_BITS // 8

_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/api/v1"
_DEFAULT_API_KEY_ENV = "GCMW_CLOUD_API_KEY"

# wire events we treat as benign bookkeeping (acknowledged, not forwarded)
_BOOKKEEPING_EVENTS = {
    "session.created",
    "session.updated",
    "conversation.item.created",
    "conversation.item.updated",
    "conversation.created",
    "input_audio_buffer.committed",
    "response.created",
    "response.updated",
}

# vendor error code -> stable registry code (vendor text never forwarded)
_VENDOR_ERROR_MAP = {
    "invalid_api_key": ErrorCode.AUTH_INVALID_CREDENTIALS,
    "unauthorized": ErrorCode.AUTH_INVALID_CREDENTIALS,
    "rate_limit_exceeded": ErrorCode.RATE_LIMIT_EXCEEDED,
    "model_not_found": ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED,
    "timeout": ErrorCode.TIMEOUT_PROVIDER,
    "overloaded": ErrorCode.UNAVAILABLE_OVERLOADED,
}


class RealtimeWire(Protocol):
    """Injected WebSocket transport. Implementations never see model
    prompts/outputs beyond the JSON payloads and never log them."""

    async def open(self, url: str, api_key: str) -> None: ...

    async def send(self, payload: dict[str, Any]) -> None: ...

    async def recv(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def _mime_params(mime_type: str | None) -> tuple[str, dict[str, str]]:
    mime = (mime_type or "").lower()
    head, _, query = mime.partition(";")
    params: dict[str, str] = {}
    for piece in query.split(";"):
        piece = piece.strip()
        if "=" in piece:
            key, _, value = piece.partition("=")
            params[key.strip()] = value.strip()
    return head.strip(), params


def validate_audio_part(part: ContentPart) -> None:
    """Reject anything but PCM 16 kHz / mono / 16-bit with a structured error.

    Unsupported formats are never downgraded or sent (issue #51).
    """
    head, params = _mime_params(part.mime_type)
    if head not in {"audio/pcm", "audio/l16", "audio/x-pcm"}:
        if (part.mime_type is None and part.media_ref or "").lower().endswith(".pcm"):
            head = "audio/pcm"  # bare .pcm reference: official raw format
        else:
            raise ModelGatewayError(
                ErrorCode.VALIDATION_UNSUPPORTED_FORMAT,
                "audio must be raw PCM; other container formats are not accepted",
            )
    try:
        rate = int(params.get("rate", params.get("samplerate", AUDIO_RATE_HZ)))
        channels = int(params.get("channels", params.get("ch", AUDIO_CHANNELS)))
        bits = int(params.get("bits", params.get("samplebits", AUDIO_BITS)))
    except (TypeError, ValueError):
        raise ModelGatewayError(
            ErrorCode.VALIDATION_UNSUPPORTED_FORMAT,
            "audio format parameters must be integers",
        ) from None
    if (rate, channels, bits) != (AUDIO_RATE_HZ, AUDIO_CHANNELS, AUDIO_BITS):
        raise ModelGatewayError(
            ErrorCode.VALIDATION_UNSUPPORTED_FORMAT,
            f"audio must be PCM 16kHz mono 16-bit; got {rate}Hz/{channels}ch/{bits}bit",
        )


def _session_update(model: str) -> dict[str, Any]:
    """Client session config: pinned snapshot + built-in web search disabled."""
    return {
        "type": "session.update",
        "session": {
            "model": model,
            "enable_search": False,
            "modalities": ["text", "audio"],
            "audio": {
                "input": {
                    "format": "pcm",
                    "sample_rate": AUDIO_RATE_HZ,
                    "channels": AUDIO_CHANNELS,
                    "bits": AUDIO_BITS,
                },
                "turn": {"type": "vad"},
            },
        },
    }


def normalize_server_event(payload: dict[str, Any]) -> ModelEvent | None:
    """Map one vendor event onto an internal ModelEvent (never verbatim).

    Bookkeeping events return None (acknowledged internally). Unknown event
    types raise a structured error — vendor passthrough is forbidden.
    """
    event_type = payload.get("type", "")
    if event_type in _BOOKKEEPING_EVENTS:
        return None
    if (
        event_type == "response.text.delta"
        or event_type == "response.output_text.delta"
    ):
        delta = payload.get("delta", "")
        return ModelEvent(
            type=ModelEventType.DELTA,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
            data={"delta": delta},
        )
    if event_type in {"response.done", "response.completed", "message.completed"}:
        return ModelEvent(
            type=ModelEventType.DONE,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
        )
    if event_type == "conversation.item.input_audio_transcription.completed":
        return ModelEvent(
            type=ModelEventType.DELTA,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
            data={"recognition": payload.get("transcript", "")},
        )
    if event_type in {
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
    }:
        return ModelEvent(
            type=ModelEventType.DELTA,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
            data={"vad": event_type.rsplit(".", 1)[1]},
        )
    if event_type == "function_call.done":
        name = payload.get("name") or (payload.get("function") or {}).get("name", "")
        arguments = payload.get("arguments") or ""
        return ModelEvent(
            type=ModelEventType.TOOL_CALL,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
            data={"name": name, "arguments": arguments},
        )
    if event_type == "error":
        vendor_code = (payload.get("error") or {}).get("code", "")
        mapped = _VENDOR_ERROR_MAP.get(vendor_code, ErrorCode.PROVIDER_UNREACHABLE)
        return ModelEvent(
            type=ModelEventType.ERROR,
            provider_id="cloud",
            model_id=CLOUD_MODEL_PINNED_SNAPSHOT,
            model_version=CLOUD_MODEL_PINNED_SNAPSHOT,
            data={"error_code": mapped.value},  # vendor text is never forwarded
        )
    raise ModelGatewayError(
        ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED,
        f"unexpected vendor event {event_type!r}",
    )


class CloudRealtimeProvider:
    """Deterministic realtime adapter over an injected wire transport."""

    provider_id = "cloud"

    def __init__(
        self,
        *,
        wire_factory: Callable[[], RealtimeWire] | None = None,
        model: str = CLOUD_MODEL_PINNED_SNAPSHOT,
        api_base: str = _DEFAULT_API_BASE,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        available: bool = True,
        latency_ms: int = 0,
        connect_timeout_ms: int = 5_000,
    ) -> None:
        if model == CLOUD_MODEL_FLOATING_ALIAS:
            raise ModelGatewayError(
                ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED,
                "floating alias is not a session target; pin a snapshot",
            )
        self.model_id = model
        self.model_version = model
        self._wire_factory = wire_factory
        self._api_base = api_base
        self._api_key_env = api_key_env
        self._available = available
        self._latency_ms = latency_ms
        self._connect_timeout_ms = connect_timeout_ms
        self._sessions: dict[str, RealtimeWire] = {}
        self._lock = asyncio.Lock()
        self.sessions_opened = 0
        self.cancelled = 0

    # -- wire plumbing --------------------------------------------------------

    def _new_wire(self) -> RealtimeWire:
        if self._wire_factory is None:
            raise ModelGatewayError(
                ErrorCode.PROVIDER_UNREACHABLE,
                "no wire transport configured for the cloud provider",
            )
        return self._wire_factory()

    async def _open_wire(self) -> RealtimeWire:
        wire = self._new_wire()
        token = os.getenv(self._api_key_env, "")
        try:
            await asyncio.wait_for(
                wire.open(f"{self._api_base}/realtime", token),
                timeout=self._connect_timeout_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            try:
                await wire.close()  # never leak a half-open connection
            except Exception:
                _LOGGER.debug("wire close after connect timeout failed", exc_info=True)
            raise ModelGatewayError(ErrorCode.TIMEOUT_PROVIDER) from exc
        return wire

    # -- ProviderAdapter: chat / stream ---------------------------------------

    async def chat(self, request: ModelRequest) -> ModelResponse:
        """Text round-trip over the realtime wire; audio entries are validated
        and rejected unless PCM16k/mono/16bit (no raw bytes are sent here —
        media_ref audio is streamed via realtime sessions)."""
        self._reject_non_text_media(request)
        wire = await self._open_wire()
        try:
            await wire.send(_session_update(self.model_id))
            await self._wait_bookkeeping(wire, "session.created")
            text = self._last_user_text(request)
            await wire.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            await wire.send(
                {"type": "response.create", "response": {"modalities": ["text"]}}
            )
            content = ""
            while True:
                event = normalize_server_event(await wire.recv())
                if event is None:
                    continue
                if event.type is ModelEventType.DELTA and "delta" in event.data:
                    content += event.data["delta"]
                elif event.type is ModelEventType.DONE:
                    break
                elif event.type is ModelEventType.ERROR:
                    raise ModelGatewayError(
                        ErrorCode(event.data["error_code"]),
                        "vendor error (see code)",
                    )
        finally:
            await wire.close()
        return ModelResponse(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_version=self.model_version,
            content=content,
            finish_reason="stop",
            latency_ms=self._latency_ms,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Streaming text over the realtime wire; cancellation closes the wire
        promptly and propagates CancelledError to the gateway."""
        self._reject_non_text_media(request)
        wire = await self._open_wire()
        try:
            await wire.send(_session_update(self.model_id))
            await self._wait_bookkeeping(wire, "session.created")
            text = self._last_user_text(request)
            await wire.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            await wire.send(
                {"type": "response.create", "response": {"modalities": ["text"]}}
            )
            while True:
                event = normalize_server_event(await wire.recv())
                if event is None:
                    continue
                yield event
                if event.type in (ModelEventType.DONE, ModelEventType.ERROR):
                    return
        finally:
            await wire.close()

    def _reject_non_text_media(self, request: ModelRequest) -> None:
        for part in request.content_parts:
            if part.type is not ContentType.TEXT:
                raise ModelGatewayError(
                    ErrorCode.VALIDATION_UNSUPPORTED_MODALITY,
                    "chat/stream accept text only; audio rides realtime sessions",
                )

    def _last_user_text(self, request: ModelRequest) -> str:
        for message in reversed(request.messages):
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return str(part.get("text", ""))
        return ""

    def _handshake_error_code(self, payload: dict[str, Any]) -> ErrorCode | None:
        """Map a vendor error received before session.created onto a registry
        code (None when the payload is not an error)."""
        if payload.get("type") != "error":
            return None
        vendor_code = (payload.get("error") or {}).get("code", "")
        return _VENDOR_ERROR_MAP.get(vendor_code, ErrorCode.PROVIDER_UNREACHABLE)

    async def _wait_bookkeeping(self, wire: RealtimeWire, expected: str) -> None:
        while True:
            payload = await wire.recv()
            if payload.get("type") == expected:
                return
            code = self._handshake_error_code(payload)
            if code is not None:
                raise ModelGatewayError(code, "vendor rejected the session handshake")
            normalize_server_event(payload)  # validate but keep waiting

    # -- realtime sessions (audio in, VAD turns) ------------------------------

    async def open_realtime_session(self, request: ModelRequest) -> RealtimeSessionInfo:
        for part in request.content_parts:
            if part.type is ContentType.AUDIO:
                validate_audio_part(part)
        wire = await self._open_wire()
        try:
            await wire.send(_session_update(self.model_id))
            while True:
                payload = await wire.recv()
                if payload.get("type") == "session.created":
                    break
                code = self._handshake_error_code(payload)
                if code is not None:
                    raise ModelGatewayError(
                        code, "vendor rejected the session handshake"
                    )
                normalize_server_event(payload)
        except Exception:
            await wire.close()
            raise
        session_id = f"cloud-{uuid.uuid4().hex[:16]}"
        async with self._lock:
            self._sessions[session_id] = wire
        self.sessions_opened += 1
        return RealtimeSessionInfo(
            session_id=session_id,
            provider_id=self.provider_id,
            model_id=self.model_id,
        )

    async def append_audio(self, session_id: str, pcm: bytes) -> None:
        """Append one validated PCM frame (16kHz/mono/16-bit) to the session."""
        if len(pcm) % AUDIO_BYTES_PER_SAMPLE:
            raise ModelGatewayError(
                ErrorCode.VALIDATION_INVALID_INPUT,
                "PCM payload length must be a multiple of 2 bytes",
            )
        wire = self._require_session(session_id)
        await wire.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def complete_turn(self, session_id: str) -> None:
        """Commit the buffered audio and request a response (VAD turn)."""
        wire = self._require_session(session_id)
        await wire.send({"type": "input_audio_buffer.commit"})
        await wire.send(
            {"type": "response.create", "response": {"modalities": ["text"]}}
        )

    async def read_session_events(
        self, session_id: str, max_events: int = 8
    ) -> list[ModelEvent]:
        """Read up to ``max_events`` normalized session events (voice/text
        deltas, VAD markers, function calls). Reading stops as soon as a turn
        reaches a terminal event (DONE/ERROR)."""
        wire = self._require_session(session_id)
        events: list[ModelEvent] = []
        while len(events) < max_events:
            payload = await wire.recv()
            event = normalize_server_event(payload)
            if event is None:
                continue
            events.append(event)
            if event.type in (ModelEventType.DONE, ModelEventType.ERROR):
                break
        return events

    async def cancel_turn(self, session_id: str) -> None:
        """Cancel the in-flight response; the session stays open."""
        wire = self._require_session(session_id)
        await wire.send({"type": "response.cancel"})
        self.cancelled += 1

    async def close_session(self, session_id: str) -> None:
        wire = self._sessions.pop(session_id, None)
        if wire is not None:
            await wire.close()

    def _require_session(self, session_id: str) -> RealtimeWire:
        wire = self._sessions.get(session_id)
        if wire is None:
            raise ModelGatewayError(
                ErrorCode.NOT_FOUND_RUN,
                f"no open realtime session {session_id!r}",
            )
        return wire

    # -- availability / info ---------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            status=ProviderStatus.AVAILABLE
            if self._available
            else ProviderStatus.UNAVAILABLE,
            latency_ms=self._latency_ms,
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_version=self.model_version,
            supported_modalities=[ContentType.TEXT, ContentType.AUDIO],
            supports_function_calling=True,
            supports_streaming=True,
            supports_realtime=True,
            supports_json_schema=False,
            snapshot=CLOUD_MODEL_PINNED_SNAPSHOT,
        )
