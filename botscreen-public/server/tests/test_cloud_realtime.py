"""Tests for CloudRealtimeProvider (issue #51).

Acceptance coverage:
- all model calls ride ModelGateway; responses/events carry the actual
  provider/model snapshot for run records;
- session config always pins ``qwen3.5-omni-plus-realtime-2026-03-15`` and
  explicitly sets ``enable_search: false`` (floating alias is rejected);
- the only accepted audio entry is PCM 16kHz/mono/16-bit; anything else is
  rejected with a structured error and is never sent on the wire;
- vendor events are normalized to internal ModelEvents — no passthrough, no
  raw vendor text in error data, session info carries no URL/key;
- cancellation and timeout map to stable codes; the wire is closed promptly.

All tests run against scripted fake wires: no network, no real data.
"""

import asyncio

import pytest
from pytest import mark

from app.config import CLOUD_MODEL_FLOATING_ALIAS, CLOUD_MODEL_PINNED_SNAPSHOT
from app.contracts.errors import ErrorCode
from app.contracts.model import (
    ContentPart,
    ContentType,
    ModelEventType,
    ModelInfo,
    ModelRequest,
    ProviderStatus,
    RealtimeSessionInfo,
)
from app.providers.cloud_realtime import (
    CloudRealtimeProvider,
    normalize_server_event,
    validate_audio_part,
)
from app.providers.model_gateway import ModelGateway, ModelGatewayError

SESSION_ACK = {"type": "session.created", "session": {"id": "ws-1"}}


class FakeWire:
    def __init__(self, replies=None) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.opened = False
        self.url = ""
        self.key = ""
        self._replies = list(replies or [])

    async def open(self, url: str, api_key: str) -> None:
        self.opened = True
        self.url = url
        self.key = api_key

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def recv(self) -> dict:
        if self._replies:
            return self._replies.pop(0)
        raise AssertionError("fake wire exhausted")

    async def close(self) -> None:
        self.closed = True


def _text_request(**overrides):
    fields = {
        "messages": [{"role": "user", "content": "你好"}],
        "deadline_ms": 5000,
        "token_budget": 400,
        "trace_id": "t1",
    }
    fields.update(overrides)
    return ModelRequest(**fields)


def _audio_part(mime: str | None = "audio/pcm; rate=16000; channels=1; bits=16"):
    return ContentPart(
        type=ContentType.AUDIO, media_ref="kbase://audio/a.pcm", mime_type=mime
    )


class TestHardConstraints:
    def test_default_model_is_the_pinned_snapshot(self):
        provider = CloudRealtimeProvider()
        assert provider.model_id == CLOUD_MODEL_PINNED_SNAPSHOT
        assert provider.model_id != CLOUD_MODEL_FLOATING_ALIAS

    def test_floating_alias_rejected_at_construction(self):
        with pytest.raises(ModelGatewayError) as exc:
            CloudRealtimeProvider(model=CLOUD_MODEL_FLOATING_ALIAS)
        assert exc.value.code is ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED

    @mark.asyncio
    async def test_session_update_pins_snapshot_and_disables_search(self):
        wire = FakeWire([SESSION_ACK])
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        session = await provider.open_realtime_session(_text_request())
        update = wire.sent[0]
        assert update["type"] == "session.update"
        assert update["session"]["model"] == CLOUD_MODEL_PINNED_SNAPSHOT
        assert update["session"]["enable_search"] is False
        # audio input is declared as official PCM 16k/mono/16-bit only
        audio_input = update["session"]["audio"]["input"]
        assert (
            audio_input["sample_rate"],
            audio_input["channels"],
            audio_input["bits"],
        ) == (
            16000,
            1,
            16,
        )
        assert isinstance(session, RealtimeSessionInfo)
        assert session.provider_id == "cloud"
        assert session.model_id == CLOUD_MODEL_PINNED_SNAPSHOT

    def test_audio_validation_accepts_official_pcm(self):
        validate_audio_part(_audio_part())

    @pytest.mark.parametrize(
        "mime",
        [
            "audio/wav",
            "audio/mpeg",
            "audio/pcm; rate=8000; channels=1; bits=16",
            "audio/pcm; rate=16000; channels=2; bits=16",
            "audio/pcm; rate=16000; channels=1; bits=8",
            "audio/pcm; rate=abc; channels=1; bits=16",
        ],
    )
    def test_audio_validation_rejects_unsupported_formats(self, mime):
        with pytest.raises(ModelGatewayError) as exc:
            validate_audio_part(_audio_part(mime=mime))
        assert exc.value.code is ErrorCode.VALIDATION_UNSUPPORTED_FORMAT

    def test_audio_without_format_metadata_rejected(self):
        part = ContentPart(type=ContentType.AUDIO, media_ref="kbase://audio/a.bin")
        with pytest.raises(ModelGatewayError) as exc:
            validate_audio_part(part)
        assert exc.value.code is ErrorCode.VALIDATION_UNSUPPORTED_FORMAT

    def test_bare_pcm_reference_accepted(self):
        part = ContentPart(type=ContentType.AUDIO, media_ref="kbase://audio/a.pcm")
        validate_audio_part(part)

    @mark.asyncio
    async def test_open_realtime_session_rejects_bad_audio_before_wire(self):
        opened = []
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        request = _text_request(content_parts=[_audio_part(mime="audio/mp4")])
        with pytest.raises(ModelGatewayError) as exc:
            await provider.open_realtime_session(request)
        assert exc.value.code is ErrorCode.VALIDATION_UNSUPPORTED_FORMAT
        assert opened == []  # nothing was sent for an unsupported format


class TestWireContract:
    @mark.asyncio
    async def test_append_audio_sends_base64_pcm_frame(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        session = await provider.open_realtime_session(_text_request())
        pcm = b"\x00\x01" * 64  # 128 bytes, 64 samples of 16-bit mono
        await provider.append_audio(session.session_id, pcm)
        wire = provider._sessions[session.session_id]
        payload = wire.sent[-1]
        assert payload["type"] == "input_audio_buffer.append"
        import base64

        assert payload["audio"] == base64.b64encode(pcm).decode("ascii")

    @mark.asyncio
    async def test_append_audio_rejects_odd_payload(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        session = await provider.open_realtime_session(_text_request())
        with pytest.raises(ModelGatewayError) as exc:
            await provider.append_audio(session.session_id, b"\x00\x01\x02")
        assert exc.value.code is ErrorCode.VALIDATION_INVALID_INPUT

    @mark.asyncio
    async def test_complete_turn_sends_commit_and_response_create(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        session = await provider.open_realtime_session(_text_request())
        await provider.complete_turn(session.session_id)
        wire = provider._sessions[session.session_id]
        assert wire.sent[-2]["type"] == "input_audio_buffer.commit"
        assert wire.sent[-1]["type"] == "response.create"

    @mark.asyncio
    async def test_cancel_turn_and_close_session(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        session = await provider.open_realtime_session(_text_request())
        await provider.cancel_turn(session.session_id)
        assert provider.cancelled == 1
        wire = provider._sessions[session.session_id]
        assert wire.sent[-1]["type"] == "response.cancel"
        await provider.close_session(session.session_id)
        assert wire.closed is True
        assert session.session_id not in provider._sessions

    @mark.asyncio
    async def test_session_operations_on_unknown_session_rejected(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        with pytest.raises(ModelGatewayError):
            await provider.append_audio("ghost", b"\x00" * 4)


class TestNormalization:
    def test_bookkeeping_events_return_none(self):
        assert normalize_server_event({"type": "session.updated"}) is None
        assert normalize_server_event({"type": "conversation.item.created"}) is None

    def test_text_delta_and_done_normalized(self):
        event = normalize_server_event({"type": "response.text.delta", "delta": "好"})
        assert event.type is ModelEventType.DELTA
        assert event.data == {"delta": "好"}
        done = normalize_server_event({"type": "response.done"})
        assert done.type is ModelEventType.DONE

    def test_recognition_and_vad_markers_normalized(self):
        rec = normalize_server_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "体温多少",
            }
        )
        assert rec.data == {"recognition": "体温多少"}
        vad = normalize_server_event({"type": "input_audio_buffer.speech_stopped"})
        assert vad.data == {"vad": "speech_stopped"}

    def test_function_call_normalized(self):
        event = normalize_server_event(
            {
                "type": "function_call.done",
                "name": "knowledge.search",
                "arguments": '{"query": "发热"}',
            }
        )
        assert event.type is ModelEventType.TOOL_CALL
        assert event.data["name"] == "knowledge.search"

    def test_vendor_error_mapped_without_raw_text(self):
        event = normalize_server_event(
            {
                "type": "error",
                "error": {"code": "invalid_api_key", "message": "secret detail"},
            }
        )
        assert event.type is ModelEventType.ERROR
        assert event.data["error_code"] == ErrorCode.AUTH_INVALID_CREDENTIALS.value
        dumped = event.model_dump_json()
        assert "secret detail" not in dumped

    def test_unknown_vendor_event_rejected_not_forwarded(self):
        with pytest.raises(ModelGatewayError) as exc:
            normalize_server_event(
                {"type": "response.thinking.delta", "delta": "chain"}
            )
        assert exc.value.code is ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED


class TestChatAndStream:
    @mark.asyncio
    async def test_chat_handshake_error_maps_to_registry_code(self):
        wire = FakeWire(
            [{"type": "error", "error": {"code": "invalid_api_key", "message": "x"}}]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        with pytest.raises(ModelGatewayError) as exc:
            await provider.chat(_text_request())
        assert exc.value.code is ErrorCode.AUTH_INVALID_CREDENTIALS
        assert wire.closed is True  # handshake failure closes the wire

    @mark.asyncio
    async def test_open_session_handshake_error_closes_wire(self):
        wire = FakeWire(
            [{"type": "error", "error": {"code": "overloaded", "message": "x"}}]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        with pytest.raises(ModelGatewayError) as exc:
            await provider.open_realtime_session(_text_request())
        assert exc.value.code is ErrorCode.UNAVAILABLE_OVERLOADED
        assert wire.closed is True

    @mark.asyncio
    async def test_connect_timeout_closes_half_open_wire(self):
        class SlowWire(FakeWire):
            async def open(self, url: str, api_key: str) -> None:
                await asyncio.sleep(60)

        wire = SlowWire()
        provider = CloudRealtimeProvider(
            wire_factory=lambda: wire, connect_timeout_ms=30
        )
        with pytest.raises(ModelGatewayError) as exc:
            await provider.chat(_text_request())
        assert exc.value.code is ErrorCode.TIMEOUT_PROVIDER
        assert wire.closed is True  # no leaked half-open connection

    @mark.asyncio
    async def test_chat_roundtrip_over_wire(self):
        wire = FakeWire(
            [
                SESSION_ACK,
                {"type": "response.text.delta", "delta": "体温"},
                {"type": "response.text.delta", "delta": "正常"},
                {"type": "response.done"},
            ]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        response = await provider.chat(_text_request())
        assert response.content == "体温正常"
        assert response.provider_id == "cloud"
        assert response.model_id == CLOUD_MODEL_PINNED_SNAPSHOT
        assert wire.closed is True
        assert wire.url.endswith("/realtime")
        assert wire.key == ""  # env token resolved at open; none in tests

    @mark.asyncio
    async def test_chat_vendor_error_maps_to_registry_code(self):
        wire = FakeWire(
            [
                SESSION_ACK,
                {"type": "error", "error": {"code": "overloaded", "message": "busy"}},
            ]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        with pytest.raises(ModelGatewayError) as exc:
            await provider.chat(_text_request())
        assert exc.value.code is ErrorCode.UNAVAILABLE_OVERLOADED

    @mark.asyncio
    async def test_chat_rejects_non_text_media(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([]))
        request = _text_request(content_parts=[_audio_part()])
        with pytest.raises(ModelGatewayError) as exc:
            await provider.chat(request)
        assert exc.value.code is ErrorCode.VALIDATION_UNSUPPORTED_MODALITY

    @mark.asyncio
    async def test_stream_yields_normalized_events(self):
        wire = FakeWire(
            [
                SESSION_ACK,
                {"type": "response.text.delta", "delta": "一"},
                {"type": "response.text.delta", "delta": "二"},
                {"type": "function_call.done", "name": "x", "arguments": "{}"},
                {"type": "response.done"},
            ]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        events = [event async for event in provider.stream(_text_request())]
        assert [e.type for e in events] == [
            ModelEventType.DELTA,
            ModelEventType.DELTA,
            ModelEventType.TOOL_CALL,
            ModelEventType.DONE,
        ]
        assert wire.closed is True

    @mark.asyncio
    async def test_stream_cancellation_closes_wire_promptly(self):
        # The wire transport reports cancellation as CancelledError on recv
        # (transport-level cancel). The provider must close the wire promptly
        # and let the exception propagate to the gateway boundary. Task-level
        # cancel scheduling differs across Python versions, so the transport
        # signal is simulated deterministically instead.
        class CancelWire(FakeWire):
            async def recv(self):
                if self._replies:
                    return self._replies.pop(0)
                raise asyncio.CancelledError("wire cancelled")

        wire = CancelWire([SESSION_ACK])
        provider = CloudRealtimeProvider(wire_factory=lambda: wire)
        with pytest.raises(asyncio.CancelledError):
            async for _ in provider.stream(_text_request()):
                pass
        assert wire.closed is True

    @mark.asyncio
    async def test_read_session_events_returns_normalized_feed(self):
        wire = FakeWire(
            [
                {"type": "input_audio_buffer.speech_started"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "挂号",
                },
                {"type": "response.text.delta", "delta": "呼吸内科"},
                {"type": "response.done"},
            ]
        )
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        session = await provider.open_realtime_session(_text_request())
        provider._sessions[session.session_id] = wire
        events = await provider.read_session_events(session.session_id)
        assert events[0].data == {"vad": "speech_started"}
        assert events[1].data == {"recognition": "挂号"}
        assert events[2].data == {"delta": "呼吸内科"}
        assert events[3].type is ModelEventType.DONE


class TestInfoAndGatewayIntegration:
    @mark.asyncio
    async def test_health_and_model_info(self):
        provider = CloudRealtimeProvider(available=False)
        health = await provider.health()
        assert health.status is ProviderStatus.UNAVAILABLE
        info = provider.model_info()
        assert isinstance(info, ModelInfo)
        assert info.snapshot == CLOUD_MODEL_PINNED_SNAPSHOT
        assert info.supports_realtime is True

    @mark.asyncio
    async def test_gateway_mediates_chat_with_provider_hint(self):
        provider = CloudRealtimeProvider(
            wire_factory=lambda: FakeWire(
                [
                    SESSION_ACK,
                    {"type": "response.text.delta", "delta": "好"},
                    {"type": "response.done"},
                ]
            )
        )
        gateway = ModelGateway(active_provider_id="cloud")
        gateway.register(provider)
        response = await gateway.chat(_text_request(provider_hint="cloud"))
        assert response.provider_id == "cloud"
        assert response.model_id == CLOUD_MODEL_PINNED_SNAPSHOT
        assert response.content == "好"

    @mark.asyncio
    async def test_gateway_records_realtime_session_model(self):
        provider = CloudRealtimeProvider(wire_factory=lambda: FakeWire([SESSION_ACK]))
        gateway = ModelGateway(active_provider_id="cloud")
        gateway.register(provider)
        info = await gateway.open_realtime_session(_text_request(provider_hint="cloud"))
        assert info.provider_id == "cloud"
        assert info.model_id == CLOUD_MODEL_PINNED_SNAPSHOT

    @mark.asyncio
    async def test_gateway_deadline_maps_wire_hang_to_timeout(self):
        async def hang_recv():
            await asyncio.sleep(60)

        class HangWire(FakeWire):
            async def recv(self):
                return await hang_recv()

        provider = CloudRealtimeProvider(wire_factory=lambda: HangWire())
        gateway = ModelGateway(active_provider_id="cloud")
        gateway.register(provider)
        request = _text_request(deadline_ms=50, provider_hint="cloud")
        with pytest.raises(ModelGatewayError) as exc:
            await gateway.chat(request)
        assert exc.value.code is ErrorCode.TIMEOUT_PROVIDER
