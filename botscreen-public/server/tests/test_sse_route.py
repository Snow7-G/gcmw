"""Public SSE route tests (#65B-2 slice B2-B).

Scope guard — this slice ships the ROUTE only:
- authentication and ownership complete BEFORE the first byte;
- ``Last-Event-ID`` resume (plus the explicit ``after_seq`` cursor);
- ``Cache-Control: no-store``;
- a fixed server-side heartbeat that client input cannot influence;
- structured error mapping (JSON envelopes before the stream, one
  ``stream.error`` envelope frame after it).

Deliberately NOT here (B2-C): subscriber leases, connection counting, reconnect
grace and disconnect->cancel propagation. The disconnect test below asserts
that boundary instead of assuming it.

Two test vehicles, on purpose:
* FINITE streams (terminal run, pre-stream errors) go through ``TestClient``;
* OPEN-ENDED streams (idle run + heartbeat, early disconnect) call the real
  route coroutine and read a bounded number of chunks, because Starlette's
  ``TestClient`` buffers a whole response body before returning and could never
  observe a stream that is still open.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from api_harness import (
    OTHER_DEVICE,
    OTHER_TENANT,
    PRINCIPAL,
    Harness,
    new_run,
    new_session,
    running_app,
)
from fastapi import Request
from sse_frames import (
    event_ids,
    event_names,
    event_payload,
    parse_frames,
    protocol_frames,
)

from app.api.v1 import agent_api
from app.api.v1.auth import get_device_principal
from app.contracts.api import Channel, CreateRunRequest, CreateSessionRequest, RunInput
from app.contracts.errors import ErrorEnvelope
from app.contracts.run import RunState
from app.storage.run_repository import (
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
)


@pytest.fixture
def harness() -> Harness:
    with running_app() as h:
        yield h


def _events_url(run_id: str) -> str:
    return f"/api/v1/agent/runs/{run_id}/events"


# ---------------------------------------------------------------------------
# Direct-route vehicle for open-ended streams
# ---------------------------------------------------------------------------


def _request(headers: dict[str, str] | None = None, query: str = "") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/agent/runs/x/events",
            "query_string": query.encode(),
            "headers": raw,
            "state": {},
        }
    )


async def _seed_run(harness: Harness, text: str = "hi", key: str = "k") -> dict:
    """Create session + run through the same service the routes use."""
    service = harness.service
    session_id = service.create_session(
        PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
    ).session_id
    snap = await service.create_run(
        PRINCIPAL,
        CreateRunRequest(
            session_id=session_id,
            input=RunInput(type="text", text=text),
            idempotency_key=key,
        ),
        "req",
        "trace",
    )
    return {"run_id": snap.run_id, "session_id": session_id}


async def _open_stream(
    harness: Harness, run_id: str, request: Request, after_seq: int = 0
):
    """Call the REAL route coroutine and return its streaming response."""
    return await agent_api.stream_run_events(
        run_id,
        request,
        PRINCIPAL,
        after_seq=after_seq,
        last_event_id=None,
        service=harness.service,
        leases=harness.app.state.stream_leases,
        limiter=harness.app.state.rate_limiter,
    )


async def _read(response, *, chunks: int = 1, timeout_s: float = 2.0) -> list[str]:
    """Read a BOUNDED number of chunks from an open stream."""
    collected: list[str] = []

    async def pump() -> None:
        async for chunk in response.body_iterator:
            collected.append(chunk)
            if len(collected) >= chunks:
                return

    await asyncio.wait_for(pump(), timeout_s)
    return collected


async def _disconnect(response) -> None:
    """Close the body iterator the way a disconnecting client does."""
    aclose = getattr(response.body_iterator, "aclose", None)
    if aclose is not None:
        await aclose()


class TestStreamAuthBoundary:
    """Nothing streams for an unauthenticated, foreign or unknown run."""

    def test_anonymous_is_rejected_with_json_envelope(self):
        with running_app(overrides=False) as h:
            res = h.client.get(_events_url("whatever"))
        assert res.status_code == 401
        assert res.json()["code"] == "E_AUTH_MISSING_CREDENTIALS"
        assert res.headers["content-type"].startswith("application/json")
        assert "data:" not in res.text

    def test_foreign_device_is_forbidden_before_first_byte(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.app.dependency_overrides[get_device_principal] = lambda: OTHER_DEVICE
        try:
            res = harness.client.get(_events_url(run["run_id"]))
        finally:
            harness.app.dependency_overrides[get_device_principal] = lambda: PRINCIPAL
        assert res.status_code == 403
        assert harness.env(res).code == "E_AUTHZ_FORBIDDEN"
        assert res.headers["content-type"].startswith("application/json")
        assert "data:" not in res.text

    def test_foreign_tenant_is_forbidden_before_first_byte(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.app.dependency_overrides[get_device_principal] = lambda: OTHER_TENANT
        try:
            res = harness.client.get(_events_url(run["run_id"]))
        finally:
            harness.app.dependency_overrides[get_device_principal] = lambda: PRINCIPAL
        assert res.status_code == 403
        assert harness.env(res).code == "E_AUTHZ_FORBIDDEN"

    def test_unknown_run_is_not_found_before_first_byte(self, harness):
        res = harness.client.get(_events_url("ghost"))
        assert res.status_code == 404
        assert harness.env(res).code == "E_NOT_FOUND_RUN"
        assert res.headers["content-type"].startswith("application/json")

    def test_authorization_happens_exactly_once_per_stream(self, harness):
        """One ownership read feeds the whole stream (no per-page re-check)."""
        run = new_run(harness, new_session(harness)["session_id"])
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").status_code
            == 200
        )
        calls = {"n": 0}
        original = harness.service.authorize_stream

        async def counting(principal, run_id):
            calls["n"] += 1
            return await original(principal, run_id)

        harness.service.authorize_stream = counting
        try:
            res = harness.client.get(_events_url(run["run_id"]))
        finally:
            harness.service.authorize_stream = original
        assert res.status_code == 200
        assert calls["n"] == 1


class TestFiniteStreams:
    """Terminal runs close the stream, so the whole body is observable."""

    def test_terminal_run_streams_validated_frames_and_closes(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").status_code
            == 200
        )

        res = harness.client.get(_events_url(run["run_id"]))
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        assert res.headers["cache-control"] == "no-store"

        frames = protocol_frames(res.text)
        assert event_names(frames) == ["run.accepted", "run.completed"]
        assert event_ids(frames) == [1, 2]
        # the wire id IS the run seq (resume authority), and the payload agrees
        for frame in frames:
            assert frame["data"]["seq"] == frame["id"]
            assert frame["data"]["run_id"] == run["run_id"]
            assert frame["data"]["protocol_version"] == "1.0"
        # the terminal status is derived from the committed target state
        assert event_payload(frames[-1]) == {"status": "cancelled"}

    def test_last_event_id_resumes_without_replay(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")

        res = harness.client.get(
            _events_url(run["run_id"]), headers={"Last-Event-ID": "1"}
        )
        frames = protocol_frames(res.text)
        assert event_ids(frames) == [2]
        assert event_names(frames) == ["run.completed"]

    def test_after_seq_query_resumes(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(_events_url(run["run_id"]) + "?after_seq=1")
        assert event_ids(protocol_frames(res.text)) == [2]

    def test_malformed_last_event_id_falls_back_to_the_cursor(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(
            _events_url(run["run_id"]), headers={"Last-Event-ID": "not-a-seq"}
        )
        assert event_ids(protocol_frames(res.text)) == [1, 2]

    def test_max_of_header_and_cursor_wins(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(
            _events_url(run["run_id"]) + "?after_seq=0",
            headers={"Last-Event-ID": "1"},
        )
        assert event_ids(protocol_frames(res.text)) == [2]

    def test_cursor_never_goes_backwards_with_a_lower_header(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(
            _events_url(run["run_id"]) + "?after_seq=2",
            headers={"Last-Event-ID": "1"},
        )
        assert protocol_frames(res.text) == []  # terminal frame already held

    def test_negative_after_seq_is_rejected_as_a_validation_envelope(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        res = harness.client.get(_events_url(run["run_id"]) + "?after_seq=-1")
        assert res.status_code == 400  # never a framework 422
        assert harness.env(res).code == "E_VALIDATION_INVALID_INPUT"


class TestOpenStreams:
    """Idle streams: first frame, heartbeat, client-controlled window."""

    def test_first_frame_of_a_non_terminal_run_is_accepted(self, harness):
        async def main():
            run = await _seed_run(harness, key="first")
            response = await _open_stream(harness, run["run_id"], _request())
            chunks = await _read(response, chunks=1)
            await _disconnect(response)
            await harness.app.state.stream_leases.shutdown()  # no timer outlives
            return chunks

        frames = protocol_frames("".join(asyncio.run(main())))
        assert event_names(frames) == ["run.accepted"]
        assert event_ids(frames) == [1]

    def test_idle_stream_emits_comment_keepalive_without_seq_progress(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr(agent_api, "SSE_HEARTBEAT_S", 0.05)

        async def main():
            run = await _seed_run(harness, key="hb")
            # cursor already at the newest seq: the only legal traffic is a
            # heartbeat, and it must not carry an id/seq
            response = await _open_stream(
                harness, run["run_id"], _request(), after_seq=1
            )
            chunks = await _read(response, chunks=1)
            await _disconnect(response)
            await harness.app.state.stream_leases.shutdown()
            return chunks

        frames = parse_frames("".join(asyncio.run(main())))
        assert frames and frames[0].get("comment") == "keep-alive"
        assert "id" not in frames[0] and "event" not in frames[0]

    def test_heartbeat_window_is_a_server_constant(self):
        """The window is a server constant: request input cannot change it."""
        params = inspect.signature(agent_api.stream_run_events).parameters
        assert set(params) == {
            "run_id",
            "request",
            "principal",
            "after_seq",
            "last_event_id",
            "service",
            "leases",
            "limiter",
        }
        assert agent_api.SSE_HEARTBEAT_S == 15.0

    def test_client_supplied_heartbeat_is_ignored(self, harness, monkeypatch):
        """A huge ``heartbeat_s`` in the query must NOT delay the keep-alive."""
        monkeypatch.setattr(agent_api, "SSE_HEARTBEAT_S", 0.05)

        async def main():
            run = await _seed_run(harness, key="hb2")
            response = await _open_stream(
                harness,
                run["run_id"],
                _request(query="heartbeat_s=99999"),
                after_seq=1,
            )
            chunks = await _read(response, chunks=1, timeout_s=1.0)
            await _disconnect(response)
            await harness.app.state.stream_leases.shutdown()
            return chunks

        frames = parse_frames("".join(asyncio.run(main())))
        assert frames and frames[0].get("comment") == "keep-alive"


class TestDisconnectBoundary:
    """A disconnect releases the lease and DEFERS the cancel by the grace."""

    def test_disconnect_writes_nothing_before_the_grace_expires(self, harness):
        async def main():
            run = await _seed_run(harness, key="dc")
            touched: list[str] = []
            repository = harness.repository
            original_commit = repository.commit_transition
            original_delete = repository.delete

            async def spy_commit(identity, **kwargs):
                touched.append("commit")
                return await original_commit(identity, **kwargs)

            async def spy_delete(identity):
                touched.append("delete")
                return await original_delete(identity)

            repository.commit_transition = spy_commit
            repository.delete = spy_delete
            leases = harness.app.state.stream_leases
            try:
                response = await _open_stream(harness, run["run_id"], _request())
                await _read(response, chunks=1)
                assert leases.subscribers(run["run_id"]) == 1  # lease held
                await _disconnect(response)
                assert leases.subscribers(run["run_id"]) == 0
                assert leases.pending_expiries() == 1  # grace is now pending
                for _ in range(50):  # the grace has NOT expired yet (2s default)
                    await asyncio.sleep(0)
                identity = RunIdentity(
                    run_id=run["run_id"],
                    tenant_id=PRINCIPAL.tenant_id,
                    device_id=PRINCIPAL.device_id,
                    session_id=run["session_id"],
                )
                state = await repository.state(identity)
                await leases.shutdown()
            finally:
                repository.commit_transition = original_commit
                repository.delete = original_delete
            return touched, state

        touched, durable_state = asyncio.run(main())
        assert touched == []  # the cancel is deferred, not immediate
        assert durable_state is RunState.ACCEPTED

    def test_leases_are_counted_per_stream_not_per_service(self, harness):
        """Connection state lives in the lease registry, not in the service."""
        registry = harness.app.state.stream_leases
        assert registry.tracked() == 0
        for attr in ("subscribers", "leases", "connections", "subscriber_count"):
            assert not hasattr(harness.service, attr)
            assert not hasattr(harness.repository, attr)


class TestStructuredErrorMapping:
    def test_storage_fault_mid_stream_becomes_one_envelope_frame(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])

        async def failing_snapshot(identity, cursor, timeout_s):
            raise RunRepositoryError(RunRepositoryFault.UNAVAILABLE, "redis is down")

        harness.repository.snapshot = failing_snapshot
        res = harness.client.get(_events_url(run["run_id"]))

        frames = parse_frames(res.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "stream.error"
        # a failed read is never progress: no id line at all
        assert "id" not in frames[0]
        envelope = ErrorEnvelope.model_validate(frames[0]["data"])
        assert envelope.code == "E_UNAVAILABLE_OVERLOADED"
        assert envelope.retryable is True
        assert envelope.request_id and envelope.trace_id

    def test_unexpected_fault_never_leaks_exception_text(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])

        async def exploding_snapshot(identity, cursor, timeout_s):
            raise ValueError("secret provider detail /etc/passwd")

        harness.repository.snapshot = exploding_snapshot
        res = harness.client.get(_events_url(run["run_id"]))

        frames = parse_frames(res.text)
        assert len(frames) == 1
        envelope = ErrorEnvelope.model_validate(frames[0]["data"])
        assert envelope.code == "E_INTERNAL_UNKNOWN"
        assert "secret" not in res.text and "/etc/passwd" not in res.text

    def test_cursor_ahead_is_reported_structurally(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(_events_url(run["run_id"]) + "?after_seq=99")

        frames = parse_frames(res.text)
        assert frames and frames[-1]["event"] == "stream.error"
        assert "id" not in frames[-1]
        envelope = ErrorEnvelope.model_validate(frames[-1]["data"])
        assert envelope.code == "E_INTERNAL_UNKNOWN"

    def test_expired_run_mid_stream_maps_to_not_found(self, harness):
        """A run that vanishes (TTL) after authorization ends structurally."""
        run = new_run(harness, new_session(harness)["session_id"])

        async def gone(identity, cursor, timeout_s):
            raise RunRepositoryError(RunRepositoryFault.NOT_FOUND, "expired")

        harness.repository.snapshot = gone
        frames = parse_frames(harness.client.get(_events_url(run["run_id"])).text)
        assert frames[-1]["event"] == "stream.error"
        envelope = ErrorEnvelope.model_validate(frames[-1]["data"])
        assert envelope.code == "E_NOT_FOUND_RUN"

    def test_error_frame_is_valid_json_line_protocol(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(_events_url(run["run_id"]) + "?after_seq=99")
        line = [ln for ln in res.text.split("\n") if ln.startswith("data: ")][-1]
        json.loads(line[len("data: ") :])  # must be a single valid JSON object

    def test_error_frame_ids_match_the_request(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")

        async def boom(identity, cursor, timeout_s):
            raise RunRepositoryError(RunRepositoryFault.UNAVAILABLE, "down")

        harness.repository.snapshot = boom
        res = harness.client.get(
            _events_url(run["run_id"]), headers={"X-Request-ID": "trace-me-42"}
        )
        envelope = ErrorEnvelope.model_validate(parse_frames(res.text)[0]["data"])
        assert envelope.request_id == "trace-me-42"
        assert envelope.trace_id

    def test_stream_failure_does_not_touch_run_state(self, harness):
        """A broken read is a transport problem, not a run transition."""
        run = new_run(harness, new_session(harness)["session_id"])

        async def boom(identity, cursor, timeout_s):
            raise RunRepositoryError(RunRepositoryFault.UNAVAILABLE, "down")

        harness.repository.snapshot = boom
        harness.client.get(_events_url(run["run_id"]))
        assert (
            harness.client.get(f"/api/v1/agent/runs/{run['run_id']}").json()["state"]
            == "ACCEPTED"
        )
        # the run is still cancellable through the normal write path
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").status_code
            == 200
        )
