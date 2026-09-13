"""Entry-guard behaviour that only an ASGI-level probe can observe (#66).

Two review findings live here:
- a client that disconnects MID-BODY must never have its partial upload handed
  to the route as a complete request (nor may a write keep executing after the
  caller left) — asserted both at the ASGI boundary and over a real TCP socket;
- a slow upload is bounded by a wall-clock timeout (408), so a stalled client
  cannot hold a worker hostage.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any

import pytest
import uvicorn
from api_harness import PRIMARY_TOKEN, running_app

from app.api.v1.entry_guard import EntryGuardMiddleware
from app.config import Settings
from app.contracts.errors import ErrorCode
from app.main import PUBLIC_PATHS, create_app


class ProbeApp:
    """Downstream ASGI app that records every call it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.calls.append(scope)
        self.bodies.append(body)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})


def _scope(method: str, path: str, *, headers: list[tuple[bytes, bytes]] = ()) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": list(headers),
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _drive(
    guard, scope, messages: list[dict], *, on_empty: str = "disconnect"
) -> list[dict]:
    """Run the middleware with a scripted receive and capture what it sends.

    ``on_empty="hang"`` models a client that simply stops sending (a stalled
    upload), while the default models a client that went away.
    """
    pending = list(messages)
    sent: list[dict] = []

    async def receive() -> dict:
        if pending:
            return pending.pop(0)
        if on_empty == "hang":
            await asyncio.sleep(30)  # never delivers: only the timeout can fire
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await guard(scope, receive, send)
    return sent


def _guard_for(app, downstream: ProbeApp, **kwargs) -> EntryGuardMiddleware:
    return EntryGuardMiddleware(
        downstream,
        public_paths=PUBLIC_PATHS,
        max_body_bytes=kwargs.get("max_body_bytes", 4096),
        body_timeout_s=kwargs.get("body_timeout_s", 5.0),
    )


class TestDisconnectMidBody:
    def test_partial_upload_is_never_forwarded_to_the_route(self):
        """Review P1: a mid-body disconnect must abort, not "complete" the body."""
        with running_app() as h:
            downstream = ProbeApp()

            async def main():
                scope = _scope(
                    "POST",
                    "/api/v1/sessions",
                    headers=[
                        (b"authorization", f"Bearer {PRIMARY_TOKEN}".encode()),
                        (b"content-type", b"application/json"),
                        (b"content-length", b"40"),
                    ],
                )
                scope["app"] = h.app
                guard = _guard_for(h.app, downstream)
                return await _drive(
                    guard,
                    scope,
                    [
                        {
                            "type": "http.request",
                            "body": b'{"channel": "text"',
                            "more_body": True,
                        },
                        {"type": "http.disconnect"},
                    ],
                )

            sent = asyncio.run(main())
            assert downstream.calls == []  # the route was never entered
            assert sent == []  # nothing answered to a client that is gone
            assert h.service.sessions == {}  # no session was created
            assert h.service.runs == {}

    def test_complete_body_is_still_forwarded(self):
        """Control: the same probe with a complete body does reach the route."""
        with running_app() as h:
            downstream = ProbeApp()

            async def main():
                scope = _scope(
                    "POST",
                    "/api/v1/sessions",
                    headers=[
                        (b"authorization", f"Bearer {PRIMARY_TOKEN}".encode()),
                        (b"content-type", b"application/json"),
                    ],
                )
                scope["app"] = h.app
                guard = _guard_for(h.app, downstream)
                return await _drive(
                    guard,
                    scope,
                    [
                        {
                            "type": "http.request",
                            "body": b'{"channel": "text"}',
                            "more_body": False,
                        }
                    ],
                )

            asyncio.run(main())
        assert len(downstream.calls) == 1
        assert downstream.bodies == [b'{"channel": "text"}']


class TestSlowUpload:
    def test_a_stalled_body_is_answered_with_408(self):
        with running_app(settings_kwargs={"request_body_timeout_s": 0.05}) as h:
            downstream = ProbeApp()

            async def main():
                scope = _scope(
                    "POST",
                    "/api/v1/sessions",
                    headers=[
                        (b"authorization", f"Bearer {PRIMARY_TOKEN}".encode()),
                        (b"content-type", b"application/json"),
                    ],
                )
                scope["app"] = h.app
                guard = _guard_for(h.app, downstream, body_timeout_s=0.05)
                return await _drive(
                    guard,
                    scope,
                    [{"type": "http.request", "body": b"{", "more_body": True}],
                    on_empty="hang",
                )

            sent = asyncio.run(main())
        assert downstream.calls == []  # the route never ran
        start = sent[0]
        assert start["status"] == 408
        payload = json.loads(sent[1]["body"])
        assert payload["code"] == ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT.value

    def test_oversized_body_is_answered_with_413(self):
        with running_app() as h:
            downstream = ProbeApp()

            async def main():
                scope = _scope(
                    "POST",
                    "/api/v1/sessions",
                    headers=[
                        (b"authorization", f"Bearer {PRIMARY_TOKEN}".encode()),
                        (b"content-type", b"application/json"),
                    ],
                )
                scope["app"] = h.app
                guard = _guard_for(h.app, downstream, max_body_bytes=16)
                return await _drive(
                    guard,
                    scope,
                    [
                        {
                            "type": "http.request",
                            "body": b"x" * 64,
                            "more_body": False,
                        }
                    ],
                )

            sent = asyncio.run(main())
        assert downstream.calls == []
        assert sent[0]["status"] == 413
        assert (
            json.loads(sent[1]["body"])["code"]
            == ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE.value
        )


@pytest.fixture(scope="module")
def real_server():
    """A real uvicorn server for the raw-socket disconnect probe."""
    pytest.importorskip("uvicorn")
    token = "dev-rawsocket-000000000"
    settings = Settings(environment="test")
    env_name = settings.auth_credentials_env
    import os

    previous = os.environ.get(env_name)
    os.environ[env_name] = json.dumps(
        [{"tenant_id": "t1", "device_id": "d1", "token": token}]
    )
    app = create_app(settings=settings)
    instance = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    )
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not instance.started and time.time() < deadline:
        time.sleep(0.05)
    assert instance.started
    port = instance.servers[0].sockets[0].getsockname()[1]
    try:
        yield app, port, token
    finally:
        instance.should_exit = True
        thread.join(timeout=15)
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


class TestRealTcpDisconnect:
    def test_partial_upload_then_close_creates_nothing(self, real_server):
        """A real socket that drops mid-upload must not leave a session behind."""
        app, port, token = real_server
        body = b'{"channel": "text"'
        request = (
            b"POST /api/v1/sessions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Authorization: Bearer {token}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body) + 20}\r\n".encode()  # promise more
            + b"\r\n"
            + body
        )
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(request)
            time.sleep(0.2)
        time.sleep(0.4)  # give the server a moment to react to the close
        service = app.state.agent_service
        assert service.sessions == {}
        assert service.runs == {}

    def test_a_complete_raw_request_still_works(self, real_server):
        app, port, token = real_server
        body = json.dumps({"channel": "text"}).encode()
        request = (
            b"POST /api/v1/sessions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Authorization: Bearer {token}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(request)
            response = sock.recv(65536).decode(errors="replace")
        assert "201" in response.split("\r\n", 1)[0]
        assert len(app.state.agent_service.sessions) == 1
