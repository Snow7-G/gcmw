"""Real-network disconnect semantics (#65B-2 slice B2-C).

``TestClient`` buffers a whole response body and never exercises a real socket,
so the lease behaviour is verified against a REAL ASGI server: a genuine TCP
disconnect must release the lease and, after the reconnect grace, cancel the run
through the lifecycle service — while a reconnect inside the window revokes that
cancel. The server runs once per module on an ephemeral port.
"""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from app.api.v1.auth import DevicePrincipal, get_device_principal
from app.config import Settings
from app.main import create_app

#: real-network timing: short enough for CI, long enough to reconnect inside
RECONNECT_GRACE_S = 0.5
POLL_DEADLINE_S = 15.0

PRINCIPAL = DevicePrincipal(tenant_id="t1", device_id="d1")
#: deterministic low-entropy credential for the real-server fixture
TOKEN = "dev-realserver-000000000"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def server():
    uvicorn = pytest.importorskip("uvicorn")
    settings = Settings(environment="test")
    env_name = settings.auth_credentials_env
    previous = os.environ.get(env_name)
    os.environ[env_name] = json.dumps(
        [{"tenant_id": "t1", "device_id": "d1", "token": TOKEN}]
    )
    app = create_app(
        settings=settings,
        reconnect_grace_s=RECONNECT_GRACE_S,
    )
    app.dependency_overrides[get_device_principal] = lambda: PRINCIPAL
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not instance.started and time.time() < deadline:
        time.sleep(0.05)
    assert instance.started, "uvicorn did not start in time"
    port = instance.servers[0].sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(
            app=app,
            base=f"http://127.0.0.1:{port}/api/v1",
            leases=app.state.stream_leases,
        )
    finally:
        instance.should_exit = True
        thread.join(timeout=15)
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def _new_run(client: httpx.Client, base: str, key: str) -> dict:
    session = client.post(
        f"{base}/sessions", json={"channel": "text"}, headers=AUTH
    ).json()
    res = client.post(
        f"{base}/agent/runs",
        json={
            "session_id": session["session_id"],
            "input": {"type": "text", "text": "real network"},
            "idempotency_key": key,
        },
        headers=AUTH,
    )
    assert res.status_code == 200, res.text
    return res.json()


def _state(client: httpx.Client, base: str, run_id: str) -> str:
    res = client.get(f"{base}/agent/runs/{run_id}", headers=AUTH)
    assert res.status_code == 200, res.text
    return res.json()["state"]


def _wait_state(client: httpx.Client, base: str, run_id: str, expected: str) -> str:
    deadline = time.time() + POLL_DEADLINE_S
    state = _state(client, base, run_id)
    while state != expected and time.time() < deadline:
        time.sleep(0.05)
        state = _state(client, base, run_id)
    return state


def _connect_and_drop(base: str, run_id: str) -> str:
    """Open a stream, prove a frame flows, then DROP the connection.

    Note for the holding tests below: consuming ``iter_lines()`` (even by
    breaking out of it) closes the httpx response, i.e. produces a real
    disconnect. A subscriber that must STAY connected therefore simply does not
    consume frames while it holds the socket open.
    """
    with httpx.stream(
        "GET", f"{base}/agent/runs/{run_id}/events", timeout=30, headers=AUTH
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        first = next(res.iter_lines())
        assert first.startswith("id: 1")
        res.close()  # the link drops here
        return first


class TestRealDisconnect:
    def test_disconnect_cancels_the_run_after_the_grace(self, server):
        with httpx.Client(timeout=10) as client:
            run = _new_run(client, server.base, "real-dc")
            _connect_and_drop(server.base, run["run_id"])
            # the socket is closed; the run survives the grace window…
            assert _wait_state(client, server.base, run["run_id"], "CANCELLED") == (
                "CANCELLED"
            )

    def test_run_stays_active_while_a_subscriber_is_connected(self, server):
        with httpx.Client(timeout=10) as client:
            run = _new_run(client, server.base, "real-hold")
            # hold the socket WITHOUT consuming frames: the lease stays open
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=30,
                headers=AUTH,
            ) as res:
                assert res.status_code == 200
                time.sleep(RECONNECT_GRACE_S * 2)  # well past the grace
                assert _state(client, server.base, run["run_id"]) == "ACCEPTED"
            # leaving for good now starts the grace, and then the cancel
            assert _wait_state(client, server.base, run["run_id"], "CANCELLED") == (
                "CANCELLED"
            )

    def test_reconnect_inside_the_grace_revokes_the_cancel(self, server):
        with httpx.Client(timeout=10) as client:
            run = _new_run(client, server.base, "real-reconnect")
            _connect_and_drop(server.base, run["run_id"])  # first link drops

            # reconnect immediately, inside the grace window, and hold it
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=30,
                headers=AUTH,
            ) as res:
                assert res.status_code == 200
                time.sleep(RECONNECT_GRACE_S * 2)
                assert _state(client, server.base, run["run_id"]) == "ACCEPTED"
            assert _wait_state(client, server.base, run["run_id"], "CANCELLED") == (
                "CANCELLED"
            )

    def test_stream_of_a_terminal_run_never_cancels_twice(self, server):
        with httpx.Client(timeout=10) as client:
            run = _new_run(client, server.base, "real-terminal")
            assert (
                client.delete(
                    f"{server.base}/agent/runs/{run['run_id']}", headers=AUTH
                ).status_code
                == 200
            )
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=10,
                headers=AUTH,
            ) as res:
                body = "".join(res.iter_text())
            assert body.count("event: run.completed") == 1
            time.sleep(RECONNECT_GRACE_S * 2)  # the lease expiry must be a no-op
            assert _state(client, server.base, run["run_id"]) == "CANCELLED"
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=10,
                headers=AUTH,
            ) as res:
                assert "".join(res.iter_text()).count("event: run.completed") == 1

    def test_rapid_drop_reconnect_drop_cancels_exactly_once(self, server):
        """End-to-end version of the revocation race (review P1).

        Drop the link, reconnect INSIDE the grace, then drop again: the second
        grace must survive the revoked first timer's cleanup and cancel the run
        exactly once.
        """
        with httpx.Client(timeout=10) as client:
            run = _new_run(client, server.base, "real-rapid")
            _connect_and_drop(server.base, run["run_id"])  # first link drops
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=30,
                headers=AUTH,
            ) as res:
                assert res.status_code == 200  # reconnect, inside the grace
                time.sleep(RECONNECT_GRACE_S / 2)
                assert _state(client, server.base, run["run_id"]) == "ACCEPTED"
            # the second drop arms a fresh grace -> exactly one cancel
            assert _wait_state(client, server.base, run["run_id"], "CANCELLED") == (
                "CANCELLED"
            )
            with httpx.stream(
                "GET",
                f"{server.base}/agent/runs/{run['run_id']}/events",
                timeout=30,
                headers=AUTH,
            ) as res:
                body = "".join(res.iter_text())
            assert body.count("event: run.completed") == 1
            assert server.leases.tracked() == 0

    def test_server_side_fault_never_cancels_the_run(self, server):
        """A storage fault ends the stream with stream.error — the run lives on.

        Over a real socket this is the case that used to be mistaken for a
        disconnect: the server closes the response, but the client never left.
        """
        from app.storage.run_repository import (
            RunRepositoryError,
            RunRepositoryFault,
        )

        repository = server.app.state.agent_service.repository
        original = repository.snapshot
        failing: set[str] = set()

        async def flaky_snapshot(identity, cursor, timeout_s):
            if identity.run_id in failing:
                raise RunRepositoryError(RunRepositoryFault.UNAVAILABLE, "down")
            return await original(identity, cursor, timeout_s)

        repository.snapshot = flaky_snapshot
        try:
            with httpx.Client(timeout=10) as client:
                run = _new_run(client, server.base, "real-fault")
                failing.add(run["run_id"])
                with httpx.stream(
                    "GET",
                    f"{server.base}/agent/runs/{run['run_id']}/events",
                    timeout=30,
                    headers=AUTH,
                ) as res:
                    body = "".join(res.iter_text())
                assert "stream.error" in body
                assert server.leases.subscribers(run["run_id"]) == 0
                assert server.leases.pending_expiries() == 0  # no grace armed
                time.sleep(RECONNECT_GRACE_S * 2)
                assert _state(client, server.base, run["run_id"]) == "ACCEPTED"
                # and the run is still streamable/cancellable normally
                assert (
                    client.delete(
                        f"{server.base}/agent/runs/{run['run_id']}", headers=AUTH
                    ).status_code
                    == 200
                )
        finally:
            repository.snapshot = original

    def test_no_lease_leaks_after_every_stream_ended(self, server):
        with httpx.Client(timeout=10) as client:
            for i in range(5):
                run = _new_run(client, server.base, f"real-leak-{i}")
                _connect_and_drop(server.base, run["run_id"])
                _wait_state(client, server.base, run["run_id"], "CANCELLED")
            deadline = time.time() + POLL_DEADLINE_S
            while server.leases.tracked() != 0 and time.time() < deadline:
                time.sleep(0.05)
            assert server.leases.tracked() == 0
            assert server.leases.pending_expiries() == 0

    def test_unauthenticated_request_takes_no_lease(self, server):
        server.app.dependency_overrides.clear()
        try:
            with httpx.Client(timeout=10) as client:
                res = client.get(f"{server.base}/agent/runs/ghost/events")  # no header
                assert res.status_code == 401
                assert res.json()["code"] == "E_AUTH_MISSING_CREDENTIALS"
                assert server.leases.tracked() == 0
        finally:
            server.app.dependency_overrides[get_device_principal] = lambda: PRINCIPAL
