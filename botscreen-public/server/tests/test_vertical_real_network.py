"""#55A vertical slice over a REAL socket (bounded reads, reconnect, replay).

``TestClient`` buffers whole responses and never exercises a real socket, so
the streaming contract of the vertical slice is verified against a REAL ASGI
server: an in-flight run streamed over genuine TCP, a client that drops after a
bounded number of frames, and a reconnect that resumes from ``Last-Event-ID``
without gaps or duplicated terminal events.
"""

from __future__ import annotations

import asyncio
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
from app.storage.run_repository import MemoryRunRepository
from tests.sse_frames import parse_frames, protocol_frames

PRINCIPAL = DevicePrincipal(tenant_id="t1", device_id="d1")
TOKEN = "dev-realserver-000000000"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
POLL_DEADLINE_S = 15.0

#: holds the run in VERIFYING long enough to stream mid-flight and reconnect
STALL_S = 1.2


class _StallRepository(MemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.stalled_runs: set[str] = set()

    async def commit_transition(
        self, identity, *, expected_state, next_state, data=None
    ):
        # one stall PER RUN: the module-scoped server serves several runs, and
        # each one must be observable in flight
        # stall BEFORE the DRAFTING commit, so the durable state visibly rests
        # in RETRIEVING (VERIFYING exists only for microseconds)
        if next_state.value == "DRAFTING" and identity.run_id not in self.stalled_runs:
            self.stalled_runs.add(identity.run_id)
            await asyncio.sleep(STALL_S)
        return await super().commit_transition(
            identity, expected_state=expected_state, next_state=next_state, data=data
        )


@pytest.fixture(scope="module")
def server():
    uvicorn = pytest.importorskip("uvicorn")
    # the vertical slice is the subject here, not throttling (covered by
    # test_rate_limit.py): raise the windows so state polling cannot exhaust them
    settings = Settings(
        environment="test",
        rate_limit_tenant_per_minute=10_000,
        rate_limit_device_per_minute=10_000,
        rate_limit_session_per_minute=10_000,
    )
    env_name = settings.auth_credentials_env
    previous = os.environ.get(env_name)
    os.environ[env_name] = json.dumps(
        [{"tenant_id": "t1", "device_id": "d1", "token": TOKEN}]
    )
    repository = _StallRepository()
    app = create_app(
        settings=settings,
        repository_factory=lambda _settings: repository,
        agent_executor=True,
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
    # ONE session for the whole module: session creation is rate limited per
    # device, and a terminal run never blocks a new run in the same session
    with httpx.Client(timeout=30) as client:
        session = client.post(
            f"http://127.0.0.1:{port}/api/v1/sessions",
            json={"channel": "text"},
            headers=AUTH,
        ).json()
    try:
        yield SimpleNamespace(
            app=app,
            repository=repository,
            base=f"http://127.0.0.1:{port}/api/v1",
            session_id=session["session_id"],
        )
    finally:
        instance.should_exit = True
        thread.join(timeout=15)
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def _new_run(
    client: httpx.Client, base: str, text: str, key: str, session_id: str
) -> dict:
    res = client.post(
        f"{base}/agent/runs",
        json={
            "session_id": session_id,
            "input": {"type": "text", "text": text},
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


def _wait_terminal(client: httpx.Client, base: str, run_id: str) -> str:
    deadline = time.time() + POLL_DEADLINE_S
    state = _state(client, base, run_id)
    while state not in {"COMPLETED", "DEGRADED", "HANDOFF", "FAILED", "CANCELLED"}:
        assert time.time() < deadline, f"run {run_id} stuck in {state}"
        time.sleep(0.05)
        state = _state(client, base, run_id)
    return state


def test_answered_run_streams_both_layers_and_closes_cleanly(server):
    with httpx.Client(timeout=30) as client:
        run = _new_run(
            client,
            server.base,
            "发热怎么办",
            key="real-1",
            session_id=server.session_id,
        )
        assert _wait_terminal(client, server.base, run["run_id"]) == "COMPLETED"
        res = client.get(
            f"{server.base}/agent/runs/{run['run_id']}/events", headers=AUTH
        )
        frames = protocol_frames(res.text)
        ids = [f["id"] for f in frames]
        names = [f["event"] for f in frames]
        assert ids == list(range(1, len(ids) + 1))  # no gaps, increasing
        assert names[0] == "run.accepted" and names[-1] == "run.completed"
        assert names.count("run.completed") == 1
        layers = {f["data"]["layer"] for f in frames}
        assert {"process", "answer"} <= layers  # DUAL LAYER on the wire
        completed = [f for f in frames if f["event"] == "answer.completed"]
        assert completed[0]["data"]["data"]["citations"][0]["content_hash"]
        assert completed[0]["data"]["data"]["content_origin"] == "approved_faq"


def test_a_mid_run_drop_reconnects_from_last_event_id(server):
    with httpx.Client(timeout=30) as client:
        run = _new_run(
            client,
            server.base,
            "发热怎么办",
            key="real-2",
            session_id=server.session_id,
        )
        # wait until the run is parked inside the stalled VERIFYING transition
        deadline = time.time() + POLL_DEADLINE_S
        while _state(client, server.base, run["run_id"]) != "RETRIEVING":
            assert time.time() < deadline, "run never reached RETRIEVING"
            time.sleep(0.05)

        seen: list[dict] = []
        with httpx.stream(
            "GET",
            f"{server.base}/agent/runs/{run['run_id']}/events",
            timeout=30,
            headers=AUTH,
        ) as res:
            assert res.status_code == 200
            for line in res.iter_lines():  # bounded read: 3 frames, then DROP
                seen.append(line)
                if len([x for x in seen if x.startswith("event:")]) >= 3:
                    break
            res.close()  # the TCP link drops here

        cursor = max(f["id"] for f in parse_frames("\n".join(seen)) if "id" in f)
        assert cursor >= 1

        assert _wait_terminal(client, server.base, run["run_id"]) == "COMPLETED"
        res = client.get(
            f"{server.base}/agent/runs/{run['run_id']}/events",
            headers={**AUTH, "Last-Event-ID": str(cursor)},
        )
        replayed = protocol_frames(res.text)
        replay_ids = [f["id"] for f in replayed]
        assert replay_ids == list(range(cursor + 1, cursor + 1 + len(replay_ids)))
        assert replayed[-1]["event"] == "run.completed"
        names = [f["event"] for f in replayed]
        assert names.count("run.completed") == 1  # terminal exactly once
        assert "answer.completed" in names  # the answer survived the reconnect


def test_an_anonymous_real_request_gets_no_stream_bytes(server):
    with httpx.Client(timeout=30) as client:
        res = client.get(f"{server.base}/agent/runs/whatever/events")
        assert res.status_code == 401
        assert "data:" not in res.text  # zero stream bytes
