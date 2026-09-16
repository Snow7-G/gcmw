"""#55A vertical-slice acceptance over the REAL application entry.

These tests go through ``create_app`` + the HTTP boundary (TestClient), not
through the executor directly: a run is created the way a client would, the
executor drives it in the application's own event loop, and the assertions are
made on what the SSE route actually serves — dual-layer events, increasing
``id`` values, ``Last-Event-ID`` replay, and zero stream bytes for an anonymous
or unauthorised caller.
"""

from __future__ import annotations

import asyncio
import time

from app.contracts.common import Channel  # noqa: F401 - documents the contract
from app.storage.run_repository import MemoryRunRepository
from tests.api_harness import (
    OTHER_DEVICE_TOKEN,
    running_app,
)
from tests.sse_frames import event_ids, event_names, protocol_frames


def _wait_terminal(harness, run_id: str, timeout_s: float = 10.0) -> str:
    """Poll the durable state the way a client would."""
    deadline = time.monotonic() + timeout_s
    state = ""
    while time.monotonic() < deadline:
        res = harness.client.get(f"/api/v1/agent/runs/{run_id}")
        assert res.status_code == 200, res.text
        state = res.json()["state"]
        if state in {"COMPLETED", "DEGRADED", "HANDOFF", "FAILED", "CANCELLED"}:
            return state
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached a terminal state ({state})")


def _stream(harness, run_id: str, **kwargs):
    res = harness.client.get(f"/api/v1/agent/runs/{run_id}/events", **kwargs)
    return res, protocol_frames(res.text)


class TestAnsweredOverHttp:
    def test_a_verified_answer_streams_both_layers_with_citations(self):
        with running_app(agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k1",
                },
            ).json()
            state = _wait_terminal(harness, run["run_id"])
            assert state == "COMPLETED"

            res, frames = _stream(harness, run["run_id"])
            assert res.status_code == 200
            names = event_names(frames)
            ids = event_ids(frames)

            # strictly increasing business ids, no gaps, terminal exactly once
            assert ids == list(range(1, len(ids) + 1))
            assert names.count("run.completed") == 1
            assert names[0] == "run.accepted"

            # DUAL LAYER: process progress AND answer content both streamed,
            # and every frame self-describes its layer
            layers = {f["data"]["layer"] for f in frames}
            assert {"process", "answer"} <= layers
            stages = [
                f["data"]["data"]["stage"]
                for f in frames
                if f["event"] == "process.status"
            ]
            assert stages == [
                "guarding",
                "routing",
                "retrieving",
                "drafting",
                "verifying",
                "streaming",
            ]
            deltas = [f for f in frames if f["event"] == "answer.delta"]
            assert deltas, "the answer must be streamed on the answer layer"
            completed = [f for f in frames if f["event"] == "answer.completed"]
            assert len(completed) == 1
            citation = completed[0]["data"]["data"]["citations"][0]
            assert citation["source_id"] == "faq-fever"
            assert citation["content_hash"], (
                "the hash lets the client verify the source"
            )
            assert completed[0]["data"]["data"]["content_origin"] == "approved_faq"

            # the deltas reassemble into the delivered answer
            streamed = "".join(f["data"]["data"]["delta"] for f in deltas)
            assert "体温超过38.5建议门诊就诊" in streamed

            terminal = [f for f in frames if f["event"] == "run.completed"]
            assert terminal[0]["data"]["data"]["result"] == "answered"

    def test_last_event_id_replay_resumes_without_duplication(self):
        with running_app(agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k2",
                },
            ).json()
            _wait_terminal(harness, run["run_id"])

            _, frames = _stream(harness, run["run_id"])
            all_ids = event_ids(frames)
            cursor = all_ids[2]  # the client remembers the third event
            res = harness.client.get(
                f"/api/v1/agent/runs/{run['run_id']}/events",
                headers={"Last-Event-ID": str(cursor)},
            )
            assert res.status_code == 200
            replayed = event_ids(protocol_frames(res.text))
            assert replayed == [i for i in all_ids if i > cursor]


class TestRefusalAndEscalationOverHttp:
    def test_a_question_without_evidence_fails_without_answer_events(self):
        with running_app(agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "近视激光手术多少钱"},
                    "idempotency_key": "k3",
                },
            ).json()
            assert _wait_terminal(harness, run["run_id"]) == "FAILED"
            _, frames = _stream(harness, run["run_id"])
            names = event_names(frames)
            assert "answer.completed" not in names
            assert "answer.delta" not in names
            terminal = [f for f in frames if f["event"] == "run.completed"]
            assert terminal[0]["data"]["data"]["result"] == "refused_no_answer"

    def test_a_red_flag_question_is_escalated_to_a_human(self):
        with running_app(agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "我想自杀"},
                    "idempotency_key": "k4",
                },
            ).json()
            assert _wait_terminal(harness, run["run_id"]) == "HANDOFF"
            _, frames = _stream(harness, run["run_id"])
            names = event_names(frames)
            assert "answer.completed" not in names and "answer.delta" not in names
            terminal = [f for f in frames if f["event"] == "run.completed"]
            assert terminal[0]["data"]["data"]["result"] == "escalated_to_human"
            assert terminal[0]["data"]["data"]["status"] == "handoff"


class TestCancelOverHttp:
    def test_cancelling_a_running_run_stops_it_without_a_late_answer(self):
        stall = {"armed": True}

        class _StallRepository(MemoryRunRepository):
            """Holds the VERIFYING transition once, so the run is clearly
            in-flight while the test cancels it."""

            async def commit_transition(
                self, identity, *, expected_state, next_state, data=None
            ):
                if stall["armed"] and next_state.value == "VERIFYING":
                    stall["armed"] = False
                    await asyncio.sleep(0.8)
                return await super().commit_transition(
                    identity,
                    expected_state=expected_state,
                    next_state=next_state,
                    data=data,
                )

        with running_app(repository=_StallRepository(), agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k5",
                },
            ).json()
            run_id = run["run_id"]
            time.sleep(0.15)  # the executor is inside the stalled transition
            cancelled = harness.client.delete(f"/api/v1/agent/runs/{run_id}")
            assert cancelled.status_code == 200
            assert cancelled.json()["state"] == "CANCELLED"

            assert _wait_terminal(harness, run_id) == "CANCELLED"
            _, frames = _stream(harness, run_id)
            names = event_names(frames)
            assert "answer.completed" not in names and "answer.delta" not in names
            assert names.count("run.completed") == 1
            ids = event_ids(frames)
            assert ids == list(range(1, len(ids) + 1))


class TestStreamAuthorisationOverHttp:
    def test_an_anonymous_caller_gets_no_stream_bytes(self):
        with running_app(agent_executor=True, default_credential=None) as harness:
            res = harness.client.get("/api/v1/agent/runs/whatever/events")
            assert res.status_code == 401
            assert "data:" not in res.text  # zero stream bytes

    def test_a_foreign_device_gets_no_stream_bytes(self):
        with running_app(agent_executor=True) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k6",
                },
            ).json()
            with harness.as_token(OTHER_DEVICE_TOKEN):
                res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
            assert res.status_code == 403
            assert "data:" not in res.text  # zero stream bytes


class TestSessionTeardownStopsTheExecutor:
    """A deleted or expired session must not leave a background task answering
    a question nobody can read any more."""

    @staticmethod
    def _pending_run_tasks(harness, run_id: str) -> int:
        name = f"gcmw-run:{run_id}"
        return len(
            [t for t in harness.app.state.run_executor._tasks if t.get_name() == name]
        )

    def _slow_stall_repository(self) -> MemoryRunRepository:
        stall = {"armed": True}

        class _StallRepository(MemoryRunRepository):
            async def commit_transition(
                self, identity, *, expected_state, next_state, data=None
            ):
                if stall["armed"] and next_state.value == "VERIFYING":
                    stall["armed"] = False
                    await asyncio.sleep(1.5)
                return await super().commit_transition(
                    identity,
                    expected_state=expected_state,
                    next_state=next_state,
                    data=data,
                )

        return _StallRepository()

    def test_deleting_the_session_interrupts_the_run_task(self):
        with running_app(
            repository=self._slow_stall_repository(), agent_executor=True
        ) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k7",
                },
            ).json()
            run_id = run["run_id"]
            time.sleep(0.15)  # the task is inside the stalled transition now
            assert self._pending_run_tasks(harness, run_id) == 1

            res = harness.client.delete(f"/api/v1/sessions/{session['session_id']}")
            assert res.status_code == 204

            # the durable run and every index are gone...
            assert harness.client.get(f"/api/v1/agent/runs/{run_id}").status_code == 404
            # ...and the task did NOT survive the delete: it must unwind well
            # before the stall (1.5 s) would have elapsed
            deadline = time.monotonic() + 1.0
            while (
                self._pending_run_tasks(harness, run_id) and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert self._pending_run_tasks(harness, run_id) == 0

    def test_an_expired_session_interrupts_the_run_task_too(self):
        with running_app(
            repository=self._slow_stall_repository(),
            agent_executor=True,
        ) as harness:
            session = harness.client.post(
                "/api/v1/sessions", json={"channel": "text"}
            ).json()
            run = harness.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "发热怎么办"},
                    "idempotency_key": "k8",
                },
            ).json()
            run_id = run["run_id"]
            time.sleep(0.15)
            assert self._pending_run_tasks(harness, run_id) == 1

            harness.clock.advance(1801)  # the session TTL (1800 s) has passed
            res = harness.client.get(f"/api/v1/agent/runs/{run_id}")
            assert res.status_code == 404  # expiry purged the session

            deadline = time.monotonic() + 1.0
            while (
                self._pending_run_tasks(harness, run_id) and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert self._pending_run_tasks(harness, run_id) == 0
