"""Tests for Agent API, admission service, auth boundary (issue #36).

Round B2-B (review round 2): the admission service keeps LIFECYCLE bookkeeping
only. Run state and sequences belong to ``RunRepository``, and every state
decision in the API layer is answered by the repository — never by a local
mirror that another worker (or a future agent) could have outdated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import pytest
from api_harness import (
    OTHER_DEVICE_TOKEN,
    OTHER_TENANT_TOKEN,
    PRINCIPAL,
    FakeClock,
    ForcedStateRepository,
    Harness,
    VanishingRepository,
    new_run,
    new_session,
    running_app,
)
from sse_frames import event_ids, event_names, protocol_frames

from app.api.v1.agent_api import RunAdmissionService
from app.api.v1.errors import AppError
from app.contracts.api import Channel, CreateRunRequest, CreateSessionRequest, RunInput
from app.contracts.errors import ErrorCode
from app.contracts.run import RunState
from app.storage.run_repository import (
    MemoryRunRepository,
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
)


@pytest.fixture
def harness() -> Harness:
    with running_app() as h:
        yield h


async def _durable_seqs(repository, identity: RunIdentity) -> list[int]:
    page = await repository.snapshot(identity, 0, 0.0)
    return [event.seq for event in page.events]


def _stream_seqs(harness: Harness, run_id: str, **kwargs) -> list[int]:
    """Read a TERMINAL run stream to completion (a non-terminal one is open)."""
    res = harness.client.get(f"/api/v1/agent/runs/{run_id}/events", **kwargs)
    assert res.status_code == 200, res.text
    return event_ids(protocol_frames(res.text))


class TestHealth:
    def test_live(self, harness):
        assert harness.client.get("/api/v1/health/live").json() == {"status": "alive"}

    def test_ready_reports_backends_without_counts(self, harness):
        body = harness.client.get("/api/v1/health/ready").json()
        assert body["status"] == "ready"
        assert body["checks"]["run_repository"] == "memory"
        assert body["checks"]["admission_store"] == "memory"
        assert body["problems"] == []
        dump = json.dumps(body)
        assert "sessions" not in dump and "runs" not in dump


class TestAuthBoundary:
    def test_default_deny_without_principal(self):
        with running_app(default_credential=None) as h:
            res = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert res.status_code == 401
        assert ErrorCode.AUTH_MISSING_CREDENTIALS.value == "E_AUTH_MISSING_CREDENTIALS"
        assert res.json()["code"] == "E_AUTH_MISSING_CREDENTIALS"


class TestSessions:
    def test_create_session_derives_identity(self, harness):
        body = new_session(harness)
        assert body["tenant_id"] == "t1"
        assert body["device_id"] == "d1"

    def test_validation_error_is_envelope(self, harness):
        res = harness.client.post("/api/v1/sessions", json={"channel": "nope"})
        assert res.status_code == 400
        assert harness.env(res).code == "E_VALIDATION_INVALID_INPUT"

    def test_delete_cleans_runs_and_idempotency(self, harness):
        session = new_session(harness)
        new_run(harness, session["session_id"])
        assert len(harness.service.idempotency) == 1
        res = harness.client.delete(f"/api/v1/sessions/{session['session_id']}")
        assert res.status_code == 204
        assert harness.service.sessions == {}
        assert harness.service.runs == {}
        assert harness.service.idempotency == {}

    def test_delete_foreign_session_forbidden(self, harness):
        session = new_session(harness)
        with harness.as_token(OTHER_DEVICE_TOKEN):
            res = harness.client.delete(f"/api/v1/sessions/{session['session_id']}")
        assert res.status_code == 403
        assert harness.env(res).code == "E_AUTHZ_FORBIDDEN"


class TestExpiry:
    def test_expired_session_rejects_new_runs_and_purges_snapshots(self, harness):
        session = new_session(harness)
        run = new_run(
            harness,
            session["session_id"],
            text="敏感医疗问题-必须随过期消失",
            key="ttl",
        )
        assert harness.service.runs[run["run_id"]].snapshot.text.startswith("敏感")
        harness.clock.advance(harness.service.sessions[session["session_id"]].ttl_s + 1)

        assert (
            harness.client.get(f"/api/v1/agent/runs/{run['run_id']}").status_code == 404
        )
        res = harness.client.post(
            "/api/v1/agent/runs",
            json={
                "session_id": session["session_id"],
                "input": {"type": "text", "text": "new"},
                "idempotency_key": "new",
            },
        )
        assert res.status_code == 404
        assert harness.env(res).code == "E_NOT_FOUND_SESSION"
        assert harness.service.sessions == {}
        assert harness.service.runs == {}
        assert harness.service.idempotency == {}

    def test_expired_session_delete_returns_not_found(self, harness):
        session = new_session(harness)
        harness.clock.advance(3600)
        res = harness.client.delete(f"/api/v1/sessions/{session['session_id']}")
        assert res.status_code == 404


class TestRuns:
    def test_snapshot_preserves_input_and_ids(self, harness):
        session = new_session(harness)
        res = harness.client.post(
            "/api/v1/agent/runs",
            json={
                "session_id": session["session_id"],
                "input": {"type": "text", "text": "孩子近视后需要复查吗"},
                "idempotency_key": "snap",
            },
            headers={"X-Request-ID": "my-req-42"},
        )
        snap = harness.service.runs[res.json()["run_id"]].snapshot
        assert snap.text == "孩子近视后需要复查吗"
        assert snap.request_id == "my-req-42"
        assert snap.trace_id

    def test_payload_hash_never_contains_plaintext(self, harness):
        session = new_session(harness)
        secret = "疑似青光眼-20260907-秘密问题"
        run = new_run(harness, session["session_id"], text=secret, key="hash")
        stored = harness.service.runs[run["run_id"]]
        assert secret not in stored.payload_hash
        assert len(stored.payload_hash) == 64
        canonical = json.dumps(
            {"text": secret, "locale": "zh-CN", "channel": "text"},
            sort_keys=True,
            ensure_ascii=False,
        )
        assert stored.payload_hash == hashlib.sha256(canonical.encode()).hexdigest()

    def test_admission_record_holds_no_state_or_sequence(self, harness):
        session = new_session(harness)
        run = new_run(harness, session["session_id"])
        record = harness.service.runs[run["run_id"]]
        assert not hasattr(record, "state")
        assert not hasattr(record, "machine")
        assert not hasattr(record, "sse_events")
        assert run["state"] == "ACCEPTED"
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").json()["state"]
            == "CANCELLED"
        )
        assert _stream_seqs(harness, run["run_id"]) == [1, 2]

    def test_repository_is_the_only_state_and_seq_authority(self, harness):
        session = new_session(harness)
        run = new_run(harness, session["session_id"], key="authority")
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").status_code
            == 200
        )
        res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        frames = protocol_frames(res.text)
        assert event_names(frames) == ["run.accepted", "run.completed"]
        assert event_ids(frames) == [1, 2]  # no gaps, terminal event once

    def test_missing_session_run(self, harness):
        res = harness.client.post(
            "/api/v1/agent/runs",
            json={
                "session_id": "ghost",
                "input": {"type": "text", "text": "hi"},
                "idempotency_key": "k",
            },
        )
        assert res.status_code == 404
        assert harness.env(res).code == "E_NOT_FOUND_SESSION"

    def test_idempotent_replay_same_run_no_duplicate_events(self, harness):
        session = new_session(harness)
        payload = {
            "session_id": session["session_id"],
            "input": {"type": "text", "text": "hi"},
            "idempotency_key": "same-key",
        }
        first = harness.client.post("/api/v1/agent/runs", json=payload).json()
        replay = harness.client.post("/api/v1/agent/runs", json=payload)
        assert replay.status_code == 200
        assert replay.json()["run_id"] == first["run_id"]
        assert (
            harness.client.delete(f"/api/v1/agent/runs/{first['run_id']}").status_code
            == 200
        )
        assert _stream_seqs(harness, first["run_id"]) == [1, 2]

    def test_same_key_different_payload_conflicts(self, harness):
        session = new_session(harness)
        payload = {
            "session_id": session["session_id"],
            "input": {"type": "text", "text": "first"},
            "idempotency_key": "same-key-2",
        }
        assert (
            harness.client.post("/api/v1/agent/runs", json=payload).status_code == 200
        )
        changed = dict(payload, input={"type": "text", "text": "second"})
        res = harness.client.post("/api/v1/agent/runs", json=changed)
        assert res.status_code == 409
        assert harness.env(res).code == "E_CONFLICT_IDEMPOTENCY"

    def test_ownership_enforced(self, harness):
        """Another device with a VALID credential still cannot touch the run."""
        session = new_session(harness)
        run = new_run(harness, session["session_id"])
        run_id = run["run_id"]
        with harness.as_token(OTHER_DEVICE_TOKEN):
            assert harness.client.get(f"/api/v1/agent/runs/{run_id}").status_code == 403
            assert (
                harness.client.get(f"/api/v1/agent/runs/{run_id}/events").status_code
                == 403
            )
            assert (
                harness.client.delete(f"/api/v1/agent/runs/{run_id}").status_code == 403
            )

    def test_cancel_terminal_once_and_events_no_gap(self, harness):
        session = new_session(harness)
        run = new_run(harness, session["session_id"])
        cancelled = harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").json()
        assert cancelled["state"] == "CANCELLED"
        again = harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").json()
        assert again["state"] == "CANCELLED"
        res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        frames = protocol_frames(res.text)
        assert event_names(frames) == ["run.accepted", "run.completed"]
        assert event_ids(frames) == [1, 2]

    def test_missing_run(self, harness):
        res = harness.client.get("/api/v1/agent/runs/ghost")
        assert res.status_code == 404
        assert harness.env(res).code == "E_NOT_FOUND_RUN"


class TestRepositoryAuthority:
    """Review probes: the durable state must win in every decision."""

    def test_durable_terminal_state_frees_the_session_without_any_get(self):
        """A run finished by ANOTHER component must not block the next question."""
        repository = ForcedStateRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            first = new_run(h, session["session_id"], key="first")
            # an external writer (worker/ManagerAgent) finishes the run durably
            repository.forced[first["run_id"]] = RunState.FAILED

            second = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "follow-up"},
                    "idempotency_key": "second",
                },
            )
            assert second.status_code == 200, second.text
            assert second.json()["run_id"] != first["run_id"]

    def test_idempotent_replay_returns_the_durable_state(self):
        repository = ForcedStateRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            payload = {
                "session_id": session["session_id"],
                "input": {"type": "text", "text": "hi"},
                "idempotency_key": "replay",
            }
            first = h.client.post("/api/v1/agent/runs", json=payload).json()
            assert first["state"] == "ACCEPTED"
            repository.forced[first["run_id"]] = RunState.FAILED

            replay = h.client.post("/api/v1/agent/runs", json=payload)
            assert replay.status_code == 200
            assert replay.json()["run_id"] == first["run_id"]
            # the replay reports what the repository says NOW, not a local mirror
            assert replay.json()["state"] == "FAILED"

    def test_replay_of_a_run_that_vanished_durably_creates_a_new_one(self):
        repository = VanishingRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], key="vanish")
            # the durable run expires (TTL) while admission still lists it
            repository.vanished.add(run["run_id"])

            replay = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "hi"},
                    "idempotency_key": "vanish",
                },
            )
            assert replay.status_code == 200, replay.text
            assert replay.json()["run_id"] != run["run_id"]
            # the stale bookkeeping entry went with it
            assert run["run_id"] not in h.service.runs

    def test_stale_run_read_is_never_answered_from_memory(self):
        repository = ForcedStateRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"])
            repository.forced[run["run_id"]] = RunState.CANCELLED
            body = h.client.get(f"/api/v1/agent/runs/{run['run_id']}").json()
            assert body["state"] == "CANCELLED"
            assert body["cancelled"] is True


class StubRepository:
    """Duck-typed repository that fails exactly where a test wants it to."""

    def __init__(self, fault: RunRepositoryFault) -> None:
        self.fault = fault
        self.state_value = RunState.ACCEPTED

    async def create(self, identity) -> int:
        return 1

    async def commit_transition(self, identity, **kwargs) -> int:
        raise RunRepositoryError(self.fault, "stub failure")

    async def state(self, identity) -> RunState:
        return self.state_value

    async def delete(self, identity) -> None:
        return None

    async def snapshot(self, identity, cursor, timeout_s):
        raise AssertionError("streaming is not part of this stub")


class TestRepositoryErrorMapping:
    """Repository failures become mapped ErrorCodes and never move state."""

    async def _cancel_with(self, fault: RunRepositoryFault):
        service = RunAdmissionService(repository=StubRepository(fault))
        session_id = service.create_session(
            PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
        ).session_id
        run = await service.create_run(
            PRINCIPAL,
            CreateRunRequest(
                session_id=session_id,
                input=RunInput(type="text", text="mapped"),
                idempotency_key="m",
            ),
            "req",
            "trace",
        )
        with pytest.raises(AppError) as excinfo:
            await service.cancel_run(PRINCIPAL, run.run_id)
        state = await service.repository.state(service.runs[run.run_id].identity)
        return excinfo.value.code, state

    def test_invariant_failure_maps_and_leaves_state_untouched(self):
        code, state = asyncio.run(self._cancel_with(RunRepositoryFault.INVARIANT))
        assert code is ErrorCode.INTERNAL_UNKNOWN
        assert state is RunState.ACCEPTED

    def test_unavailable_failure_maps_to_overloaded(self):
        code, state = asyncio.run(self._cancel_with(RunRepositoryFault.UNAVAILABLE))
        assert code is ErrorCode.UNAVAILABLE_OVERLOADED
        assert state is RunState.ACCEPTED

    def test_persistent_cas_conflict_is_bounded_not_infinite(self):
        async def main():
            service = RunAdmissionService(
                repository=StubRepository(RunRepositoryFault.CAS_CONFLICT)
            )
            session_id = service.create_session(
                PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
            ).session_id
            run = await service.create_run(
                PRINCIPAL,
                CreateRunRequest(
                    session_id=session_id,
                    input=RunInput(type="text", text="cas"),
                    idempotency_key="cas",
                ),
                "req",
                "trace",
            )
            with pytest.raises(AppError) as excinfo:
                await service.cancel_run(PRINCIPAL, run.run_id)
            return excinfo.value.code

        assert asyncio.run(main()) is ErrorCode.CONFLICT_ACTIVE_RUN

    def test_status_state_cancelled_always_consistent(self, harness):
        session = new_session(harness)
        run = new_run(harness, session["session_id"], key="sc-1")
        before = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}").json()
        assert before["state"] == "ACCEPTED" and before["cancelled"] is False
        cancelled = harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}").json()
        assert cancelled["state"] == "CANCELLED" and cancelled["cancelled"] is True
        after = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}").json()
        assert after["state"] == "CANCELLED" and after["cancelled"] is True


class SessionScopedLockRepository(MemoryRunRepository):
    """Blocks ``create`` for ONE session, to prove locks do not cross sessions."""

    def __init__(self, slow_session_id: str) -> None:
        super().__init__()
        self.slow_session_id = slow_session_id
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, identity) -> int:
        if identity.session_id == self.slow_session_id:
            self.entered.set()
            await self.release.wait()
        return await super().create(identity)


class BlockingDeleteRepository(MemoryRunRepository):
    """Blocks inside ``delete`` so another request can queue on the session lock."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def delete(self, identity) -> None:
        self.entered.set()
        await self.release.wait()
        return await super().delete(identity)


class TestConcurrency:
    """Concurrency is asserted on the service directly, each test with its own
    service + repository inside one event loop."""

    @staticmethod
    def _service(repository=None) -> RunAdmissionService:
        return RunAdmissionService(repository=repository or MemoryRunRepository())

    @staticmethod
    def _session(service: RunAdmissionService) -> str:
        req = CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
        return service.create_session(PRINCIPAL, req).session_id

    def test_concurrent_same_key_same_payload_single_run(self):
        async def main():
            service = self._service()
            session_id = self._session(service)
            req = CreateRunRequest(
                session_id=session_id,
                input=RunInput(type="text", text="同题并发"),
                idempotency_key="conc-same",
            )
            runs = await asyncio.gather(
                *(service.create_run(PRINCIPAL, req, "req", "trace") for _ in range(16))
            )
            stored = [r for r in service.runs.values() if r.session_id == session_id]
            return {r.run_id for r in runs}, len(stored)

        ids, stored = asyncio.run(main())
        assert len(ids) == 1
        assert stored == 1

    def test_concurrent_different_keys_single_active_run(self):
        async def main():
            service = self._service()
            session_id = self._session(service)

            async def fire(i):
                try:
                    await service.create_run(
                        PRINCIPAL,
                        CreateRunRequest(
                            session_id=session_id,
                            input=RunInput(type="text", text=f"q-{i}"),
                            idempotency_key=f"key-{i}",
                        ),
                        f"req-{i}",
                        "trace",
                    )
                    return "created"
                except AppError as exc:
                    return exc.code.value

            outcomes = await asyncio.gather(*(fire(i) for i in range(12)))
            active = [r for r in service.runs.values() if r.session_id == session_id]
            return outcomes, active

        outcomes, active = asyncio.run(main())
        assert outcomes.count("created") == 1
        assert all(o == "E_CONFLICT_ACTIVE_RUN" for o in outcomes if o != "created")
        assert len(active) == 1

    def test_concurrent_cancel_single_terminal_event(self):
        async def main():
            service = self._service()
            session_id = self._session(service)
            run = await service.create_run(
                PRINCIPAL,
                CreateRunRequest(
                    session_id=session_id,
                    input=RunInput(type="text", text="cancel me"),
                    idempotency_key="cc",
                ),
                "req-c",
                "trace",
            )
            await asyncio.gather(
                *(service.cancel_run(PRINCIPAL, run.run_id) for _ in range(12))
            )
            identity = RunIdentity(
                run_id=run.run_id,
                tenant_id=PRINCIPAL.tenant_id,
                device_id=PRINCIPAL.device_id,
                session_id=session_id,
            )
            return (
                await _durable_seqs(service.repository, identity),
                await service.repository.state(identity),
            )

        seqs, state = asyncio.run(main())
        assert seqs == [1, 2]  # exactly one terminal event, no gap
        assert state is RunState.CANCELLED

    def test_create_vs_delete_race_keeps_invariants(self):
        async def main():
            service = self._service()
            session_id = self._session(service)

            async def creator():
                try:
                    await service.create_run(
                        PRINCIPAL,
                        CreateRunRequest(
                            session_id=session_id,
                            input=RunInput(type="text", text="race"),
                            idempotency_key="race-key",
                        ),
                        "req-r",
                        "trace",
                    )
                    return "created"
                except AppError as exc:
                    return exc.code.value

            async def destroyer():
                try:
                    await service.delete_session(PRINCIPAL, session_id)
                    return "deleted"
                except AppError as exc:
                    return exc.code.value

            outcomes = await asyncio.gather(
                *(creator() for _ in range(8)), *(destroyer() for _ in range(8))
            )
            orphans = [
                run_id
                for run_id, run in service.runs.items()
                if run.session_id not in service.sessions
            ]
            return outcomes, orphans

        outcomes, orphans = asyncio.run(main())
        allowed = {"created", "deleted", "E_CONFLICT_ACTIVE_RUN", "E_NOT_FOUND_SESSION"}
        assert set(outcomes) <= allowed
        assert orphans == []

    def test_slow_session_never_blocks_another_session(self):
        """Per-session locks: no cross-tenant head-of-line blocking."""

        async def main():
            repository = SessionScopedLockRepository(slow_session_id="placeholder")
            service = RunAdmissionService(repository=repository)
            slow = self._session(service)
            fast = self._session(service)
            repository.slow_session_id = slow

            async def create(session_id: str, key: str):
                return await service.create_run(
                    PRINCIPAL,
                    CreateRunRequest(
                        session_id=session_id,
                        input=RunInput(type="text", text=key),
                        idempotency_key=key,
                    ),
                    "req",
                    "trace",
                )

            blocked = asyncio.create_task(create(slow, "slow"))
            await repository.entered.wait()
            started = asyncio.get_running_loop().time()
            # the unrelated session completes while the first one is stuck
            await asyncio.wait_for(create(fast, "fast"), timeout=1.0)
            elapsed = asyncio.get_running_loop().time() - started
            assert blocked.done() is False  # still blocked in storage
            repository.release.set()
            await blocked
            return elapsed

        assert asyncio.run(main()) < 1.0


class TestResumableSessionDeletion:
    """Two runs, the second delete fails, the retry completes."""

    class FlakyDeleteRepository(MemoryRunRepository):
        def __init__(self) -> None:
            super().__init__()
            self.fail_for: set[str] = set()
            self.deleted: list[str] = []

        async def delete(self, identity) -> None:
            if identity.run_id in self.fail_for:
                raise RunRepositoryError(RunRepositoryFault.UNAVAILABLE, "storage down")
            self.deleted.append(identity.run_id)
            return await super().delete(identity)

    def test_partial_failure_keeps_memory_and_storage_in_step(self):
        repository = self.FlakyDeleteRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            first = new_run(h, session["session_id"], key="first")
            assert (
                h.client.delete(f"/api/v1/agent/runs/{first['run_id']}").status_code
                == 200
            )
            second = new_run(h, session["session_id"], key="second")
            repository.fail_for.add(second["run_id"])

            res = h.client.delete(f"/api/v1/sessions/{session['session_id']}")
            assert res.status_code == 503
            assert h.env(res).code == "E_UNAVAILABLE_OVERLOADED"

            # exactly as durable as storage is: first is gone everywhere…
            assert first["run_id"] not in h.service.runs
            assert (
                h.client.get(f"/api/v1/agent/runs/{first['run_id']}").status_code == 404
            )
            # …and the still-durable run is still reachable and still listed
            assert second["run_id"] in h.service.runs
            assert (
                h.client.get(f"/api/v1/agent/runs/{second['run_id']}").status_code
                == 200
            )
            # the session survives, so the client can retry the same request
            assert session["session_id"] in h.service.sessions

            # retry: it resumes at the run that was still durable
            repository.fail_for.clear()
            retry = h.client.delete(f"/api/v1/sessions/{session['session_id']}")
            assert retry.status_code == 204
            assert set(repository.deleted) == {first["run_id"], second["run_id"]}
            assert h.service.sessions == {}
            assert h.service.runs == {}
            assert h.service.idempotency == {}
            assert (
                h.client.get(f"/api/v1/agent/runs/{second['run_id']}").status_code
                == 404
            )
            assert (
                h.client.delete(f"/api/v1/sessions/{session['session_id']}").status_code
                == 404
            )

    def test_single_run_failure_then_retry(self):
        repository = self.FlakyDeleteRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], key="only")
            repository.fail_for.add(run["run_id"])

            assert (
                h.client.delete(f"/api/v1/sessions/{session['session_id']}").status_code
                == 503
            )
            assert run["run_id"] in h.service.runs  # nothing was dropped
            assert h.service.sessions != {}

            repository.fail_for.clear()
            assert (
                h.client.delete(f"/api/v1/sessions/{session['session_id']}").status_code
                == 204
            )
            assert h.service.runs == {} and h.service.sessions == {}


class TestSessionLockLifecycle:
    """The session guard must not outlive its users (review P1)."""

    @staticmethod
    def _service(repository=None) -> RunAdmissionService:
        return RunAdmissionService(repository=repository or MemoryRunRepository())

    @staticmethod
    def _new_session(service: RunAdmissionService) -> str:
        return service.create_session(
            PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
        ).session_id

    def test_thousand_sessions_leave_no_guard_behind(self):
        """Create + run + delete 1000 sessions: the lock table must be empty."""

        async def main():
            service = self._service()
            for i in range(1000):
                session_id = self._new_session(service)
                await service.create_run(
                    PRINCIPAL,
                    CreateRunRequest(
                        session_id=session_id,
                        input=RunInput(type="text", text=f"q{i}"),
                        idempotency_key=f"k{i}",
                    ),
                    "req",
                    "trace",
                )
                await service.delete_session(PRINCIPAL, session_id)
            return (
                len(service._session_guards),
                len(service.sessions),
                len(service.runs),
                len(service.idempotency),
            )

        guards, sessions, runs, idempotency = asyncio.run(main())
        assert (guards, sessions, runs, idempotency) == (0, 0, 0, 0)

    def test_expiry_path_also_releases_the_guard(self):
        async def main():
            service = self._service()
            clock = FakeClock()
            service._clock = clock
            session_id = self._new_session(service)
            run = await service.create_run(
                PRINCIPAL,
                CreateRunRequest(
                    session_id=session_id,
                    input=RunInput(type="text", text="ttl"),
                    idempotency_key="ttl",
                ),
                "req",
                "trace",
            )
            clock.advance(3600)
            with pytest.raises(AppError):
                await service.get_run(PRINCIPAL, run.run_id)
            return len(service._session_guards), service.sessions

        guards, sessions = asyncio.run(main())
        assert guards == 0 and sessions == {}

    def test_waiter_shares_one_guard_then_gets_not_found_and_clears(self):
        async def main():
            repository = BlockingDeleteRepository()
            service = self._service(repository)
            session_id = self._new_session(service)
            run = await service.create_run(
                PRINCIPAL,
                CreateRunRequest(
                    session_id=session_id,
                    input=RunInput(type="text", text="queue"),
                    idempotency_key="queue",
                ),
                "req",
                "trace",
            )

            deleter = asyncio.create_task(service.delete_session(PRINCIPAL, session_id))
            await repository.entered.wait()
            guard = service._session_guards[session_id]
            assert guard.lock.locked() and guard.refs == 1

            # a second request for the SAME session queues on that very guard
            waiter = asyncio.create_task(service.get_run(PRINCIPAL, run.run_id))
            for _ in range(1000):
                if guard.refs == 2:
                    break
                await asyncio.sleep(0)
            assert guard.refs == 2
            assert list(service._session_guards) == [session_id]
            assert service._session_guards[session_id] is guard  # never a 2nd lock
            assert waiter.done() is False  # it is queueing, not bypassing

            repository.release.set()
            await deleter
            with pytest.raises(AppError) as excinfo:
                await waiter
            assert excinfo.value.code is ErrorCode.NOT_FOUND_RUN
            return guard.refs, dict(service._session_guards)

        refs, guards = asyncio.run(main())
        assert refs == 0
        assert guards == {}  # released once the last waiter left

    def test_queued_request_after_delete_gets_a_fresh_guard(self):
        """A later request starts a new guard (the old one is gone), and the
        session is simply absent — never a second lock racing the first."""

        async def main():
            service = self._service()
            session_id = self._new_session(service)
            await service.delete_session(PRINCIPAL, session_id)
            assert service._session_guards == {}
            with pytest.raises(AppError) as excinfo:
                await service.delete_session(PRINCIPAL, session_id)
            return excinfo.value.code, dict(service._session_guards)

        code, guards = asyncio.run(main())
        assert code is ErrorCode.NOT_FOUND_SESSION
        assert guards == {}

    def test_concurrent_deletes_share_a_single_guard(self):
        async def main():
            service = self._service()
            session_id = self._new_session(service)
            results = await asyncio.gather(
                *(service.delete_session(PRINCIPAL, session_id) for _ in range(8)),
                return_exceptions=True,
            )
            codes = sorted(r.code.value for r in results if isinstance(r, AppError))
            return codes, dict(service._session_guards)

        codes, guards = asyncio.run(main())
        assert codes == ["E_NOT_FOUND_SESSION"] * 7  # exactly one winner
        assert guards == {}


class TestRequestIds:
    def test_malformed_x_request_id_replaced(self, harness):
        res = harness.client.post(
            "/api/v1/sessions",
            json={"channel": "text"},
            headers={"X-Request-ID": "bad id <x>"},
        )
        echoed = res.headers.get("x-request-id")
        assert echoed and "<x>" not in echoed

    def test_ids_echoed(self, harness):
        res = harness.client.post(
            "/api/v1/sessions",
            json={"channel": "text"},
            headers={"X-Request-ID": "good-req-1"},
        )
        assert res.headers.get("x-request-id") == "good-req-1"
        assert res.headers.get("x-trace-id")


class TestErrorBoundary:
    def test_uncaught_exception_maps_to_internal_envelope(self, harness):
        app = harness.app

        @app.get("/__boom")
        async def boom():
            raise ValueError("sensitive provider detail: /etc/passwd")

        try:
            res = harness.client.get("/__boom")
        finally:
            for route in list(app.routes):
                if getattr(route, "path", None) == "/__boom":
                    app.routes.remove(route)
        assert res.status_code == 500
        body = res.json()
        assert body["code"] == "E_INTERNAL_UNKNOWN"
        assert "sensitive" not in res.text
        assert set(body.keys()) == {
            "code",
            "message",
            "request_id",
            "trace_id",
            "retryable",
            "retry_after_ms",
        }


class TestServiceScope:
    def test_each_application_run_builds_its_own_service(self):
        with running_app() as first:
            service_a = first.service
            session = new_session(first)
            assert session["session_id"] in service_a.sessions
        with running_app() as second:
            assert second.service is not service_a
            assert second.service.sessions == {}  # no state leaks across runs
        assert first.app.state.agent_service is None  # lifespan cleaned up
        assert first.app.state.readiness is None

    def test_service_is_not_a_module_level_singleton(self):
        from app.api.v1 import agent_api

        assert not hasattr(agent_api, "SERVICE")
        assert not hasattr(agent_api.RunAdmissionService, "_locks")


# ---------------------------------------------------------------------------
# P1-1B（第四轮）：Session 幂等契约
# ---------------------------------------------------------------------------


class TestSessionIdempotency:
    def post_session(self, harness, key=None, channel="text", locale="zh-CN"):
        body = {"channel": channel, "locale": locale}
        if key is not None:
            body["idempotency_key"] = key
        return harness.client.post("/api/v1/sessions", json=body)

    def test_same_key_same_payload_replays_original_session(self, harness):
        first = self.post_session(harness, key="voice-session-abc")
        assert first.status_code == 201
        second = self.post_session(harness, key="voice-session-abc")
        assert second.status_code == 201  # replay keeps the 201 contract
        assert second.json()["session_id"] == first.json()["session_id"]
        assert second.json()["created_at"] == first.json()["created_at"]
        assert len(harness.service.sessions) == 1  # no second session created
        assert len(harness.service.session_idempotency) == 1

    def test_same_key_different_payload_is_structured_conflict(self, harness):
        first = self.post_session(harness, key="voice-session-abc")
        assert first.status_code == 201
        res = self.post_session(harness, key="voice-session-abc", locale="en-US")
        assert res.status_code == 409
        assert harness.env(res).code == "E_CONFLICT_IDEMPOTENCY"
        assert len(harness.service.sessions) == 1  # zero extra writes

    def test_same_key_across_tenants_is_independent(self, harness):
        first = self.post_session(harness, key="shared-key")
        with harness.as_token(OTHER_TENANT_TOKEN):
            second = self.post_session(harness, key="shared-key")
        assert first.status_code == 201 and second.status_code == 201
        assert second.json()["session_id"] != first.json()["session_id"]
        assert second.json()["tenant_id"] == "t2" != first.json()["tenant_id"]

    def test_same_key_across_devices_is_independent(self, harness):
        first = self.post_session(harness, key="shared-key")
        with harness.as_token(OTHER_DEVICE_TOKEN):
            second = self.post_session(harness, key="shared-key")
        assert first.status_code == 201 and second.status_code == 201
        assert second.json()["device_id"] == "OTHER" != first.json()["device_id"]

    def test_absent_key_keeps_old_behaviour(self, harness):
        first = self.post_session(harness)
        second = self.post_session(harness)
        assert first.status_code == 201 and second.status_code == 201
        assert (
            second.json()["session_id"] != first.json()["session_id"]
        )  # two distinct sessions, no idempotency

    def test_key_length_bounds_enforced(self, harness):
        res = self.post_session(harness, key="x" * 129)
        assert res.status_code == 400
        assert harness.env(res).code == "E_VALIDATION_INVALID_INPUT"

    def test_key_never_accepts_identity_fields(self, harness):
        """tenant/device/session 身份字段一律拒绝（extra=forbid）。"""
        res = harness.client.post(
            "/api/v1/sessions",
            json={"idempotency_key": "k", "tenant_id": "t2", "device_id": "d2"},
        )
        assert res.status_code == 400


class TestSessionIdempotencyLifecycle:
    def test_explicit_delete_clears_the_index_and_key_is_reusable(self, harness):
        first = harness.client.post(
            "/api/v1/sessions", json={"idempotency_key": "voice-session-abc"}
        )
        session_id = first.json()["session_id"]
        res = harness.client.delete(f"/api/v1/sessions/{session_id}")
        assert res.status_code == 204
        assert harness.service.session_idempotency == {}  # index follows session
        # 同一 key 之后可以安全复用：创建的是全新 Session
        again = harness.client.post(
            "/api/v1/sessions", json={"idempotency_key": "voice-session-abc"}
        )
        assert again.status_code == 201
        assert again.json()["session_id"] != session_id

    def test_ttl_expiry_clears_the_index_and_key_is_reusable(self, harness):
        first = harness.client.post(
            "/api/v1/sessions", json={"idempotency_key": "voice-session-abc"}
        )
        session_id = first.json()["session_id"]
        harness.clock.advance(harness.service.sessions[session_id].ttl_s + 1)
        # 触发路径无关紧要：同 key 重放同样会先走过期检查
        res = harness.client.post(
            "/api/v1/sessions", json={"idempotency_key": "voice-session-abc"}
        )
        assert res.status_code == 201
        assert res.json()["session_id"] != session_id
        # 旧索引已随过期失效：key 现在绑定的是全新 Session（不允许旧条目残留）
        entry = harness.service.session_idempotency[("t1", "d1", "voice-session-abc")]
        assert entry[0] == res.json()["session_id"]
        assert all(
            sid != session_id
            for sid, _h, _r in harness.service.session_idempotency.values()
        )


# ---------------------------------------------------------------------------
# P1-2（第四轮）：主动 TTL sweeper
# ---------------------------------------------------------------------------


class UnavailableOnceRepository(MemoryRunRepository):
    """delete 第一次抛真实故障（UNAVAILABLE），之后恢复。"""

    def __init__(self) -> None:
        super().__init__()
        self.fail_first_delete = True
        self.delete_attempts = 0

    async def delete(self, identity):
        self.delete_attempts += 1
        if self.fail_first_delete:
            self.fail_first_delete = False
            raise RunRepositoryError(
                RunRepositoryFault.UNAVAILABLE, "storage temporarily down"
            )
        return await super().delete(identity)


class TestActiveSweeper:
    def test_sweep_reclaims_expired_session_without_any_client_access(self, harness):
        """③①：无人访问目标 Session/Run，只跑 sweeper——四张表全部归零，
        仓储里的 Run 不可再读。"""
        session = new_session(harness)
        run = new_run(harness, session["session_id"], text="随过期消失", key="k")
        harness.clock.advance(harness.service.sessions[session["session_id"]].ttl_s + 1)

        reclaimed = asyncio.run(harness.service.expire_due_sessions())

        assert reclaimed == 1
        assert harness.service.sessions == {}
        assert harness.service.runs == {}
        assert harness.service.idempotency == {}
        assert harness.service.session_idempotency == {}
        identity = RunIdentity(
            run_id=run["run_id"],
            tenant_id="t1",
            device_id="d1",
            session_id=session["session_id"],
        )
        with pytest.raises(RunRepositoryError) as exc:
            asyncio.run(harness.repository.state(identity))
        assert exc.value.fault is RunRepositoryFault.NOT_FOUND

    def test_sweep_touches_unexpired_sessions_with_zero_writes(self, harness):
        """⑪：未过期 Session 零写入，Run/事件保持完整。"""
        session = new_session(harness)
        run = new_run(harness, session["session_id"])
        before_runs = dict(harness.service.runs)
        before_sessions = dict(harness.service.sessions)

        reclaimed = asyncio.run(harness.service.expire_due_sessions())

        assert reclaimed == 0
        assert harness.service.runs == before_runs
        assert harness.service.sessions == before_sessions
        state = asyncio.run(
            harness.repository.state(
                RunIdentity(
                    run_id=run["run_id"],
                    tenant_id="t1",
                    device_id="d1",
                    session_id=session["session_id"],
                )
            )
        )
        assert state is not None

    def test_real_storage_fault_keeps_session_for_next_sweep(self, harness):
        """⑫：仓储删除暂时故障 → 不丢内存索引、不假装成功；故障恢复后
        下一轮 sweep 完成，状态与仓储始终一致。"""
        repository = UnavailableOnceRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            new_run(h, session["session_id"], text="敏感快照", key="k")
            h.clock.advance(h.service.sessions[session["session_id"]].ttl_s + 1)

            # 第一轮：真实故障 → Session 保留（可重试）
            reclaimed = asyncio.run(h.service.expire_due_sessions())
            assert reclaimed == 0
            assert len(h.service.sessions) == 1  # 内存索引未丢
            assert len(h.service.runs) == 1

            # 第二轮：故障已恢复 → 完成回收，状态与仓储一致
            reclaimed = asyncio.run(h.service.expire_due_sessions())
            assert reclaimed == 1
            assert h.service.sessions == {}
            assert h.service.runs == {}
            assert h.service.session_idempotency == {}

    def test_lifespan_sweeper_runs_and_shutdown_cancels_it(self):
        """⑬③：真实 lifespan 里 sweeper 周期运行并主动回收；应用关闭时
        cancel + await，shutdown 后不再修改状态。"""
        repository = MemoryRunRepository()
        with running_app(
            repository=repository, app_kwargs={"session_sweep_interval_s": 0.02}
        ) as h:
            sweeper = h.app.state.session_sweeper
            assert sweeper is not None and not sweeper.done()
            session = new_session(h)
            h.clock.advance(h.service.sessions[session["session_id"]].ttl_s + 1)
            deadline = time.monotonic() + 5
            while h.service.sessions and time.monotonic() < deadline:
                time.sleep(0.02)  # 等真实后台 sweeper 到下一个 tick
            assert h.service.sessions == {}, "后台 sweeper 未主动回收"
            svc = h.service
        # shutdown 之后：任务已取消结束，状态不再变化
        assert sweeper.done()
        assert sweeper.cancelled()
        assert svc.sessions == {}


# ---------------------------------------------------------------------------
# 第六轮 P1：过期回收必须「先删仓储、再清内存」，且所有入口共用同一条流程
# ---------------------------------------------------------------------------


class TestExpiryReclaimIsSharedAndOrdered:
    """审查复现的缺口：请求路径先把内存索引清掉，sweeper 便再也找不到那些 Run，
    持久记录成了孤儿（"重启才消失"不算回收）。

    矩阵：过期后「先 GET Run / 先 POST Run / 先 DELETE Session / 只跑 sweeper」
    四条入口，每一条都必须把持久记录清干净；再叠加「存储故障保留重试」与
    「多 Run 部分失败后可续跑」两条异常路径。
    """

    @staticmethod
    def _expire(h: Harness, session: dict) -> None:
        h.clock.advance(h.service.sessions[session["session_id"]].ttl_s + 1)

    @staticmethod
    def _identity(session: dict, run: dict) -> RunIdentity:
        return RunIdentity(
            run_id=run["run_id"],
            tenant_id="t1",
            device_id="d1",
            session_id=session["session_id"],
        )

    def _assert_durable_gone(self, repository, session: dict, run: dict) -> None:
        with pytest.raises(RunRepositoryError) as exc:
            asyncio.run(repository.state(self._identity(session, run)))
        assert exc.value.fault is RunRepositoryFault.NOT_FOUND

    def _assert_durable_present(self, repository, session: dict, run: dict) -> None:
        assert isinstance(
            asyncio.run(repository.state(self._identity(session, run))), RunState
        )

    def _assert_memory_empty(self, h: Harness) -> None:
        assert h.service.sessions == {}
        assert h.service.runs == {}
        assert h.service.idempotency == {}
        assert h.service.session_idempotency == {}

    def _assert_memory_intact(self, h: Harness) -> None:
        assert len(h.service.sessions) == 1
        assert len(h.service.runs) == 1

    @pytest.mark.parametrize(
        "first_move", ["get_run", "post_run", "delete_session", "sweeper"]
    )
    def test_every_entry_point_fully_reclaims(self, first_move):
        repository = MemoryRunRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], key="k")
            self._expire(h, session)

            status = self._first_move(h, first_move, session, run)
            if status is not None:
                assert status == 404, first_move

            # 第一条入口之后持久记录就必须没了 —— 不是"等 sweeper 再说"
            self._assert_durable_gone(repository, session, run)
            self._assert_memory_empty(h)
            # 再跑 sweeper 无事可做：证明清理没有被推迟
            assert asyncio.run(h.service.expire_due_sessions()) == 0

    def _first_move(self, h: Harness, move: str, session: dict, run: dict):
        """执行"过期后的第一个动作"，返回 HTTP 状态码（sweeper 返回 None）。"""
        if move == "get_run":
            return h.client.get(f"/api/v1/agent/runs/{run['run_id']}").status_code
        if move == "post_run":
            return h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "再问一次"},
                    "idempotency_key": "k2",
                },
            ).status_code
        if move == "delete_session":
            return h.client.delete(
                f"/api/v1/sessions/{session['session_id']}"
            ).status_code
        assert move == "sweeper"
        assert asyncio.run(h.service.expire_due_sessions()) == 1
        return None

    @pytest.mark.parametrize(
        "first_move", ["get_run", "post_run", "delete_session", "sweeper"]
    )
    def test_storage_fault_keeps_records_and_next_sweep_finishes(self, first_move):
        """真实存储故障：请求被拒、内存索引与持久记录**都留下**，
        故障恢复后 sweeper 清完。这里同时钉住"故障时绝不假装回收成功"。"""
        repository = UnavailableOnceRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], key="k")
            self._expire(h, session)

            if first_move == "sweeper":
                # 故障让这一轮 sweep 一个都收不回来（绝不假装成功）
                assert asyncio.run(h.service.expire_due_sessions()) == 0
            else:
                status = self._first_move(h, first_move, session, run)
                assert status == 404, first_move
            assert repository.delete_attempts == 1

            # 内存索引完整保留：这就是重试入口，不能提前丢
            self._assert_memory_intact(h)
            self._assert_durable_present(repository, session, run)

            # 故障恢复 → sweeper 清完
            assert asyncio.run(h.service.expire_due_sessions()) == 1
            self._assert_memory_empty(h)
            self._assert_durable_gone(repository, session, run)

    def test_partial_failure_resumes_and_cancels_each_task_once(self):
        """两个 Run、第二个删除失败：已删的不再被声称存在、未删的仍是唯一重试
        线索、会话保留；恢复后继续跑完。后台任务在唯一 choke point 被中断，
        且**每个任务恰好一次**（不重复、不泄漏）。"""

        class FlakySecondDelete(MemoryRunRepository):
            def __init__(self) -> None:
                super().__init__()
                self.fail_for: set[str] = set()
                self.attempts: list[str] = []

            async def delete(self, identity) -> None:
                self.attempts.append(identity.run_id)
                if identity.run_id in self.fail_for:
                    raise RunRepositoryError(
                        RunRepositoryFault.UNAVAILABLE, "storage down"
                    )
                return await super().delete(identity)

        class CancelRecorder:
            def __init__(self) -> None:
                self.cancelled: list[str] = []

            def cancel(self, run_id: str) -> None:
                self.cancelled.append(run_id)

        repository = FlakySecondDelete()
        with running_app(repository=repository) as h:
            session = new_session(h)
            first = new_run(h, session["session_id"], key="first")
            # Terminal-out the first run through the REPOSITORY (simulating an
            # external writer), not through the cancel route: the route would
            # cancel the executor task itself, and then "each task is interrupted
            # exactly once by the reclaim" would no longer be what we measure.
            asyncio.run(
                repository.commit_transition(
                    self._identity(session, first),
                    expected_state=RunState.ACCEPTED,
                    next_state=RunState.CANCELLED,
                )
            )
            second = new_run(h, session["session_id"], key="second")
            recorder = CancelRecorder()
            h.service.executor = recorder
            repository.fail_for.add(second["run_id"])
            self._expire(h, session)

            assert asyncio.run(h.service.expire_due_sessions()) == 0  # 会话保留

            # 已删的：内存与仓储都不再声称它存在；任务已中断
            assert first["run_id"] not in h.service.runs
            self._assert_durable_gone(repository, session, first)
            # 未删的：仍在内存里，是唯一的重试线索
            assert second["run_id"] in h.service.runs
            self._assert_durable_present(repository, session, second)
            assert session["session_id"] in h.service.sessions

            repository.fail_for.clear()
            assert asyncio.run(h.service.expire_due_sessions()) == 1
            self._assert_memory_empty(h)
            self._assert_durable_gone(repository, session, second)

        assert sorted(recorder.cancelled) == sorted(
            [first["run_id"], second["run_id"]]
        ), "每个 Run 的后台任务必须恰好被中断一次"

    def test_idempotent_replay_after_expiry_keeps_the_old_session_reclaimable(self):
        """幂等重放：同 key 拿到新会话，但**旧会话不能被丢掉** —— 它是旧
        持久 Run 的清理索引，之后由 sweeper 正常清完。"""
        repository = MemoryRunRepository()
        with running_app(repository=repository) as h:
            key = "voice-session-abc"
            original = h.client.post(
                "/api/v1/sessions", json={"idempotency_key": key}
            ).json()
            run = new_run(h, original["session_id"], key="k")
            self._expire(h, original)

            replay = h.client.post("/api/v1/sessions", json={"idempotency_key": key})
            assert replay.status_code == 201
            assert replay.json()["session_id"] != original["session_id"]

            # 旧会话与它的 Run 仍在内存里（清理索引没被提前丢弃）
            assert original["session_id"] in h.service.sessions
            assert run["run_id"] in h.service.runs
            self._assert_durable_present(repository, original, run)

            # 于是 sweeper 仍能把它清干净
            assert asyncio.run(h.service.expire_due_sessions()) == 1
            self._assert_durable_gone(repository, original, run)
            assert h.service.sessions.keys() == {replay.json()["session_id"]}

    def test_expired_session_is_never_served_on_any_path(self):
        """正常过期（未注入故障）：会话已过期，三条读/写入口一律 404，且都被
        彻底回收 —— 内存与仓储一致。

        本用例刻意关闭故障注入，证明的是「过期即拒绝服务」；存储故障下
        「记录与内存索引都保留、恢复后再清完」由参数化用例
        ``test_storage_fault_keeps_records_and_next_sweep_finishes`` 覆盖。
        """
        repository = UnavailableOnceRepository()
        with running_app(repository=repository) as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], key="k")
            repository.fail_first_delete = False  # 本次不注入故障
            self._expire(h, session)

            assert (
                h.client.get(f"/api/v1/agent/runs/{run['run_id']}").status_code == 404
            )
            assert (
                h.client.post(
                    "/api/v1/agent/runs",
                    json={
                        "session_id": session["session_id"],
                        "input": {"type": "text", "text": "再问一次"},
                        "idempotency_key": "k2",
                    },
                ).status_code
                == 404
            )
            assert (
                h.client.delete(f"/api/v1/sessions/{session['session_id']}").status_code
                == 404
            )
            # 三条路径都把该会话彻底回收了：内存与仓储一致
            self._assert_memory_empty(h)
            self._assert_durable_gone(repository, session, run)
