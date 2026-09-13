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

import pytest
from api_harness import (
    OTHER_DEVICE_TOKEN,
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
