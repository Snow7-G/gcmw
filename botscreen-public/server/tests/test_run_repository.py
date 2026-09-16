"""Tests for RunRepository — the single durable authority (#65B-2 slice B2-A).

Unit tests run against an async fake whose ``eval`` twins mirror the Lua
scripts exactly (explicit ``<seq>-0`` ids, tenant-scoped keys, whitelist
enforcement, atomic snapshot payload). The real-Redis class runs when
``GCMW_REDIS_TEST_URL`` is set (CI provides a redis service) and covers the
reviewer's mandated regressions: 16-way concurrent append, transition/append
race, wrong device/session, answer-order safety, tampering detection,
cursor_ahead without waiting, operation timeout, TTL recreation, immutability
and Memory/Redis window parity.
"""

import asyncio
import inspect
import json as _json
import os
import time

import pytest
import pytest_asyncio
from pydantic import ValidationError
from pytest import mark

from app.api.v1.sse_stream import StreamFault, stream_engine
from app.contracts.events import SSEEvent, SSEEventType
from app.contracts.run import RunState
from app.orchestration.state_machine import is_allowed_transition
from app.storage.run_repository import (
    _LUA_APPEND,
    _LUA_COMMIT,
    _LUA_CREATE,
    _LUA_DELETE,
    _LUA_SNAPSHOT,
    MemoryRunRepository,
    RedisRunRepository,
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
    build_event,
)

T1 = RunIdentity(run_id="r1", tenant_id="t1", device_id="d1", session_id="s1")
T1_WRONG_DEVICE = RunIdentity(
    run_id="r1", tenant_id="t1", device_id="dX", session_id="s1"
)
T1_WRONG_SESSION = RunIdentity(
    run_id="r1", tenant_id="t1", device_id="d1", session_id="sX"
)
T2 = RunIdentity(run_id="r1", tenant_id="t2", device_id="d1", session_id="s1")
R2 = RunIdentity(run_id="r2", tenant_id="t1", device_id="d1", session_id="s1")

ACCEPTED = RunState.ACCEPTED
GUARDING = RunState.GUARDING
ROUTING = RunState.ROUTING
RETRIEVING = RunState.RETRIEVING
DRAFTING = RunState.DRAFTING
VERIFYING = RunState.VERIFYING
STREAMING = RunState.STREAMING
COMPLETED = RunState.COMPLETED
FAILED = RunState.FAILED

LEGAL_PATH = [GUARDING, ROUTING, DRAFTING, VERIFYING, STREAMING]


async def _to_streaming(repo, identity=T1) -> int:
    """Advance a fresh run to STREAMING legally; returns the last seq."""
    await repo.create(identity)
    current = ACCEPTED
    seq = 1
    for step in LEGAL_PATH:
        seq = await repo.commit_transition(
            identity, expected_state=current, next_state=step
        )
        current = step
    return seq


class TestIdentityAndConfig:
    def test_identity_fields_must_be_non_empty(self):
        for field in ("run_id", "tenant_id", "device_id", "session_id"):
            kwargs = {
                "run_id": "r",
                "tenant_id": "t",
                "device_id": "d",
                "session_id": "s",
                field: "  ",
            }
            with pytest.raises(ValueError):
                RunIdentity(**kwargs)

    @pytest.mark.parametrize("max_events", [0, -1])
    def test_max_events_validated(self, max_events):
        with pytest.raises(ValueError):
            MemoryRunRepository(max_events=max_events)
        with pytest.raises(ValueError):
            RedisRunRepository(FakeRedis(), max_events=max_events)

    @pytest.mark.parametrize("ttl", [0, -5])
    def test_ttl_validated(self, ttl):
        with pytest.raises(ValueError):
            MemoryRunRepository(ttl_s=ttl)
        with pytest.raises(ValueError):
            RedisRunRepository(FakeRedis(), ttl_s=ttl)

    @pytest.mark.parametrize("op_timeout", [0, -1.0, float("inf")])
    def test_op_timeout_validated(self, op_timeout):
        with pytest.raises(ValueError):
            RedisRunRepository(FakeRedis(), op_timeout_s=op_timeout)

    def test_none_ttl_allowed(self):
        MemoryRunRepository(ttl_s=None)
        RedisRunRepository(FakeRedis(), ttl_s=None)


class TestEventValidation:
    def test_forbidden_and_unknown_data_keys_rejected(self):
        with pytest.raises(RunRepositoryError) as exc:
            build_event(T1, 2, SSEEventType.ANSWER_DELTA, {"chain_of_thought": "x"})
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        with pytest.raises(RunRepositoryError):
            build_event(T1, 2, SSEEventType.PROCESS_STATUS, {"delta": "x"})

    def test_layer_derived_from_event_type(self):
        delta = build_event(T1, 2, SSEEventType.ANSWER_DELTA, {"delta": "a"})
        assert delta.layer.value == "answer"
        completed = build_event(
            T1,
            3,
            SSEEventType.ANSWER_COMPLETED,
            {"citations": [], "content_origin": "ai_generated"},
        )
        assert completed.layer.value == "answer"
        process = build_event(T1, 2, SSEEventType.PROCESS_STATUS, {"stage": "guarding"})
        assert process.layer.value == "process"

    def test_identity_comes_from_arguments(self):
        event = build_event(T1, 2, SSEEventType.PROCESS_STATUS, {"stage": "guarding"})
        assert (event.tenant_id, event.device_id, event.session_id) == (
            "t1",
            "d1",
            "s1",
        )

    def test_nested_data_is_detached_from_caller(self):
        payload = {"citations": [{"id": "c1"}]}
        event = build_event(
            T1,
            4,
            SSEEventType.ANSWER_COMPLETED,
            {**payload, "content_origin": "ai_generated"},
        )
        payload["citations"].append({"id": "tampered"})
        assert event.data["citations"] == [{"id": "c1"}]


class TestMemoryRepository:
    @mark.asyncio
    async def test_create_writes_fixed_first_event(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert [e.event for e in snapshot.events] == [SSEEventType.RUN_ACCEPTED]
        assert snapshot.events[0].device_id == "d1"

    @mark.asyncio
    async def test_illegal_transition_rejected_without_write(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        assert is_allowed_transition(ACCEPTED, COMPLETED) is False
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=ACCEPTED, next_state=COMPLETED
            )
        assert exc.value.fault is RunRepositoryFault.ILLEGAL_TRANSITION
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert len(snapshot.events) == 1 and snapshot.state is ACCEPTED

    @mark.asyncio
    async def test_answer_events_require_streaming_and_keep_state(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT  # ACCEPTED: too early
        await repo.commit_transition(T1, expected_state=ACCEPTED, next_state=GUARDING)
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )  # GUARDING: still before retrieval/verification
        seq = await _to_streaming(repo, R2)
        assert seq == 6
        await repo.append_event(
            R2, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "体"}
        )
        await repo.append_event(
            R2, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "温"}
        )
        snapshot = await repo.snapshot(R2, 0, 0.01)
        assert snapshot.state is STREAMING
        assert [e.event.value for e in snapshot.events][-2:] == [
            "answer.delta",
            "answer.delta",
        ]
        assert all(e.layer.value == "answer" for e in snapshot.events[-2:])

    @mark.asyncio
    async def test_answer_completed_is_single_shot_and_seals_deltas(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1,
                event_type=SSEEventType.ANSWER_COMPLETED,
                data={"citations": [], "content_origin": "ai_generated"},
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )

    @mark.asyncio
    async def test_heartbeat_is_never_persisted(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(T1, event_type=SSEEventType.HEARTBEAT)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert "comment frame" in str(exc.value)

    @mark.asyncio
    async def test_state_only_event_types_are_not_appendable(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(T1, event_type=SSEEventType.RUN_COMPLETED)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_commit_after_terminal_is_invariant(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        await repo.commit_transition(T1, expected_state=ACCEPTED, next_state=GUARDING)
        await repo.commit_transition(T1, expected_state=GUARDING, next_state=FAILED)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.MIC_STATUS, data={"state": "on"}
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_wrong_device_and_session_are_not_found(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        for identity in (T1_WRONG_DEVICE, T1_WRONG_SESSION, T2):
            with pytest.raises(RunRepositoryError) as exc:
                await repo.commit_transition(
                    identity, expected_state=ACCEPTED, next_state=GUARDING
                )
            assert exc.value.fault is RunRepositoryFault.NOT_FOUND
            with pytest.raises(RunRepositoryError):
                await repo.snapshot(identity, 0, 0.01)
            with pytest.raises(RunRepositoryError):
                await repo.delete(identity)
        assert await repo.state(T1) is ACCEPTED  # untouched

    @mark.asyncio
    async def test_concurrent_transitions_single_winner(self):
        repo = MemoryRunRepository()
        await repo.create(T1)

        async def racer():
            try:
                await repo.commit_transition(
                    T1, expected_state=ACCEPTED, next_state=GUARDING
                )
                return "ok"
            except RunRepositoryError as exc:
                return exc.fault.value

        results = await asyncio.gather(*(racer() for _ in range(8)))
        assert results.count("ok") == 1
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert [e.seq for e in snapshot.events] == [1, 2]

    @mark.asyncio
    async def test_ttl_expiry_allows_direct_recreate(self):
        clock = {"now": 100.0}
        repo = MemoryRunRepository(ttl_s=10, monotonic=lambda: clock["now"])
        await repo.create(T1)
        clock["now"] = 200.0
        await repo.create(T1)  # no read needed first: create purges expired
        assert await repo.state(T1) is ACCEPTED

    @mark.asyncio
    async def test_stored_nested_data_and_returned_events_are_immutable(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        citations = [{"id": "c1"}]
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": citations, "content_origin": "ai_generated"},
        )
        citations.append({"id": "tampered"})  # mutate the caller's object
        first = await repo.snapshot(T1, 0, 0.01)
        assert first.events[-1].data["citations"] == [{"id": "c1"}]
        first.events[-1].data["citations"].append({"id": "reverse"})  # mutating read
        second = await repo.snapshot(T1, 0, 0.01)
        assert second.events[-1].data["citations"] == [{"id": "c1"}]

    @mark.asyncio
    async def test_max_events_keeps_tail_but_state_intact(self):
        repo = MemoryRunRepository(max_events=3)
        await _to_streaming(repo)
        for _ in range(4):
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        seqs = [e.seq for e in snapshot.events]
        assert len(seqs) == 3 and seqs[-1] == snapshot.latest_seq
        assert snapshot.oldest_available_seq == seqs[0]
        assert snapshot.state is STREAMING


class TestMemoryBlockingSnapshot:
    @mark.asyncio
    async def test_wait_wakes_on_commit(self):
        repo = MemoryRunRepository()
        await repo.create(T1)

        async def later():
            await asyncio.sleep(0.02)
            await repo.commit_transition(
                T1, expected_state=ACCEPTED, next_state=GUARDING
            )

        task = asyncio.create_task(later())
        snapshot = await repo.snapshot(T1, 1, timeout_s=2.0)
        await task
        assert snapshot.timed_out is False
        assert [e.seq for e in snapshot.events] == [2]

    @mark.asyncio
    async def test_timeout_snapshot_invariants(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 1, timeout_s=0.01)
        assert snapshot.timed_out is True
        assert snapshot.events == ()
        assert snapshot.latest_seq == 1
        assert snapshot.state is ACCEPTED

    @mark.asyncio
    async def test_cursor_ahead_returns_immediately(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 99, timeout_s=5.0)
        assert snapshot.timed_out is False
        assert snapshot.latest_seq == 1 and snapshot.events == ()

    @mark.asyncio
    async def test_terminal_run_never_times_out(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        await repo.commit_transition(T1, expected_state=ACCEPTED, next_state=GUARDING)
        await repo.commit_transition(T1, expected_state=GUARDING, next_state=FAILED)
        snapshot = await repo.snapshot(T1, 3, timeout_s=0.01)
        assert snapshot.timed_out is False and snapshot.terminal_seq == 3


class TestEngineIntegration:
    @mark.asyncio
    async def test_engine_replays_dual_layer_to_terminal(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        await repo.append_event(
            T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "体温"}
        )
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        frames = [
            f
            async for f in stream_engine(
                wait_page=lambda cursor, timeout: repo.snapshot(T1, cursor, timeout),
                heartbeat_s=0.5,
            )
        ]
        events = []
        for chunk in frames:
            for line in chunk.split("\n"):
                if line.startswith("event: "):
                    events.append(line[len("event: ") :])
        assert events == [
            "run.accepted",
            "process.status",
            "process.status",
            "process.status",
            "process.status",
            "process.status",
            "answer.delta",
            "answer.completed",
            "run.completed",
        ]

    @mark.asyncio
    async def test_engine_cursor_ahead_is_classified_by_engine(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        with pytest.raises(Exception) as exc:
            async for _ in stream_engine(
                wait_page=lambda cursor, timeout: repo.snapshot(T1, cursor, timeout),
                after_seq=42,
                heartbeat_s=5.0,
            ):
                pass
        assert (
            getattr(getattr(exc.value, "fault", None), "value", "")
            == StreamFault.CURSOR_AHEAD.value
        )

    @mark.asyncio
    async def test_engine_heartbeats_while_idle_then_terminal(self):
        repo = MemoryRunRepository()
        await repo.create(T1)

        async def later():
            await asyncio.sleep(0.05)
            await repo.commit_transition(
                T1, expected_state=ACCEPTED, next_state=GUARDING
            )
            await repo.commit_transition(T1, expected_state=GUARDING, next_state=FAILED)

        task = asyncio.create_task(later())
        frames = [
            f
            async for f in stream_engine(
                wait_page=lambda cursor, timeout: repo.snapshot(T1, cursor, timeout),
                heartbeat_s=0.01,
            )
        ]
        await task
        assert ": keep-alive\n\n" in frames
        assert frames[-1].startswith("id: 3\nevent: run.completed")


class TestWindowParity:
    @mark.asyncio
    async def test_oldest_available_seq_matches_across_implementations(self):
        memory = MemoryRunRepository(max_events=3)
        redis_backed = RedisRunRepository(FakeRedis(), max_events=3)
        windows = []
        for repo, identity in ((memory, T1), (redis_backed, T2)):
            await repo.create(identity)
            current = ACCEPTED
            for step in LEGAL_PATH:
                await repo.commit_transition(
                    identity, expected_state=current, next_state=step
                )
                current = step
            for _ in range(4):
                await repo.append_event(
                    identity,
                    event_type=SSEEventType.ANSWER_DELTA,
                    data={"delta": "x"},
                )
            windows.append(await repo.snapshot(identity, 0, 0.01))
        # create(1) + 5 transitions(2..6) + 4 deltas(7..10), window = 3 tail
        assert [
            (w.oldest_available_seq, w.latest_seq, len(w.events)) for w in windows
        ] == [(8, 10, 3), (8, 10, 3)]  # Memory and Redis agree exactly


class FakeRedis:
    """Async redis duck: python twins of the repository Lua scripts."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.wrong_type: set[str] = set()
        self.unavailable = False
        self.hang = False
        self.xread_calls = 0
        self.maxlen: dict[str, int] = {}
        self._wake: dict[str, asyncio.Event] = {}

    # -- twins -----------------------------------------------------------------

    def _event_of(self, name: str) -> asyncio.Event:
        return self._wake.setdefault(name, asyncio.Event())

    def _ids(self, name: str) -> list[str]:
        return [entry_id for entry_id, _ in self.streams.get(name, [])]

    def _trim(self, name: str) -> None:
        limit = self.maxlen.get(name)
        if limit is not None:
            stream = self.streams.get(name, [])
            if len(stream) > limit:
                del stream[: len(stream) - limit]

    def _twin_create(self, keys, args):
        state_key, stream_key = keys
        tenant, device, session, _ttl, maxlen, event_json = args[:6]
        if state_key in self.hashes or stream_key in self.streams:
            return 0
        self.hashes[state_key] = {
            "state": "ACCEPTED",
            "tenant_id": tenant,
            "device_id": device,
            "session_id": session,
            "latest_seq": "1",
        }
        self.streams.setdefault(stream_key, []).append(("1-0", {"event": event_json}))
        self.maxlen[stream_key] = int(maxlen)
        self._trim(stream_key)
        self._event_of(stream_key).set()
        return 1

    def _guard(self, state_key, stream_key, args):
        """Shared ownership/terminal checks; returns a control code or None."""
        if state_key not in self.hashes:
            return -1
        fields = self.hashes[state_key]
        if fields["tenant_id"] != args[0]:
            return -2
        if fields["device_id"] != args[1] or fields["session_id"] != args[2]:
            return -4
        if stream_key not in self.streams:
            return -5
        if "terminal_seq" in fields:
            return -3
        return None

    def _twin_commit(self, keys, args):
        state_key, stream_key = keys
        guard = self._guard(state_key, stream_key, args)
        if guard is not None:
            return guard
        fields = self.hashes[state_key]
        expected_state, expected_seq = args[3], int(args[4])
        if fields["state"] != expected_state:
            return -7
        if int(fields["latest_seq"]) != expected_seq:
            return -6
        if args[10] == "1":
            entries = self.streams[stream_key]
            if not entries:
                return -9
            if entries[-1][0] != f"{expected_seq}-0":  # physical id must match
                return -9
            try:
                newest = _json.loads(entries[-1][1]["event"])
            except Exception:  # noqa: BLE001
                return -9
            origins = set(args[13].split(","))
            if (
                newest.get("event") != "answer.completed"
                or str(newest.get("seq")) != str(expected_seq)
                or newest.get("tenant_id") != args[0]
                or newest.get("device_id") != args[1]
                or newest.get("session_id") != args[2]
                or newest.get("run_id") != args[11]
                or newest.get("layer") != "answer"
                or newest.get("protocol_version") != args[12]
                or (newest.get("data") or {}).get("content_origin") not in origins
            ):
                return -9  # no real answer.completed proves the seal
        seq = expected_seq + 1
        self.streams[stream_key].append((f"{seq}-0", {"event": args[6]}))
        fields["state"] = args[5]
        fields["latest_seq"] = str(seq)
        if args[7] == "1":
            fields["terminal_seq"] = str(seq)
        self.maxlen[stream_key] = int(args[8])
        self._trim(stream_key)
        self._event_of(stream_key).set()
        return seq

    def _twin_append(self, keys, args):
        state_key, stream_key = keys
        guard = self._guard(state_key, stream_key, args)
        if guard is not None:
            return guard
        fields = self.hashes[state_key]
        entries = self.streams[stream_key]
        if entries:
            latest = int(fields["latest_seq"])
            if entries[-1][0] != f"{latest}-0":
                return -10  # physical id does not match the business seq
            payload = entries[-1][1].get("event")
            if payload is None:
                return -10
            try:
                newest = _json.loads(payload)
            except Exception:  # noqa: BLE001
                return -10  # corrupt tail: fail closed, never append
            if (
                str(newest.get("seq")) != str(latest)
                or newest.get("tenant_id") != args[0]
                or newest.get("device_id") != args[1]
                or newest.get("session_id") != args[2]
                or newest.get("run_id") != args[10]
            ):
                return -10
            if newest.get("event") == "answer.completed":
                return -8  # the answer is sealed: no further answer events
        if fields["state"] not in set(args[7].split(",")):
            return -8
        expected_seq = int(args[3])
        if int(fields["latest_seq"]) != expected_seq:
            return -6
        seq = expected_seq + 1
        self.streams[stream_key].append((f"{seq}-0", {"event": args[4]}))
        fields["latest_seq"] = str(seq)
        self.maxlen[stream_key] = int(args[6])
        self._trim(stream_key)
        self._event_of(stream_key).set()
        return seq

    def _twin_delete(self, keys, args):
        state_key, stream_key = keys
        has_state = state_key in self.hashes
        has_stream = stream_key in self.streams
        if not has_state and not has_stream:
            return -1
        if not has_state or not has_stream:
            return -5  # symmetric orphan gate: refuse, zero writes
        fields = self.hashes[state_key]
        if (
            fields["tenant_id"] != args[0]
            or fields["device_id"] != args[1]
            or fields["session_id"] != args[2]
        ):
            return -2
        self.hashes.pop(state_key, None)
        self.streams.pop(stream_key, None)
        return 1

    def _twin_snapshot(self, keys, args):
        state_key, stream_key = keys
        tenant, device, session, start, limit = args[:5]
        if state_key not in self.hashes:
            if stream_key in self.streams:
                raise RuntimeError("ORPHAN_STREAM")
            raise RuntimeError("NOT_FOUND")
        if state_key in self.wrong_type:
            raise RuntimeError("WRONGTYPE operation")
        fields = self.hashes[state_key]
        if (
            fields["tenant_id"] != tenant
            or fields["device_id"] != device
            or fields["session_id"] != session
        ):
            raise RuntimeError("NOT_FOUND")
        if stream_key not in self.streams:
            raise RuntimeError("ORPHAN_STATE")
        entries = self.streams[stream_key]
        for _entry_id, entry_payload in entries:
            if "event" not in entry_payload:
                raise RuntimeError("MALFORMED_ENTRY")
        oldest = entries[0][0] if entries else ""
        newest = entries[-1][0] if entries else ""
        newest_payload = entries[-1][1]["event"] if entries else ""
        terminal = fields.get("terminal_seq", "")
        terminal_id = ""
        terminal_payload = ""
        if terminal:
            for entry_id, entry_payload in entries:
                if entry_id.split("-")[0] == terminal:
                    terminal_id = entry_id
                    terminal_payload = entry_payload["event"]
            if not terminal_id:
                raise RuntimeError("MALFORMED_TERMINAL")
            try:
                decoded = _json.loads(terminal_payload)
            except Exception as exc:
                raise RuntimeError("MALFORMED_TERMINAL") from exc
            if decoded.get("event") != "run.completed":
                raise RuntimeError("MALFORMED_TERMINAL")
        page = entries
        if start != "-":
            cursor = int(start[1:].split("-")[0])
            page = [e for e in entries if int(e[0].split("-")[0]) > cursor]
        page = page[: int(limit)]
        out = [
            fields["state"],
            fields["latest_seq"],
            terminal,
            fields["device_id"],
            fields["session_id"],
            oldest,
            newest,
            newest_payload,
            terminal_id,
            terminal_payload,
        ]
        for entry_id, entry_payload in page:
            out.append(entry_id)
            out.append(entry_payload["event"])
        return out

    # -- client duck -----------------------------------------------------------

    async def eval(self, script, numkeys, *keys_and_args):
        if self.hang:
            await asyncio.sleep(3600)
        if self.unavailable:
            raise ConnectionError("redis down")
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        if script == _LUA_CREATE:
            return self._twin_create(keys, args)
        if script == _LUA_COMMIT:
            return self._twin_commit(keys, args)
        if script == _LUA_APPEND:
            return self._twin_append(keys, args)
        if script == _LUA_DELETE:
            return self._twin_delete(keys, args)
        if script == _LUA_SNAPSHOT:
            return self._twin_snapshot(keys, args)
        raise AssertionError("fake does not understand this script")

    async def xread(self, streams, count=None, block=None):
        self.xread_calls += 1
        if self.hang:
            await asyncio.sleep(3600)
        if self.unavailable:
            raise ConnectionError("redis down")
        name, after = next(iter(streams.items()))
        entries = [e for e in self.streams.get(name, []) if e[0] > after]
        if entries:
            return [(name, entries[:count])]
        event = self._event_of(name)
        event.clear()
        try:
            await asyncio.wait_for(event.wait(), timeout=(block or 1) / 1000)
        except TimeoutError:
            return None
        entries = [e for e in self.streams.get(name, []) if e[0] > after]
        return [(name, entries[:count])] if entries else None

    async def delete(self, *names):
        for name in names:
            self.hashes.pop(name, None)
            self.streams.pop(name, None)


class TamperingClient:
    """Wraps a redis client and mutates the store right before one script runs.

    Simulates the reviewer's TOCTOU probe: the Python pre-read already
    happened, so only the atomic Lua gate can catch the tampering.
    """

    def __init__(self, inner, tamper=None) -> None:
        self._inner = inner
        self._tamper = tamper

    async def eval(self, script, numkeys, *keys_and_args):
        if self._tamper is not None:
            callback, self._tamper = self._tamper, None
            result = callback()
            if inspect.isawaitable(result):
                await result
        return await self._inner.eval(script, numkeys, *keys_and_args)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _redis_repo(redis=None, **kwargs):
    return RedisRunRepository(redis or FakeRedis(), **kwargs)


class TestRedisRepository:
    @mark.asyncio
    async def test_keys_are_tenant_scoped_and_same_slot(self):
        repo = _redis_repo()
        state_key, stream_key = repo.keys(T1)
        assert "t1" in state_key and "r1" in state_key
        assert state_key.split("}")[0] == stream_key.split("}")[0]  # same hash tag
        assert repo.keys(T2)[0] != state_key  # another tenant → different key

    @mark.asyncio
    async def test_create_commit_and_snapshot(self):
        repo = _redis_repo()
        await repo.create(T1)
        assert (
            await repo.commit_transition(
                T1, expected_state=ACCEPTED, next_state=GUARDING
            )
            == 2
        )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert [e.seq for e in snapshot.events] == [1, 2]
        assert snapshot.state is GUARDING

    @mark.asyncio
    async def test_concurrent_appends_produce_contiguous_returns(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        results = await asyncio.gather(
            *(
                repo.append_event(
                    T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": f"d{i}"}
                )
                for i in range(16)
            ),
            return_exceptions=True,
        )
        seqs = sorted(r for r in results if isinstance(r, int))
        # no duplicate seqs, no gaps: some may retry-conflict out, but every
        # successful return must exist in the store
        assert len(seqs) == len(set(seqs))
        snapshot = await repo.snapshot(T1, 0, 0.01)
        stored = [e.seq for e in snapshot.events]
        assert set(seqs).issubset(set(stored))
        assert stored == list(range(stored[0], stored[0] + len(stored)))
        assert stored[-1] == snapshot.latest_seq
        for failure in (r for r in results if isinstance(r, Exception)):
            assert isinstance(failure, RunRepositoryError)

    @mark.asyncio
    async def test_transition_and_append_race_has_no_duplicate_seq(self):
        repo = _redis_repo()
        await repo.create(T1)

        async def transition():
            try:
                return (
                    "t",
                    await repo.commit_transition(
                        T1, expected_state=ACCEPTED, next_state=GUARDING
                    ),
                )
            except RunRepositoryError:
                return ("t", None)

        async def append():
            try:
                return (
                    "a",
                    await repo.append_event(
                        T1, event_type=SSEEventType.MIC_STATUS, data={"state": "on"}
                    ),
                )
            except RunRepositoryError:
                return ("a", None)

        results = await asyncio.gather(transition(), append())
        successes = [seq for _, seq in results if seq]
        assert len(successes) == len(set(successes))  # never a duplicate seq
        snapshot = await repo.snapshot(T1, 0, 0.01)
        stored = [e.seq for e in snapshot.events]
        assert stored == list(range(1, len(stored) + 1))  # contiguous, no gaps
        assert (
            sum(1 for e in snapshot.events if e.event is SSEEventType.PROCESS_STATUS)
            == 1
        )

    @mark.asyncio
    async def test_wrong_device_or_session_refused_everywhere(self):
        repo = _redis_repo()
        await repo.create(T1)
        for identity in (T1_WRONG_DEVICE, T1_WRONG_SESSION, T2):
            for call in (
                lambda i=identity: repo.state(i),
                lambda i=identity: repo.snapshot(i, 0, 0.01),
                lambda i=identity: repo.delete(i),
                lambda i=identity: repo.commit_transition(
                    i, expected_state=ACCEPTED, next_state=GUARDING
                ),
                lambda i=identity: repo.append_event(
                    i, event_type=SSEEventType.MIC_STATUS, data={"state": "on"}
                ),
            ):
                with pytest.raises(RunRepositoryError) as exc:
                    await call()
                assert exc.value.fault is RunRepositoryFault.NOT_FOUND
        assert await repo.state(T1) is ACCEPTED

    @mark.asyncio
    async def test_answer_safety_whitelist_in_redis(self):
        repo = _redis_repo()
        await repo.create(T1)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        with pytest.raises(RunRepositoryError):
            await repo.append_event(T1, event_type=SSEEventType.HEARTBEAT)
        await _to_streaming(repo, R2)
        await repo.append_event(
            R2,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                R2, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "late"}
            )

    @mark.asyncio
    async def test_tampered_stream_id_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        _, stream_key = repo.keys(T1)
        _entry_id, payload = redis.streams[stream_key][0]
        redis.streams[stream_key][0] = ("7-0", payload)  # id no longer matches seq
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_tampered_event_identity_is_invariant(self):
        import json

        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        _, stream_key = repo.keys(T1)
        _, payload = redis.streams[stream_key][0]
        forged = json.loads(payload["event"])
        forged["tenant_id"] = "t2"  # another tenant's event planted in the stream
        redis.streams[stream_key][0] = ("1-0", {"event": json.dumps(forged)})
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_tampered_latest_seq_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        state_key, _ = repo.keys(T1)
        redis.hashes[state_key]["latest_seq"] = "9"  # hash ahead of the stream
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_terminal_seq_without_terminal_event_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        await repo.commit_transition(T1, expected_state=ACCEPTED, next_state=GUARDING)
        state_key, stream_key = repo.keys(T1)
        # plant a process.status event and claim it is the terminal position
        redis.streams[stream_key].append(
            ("2-0", {"event": redis.streams[stream_key][-1][1]["event"]})
        )
        redis.hashes[state_key]["state"] = "COMPLETED"
        redis.hashes[state_key]["terminal_seq"] = "2"
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_cursor_ahead_does_not_wait(self):
        redis = FakeRedis()
        repo = _redis_repo(redis, op_timeout_s=1.0)
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 99, timeout_s=5.0)
        assert snapshot.events == () and snapshot.latest_seq == 1
        assert redis.xread_calls == 0  # no blocking wait for a bad cursor

    @mark.asyncio
    async def test_operation_timeout_is_explicit(self):
        redis = FakeRedis()
        redis.hang = True
        repo = _redis_repo(redis, op_timeout_s=0.05)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.create(T1)
        assert exc.value.fault is RunRepositoryFault.UNAVAILABLE
        assert "timed out" in str(exc.value)

    @mark.asyncio
    async def test_blocking_read_wakes_on_commit(self):
        repo = _redis_repo()
        await repo.create(T1)

        async def later():
            await asyncio.sleep(0.02)
            await repo.commit_transition(
                T1, expected_state=ACCEPTED, next_state=GUARDING
            )

        task = asyncio.create_task(later())
        snapshot = await repo.snapshot(T1, 1, timeout_s=1.0)
        await task
        assert [e.seq for e in snapshot.events] == [2]

    @mark.asyncio
    async def test_idle_timeout_snapshot(self):
        repo = _redis_repo()
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 1, timeout_s=0.02)
        assert snapshot.timed_out is True
        assert snapshot.events == () and snapshot.latest_seq == 1

    @mark.asyncio
    async def test_long_poll_budget_is_not_cut_by_command_timeout(self):
        redis = FakeRedis()
        # command timeout far below the requested wait window: the blocking read
        # must still run to completion and yield a legal timeout snapshot
        repo = _redis_repo(redis, op_timeout_s=0.05, block_grace_s=0.5)
        await repo.create(T1)
        snapshot = await repo.snapshot(T1, 1, timeout_s=0.2)
        assert snapshot.timed_out is True
        assert snapshot.events == () and snapshot.latest_seq == 1

    @mark.asyncio
    async def test_hanging_blocking_read_fails_after_grace(self):
        redis = FakeRedis()
        repo = _redis_repo(redis, op_timeout_s=0.05, block_grace_s=0.05)
        await repo.create(T1)  # create uses the (small) command budget
        redis.hang = True  # only the subsequent blocking read hangs
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 1, timeout_s=0.05)
        assert exc.value.fault is RunRepositoryFault.UNAVAILABLE

    @mark.asyncio
    async def test_key_components_cannot_collide(self):
        repo = _redis_repo()
        first = RunIdentity(run_id="c", tenant_id="a:b", device_id="d", session_id="s")
        second = RunIdentity(run_id="b:c", tenant_id="a", device_id="d", session_id="s")
        assert repo.keys(first)[0] != repo.keys(second)[0]
        braces = RunIdentity(
            run_id="r{}", tenant_id="t{1}", device_id="d", session_id="s"
        )
        assert "{" not in repo.keys(braces)[0].split("}")[0].lstrip("{")
        with pytest.raises(ValueError):
            repo.keys(
                RunIdentity(
                    run_id="r" * 129, tenant_id="t", device_id="d", session_id="s"
                )
            )

    @mark.asyncio
    async def test_orphan_stream_delete_is_refused(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        state_key, stream_key = repo.keys(T1)
        redis.hashes.pop(state_key)  # orphan stream: state gone, stream remains
        # same tenant namespace (any principal): refuse as invariant, zero writes
        for identity in (T1, T1_WRONG_DEVICE):
            with pytest.raises(RunRepositoryError) as exc:
                await repo.delete(identity)
            assert exc.value.fault is RunRepositoryFault.INVARIANT
            assert "ORPHAN" in str(exc.value) or "orphan" in str(exc.value).lower()
        # another tenant's namespace simply has no such run
        with pytest.raises(RunRepositoryError) as exc:
            await repo.delete(T2)
        assert exc.value.fault is RunRepositoryFault.NOT_FOUND
        assert stream_key in redis.streams  # zero writes in every case

    @mark.asyncio
    async def test_malformed_snapshot_payloads_are_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        state_key, stream_key = repo.keys(T1)

        # 1. physical id with a counter ("1-999") is not a valid business id
        entry = redis.streams[stream_key][0]
        redis.streams[stream_key][0] = ("1-999", entry[1])
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        redis.streams[stream_key][0] = entry

        # 2. broken JSON never leaks a raw decode error
        redis.streams[stream_key][0] = ("1-0", {"event": "{not-json"})
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        redis.streams[stream_key][0] = entry

        # 3. non-terminal hash carrying terminal_seq must not be hidden
        redis.hashes[state_key]["terminal_seq"] = "1"
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        redis.hashes[state_key].pop("terminal_seq")

        # 4. terminal state without a terminal event
        redis.hashes[state_key]["state"] = "COMPLETED"
        redis.hashes[state_key]["terminal_seq"] = "1"
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_run_completed_event_without_terminal_state_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        state_key, stream_key = repo.keys(T1)
        _, payload = redis.streams[stream_key][0]
        import json as _json

        forged = _json.loads(payload["event"])
        forged["event"] = "run.completed"
        forged["data"] = {"status": "completed"}
        redis.streams[stream_key][0] = ("1-0", {"event": _json.dumps(forged)})
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert state_key in redis.hashes  # nothing written

    @mark.asyncio
    async def test_completed_without_answer_is_rejected_without_writes(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        before = await repo.snapshot(T1, 0, 0.01)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1,
                expected_state=STREAMING,
                next_state=COMPLETED,
                data={},
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        after = await repo.snapshot(T1, 0, 0.01)
        assert after.state is STREAMING  # zero writes
        assert [e.seq for e in after.events] == [e.seq for e in before.events]

    @mark.asyncio
    async def test_completed_with_answer_is_legal(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        seq = await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert snapshot.terminal_seq == seq == snapshot.latest_seq
        assert snapshot.events[-1].event is SSEEventType.RUN_COMPLETED

    @mark.asyncio
    @pytest.mark.parametrize("target", [RunState.HANDOFF, RunState.DEGRADED, FAILED])
    async def test_answer_less_terminals_remain_legal(self, target):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        seq = await repo.commit_transition(
            T1, expected_state=STREAMING, next_state=target
        )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert snapshot.state is target
        assert snapshot.terminal_seq == seq == snapshot.latest_seq

    @mark.asyncio
    async def test_state_only_orphan_delete_is_refused(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        state_key, stream_key = repo.keys(T1)
        redis.streams.pop(stream_key)  # orphan state: stream gone, state remains
        for identity in (T1, T1_WRONG_DEVICE):
            with pytest.raises(RunRepositoryError) as exc:
                await repo.delete(identity)
            assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert state_key in redis.hashes  # zero writes

    @mark.asyncio
    async def test_missing_event_field_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        _, stream_key = repo.keys(T1)
        redis.streams[stream_key][0] = ("1-0", {"wrong_field": "{}"})
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert exc.value.fault is not RunRepositoryFault.UNAVAILABLE

    @mark.asyncio
    async def test_bad_terminal_payload_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        state_key, stream_key = repo.keys(T1)
        terminal_seq = redis.hashes[state_key]["terminal_seq"]
        redis.streams[stream_key] = [
            (
                entry_id,
                {
                    "event": "{broken"
                    if entry_id.split("-")[0] == terminal_seq
                    else payload["event"]
                },
            )
            for entry_id, payload in redis.streams[stream_key]
        ]
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_hidden_terminal_identity_is_still_validated(self):
        import json as _json

        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        state_key, stream_key = repo.keys(T1)
        terminal_seq = int(redis.hashes[state_key]["terminal_seq"])
        forged = _json.loads(
            next(
                p["event"]
                for i, p in redis.streams[stream_key]
                if i.split("-")[0] == str(terminal_seq)
            )
        )
        forged["tenant_id"] = "t2"  # forge the hidden terminal event's identity
        redis.streams[stream_key] = [
            (
                entry_id,
                {"event": _json.dumps(forged)}
                if entry_id.split("-")[0] == str(terminal_seq)
                else payload,
            )
            for entry_id, payload in redis.streams[stream_key]
        ]
        # resume AT the terminal cursor: the page is empty, only the terminal
        # event itself can reveal the forgery
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, terminal_seq, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_malformed_oldest_outside_the_page_is_invariant(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        await repo.commit_transition(T1, expected_state=ACCEPTED, next_state=GUARDING)
        _, stream_key = repo.keys(T1)
        first_id, first_payload = redis.streams[stream_key][0]
        redis.streams[stream_key][0] = ("1-999", first_payload)  # bad oldest id
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 2, 0.01)  # page starts after the bad entry
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert first_id != "1-999"

    @mark.asyncio
    async def test_cursor_at_terminal_with_forged_terminal_identity(self):
        import json as _json

        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        state_key, stream_key = repo.keys(T1)
        terminal_seq = int(redis.hashes[state_key]["terminal_seq"])
        forged = _json.loads(
            next(
                p["event"]
                for i, p in redis.streams[stream_key]
                if i.split("-")[0] == str(terminal_seq)
            )
        )
        forged["device_id"] = "dX"
        redis.streams[stream_key] = [
            (
                entry_id,
                {"event": _json.dumps(forged)}
                if entry_id.split("-")[0] == str(terminal_seq)
                else payload,
            )
            for entry_id, payload in redis.streams[stream_key]
        ]
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, terminal_seq, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_completed_without_answer_rejected_in_redis(self):
        repo = _redis_repo()
        await _to_streaming(repo)
        before = await repo.snapshot(T1, 0, 0.01)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1,
                expected_state=STREAMING,
                next_state=COMPLETED,
                data={},
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        after = await repo.snapshot(T1, 0, 0.01)
        assert after.state is STREAMING
        assert [e.seq for e in after.events] == [e.seq for e in before.events]

    @mark.asyncio
    async def test_state_event_fields_are_derived_not_supplied(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        # caller-supplied authoritative fields are refused
        for payload in ({"status": "completed"}, {"stage": "completed"}):
            with pytest.raises(RunRepositoryError) as exc:
                await repo.commit_transition(
                    T1, expected_state=STREAMING, next_state=FAILED, data=payload
                )
            assert exc.value.fault is RunRepositoryFault.INVARIANT
        # derived status always matches the committed state
        await repo.commit_transition(T1, expected_state=STREAMING, next_state=FAILED)
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert snapshot.state is FAILED
        assert snapshot.events[-1].event is SSEEventType.RUN_COMPLETED
        assert snapshot.events[-1].data["status"] == "failed"

    @mark.asyncio
    async def test_process_status_stage_is_derived(self):
        repo = MemoryRunRepository()
        await repo.create(T1)
        await repo.commit_transition(
            T1, expected_state=ACCEPTED, next_state=GUARDING, data={"message": "ok"}
        )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert snapshot.events[-1].data == {"stage": "guarding", "message": "ok"}

    @mark.asyncio
    async def test_answer_completed_requires_content_origin(self):
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_COMPLETED, data={"citations": []}
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert "content_origin" in str(exc.value)

    @mark.asyncio
    async def test_nested_forbidden_keys_are_rejected_without_echo(self):
        marker = "marker_should_not_appear"
        cases = [
            {"sources": [{"chain_of_thought": marker}]},
            {
                "citations": [{"system_prompt": marker}],
                "content_origin": "ai_generated",
            },
            {"actions": [{"tool_arguments": marker}], "content_origin": "ai_generated"},
        ]
        with pytest.raises(RunRepositoryError) as exc:
            build_event(T1, 3, SSEEventType.ANSWER_COMPLETED, cases[1])
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert marker not in str(exc.value)  # never echo the offending value
        with pytest.raises(RunRepositoryError):
            build_event(T1, 4, SSEEventType.EVIDENCE_FOUND, cases[0])

    @mark.asyncio
    async def test_forged_answer_seal_bit_is_not_enough(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        state_key, _ = repo.keys(T1)
        redis.hashes[state_key]["answer_sealed"] = "1"  # fake the cached bit
        before = await repo.snapshot(T1, 0, 0.01)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=STREAMING, next_state=COMPLETED
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        after = await repo.snapshot(T1, 0, 0.01)
        assert after.state is STREAMING  # zero writes
        assert [e.seq for e in after.events] == [e.seq for e in before.events]

    @mark.asyncio
    async def test_terminal_payload_seq_tampering_is_invariant(self):
        import json as _json2

        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(T1, expected_state=STREAMING, next_state=COMPLETED)
        state_key, stream_key = repo.keys(T1)
        terminal_seq = redis.hashes[state_key]["terminal_seq"]
        redis.streams[stream_key] = [
            (
                entry_id,
                {
                    "event": _json2.dumps(
                        {
                            **_json2.loads(payload["event"]),
                            "seq": 999,  # payload seq no longer matches its position
                        }
                    )
                }
                if entry_id.split("-")[0] == terminal_seq
                else payload,
            )
            for entry_id, payload in redis.streams[stream_key]
        ]
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, int(terminal_seq), 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_answer_completed_without_origin_rejected_at_contract(self):
        import json as _json2

        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        state_key, stream_key = repo.keys(T1)
        entry_id, payload = redis.streams[stream_key][-1]
        forged = _json2.loads(payload["event"])
        forged["data"] = {"citations": []}  # provenance marker removed
        redis.streams[stream_key][-1] = (entry_id, {"event": _json2.dumps(forged)})
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert state_key not in ("", None)

    @mark.asyncio
    async def test_tuple_nested_sensitive_key_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SSEEvent(
                seq=2,
                tenant_id="t1",
                device_id="d1",
                session_id="s1",
                run_id="r1",
                layer="answer",
                event=SSEEventType.ANSWER_COMPLETED,
                data={
                    "content_origin": "ai_generated",
                    "sources": ({"chain_of_thought": "HIDDEN-TUPLE-VALUE"},),
                },
            )
        assert "chain_of_thought" in str(exc.value)
        assert "HIDDEN-TUPLE-VALUE" not in str(exc.value)  # value never echoed

    @mark.asyncio
    async def test_error_messages_never_echo_payload_values(self):
        marker = "marker_placeholder_value"
        with pytest.raises(RunRepositoryError) as exc:
            build_event(
                T1,
                3,
                SSEEventType.ANSWER_COMPLETED,
                {
                    "content_origin": "ai_generated",
                    "sources": [{"chain_of_thought": marker}],
                },
            )
        assert marker not in str(exc.value)
        with pytest.raises(ValidationError) as vexc:
            SSEEvent.model_validate(
                {
                    "seq": 1,
                    "tenant_id": "t1",
                    "device_id": "d1",
                    "session_id": "s1",
                    "run_id": "r1",
                    "layer": "process",
                    "event": "run.accepted",
                    # 'bogus' is the offending KEY; the marker rides as a VALUE
                    "data": {"status": "accepted", "message": marker, "bogus": "x"},
                }
            )
        assert "bogus" in str(vexc.value)
        assert marker not in str(vexc.value)  # values are never echoed

    @mark.asyncio
    async def test_tocfe_tampering_run_id_between_read_and_commit(self):
        redis = FakeRedis()
        await _to_streaming(repo := _redis_repo(redis))
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        state_key, stream_key = repo.keys(T1)

        def tamper():
            entry_id, payload = redis.streams[stream_key][-1]
            forged = _json.loads(payload["event"])
            forged["run_id"] = "other-run"  # tamper after the Python pre-read
            redis.streams[stream_key][-1] = (entry_id, {"event": _json.dumps(forged)})

        repo = _redis_repo(TamperingClient(redis, tamper))
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=STREAMING, next_state=COMPLETED
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert redis.hashes[state_key]["state"] == "STREAMING"  # zero writes

    @mark.asyncio
    async def test_tocfe_tampering_physical_id_between_read_and_commit(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        state_key, stream_key = repo.keys(T1)

        def tamper():
            entry_id, payload = redis.streams[stream_key][-1]
            seq = entry_id.split("-")[0]
            redis.streams[stream_key][-1] = (f"{seq}-999", payload)  # bad physical id

        repo = _redis_repo(TamperingClient(redis, tamper))
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=STREAMING, next_state=COMPLETED
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert redis.hashes[state_key]["state"] == "STREAMING"  # zero writes

    @mark.asyncio
    async def test_tocfe_corrupt_tail_between_read_and_append(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await _to_streaming(repo)
        _state_key, stream_key = repo.keys(T1)
        before = len(redis.streams[stream_key])

        def tamper():
            entry_id, _payload = redis.streams[stream_key][-1]
            redis.streams[stream_key][-1] = (entry_id, {"event": "{broken"})

        repo = _redis_repo(TamperingClient(redis, tamper))
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert len(redis.streams[stream_key]) == before  # zero writes

    @mark.asyncio
    async def test_snapshot_limit_is_validated(self):
        for bad in (0, -1, True):
            with pytest.raises(ValueError):
                _redis_repo(snapshot_limit=bad)

    @mark.asyncio
    async def test_orphan_and_wrong_type_and_unavailable(self):
        redis = FakeRedis()
        repo = _redis_repo(redis)
        await repo.create(T1)
        _, stream_key = repo.keys(T1)
        redis.streams.pop(stream_key)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

        redis2 = FakeRedis()
        repo2 = _redis_repo(redis2)
        await repo2.create(T1)
        redis2.wrong_type.add(repo2.keys(T1)[0])
        with pytest.raises(RunRepositoryError) as exc:
            await repo2.snapshot(T1, 0, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

        redis3 = FakeRedis()
        redis3.unavailable = True
        with pytest.raises(RunRepositoryError) as exc:
            await _redis_repo(redis3).create(T1)
        assert exc.value.fault is RunRepositoryFault.UNAVAILABLE


@pytest.mark.skipif(
    not os.getenv("GCMW_REDIS_TEST_URL"),
    reason="GCMW_REDIS_TEST_URL not set (CI redis service)",
)
class TestRealRedis:
    """End-to-end against a real Redis server (CI service container)."""

    @pytest_asyncio.fixture()
    async def repo(self):
        import redis.asyncio as aioredis

        client = aioredis.from_url(os.environ["GCMW_REDIS_TEST_URL"])
        repo = RedisRunRepository(client, prefix="gcmw:test:rr")
        for identity in (T1, R2):
            try:
                await repo.delete(identity)
            except RunRepositoryError:
                pass
        yield repo
        for identity in (T1, R2):
            try:
                await repo.delete(identity)
            except RunRepositoryError:
                pass
        await client.aclose()

    @mark.asyncio
    async def test_16_way_concurrent_append_returns_and_persists_identically(
        self, repo
    ):
        await _to_streaming(repo)
        base = (await repo.snapshot(T1, 0, 0.01)).latest_seq

        async def worker(index):
            try:
                seq = await repo.append_event(
                    T1,
                    event_type=SSEEventType.ANSWER_DELTA,
                    data={"delta": f"d{index}"},
                )
                return seq
            except RunRepositoryError as exc:
                if exc.fault is RunRepositoryFault.CONCURRENT_MODIFICATION:
                    return None
                raise

        results = await asyncio.gather(*(worker(i) for i in range(16)))
        returned = sorted(s for s in results if s is not None)
        assert len(returned) == len(set(returned))  # never a duplicate return
        snapshot = await repo.snapshot(T1, 0, 0.01)
        stored = [
            e.seq for e in snapshot.events if e.event is SSEEventType.ANSWER_DELTA
        ]
        assert set(returned).issubset(set(stored))
        # every returned seq really exists, and the stored tail is contiguous
        assert stored == list(range(stored[0], stored[0] + len(stored)))
        assert stored[-1] == snapshot.latest_seq
        assert base + len(stored) == snapshot.latest_seq

    @mark.asyncio
    async def test_transition_and_append_race_single_winner(self, repo):
        await repo.create(T1)

        async def transition():
            try:
                return await repo.commit_transition(
                    T1, expected_state=ACCEPTED, next_state=GUARDING
                )
            except RunRepositoryError:
                return None

        async def append():
            try:
                return await repo.append_event(
                    T1, event_type=SSEEventType.MIC_STATUS, data={"state": "on"}
                )
            except RunRepositoryError:
                return None

        results = await asyncio.gather(transition(), append())
        successes = [r for r in results if r]
        assert len(successes) == len(set(successes))  # never a duplicate seq
        snapshot = await repo.snapshot(T1, 0, 0.01)
        stored = [e.seq for e in snapshot.events]
        assert stored == list(range(1, len(stored) + 1))  # contiguous, no gaps
        assert (
            sum(1 for e in snapshot.events if e.event is SSEEventType.PROCESS_STATUS)
            == 1
        )

    @mark.asyncio
    async def test_wrong_device_session_refused(self, repo):
        await repo.create(T1)
        for identity in (T1_WRONG_DEVICE, T1_WRONG_SESSION):
            with pytest.raises(RunRepositoryError) as exc:
                await repo.state(identity)
            assert exc.value.fault is RunRepositoryFault.NOT_FOUND
            with pytest.raises(RunRepositoryError):
                await repo.delete(identity)
        assert await repo.state(T1) is ACCEPTED

    @mark.asyncio
    async def test_answer_order_safety(self, repo):
        await repo.create(T1)
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "early"}
            )
        with pytest.raises(RunRepositoryError):
            await repo.append_event(T1, event_type=SSEEventType.HEARTBEAT)
        await _to_streaming(repo, R2)
        await repo.append_event(
            R2,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                R2,
                event_type=SSEEventType.ANSWER_COMPLETED,
                data={"citations": [], "content_origin": "ai_generated"},
            )
        with pytest.raises(RunRepositoryError):
            await repo.append_event(
                R2, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "late"}
            )

    @mark.asyncio
    async def test_cursor_ahead_does_not_wait(self, repo):
        await repo.create(T1)
        started = time.monotonic()
        snapshot = await repo.snapshot(T1, 99, timeout_s=5.0)
        assert time.monotonic() - started < 1.0  # no heartbeat-length wait
        assert snapshot.events == ()

    @mark.asyncio
    async def test_full_legal_path_and_window_parity(self, repo):
        await _to_streaming(repo)
        await repo.append_event(
            T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "x"}
        )
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        snapshot = await repo.snapshot(T1, 0, 0.01)
        assert snapshot.state is COMPLETED
        # create(1) + 5 transitions(2..6) + delta(7) + answer.completed(8) + terminal(9)
        assert snapshot.terminal_seq == snapshot.latest_seq == 9
        assert [e.seq for e in snapshot.events] == list(range(1, 10))
        assert snapshot.oldest_available_seq == 1

    @mark.asyncio
    async def test_completed_without_answer_rejected_and_state_only_orphan(self, repo):
        await _to_streaming(repo, R2)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                R2,
                expected_state=STREAMING,
                next_state=COMPLETED,
                data={},
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert await repo.state(R2) is STREAMING  # zero writes

        # state-only orphan: deleting the stream must not silently drop the state
        await repo.create(T1)
        state_key, stream_key = repo.keys(T1)
        await repo._client.delete(stream_key)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.delete(T1)
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert await repo._client.exists(state_key) == 1  # zero writes
        await repo._client.delete(state_key)

    @mark.asyncio
    async def test_hidden_terminal_identity_is_validated_on_resume(self, repo):
        """Resume at the terminal cursor: the page is empty, so only the
        terminal event itself can reveal a forged identity."""
        import json as _json

        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(
            T1,
            expected_state=STREAMING,
            next_state=COMPLETED,
            data={},
        )
        state_key, stream_key = repo.keys(T1)
        terminal_seq = int(
            (await repo._client.hget(state_key, "terminal_seq")).decode()
        )
        entries = await repo._client.xrange(stream_key)
        await repo._client.delete(stream_key)
        for entry_id, fields in entries:
            payload = fields[b"event"].decode()
            seq = int(entry_id.decode().split("-")[0])
            if seq == terminal_seq:
                forged = _json.loads(payload)
                forged["tenant_id"] = "t2"  # forge the hidden terminal identity
                payload = _json.dumps(forged)
            await repo._client.xadd(
                stream_key, {"event": payload}, id=entry_id.decode()
            )
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, terminal_seq, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_forged_answer_seal_bit_is_not_enough(self, repo):
        await _to_streaming(repo)
        state_key, _ = repo.keys(T1)
        await repo._client.hset(state_key, "answer_sealed", "1")  # fake cached bit
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=STREAMING, next_state=COMPLETED
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT
        assert await repo.state(T1) is STREAMING  # zero writes

    @mark.asyncio
    async def test_terminal_payload_seq_tampering_is_invariant(self, repo):
        import json as _json

        await _to_streaming(repo)
        await repo.append_event(
            T1,
            event_type=SSEEventType.ANSWER_COMPLETED,
            data={"citations": [], "content_origin": "ai_generated"},
        )
        await repo.commit_transition(T1, expected_state=STREAMING, next_state=COMPLETED)
        state_key, stream_key = repo.keys(T1)
        terminal_seq = int(
            (await repo._client.hget(state_key, "terminal_seq")).decode()
        )
        entries = await repo._client.xrange(stream_key)
        await repo._client.delete(stream_key)
        for entry_id, fields in entries:
            payload = fields[b"event"].decode()
            seq = int(entry_id.decode().split("-")[0])
            if seq == terminal_seq:
                forged = _json.loads(payload)
                forged["seq"] = 999  # payload seq no longer matches its position
                payload = _json.dumps(forged)
            await repo._client.xadd(
                stream_key, {"event": payload}, id=entry_id.decode()
            )
        with pytest.raises(RunRepositoryError) as exc:
            await repo.snapshot(T1, terminal_seq, 0.01)
        assert exc.value.fault is RunRepositoryFault.INVARIANT

    @mark.asyncio
    async def test_answer_events_are_dual_layer_and_replay_stable(self, repo):
        await _to_streaming(repo)
        await repo.append_event(
            T1, event_type=SSEEventType.ANSWER_DELTA, data={"delta": "体"}
        )
        first = await repo.snapshot(T1, 0, 0.01)
        await asyncio.sleep(0.05)
        second = await repo.snapshot(T1, 0, 0.01)
        layers = {e.layer.value for e in first.events}
        assert layers == {"process", "answer"}
        assert first.events[-1].timestamp == second.events[-1].timestamp


# -- sealed-answer atomic invariant: enforced INSIDE the commit boundary ------

_SEAL_PAYLOAD = {
    "citations": [],
    "actions": [],
    "content_origin": "approved_faq",
}


async def _seal_answer(repo, identity=T1) -> None:
    """Advance a fresh run to STREAMING and append the answer.completed seal."""
    await _to_streaming(repo, identity)
    await repo.append_event(
        identity, event_type=SSEEventType.ANSWER_COMPLETED, data=dict(_SEAL_PAYLOAD)
    )


class TestSealedAnswerAtomicInvariant:
    """A FAILED over a persisted answer.completed is refused ZERO-WRITE inside
    the commit boundary, with a DISTINGUISHABLE fault — not generic INVARIANT.
    An executor-side check-then-act has a TOCTOU window this closes."""

    @mark.asyncio
    async def test_failed_over_a_sealed_answer_is_refused_zero_write(self):
        repo = MemoryRunRepository()
        await _seal_answer(repo)
        before = await repo.snapshot(T1, 0, 0.01)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=RunState.STREAMING, next_state=RunState.FAILED
            )
        assert exc.value.fault is RunRepositoryFault.ANSWER_SEALED
        after = await repo.snapshot(T1, 0, 0.01)
        assert after.state is RunState.STREAMING  # zero-write: nothing moved
        assert after.latest_seq == before.latest_seq
        assert after.terminal_seq is None

    @mark.asyncio
    async def test_failed_wins_then_the_seal_is_refused(self):
        """The mirror direction: once FAILED is the terminal, a late seal must
        not sneak in either — the run can never become answer.completed+FAILED."""
        repo = MemoryRunRepository()
        await _to_streaming(repo)
        await repo.commit_transition(
            T1, expected_state=RunState.STREAMING, next_state=RunState.FAILED
        )
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_COMPLETED, data=dict(_SEAL_PAYLOAD)
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT  # terminal run


@pytest.mark.skipif(
    not os.getenv("GCMW_REDIS_TEST_URL"),
    reason="GCMW_REDIS_TEST_URL not set (CI redis service)",
)
class TestRealRedisSealedAnswerInvariant:
    """Same invariant, exercised against the REAL Redis Lua commit boundary
    (the -11 control code inside _LUA_COMMIT). Skipped without a Redis URL."""

    @pytest_asyncio.fixture()
    async def repo(self):
        import redis.asyncio as aioredis

        client = aioredis.from_url(os.environ["GCMW_REDIS_TEST_URL"])
        repo = RedisRunRepository(client, prefix="gcmw:test:rr-seal")
        for identity in (T1, R2):
            try:
                await repo.delete(identity)
            except RunRepositoryError:
                pass
        yield repo
        for identity in (T1, R2):
            try:
                await repo.delete(identity)
            except RunRepositoryError:
                pass
        await client.aclose()

    @mark.asyncio
    async def test_failed_over_a_sealed_answer_is_refused_zero_write(self, repo):
        await _seal_answer(repo)
        before = await repo.snapshot(T1, 0, 0.01)
        with pytest.raises(RunRepositoryError) as exc:
            await repo.commit_transition(
                T1, expected_state=RunState.STREAMING, next_state=RunState.FAILED
            )
        assert exc.value.fault is RunRepositoryFault.ANSWER_SEALED
        after = await repo.snapshot(T1, 0, 0.01)
        assert after.state is RunState.STREAMING  # the Lua refusal wrote nothing
        assert after.latest_seq == before.latest_seq
        assert after.terminal_seq is None

    @mark.asyncio
    async def test_failed_wins_then_the_seal_is_refused(self, repo):
        await _to_streaming(repo)
        await repo.commit_transition(
            T1, expected_state=RunState.STREAMING, next_state=RunState.FAILED
        )
        with pytest.raises(RunRepositoryError) as exc:
            await repo.append_event(
                T1, event_type=SSEEventType.ANSWER_COMPLETED, data=dict(_SEAL_PAYLOAD)
            )
        assert exc.value.fault is RunRepositoryFault.INVARIANT  # terminal run
