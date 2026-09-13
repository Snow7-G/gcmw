"""SSE connection leases: counting, reconnect grace, disconnect cancel (B2-C).

The grace window is the only policy this slice adds: when the LAST subscriber of
a run goes away the run survives ``DEFAULT_RECONNECT_GRACE_S`` so a flaky link
can reconnect; a reconnect inside the window revokes the pending cancel. The
timing primitive is injectable, so every decision below is deterministic rather
than sleep-based.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from api_harness import PRINCIPAL, Harness, running_app

from app.api.v1 import agent_api
from app.api.v1.stream_leases import DEFAULT_RECONNECT_GRACE_S, RunLeaseRegistry
from app.contracts.run import RunState
from app.storage.run_repository import (
    RunIdentity,
    RunRepositoryError,
    RunRepositoryFault,
)


class ManualSleep:
    """Deterministic grace trigger: the timer fires when the test says so."""

    def __init__(self) -> None:
        self.gates: list[asyncio.Event] = []
        self.requested: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.requested.append(seconds)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()

    async def fire_all(self) -> None:
        """Release every registered gate (after letting started tasks settle)."""
        for _ in range(1000):
            if self.gates:
                break
            await asyncio.sleep(0)
        gates, self.gates = self.gates, []
        for gate in gates:
            gate.set()
        for _ in range(200):  # let the released expiry tasks run to completion
            await asyncio.sleep(0)

    async def wait_pending(self, count: int = 1) -> None:
        """Wait until ``count`` grace timers have reached the injected sleep."""
        for _ in range(1000):
            if len(self.gates) >= count:
                return
            await asyncio.sleep(0)
        raise AssertionError(f"only {len(self.gates)} of {count} timers started")

    async def fire(self) -> None:
        """Let the expiry task run and release its grace gate.

        The registry schedules the timer with ``create_task``, so the task may
        not have reached the injected sleep yet — wait for its gate first.
        """
        for _ in range(1000):
            if self.gates:
                break
            await asyncio.sleep(0)
        assert self.gates, "no pending grace timer to fire"
        self.gates.pop(0).set()
        for _ in range(200):  # let the expiry task run to completion
            await asyncio.sleep(0)

    def pending(self) -> int:
        return len(self.gates)


@pytest.fixture
def harness() -> Harness:
    with running_app() as h:
        yield h


def _recorder(expired: list[str]):
    async def on_expire(run_id: str) -> None:
        expired.append(run_id)

    return on_expire


class TestLeaseRegistry:
    def test_documented_grace_is_two_seconds(self):
        assert DEFAULT_RECONNECT_GRACE_S == 2.0

    def test_last_close_starts_exactly_one_grace_timer(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)

            registry.open("r1")
            registry.open("r1")
            assert registry.subscribers("r1") == 2

            registry.close(
                "r1", client_gone=True
            )  # one subscriber left: nothing scheduled
            assert registry.subscribers("r1") == 1
            assert registry.pending_expiries() == 0
            assert sleep.pending() == 0

            registry.close(
                "r1", client_gone=True
            )  # last subscriber: grace window opens
            assert registry.subscribers("r1") == 0
            assert registry.pending_expiries() == 1
            await sleep.wait_pending()
            assert sleep.requested == [DEFAULT_RECONNECT_GRACE_S]

            await sleep.fire()
            return expired, registry.tracked(), sleep.pending()

        expired, tracked, pending = asyncio.run(main())
        assert expired == ["r1"]
        assert tracked == 0 and pending == 0  # nothing left behind

    def test_reconnect_inside_the_grace_revokes_the_cancel(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)

            registry.open("r1")
            registry.close("r1", client_gone=True)
            assert registry.pending_expiries() == 1

            registry.open("r1")  # the user reconnects inside the window
            assert registry.subscribers("r1") == 1
            assert registry.pending_expiries() == 0
            assert registry.tracked() == 1

            for _ in range(50):  # the revoked timer must never fire
                await asyncio.sleep(0)
            return expired, registry.subscribers("r1")

        expired, subscribers = asyncio.run(main())
        assert expired == []
        assert subscribers == 1

    def test_runs_are_counted_independently(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)
            registry.open("r1")
            registry.open("r2")
            registry.close("r1", client_gone=True)  # only r1 enters its grace window
            assert registry.pending_expiries() == 1
            await sleep.fire()
            return expired, registry.subscribers("r2")

        expired, r2 = asyncio.run(main())
        assert expired == ["r1"]
        assert r2 == 1  # the other run keeps its subscriber

    def test_reconnect_after_the_grace_does_not_revoke_a_fired_cancel(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)
            registry.open("r1")
            registry.close("r1", client_gone=True)
            await sleep.fire()  # the decision was already taken
            registry.open("r1")  # a brand-new lease, it cannot un-cancel
            return expired

        assert asyncio.run(main()) == ["r1"]

    def test_many_streams_leave_no_lease_behind(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)
            for i in range(1000):
                run_id = f"run-{i}"
                registry.open(run_id)
                registry.close(run_id, client_gone=True)
                await sleep.fire()
            return registry.tracked(), registry.pending_expiries(), len(expired)

        tracked, pending, expired = asyncio.run(main())
        assert (tracked, pending, expired) == (0, 0, 1000)

    def test_server_side_end_never_schedules_a_cancel(self):
        """A stream that ends on the server (terminal or stream.error) is NOT a
        disconnect: the run must stay active with no grace timer at all."""

        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)
            registry.open("r1")
            registry.close("r1", client_gone=False)  # server ended the stream
            for _ in range(50):
                await asyncio.sleep(0)
            return (
                expired,
                registry.pending_expiries(),
                registry.tracked(),
                sleep.pending(),
            )

        expired, pending, tracked, timers = asyncio.run(main())
        assert expired == []  # no cancel, ever
        assert (pending, tracked, timers) == (0, 0, 0)

    def test_rapid_disconnect_reconnect_disconnect_cancels_exactly_once(self):
        """Review P1: the revoked timer must not clear the NEW timer.

        Sequence: disconnect (timer T1, already sleeping) -> reconnect (T1
        revoked) -> disconnect again (timer T2), all before T1 gets to run its
        cancellation cleanup. The old cleanup used to clear ``entry.timer`` and
        drop the entry, losing T2 entirely: no cancel would ever be applied.
        """

        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)

            registry.open("r1")
            registry.close("r1", client_gone=True)  # first disconnect: T1 armed
            await sleep.wait_pending(1)  # T1 is really sleeping in its grace

            registry.open("r1")  # reconnect revokes T1 …
            registry.close("r1", client_gone=True)  # … and the link drops again

            for _ in range(50):  # T1's cancellation cleanup runs right here
                await asyncio.sleep(0)
            after_cleanup = (registry.tracked(), registry.pending_expiries())

            await sleep.fire_all()  # T2's grace expires
            return after_cleanup, expired, registry.tracked()

        after_cleanup, expired, tracked = asyncio.run(main())
        assert after_cleanup == (1, 1)  # the OLD task did not clear the new timer
        assert expired == ["r1"]  # the second grace cancelled exactly once
        assert tracked == 0

    def test_repeated_rapid_cycles_still_cancel_exactly_once_each_time(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)

            for i in range(5):
                run_id = f"run-{i}"
                registry.open(run_id)
                registry.close(run_id, client_gone=True)
                await sleep.wait_pending(1)  # grace timer is sleeping
                registry.open(run_id)  # reconnect revokes it
                registry.close(run_id, client_gone=True)  # drops again
                for _ in range(50):
                    await asyncio.sleep(0)  # old cleanup runs
                await sleep.fire_all()
            return expired, registry.tracked(), registry.pending_expiries()

        expired, tracked, pending = asyncio.run(main())
        assert expired == [f"run-{i}" for i in range(5)]  # one cancel per run
        assert (tracked, pending) == (0, 0)

    def test_shutdown_ends_an_in_flight_callback_before_returning(self):
        """After shutdown() returns, no disconnect cancel may still run."""

        async def main():
            entered = asyncio.Event()
            release = asyncio.Event()
            applied: list[str] = []

            async def on_expire(run_id: str) -> None:
                entered.set()
                await release.wait()  # a slow storage write, mid-flight
                applied.append(run_id)

            sleep = ManualSleep()
            registry = RunLeaseRegistry(on_expire, sleep=sleep)
            registry.open("r1")
            registry.close("r1", client_gone=True)
            await sleep.fire()  # the grace expired: the callback is running
            await entered.wait()
            assert registry.in_flight_callbacks() == 1

            await registry.shutdown()  # must not return while it can still run
            at_shutdown = list(applied)
            release.set()  # would let a leaked callback finish
            for _ in range(100):
                await asyncio.sleep(0)
            return at_shutdown, list(applied), registry.tracked()

        at_shutdown, after, tracked = asyncio.run(main())
        assert at_shutdown == []  # nothing had been applied at shutdown time
        assert after == []  # and nothing continued afterwards
        assert tracked == 0

    def test_shutdown_cancels_pending_timers(self):
        async def main():
            sleep = ManualSleep()
            expired: list[str] = []
            registry = RunLeaseRegistry(_recorder(expired), sleep=sleep)
            registry.open("r1")
            registry.close("r1", client_gone=True)
            assert registry.pending_expiries() == 1
            await registry.shutdown()
            for _ in range(50):
                await asyncio.sleep(0)
            return expired, registry.tracked()

        expired, tracked = asyncio.run(main())
        assert expired == []  # shutdown must not fire the cancel
        assert tracked == 0

    @pytest.mark.parametrize("grace", [0, -1, math.inf, math.nan, True])
    def test_invalid_grace_is_rejected(self, grace):
        with pytest.raises((TypeError, ValueError)):
            RunLeaseRegistry(_recorder([]), grace_s=grace)

    def test_closing_an_unopened_lease_is_a_noop(self):
        async def main():
            registry = RunLeaseRegistry(_recorder([]), sleep=ManualSleep())
            registry.close("never-opened", client_gone=True)
            return registry.tracked(), registry.pending_expiries()

        assert asyncio.run(main()) == (0, 0)


def _request(headers: dict[str, str] | None = None):
    from fastapi import Request

    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/agent/runs/x/events",
            "query_string": b"",
            "headers": raw,
            "state": {},
        }
    )


async def _seed_run(harness: Harness, key: str) -> dict:
    from app.contracts.api import (
        Channel,
        CreateRunRequest,
        CreateSessionRequest,
        RunInput,
    )

    service = harness.service
    session_id = service.create_session(
        PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
    ).session_id
    snap = await service.create_run(
        PRINCIPAL,
        CreateRunRequest(
            session_id=session_id,
            input=RunInput(type="text", text="lease"),
            idempotency_key=key,
        ),
        "req",
        "trace",
    )
    return {
        "run_id": snap.run_id,
        "session_id": session_id,
        "identity": RunIdentity(
            run_id=snap.run_id,
            tenant_id=PRINCIPAL.tenant_id,
            device_id=PRINCIPAL.device_id,
            session_id=session_id,
        ),
    }


class TestRouteLeases:
    """The public route must hold one lease per live stream."""

    def _install(
        self, harness: Harness, sleep: ManualSleep
    ) -> tuple[RunLeaseRegistry, list[str]]:
        """Install a deterministic-timing registry on the running application.

        The expiry callback records the decision AND applies the real
        disconnect-cancel, so a test can assert both halves.
        """
        expired: list[str] = []

        async def on_expire(run_id: str) -> None:
            expired.append(run_id)
            await harness.service.cancel_for_disconnect(run_id)

        registry = RunLeaseRegistry(on_expire, sleep=sleep)
        harness.app.state.stream_leases = registry
        return registry, expired

    def test_authenticated_stream_holds_and_releases_a_lease(self, harness):
        sleep = ManualSleep()
        registry, expired = self._install(harness, sleep)

        async def main():
            run = await _seed_run(harness, "lease-1")
            response = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            assert registry.subscribers(run["run_id"]) == 0  # not started yet
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                break
            held = registry.subscribers(run["run_id"])
            await response.body_iterator.aclose()
            return run, chunks, held, registry.subscribers(run["run_id"])

        _run, chunks, held, after = asyncio.run(main())
        assert chunks and "run.accepted" in chunks[0]
        assert held == 1  # a live stream is exactly one lease
        assert after == 0  # released on disconnect
        assert expired == []  # ... and the cancel is only pending, not fired

    def test_disconnect_then_cancel_is_applied_once_the_grace_fires(self, harness):
        sleep = ManualSleep()
        registry, expired = self._install(harness, sleep)

        async def main():
            run = await _seed_run(harness, "lease-2")
            response = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            async for _chunk in response.body_iterator:
                break
            await response.body_iterator.aclose()

            assert await harness.service.repository.state(run["identity"]) is (
                RunState.ACCEPTED
            )
            await sleep.fire()
            state = await harness.service.repository.state(run["identity"])
            page = await harness.service.repository.snapshot(run["identity"], 0, 0.0)
            return state, [event.event.value for event in page.events], expired

        state, events, expired = asyncio.run(main())
        assert expired  # the grace really expired and the cancel was applied
        assert state is RunState.CANCELLED
        assert events == ["run.accepted", "run.completed"]  # single terminal event

    def test_reconnect_inside_the_grace_keeps_the_run_alive(self, harness):
        sleep = ManualSleep()
        registry, expired = self._install(harness, sleep)

        async def main():
            run = await _seed_run(harness, "lease-3")
            first = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            async for _chunk in first.body_iterator:
                break
            await first.body_iterator.aclose()

            second = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            async for _chunk in second.body_iterator:
                break
            for _ in range(50):
                await asyncio.sleep(0)  # the revoked timer must not fire
            state = await harness.service.repository.state(run["identity"])
            await second.body_iterator.aclose()
            return state, sleep.pending(), expired

        state, pending, expired = asyncio.run(main())
        assert state is RunState.ACCEPTED
        assert pending == 0  # the pending cancel was revoked by the reconnect
        assert expired == []  # ... and it never fired

    def test_unauthorised_requests_never_take_a_lease(self):
        with running_app(overrides=False) as h:
            res = h.client.get("/api/v1/agent/runs/whatever/events")
            assert res.status_code == 401
            leases = h.app.state.stream_leases
            assert leases.tracked() == 0 and leases.pending_expiries() == 0


class TestServerSideEndIsNotADisconnect:
    """Review P1: a storage fault must never look like the user leaving."""

    def test_error_frame_run_stays_active_and_no_grace_starts(self, harness):
        sleep = ManualSleep()
        expired: list[str] = []

        async def on_expire(run_id: str) -> None:
            expired.append(run_id)
            await harness.service.cancel_for_disconnect(run_id)

        registry = RunLeaseRegistry(on_expire, sleep=sleep)
        harness.app.state.stream_leases = registry

        async def main():
            run = await _seed_run(harness, "server-end")

            async def failing_snapshot(identity, cursor, timeout_s):
                raise RunRepositoryError(
                    RunRepositoryFault.UNAVAILABLE, "storage is down"
                )

            harness.repository.snapshot = failing_snapshot
            response = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            body = [chunk async for chunk in response.body_iterator]  # consumed
            state = await harness.service.repository.state(run["identity"])
            return (
                "".join(body),
                state,
                registry.pending_expiries(),
                registry.tracked(),
                sleep.pending(),
                expired,
            )

        body, state, pending, tracked, timers, expired = asyncio.run(main())
        assert "stream.error" in body  # the client got a structured failure…
        assert state is RunState.ACCEPTED  # …and the run is untouched
        assert (pending, tracked, timers) == (0, 0, 0)  # no grace was armed
        assert expired == []

    def test_terminal_stream_end_arms_nothing_either(self, harness):
        sleep = ManualSleep()
        registry = RunLeaseRegistry(harness.service.cancel_for_disconnect, sleep=sleep)
        harness.app.state.stream_leases = registry

        async def main():
            run = await _seed_run(harness, "terminal-end")
            await harness.service.cancel_run(PRINCIPAL, run["run_id"])
            response = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            body = [chunk async for chunk in response.body_iterator]
            return (
                "".join(body),
                registry.pending_expiries(),
                registry.tracked(),
                sleep.pending(),
            )

        body, pending, tracked, timers = asyncio.run(main())
        assert body.count("event: run.completed") == 1
        assert (pending, tracked, timers) == (0, 0, 0)

    def test_disconnect_after_an_error_frame_still_cancels(self, harness):
        """The grace is armed by the DISCONNECT, not by the fault before it."""
        sleep = ManualSleep()
        expired: list[str] = []

        async def on_expire(run_id: str) -> None:
            expired.append(run_id)
            await harness.service.cancel_for_disconnect(run_id)

        registry = RunLeaseRegistry(on_expire, sleep=sleep)
        harness.app.state.stream_leases = registry

        async def main():
            run = await _seed_run(harness, "error-then-drop")

            async def failing_snapshot(identity, cursor, timeout_s):
                raise RunRepositoryError(
                    RunRepositoryFault.UNAVAILABLE, "storage is down"
                )

            harness.repository.snapshot = failing_snapshot
            response = await agent_api.stream_run_events(
                run["run_id"],
                _request(),
                PRINCIPAL,
                after_seq=0,
                last_event_id=None,
                service=harness.service,
                leases=registry,
                limiter=harness.app.state.rate_limiter,
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                break  # the client walks away right after the error frame
            await response.body_iterator.aclose()
            armed = registry.pending_expiries()
            await sleep.fire()
            state = await harness.service.repository.state(run["identity"])
            return "".join(chunks), armed, state, expired

        body, armed, state, expired = asyncio.run(main())
        assert "stream.error" in body
        assert armed == 1  # the disconnect itself armed the grace
        assert expired  # and it was applied
        assert state is RunState.CANCELLED


class TestExplicitCancelStillWins:
    def test_explicit_delete_cancel_is_not_reversed_by_a_lease_expiry(self, harness):
        sleep = ManualSleep()
        registry = RunLeaseRegistry(harness.service.cancel_for_disconnect, sleep=sleep)
        harness.app.state.stream_leases = registry

        async def main():
            run = await _seed_run(harness, "explicit")
            registry.open(run["run_id"])
            registry.close(run["run_id"], client_gone=True)  # grace opens
            await harness.service.cancel_run(
                PRINCIPAL, run["run_id"]
            )  # explicit DELETE
            await sleep.fire()  # the lease expiry now finds a terminal run
            page = await harness.service.repository.snapshot(run["identity"], 0, 0.0)
            return page.state, [e.event.value for e in page.events], registry.tracked()

        state, events, tracked = asyncio.run(main())
        assert state is RunState.CANCELLED
        assert events == ["run.accepted", "run.completed"]  # never two terminals
        assert tracked == 0  # the lease entry is released either way
