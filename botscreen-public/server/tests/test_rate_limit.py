"""Per-tenant/device/session rate limiting (#66 remainder, minimal slice).

Counting is asserted with an injected clock and the HTTP behaviour with tiny
configured windows, so nothing here depends on wall-clock timing. The audit
requirement is covered by inspecting the AuditRecord the limiter emits.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from api_harness import (
    OTHER_DEVICE_TOKEN,
    OTHER_TENANT_TOKEN,
    PRINCIPAL,
    Harness,
    new_run,
    new_session,
    running_app,
)
from fastapi.testclient import TestClient

from app.api.v1.auth import DevicePrincipal
from app.api.v1.errors import AppError
from app.api.v1.rate_limit import (
    SCOPE_DEVICE,
    SCOPE_SESSION,
    SCOPE_TENANT,
    RateLimiter,
    RateLimitRule,
    rules_from_settings,
)
from app.config import Settings
from app.contracts.audit import AuditRecord
from app.contracts.errors import ErrorCode
from app.main import create_app


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


TENANT = RateLimitRule(scope=SCOPE_TENANT, limit=2, window_s=60.0)
DEVICE = RateLimitRule(scope=SCOPE_DEVICE, limit=3, window_s=60.0)
SESSION = RateLimitRule(scope=SCOPE_SESSION, limit=1, window_s=60.0)


def _limiter(rules=(TENANT,), **kwargs) -> tuple[RateLimiter, FakeClock, list]:
    clock = FakeClock()
    audit: list[AuditRecord] = []
    limiter = RateLimiter(tuple(rules), clock=clock, audit=audit.append, **kwargs)
    return limiter, clock, audit


class TestRuleValidation:
    def test_rules_from_settings_skip_disabled_scopes(self):
        settings = Settings(
            environment="test",
            rate_limit_tenant_per_minute=10,
            rate_limit_device_per_minute=0,
            rate_limit_session_per_minute=5,
        )
        rules = rules_from_settings(settings)
        assert [(r.scope, r.limit, r.window_s) for r in rules] == [
            (SCOPE_TENANT, 10, 60.0),
            (SCOPE_SESSION, 5, 60.0),
        ]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scope": "world", "limit": 1, "window_s": 1.0},
            {"scope": SCOPE_TENANT, "limit": 0, "window_s": 1.0},
            {"scope": SCOPE_TENANT, "limit": 1, "window_s": 0},
        ],
    )
    def test_invalid_rule_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RateLimitRule(**kwargs)

    def test_invalid_max_keys_is_rejected(self):
        with pytest.raises(ValueError):
            RateLimiter((TENANT,), max_keys=0)

    def test_mixed_window_lengths_are_refused(self):
        """Pruning reclaims by age against one horizon; mixing is refused until
        the pruner tracks per-window expiries (review, non-blocking item)."""
        long_window = RateLimitRule(scope=SCOPE_DEVICE, limit=1, window_s=3600.0)
        with pytest.raises(ValueError) as excinfo:
            RateLimiter((TENANT, long_window))
        assert "one window length" in str(excinfo.value)

    def test_a_single_window_length_is_accepted(self):
        limiter = RateLimiter(
            (TENANT, RateLimitRule(scope=SCOPE_DEVICE, limit=1, window_s=60.0))
        )
        assert limiter.tracked_keys() == 0


class TestWindowCounting:
    def test_limit_allows_exactly_limit_requests(self):
        limiter, _clock, _audit = _limiter()
        limiter.enforce(PRINCIPAL)
        limiter.enforce(PRINCIPAL)
        with pytest.raises(AppError) as excinfo:
            limiter.enforce(PRINCIPAL)
        assert excinfo.value.code is ErrorCode.RATE_LIMIT_EXCEEDED
        assert excinfo.value.retry_after_ms > 0

    def test_window_rolls_over_after_the_configured_seconds(self):
        limiter, clock, _audit = _limiter()
        limiter.enforce(PRINCIPAL)
        limiter.enforce(PRINCIPAL)
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL)
        clock.advance(60.0)
        limiter.enforce(PRINCIPAL)  # fresh window

    def test_retry_after_ms_counts_down_inside_the_window(self):
        limiter, clock, _audit = _limiter()
        for _ in range(2):
            limiter.enforce(PRINCIPAL)
        with pytest.raises(AppError) as first:
            limiter.enforce(PRINCIPAL)
        clock.advance(30.0)
        with pytest.raises(AppError) as second:
            limiter.enforce(PRINCIPAL)
        assert second.value.retry_after_ms < first.value.retry_after_ms
        # reported wait is rounded UP by 1ms so a client never retries too early
        assert 30_000 <= second.value.retry_after_ms <= 30_001
        assert 60_000 <= first.value.retry_after_ms <= 60_001

    def test_scopes_are_independent(self):
        """Session windows are per session: one exhausted session is not another."""
        limiter, _clock, _audit = _limiter((SESSION,))
        limiter.enforce(PRINCIPAL, session_id="s1")
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL, session_id="s1")
        limiter.enforce(PRINCIPAL, session_id="s2")  # separate window
        limiter.enforce(PRINCIPAL)  # names no session: never session-charged

    def test_narrower_scope_rejection_still_charges_the_outer_windows(self):
        """Within one request, a tenant/device budget counts attempts, so a
        hammered session cannot hide behind its own narrower window."""
        limiter, _clock, _audit = _limiter((TENANT, SESSION))
        limiter.enforce(PRINCIPAL, session_id="s1")  # tenant 1/2, session 1/1
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL, session_id="s1")  # tenant 2/2, session trips
        with pytest.raises(AppError) as excinfo:
            limiter.enforce(PRINCIPAL, session_id="s1")  # tenant is exhausted now
        assert excinfo.value.code is ErrorCode.RATE_LIMIT_EXCEEDED

    def test_session_scope_is_skipped_when_no_session_is_named(self):
        limiter, _clock, _audit = _limiter((SESSION,))
        for _ in range(5):
            limiter.enforce(PRINCIPAL)  # never session-charged
        assert limiter.tracked_keys() == 0

    def test_devices_of_the_same_tenant_have_separate_windows(self):
        limiter, _clock, _audit = _limiter((DEVICE,))
        other = DevicePrincipal(tenant_id=PRINCIPAL.tenant_id, device_id="d2")
        for _ in range(3):
            limiter.enforce(PRINCIPAL)
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL)
        limiter.enforce(other)  # unaffected

    def test_key_table_stays_bounded(self):
        limiter, clock, _audit = _limiter(max_keys=10)
        for i in range(500):
            limiter.enforce(DevicePrincipal(tenant_id=f"t{i}", device_id="d"))
            clock.advance(61.0)  # each window is expired by the next call
        assert limiter.tracked_keys() <= 10


class TestKeyScoping:
    """Review P1: keys must carry the AUTHENTICATED identity, not a bare id."""

    def test_a_foreign_device_cannot_exhaust_another_devices_session_window(self):
        rule = RateLimitRule(scope=SCOPE_SESSION, limit=2, window_s=60.0)
        limiter, _clock, _audit = _limiter((rule,))
        owner = DevicePrincipal(tenant_id="t1", device_id="d1")
        foreign_same_tenant = DevicePrincipal(tenant_id="t1", device_id="d2")
        foreign_tenant = DevicePrincipal(tenant_id="t2", device_id="d9")

        limiter.enforce(owner, session_id="shared-session")  # owner 1/2
        # both foreign callers name the SAME session id: they only charge themselves
        limiter.enforce(foreign_same_tenant, session_id="shared-session")  # 1/2
        limiter.enforce(foreign_tenant, session_id="shared-session")  # 1/2
        limiter.enforce(foreign_tenant, session_id="shared-session")  # 2/2

        limiter.enforce(owner, session_id="shared-session")  # owner 2/2, no trip
        with pytest.raises(AppError):  # only the owner's OWN window trips
            limiter.enforce(owner, session_id="shared-session")

    def test_identifiers_containing_separators_do_not_collide(self):
        """Joined strings could collide; structured tuples cannot."""
        limiter, _clock, _audit = _limiter((DEVICE,))
        a = DevicePrincipal(tenant_id="t\x1f1", device_id="d")
        b = DevicePrincipal(tenant_id="t", device_id="1\x1fd")
        for _ in range(3):  # DEVICE limit is 3
            limiter.enforce(a)
        with pytest.raises(AppError):
            limiter.enforce(a)
        limiter.enforce(b)  # a different identity, never the same window

    def test_session_key_carries_scope_tenant_and_device(self):
        limiter, _clock, _audit = _limiter((SESSION,))
        limiter.enforce(PRINCIPAL, session_id="s")
        assert limiter.tracked_keys() == 1
        # inspecting the table shape: (scope, tenant, device, session) as fields
        (entry_key,) = limiter._windows
        assert entry_key == (SCOPE_SESSION, "t1", "d1", "s")


class TestHardKeyCap:
    """Review P1: ``max_keys`` is a HARD cap (a probe saw 3 keys for max_keys=2)."""

    def test_new_keys_are_refused_once_the_table_is_full(self):
        limiter, _clock, audit = _limiter(max_keys=2)
        limiter.enforce(DevicePrincipal(tenant_id="t1", device_id="d"))
        limiter.enforce(DevicePrincipal(tenant_id="t2", device_id="d"))
        assert limiter.tracked_keys() == 2
        with pytest.raises(AppError) as excinfo:
            limiter.enforce(DevicePrincipal(tenant_id="t3", device_id="d"))
        assert excinfo.value.code is ErrorCode.UNAVAILABLE_OVERLOADED
        assert limiter.tracked_keys() == 2  # never grows past the cap
        assert len(audit) == 1
        assert audit[0].result == "overloaded"
        assert audit[0].error_code is ErrorCode.UNAVAILABLE_OVERLOADED

    def test_existing_keys_are_still_charged_when_the_table_is_full(self):
        limiter, _clock, _audit = _limiter(max_keys=1)
        limiter.enforce(DevicePrincipal(tenant_id="t1", device_id="d"))  # 1/2
        limiter.enforce(DevicePrincipal(tenant_id="t1", device_id="d"))  # 2/2
        with pytest.raises(AppError) as excinfo:
            limiter.enforce(DevicePrincipal(tenant_id="t1", device_id="d"))
        assert excinfo.value.code is ErrorCode.RATE_LIMIT_EXCEEDED  # not overload

    def test_expired_windows_are_reclaimed_before_refusing(self):
        limiter, clock, _audit = _limiter(max_keys=1)
        limiter.enforce(DevicePrincipal(tenant_id="t1", device_id="d"))
        clock.advance(61.0)
        # the old window is expired: the new key is admitted, table stays at 1
        limiter.enforce(DevicePrincipal(tenant_id="t2", device_id="d"))
        assert limiter.tracked_keys() == 1


class TestAudit:
    def test_trip_emits_one_hashed_audit_record(self):
        limiter, _clock, audit = _limiter()
        limiter.enforce(PRINCIPAL, session_id="session-1", request_id="req-7")
        limiter.enforce(PRINCIPAL, session_id="session-1", request_id="req-7")
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL, session_id="session-1", request_id="req-7")
        assert len(audit) == 1
        record = audit[0]
        assert record.tenant_id == PRINCIPAL.tenant_id
        assert record.action == "rate_limit.tenant"
        assert record.result == "throttled"
        assert record.error_code is ErrorCode.RATE_LIMIT_EXCEEDED
        assert record.request_id == "req-7"
        # identifiers are hashed, never the raw device/session id
        assert PRINCIPAL.device_id not in record.actor_id_hash
        assert len(record.actor_id_hash) == 64
        assert "session-1" not in record.session_id_hash

    def test_no_audit_sink_is_tolerated(self):
        limiter = RateLimiter((TENANT,), clock=FakeClock())
        limiter.enforce(PRINCIPAL)
        limiter.enforce(PRINCIPAL)
        with pytest.raises(AppError):
            limiter.enforce(PRINCIPAL)


class TestHttpEnforcement:
    """HTTP behaviour of the limiter (enforced by the entry guard, #66)."""

    LIMITS: ClassVar[dict] = {
        "rate_limit_tenant_per_minute": 3,
        "rate_limit_device_per_minute": 0,
        "rate_limit_session_per_minute": 0,
    }

    def test_429_envelope_carries_retry_after(self):
        with running_app(settings_kwargs=self.LIMITS) as h:
            codes = [
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                for _ in range(4)
            ]
        assert codes[:3] == [201, 201, 201]
        assert codes[3] == 429

    def test_429_body_is_a_unified_envelope_with_wait_hint(self):
        with running_app(settings_kwargs=self.LIMITS) as h:
            for _ in range(3):
                h.client.post("/api/v1/sessions", json={"channel": "text"})
            res = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert res.status_code == 429
        body = res.json()
        assert body["code"] == "E_RATE_LIMIT_EXCEEDED"
        assert body["retry_after_ms"] > 0
        assert body["retryable"] is True
        assert set(body) == {
            "code",
            "message",
            "request_id",
            "trace_id",
            "retryable",
            "retry_after_ms",
        }
        assert int(res.headers["retry-after"]) >= 1

    def test_throttled_stream_route_returns_json_not_sse(self):
        limits = {"rate_limit_tenant_per_minute": 1}
        with running_app(settings_kwargs=limits) as h:
            new_session(h)  # consumes the single token
            res = h.client.get("/api/v1/agent/runs/whatever/events")
        assert res.status_code == 429
        assert res.headers["content-type"].startswith("application/json")
        assert "data:" not in res.text and "event:" not in res.text

    def test_health_endpoints_are_not_rate_limited(self):
        limits = {"rate_limit_tenant_per_minute": 1}
        with running_app(settings_kwargs=limits) as h:
            new_session(h)
            assert h.client.get("/api/v1/health/live").status_code == 200
            assert h.client.get("/api/v1/health/ready").status_code == 200

    def test_each_request_is_charged_exactly_once(self):
        """A double charge would trip a limit of 1 on the FIRST request."""
        limits = {"rate_limit_tenant_per_minute": 1}
        with running_app(settings_kwargs=limits) as h:
            first = h.client.post("/api/v1/sessions", json={"channel": "text"})
            second = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert first.status_code == 201
        assert second.status_code == 429

    def test_device_window_is_per_device_not_per_tenant(self):
        limits = {"rate_limit_device_per_minute": 2, "rate_limit_tenant_per_minute": 0}
        with running_app(settings_kwargs=limits) as h:
            for _ in range(2):
                new_session(h)
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 429
            )
            # the second device of the same tenant is unaffected
            assert (
                h.client.post(
                    "/api/v1/sessions",
                    json={"channel": "text"},
                    headers=Harness.auth(OTHER_DEVICE_TOKEN),
                ).status_code
                == 201
            )

    def test_key_table_overload_returns_a_503_envelope(self):
        """A full limiter answers an explicit overload error, not silence."""
        with running_app() as h:
            h.app.state.rate_limiter = RateLimiter(
                (TENANT,), clock=FakeClock(), max_keys=1
            )
            h.app.state.rate_limiter.enforce(PRINCIPAL)  # occupies the only slot
            res = h.client.post(
                "/api/v1/sessions",
                json={"channel": "text"},
                headers=Harness.auth(OTHER_TENANT_TOKEN),
            )
        assert res.status_code == 503
        assert res.json()["code"] == "E_UNAVAILABLE_OVERLOADED"

    def test_session_window_follows_the_path_named_session(self):
        limits = {"rate_limit_session_per_minute": 1, "rate_limit_tenant_per_minute": 0}
        with running_app(settings_kwargs=limits) as h:
            session = new_session(h)
            other = new_session(h)
            assert (
                h.client.delete(f"/api/v1/sessions/{session['session_id']}").status_code
                == 204
            )
            second = h.client.delete(f"/api/v1/sessions/{session['session_id']}")
            assert second.status_code == 429  # same session: window exhausted
            assert (
                h.client.delete(f"/api/v1/sessions/{other['session_id']}").status_code
                == 204
            )

    def test_run_creation_is_session_charged_from_the_body(self):
        """Review P1: a session named by the PAYLOAD must be session-charged."""
        # window = 2: the run creation (body-named session) and the cancel
        # (path-named session) spend it, so the next attempt on THAT session 429s
        limits = {"rate_limit_session_per_minute": 2, "rate_limit_tenant_per_minute": 0}
        with running_app(settings_kwargs=limits) as h:
            session = new_session(h)
            first = new_run(h, session["session_id"], key="a")  # session charge 1
            assert (
                h.client.delete(f"/api/v1/agent/runs/{first['run_id']}").status_code
                == 200  # session charge 2 (path-derived)
            )
            second = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "again"},
                    "idempotency_key": "b",
                },
            )
            # a DIFFERENT session still has its own window
            other = new_session(h)
            third = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": other["session_id"],
                    "input": {"type": "text", "text": "elsewhere"},
                    "idempotency_key": "c",
                },
            )
        assert second.status_code == 429  # same session: window exhausted
        assert third.status_code == 200  # other session: unaffected

    def test_default_configuration_does_not_throttle_normal_use(self):
        with running_app() as h:
            for _ in range(20):
                assert (
                    h.client.post(
                        "/api/v1/sessions", json={"channel": "text"}
                    ).status_code
                    == 201
                )

    def test_rate_limit_settings_are_validated(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(environment="test", rate_limit_tenant_per_minute=-1)

    def test_limiter_lives_in_the_lifespan_scope(self):
        with running_app() as h:
            assert h.app.state.rate_limiter is not None
        assert h.app.state.rate_limiter is None

    def test_create_app_without_settings_uses_environment(self, monkeypatch):
        monkeypatch.setenv("GCMW_ENV", "test")
        app = create_app()
        with TestClient(app) as client:
            assert client.get("/api/v1/health/ready").status_code == 200
            assert app.state.rate_limiter is not None
