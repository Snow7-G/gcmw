"""Pre-route entry guard (#66): authenticate + charge BEFORE validation.

Acceptance for this slice (review task list):
- consecutive invalid bodies eventually answer 429;
- anonymous requests stay 401 and consume nothing;
- one request is charged exactly once;
- a foreign device never spends the owner's session quota;
- a throttled request touches no storage and streams no SSE byte.
Plus: the buffered body is replayed for the route, and an oversized body is
refused with 413 instead of being read into memory.
"""

from __future__ import annotations

import pytest
from api_harness import (
    OTHER_DEVICE_TOKEN,
    PRIMARY_TOKEN,
    new_run,
    new_session,
    running_app,
)

from app.contracts.errors import ErrorCode
from app.main import PUBLIC_PATHS

SESSION_ONLY = {
    "rate_limit_session_per_minute": 1,
    "rate_limit_tenant_per_minute": 0,
    "rate_limit_device_per_minute": 0,
}
TENANT_ONLY = {
    "rate_limit_tenant_per_minute": 1,
    "rate_limit_device_per_minute": 0,
    "rate_limit_session_per_minute": 0,
}

INVALID_SESSION_BODY = {"channel": "not-a-channel"}


class TestChargingBeforeValidation:
    def test_consecutive_invalid_bodies_eventually_answer_429(self):
        """The whole point of this slice: a malformed body is not a free try."""
        limits = {"rate_limit_tenant_per_minute": 2}
        with running_app(settings_kwargs=limits) as h:
            attempts = [
                h.client.post("/api/v1/sessions", json=INVALID_SESSION_BODY).status_code
                for _ in range(3)
            ]
            after = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert attempts == [400, 400, 429]
        assert after.status_code == 429  # the budget really was consumed

    def test_a_valid_request_after_invalid_ones_still_gets_a_chance(self):
        """Invalid attempts consume quota, they do not break the API."""
        limits = {"rate_limit_tenant_per_minute": 3}
        with running_app(settings_kwargs=limits) as h:
            assert (
                h.client.post("/api/v1/sessions", json=INVALID_SESSION_BODY).status_code
                == 400
            )
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 201
            )

    def test_one_request_is_charged_exactly_once(self):
        with running_app(settings_kwargs=TENANT_ONLY) as h:
            first = h.client.post("/api/v1/sessions", json={"channel": "text"})
            second = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert first.status_code == 201
        assert second.status_code == 429


class TestAnonymousIsFree:
    def test_anonymous_requests_are_401_and_consume_nothing(self):
        with running_app(default_credential=None, settings_kwargs=TENANT_ONLY) as h:
            for _ in range(5):
                res = h.client.post("/api/v1/sessions", json={"channel": "text"})
                assert res.status_code == 401
                assert res.json()["code"] == "E_AUTH_MISSING_CREDENTIALS"
            # the tenant budget is untouched: a real credential still works
            with h.as_token(PRIMARY_TOKEN):
                ok = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert ok.status_code == 201

    def test_invalid_credentials_are_401_and_consume_nothing(self):
        with running_app(default_credential=None, settings_kwargs=TENANT_ONLY) as h:
            with h.as_token("not-a-registered-credential"):
                res = h.client.post("/api/v1/sessions", json={"channel": "text"})
                assert res.status_code == 401
                assert res.json()["code"] == "E_AUTH_INVALID_CREDENTIALS"
            with h.as_token(PRIMARY_TOKEN):
                ok = h.client.post("/api/v1/sessions", json={"channel": "text"})
        assert ok.status_code == 201

    def test_health_probes_are_outside_the_guard(self):
        assert "/api/v1/health/live" in PUBLIC_PATHS
        with running_app(default_credential=None) as h:
            assert h.client.get("/api/v1/health/live").status_code == 200
            assert h.client.get("/api/v1/health/ready").status_code == 200


class TestSessionQuotaBelongsToTheOwner:
    def test_foreign_device_never_spends_the_owners_session_quota(self):
        with running_app(settings_kwargs=SESSION_ONLY) as h:
            session = new_session(h)
            session_id = session["session_id"]
            with h.as_token(OTHER_DEVICE_TOKEN):
                codes = [
                    h.client.delete(f"/api/v1/sessions/{session_id}").status_code
                    for _ in range(3)
                ]
            # first refusal is the ACL (403); after that the foreign device is
            # throttled by ITS OWN session window (limit 1) — never the owner's
            assert codes == [403, 429, 429]
            # the owner still has its full window and can finish the job
            assert h.client.delete(f"/api/v1/sessions/{session_id}").status_code == 204

    def test_unknown_session_id_charges_the_callers_own_window(self):
        """A path-named session is charged even when it does not exist.

        The key still carries the authenticated tenant+device, so guessing ids
        can only exhaust the caller's own window — never somebody else's — and
        probing is rate limited like any other attempt.
        """
        with running_app(settings_kwargs=SESSION_ONLY) as h:
            first = h.client.delete("/api/v1/sessions/ghost")
            second = h.client.delete("/api/v1/sessions/ghost")
            assert first.status_code == 404
            assert second.status_code == 429  # the caller's own window


class TestThrottledRequestsDoNothing:
    def test_throttled_request_touches_no_storage_and_streams_no_byte(self):
        limits = {"rate_limit_tenant_per_minute": 2}
        with running_app(settings_kwargs=limits) as h:
            session = new_session(h)
            new_run(h, session["session_id"], key="first")
            before_runs = dict(h.service.runs)
            before_idem = dict(h.service.idempotency)

            throttled = h.client.get(f"/api/v1/agent/runs/{session['session_id']}")
            stream = h.client.get("/api/v1/agent/runs/whatever/events")

            for res in (throttled, stream):
                assert res.status_code == 429
                assert res.json()["code"] == "E_RATE_LIMIT_EXCEEDED"
                assert res.headers["content-type"].startswith("application/json")
                assert "data:" not in res.text and "event:" not in res.text
            assert h.service.runs == before_runs
            assert h.service.idempotency == before_idem

    def test_bodies_are_never_parsed_for_a_throttled_request(self):
        """The guard answers before FastAPI reads the body at all."""
        with running_app(settings_kwargs=TENANT_ONLY) as h:
            new_session(h)  # spends the only token
            res = h.client.post("/api/v1/sessions", json=INVALID_SESSION_BODY)
        assert res.status_code == 429  # 429 wins over the 400 the body would earn


class TestBodyHandling:
    def test_the_buffered_body_is_replayed_to_the_route(self):
        with running_app() as h:
            session = new_session(h)
            run = new_run(h, session["session_id"], text="replayed body", key="replay")
            assert run["run_id"]
            stored = h.service.runs[run["run_id"]]
            assert stored.snapshot.text == "replayed body"  # the route saw it all

    def test_oversized_body_is_refused_with_413(self):
        limits = {"max_request_body_bytes": 512}
        with running_app(settings_kwargs=limits) as h:
            res = h.client.post(
                "/api/v1/sessions",
                json={"channel": "text", "padding": "x" * 2000},
            )
            assert res.status_code == 413
            assert res.json()["code"] == "E_VALIDATION_PAYLOAD_TOO_LARGE"
            assert h.service.sessions == {}  # nothing was created

    def test_oversized_body_still_consumed_its_attempt(self):
        limits = {"max_request_body_bytes": 512, "rate_limit_tenant_per_minute": 1}
        with running_app(settings_kwargs=limits) as h:
            big = h.client.post(
                "/api/v1/sessions", json={"channel": "text", "padding": "x" * 2000}
            )
            assert big.status_code == 413
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 429
            )

    def test_get_requests_are_not_body_buffered(self):
        with running_app() as h:
            assert h.client.get("/api/v1/health/live").status_code == 200
            assert h.client.get("/api/v1/agent/runs/ghost").status_code == 404


class TestGuardResponses:
    @pytest.mark.parametrize(
        ("headers", "expected_status", "expected_code"),
        [
            (None, 401, ErrorCode.AUTH_MISSING_CREDENTIALS),
            (
                {"Authorization": "Bearer nope-nope-nope-nope"},
                401,
                ErrorCode.AUTH_INVALID_CREDENTIALS,
            ),
        ],
        ids=["missing", "invalid"],
    )
    def test_rejections_use_the_shared_envelope(
        self, headers, expected_status, expected_code
    ):
        with running_app(default_credential=None) as h:
            res = h.client.post(
                "/api/v1/sessions", json={"channel": "text"}, headers=headers
            )
        body = res.json()
        assert res.status_code == expected_status
        assert body["code"] == expected_code.value
        assert body["request_id"] and body["trace_id"]
        assert res.headers["x-request-id"] == body["request_id"]
        assert res.headers["x-trace-id"] == body["trace_id"]
        assert set(body) == {
            "code",
            "message",
            "request_id",
            "trace_id",
            "retryable",
            "retry_after_ms",
        }

    def test_guard_decides_once_per_request(self):
        """The route dependency reuses the guard's decision (no second verify)."""
        calls = {"n": 0}
        with running_app() as h:
            store = h.app.state.credentials
            original = store.resolve

            def counting(presented):
                calls["n"] += 1
                return original(presented)

            store.resolve = counting
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 201
            )
        assert calls["n"] == 1
