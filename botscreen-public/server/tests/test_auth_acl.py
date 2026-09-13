"""Device authentication + Session/Run ACL (#66, minimal slice).

Everything here runs against the REAL auth boundary (entry guard + header): the
credential store is seeded through the environment variable named by
``Settings.auth_credentials_env``, exactly as an operator would, so no test
bypasses the production path.

Acceptance from #66: default deny is unchanged — no credential, wrong device
and cross-tenant all end in a 4xx ErrorEnvelope, and credentials never leak into
responses. Rate limiting is deliberately NOT in this slice.
"""

from __future__ import annotations

import json

import pytest
from api_harness import PRINCIPAL, running_app

from app.api.v1.agent_api import RunAdmissionService
from app.api.v1.auth import (
    MAX_CREDENTIAL_LENGTH,
    MIN_CREDENTIAL_LENGTH,
    CredentialStore,
    DeviceCredential,
    DevicePrincipal,
    digest_credential,
    presented_credential,
)
from app.contracts.api import Channel, CreateSessionRequest
from app.contracts.errors import ErrorEnvelope
from app.contracts.identity import MAX_DEVICE_ID_LENGTH, MAX_TENANT_ID_LENGTH
from app.storage.run_repository import MemoryRunRepository


def _credential(label: str) -> str:
    """Deterministic low-entropy test credential (never a real secret).

    Kept deliberately non-secret-looking so a secret scanner cannot mistake it
    for leaked material.
    """
    return f"dev-{label}-000000000000"


OWN = _credential("own")
OTHER_DEVICE = _credential("other-device")
OTHER_TENANT = _credential("other-tenant")
UNKNOWN = _credential("unknown")

CREDENTIALS = [
    {"tenant_id": "t1", "device_id": "d1", "token": OWN},
    {"tenant_id": "t1", "device_id": "d2", "token": OTHER_DEVICE},
    {"tenant_id": "t2", "device_id": "d1", "token": OTHER_TENANT},
]

ENVELOPE_FIELDS = {
    "code",
    "message",
    "request_id",
    "trace_id",
    "retryable",
    "retry_after_ms",
}

RESOURCE_ROUTES = [
    ("get", "run"),
    ("delete", "run"),
    ("get", "events"),
    ("delete", "session"),
]
ALL_ROUTES = [("post", "sessions"), ("post", "runs"), *RESOURCE_ROUTES]


def _auth(credential: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential}"}


def _envelope(response) -> ErrorEnvelope:
    return ErrorEnvelope.model_validate(response.json())


def _call_with_headers(h, method: str, route: str, headers: dict | None):
    """Call a route with resources that do not exist.

    Authentication is decided BEFORE existence/ownership, so an anonymous or
    invalid credential always produces its 401 regardless of the resource.
    """
    if route == "sessions":
        return h.client.post(
            "/api/v1/sessions", json={"channel": "text"}, headers=headers
        )
    if route == "runs":
        return h.client.post(
            "/api/v1/agent/runs",
            json={
                "session_id": "ghost",
                "input": {"type": "text", "text": "x"},
                "idempotency_key": "x",
            },
            headers=headers,
        )
    if route == "run":
        return h.client.get("/api/v1/agent/runs/ghost", headers=headers)
    if route == "events":
        return h.client.get("/api/v1/agent/runs/ghost/events", headers=headers)
    if route == "session":
        return h.client.delete("/api/v1/sessions/ghost", headers=headers)
    raise AssertionError(f"unknown route {route}")


class TestStoreHygiene:
    def test_store_keeps_digests_only(self):
        store = CredentialStore.from_json(json.dumps(CREDENTIALS))
        assert len(store) == 3
        rendered = repr(store) + repr(store._credentials)
        for token in (OWN, OTHER_DEVICE, OTHER_TENANT):
            assert token not in rendered

    def test_resolution_is_by_digest_and_exact(self):
        store = CredentialStore.from_json(json.dumps(CREDENTIALS))
        resolved = store.resolve(OWN)
        assert resolved == DeviceCredential(
            tenant_id="t1", device_id="d1", token_digest=digest_credential(OWN)
        )
        assert store.resolve(UNKNOWN) is None
        assert store.resolve(OWN + "x") is None  # prefix is not a match
        assert store.resolve("") is None

    def test_empty_store_authenticates_nobody(self):
        store = CredentialStore.from_json(None)
        assert len(store) == 0 and store.configured is False
        assert store.resolve(OWN) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            json.dumps({"tenant_id": "t1"}),  # not a list
            json.dumps([42]),  # not an object
            json.dumps([{"tenant_id": "t1", "device_id": "d1"}]),  # missing token
            json.dumps([{"tenant_id": "", "device_id": "d1", "token": OWN}]),
            json.dumps([{"tenant_id": "t1", "device_id": "d1", "token": "short"}]),
            json.dumps(
                [
                    {"tenant_id": "t1", "device_id": "d1", "token": OWN},
                    {"tenant_id": "t9", "device_id": "d9", "token": OWN},
                ]
            ),
        ],
    )
    def test_malformed_store_aborts_startup(self, raw):
        with pytest.raises((TypeError, ValueError)):
            CredentialStore.from_json(raw)

    def test_minimum_length_is_enforced(self):
        short = "x" * (MIN_CREDENTIAL_LENGTH - 1)
        with pytest.raises(ValueError):
            CredentialStore.from_json(
                json.dumps([{"tenant_id": "t1", "device_id": "d1", "token": short}])
            )

    def test_header_parsing(self):
        from fastapi import Request

        def request(header: str | None) -> Request:
            headers = [] if header is None else [(b"authorization", header.encode())]
            return Request(
                {"type": "http", "method": "GET", "path": "/", "headers": headers}
            )

        assert presented_credential(request(None)) is None
        assert presented_credential(request("Basic abc")) is None
        assert presented_credential(request("Bearer   ")) is None
        assert presented_credential(request(f"Bearer {OWN}")) == OWN
        assert presented_credential(request(f"bearer {OWN}")) == OWN


class TestIdentityBoundsAtLoadTime:
    """Review P1: a credential must be rejectable AT STARTUP, not at request time.

    The API contract bounds ``tenant_id`` to 64 and ``device_id`` to 128
    characters; an identity outside those bounds used to authenticate happily
    and then break session creation with a 500 (leaving a session behind).
    """

    @staticmethod
    def _entry(**overrides) -> dict:
        entry = {"tenant_id": "t1", "device_id": "d1", "token": OWN}
        entry.update(overrides)
        return entry

    def test_contract_bounds_match_the_shared_constants(self):
        assert (MAX_TENANT_ID_LENGTH, MAX_DEVICE_ID_LENGTH) == (64, 128)

    def test_over_long_tenant_is_rejected_at_load(self):
        entry = self._entry(tenant_id="t" * (MAX_TENANT_ID_LENGTH + 1))
        with pytest.raises(ValueError) as excinfo:
            CredentialStore.from_json(json.dumps([entry]))
        assert "tenant_id" in str(excinfo.value)
        assert OWN not in str(excinfo.value)  # never echo the credential

    def test_over_long_device_is_rejected_at_load(self):
        entry = self._entry(device_id="d" * (MAX_DEVICE_ID_LENGTH + 1))
        with pytest.raises(ValueError) as excinfo:
            CredentialStore.from_json(json.dumps([entry]))
        assert "device_id" in str(excinfo.value)

    def test_boundary_lengths_are_accepted(self):
        store = CredentialStore.from_json(
            json.dumps(
                [
                    self._entry(
                        tenant_id="t" * MAX_TENANT_ID_LENGTH,
                        device_id="d" * MAX_DEVICE_ID_LENGTH,
                    )
                ]
            )
        )
        assert len(store) == 1

    @pytest.mark.parametrize("field", ["tenant_id", "device_id", "token"])
    @pytest.mark.parametrize("value", [42, 3.5, None, True, ["t1"], {"a": 1}])
    def test_non_string_fields_are_rejected_not_coerced(self, field, value):
        with pytest.raises((TypeError, ValueError)):
            CredentialStore.from_json(json.dumps([self._entry(**{field: value})]))

    @pytest.mark.parametrize("field", ["tenant_id", "device_id", "token"])
    def test_surrounding_whitespace_is_rejected(self, field):
        with pytest.raises(ValueError):
            CredentialStore.from_json(json.dumps([self._entry(**{field: f" {OWN} "})]))

    @pytest.mark.parametrize(
        "token",
        ["with space" + "0" * 10, "line\nbreak" + "0" * 10, 'quote"' + "0" * 10],
    )
    def test_tokens_outside_the_header_alphabet_are_rejected(self, token):
        with pytest.raises(ValueError):
            CredentialStore.from_json(json.dumps([self._entry(token=token)]))

    @pytest.mark.parametrize(
        "token",
        [
            "ZGV2LXRva2VuLXRlc3Q" + "=",  # one padding char
            "ZGV2LXRva2VuLXRlc3Q" + "==",  # two padding chars (RFC 6750)
            "dev-owned-" + "0" * 6 + "===",  # three padding chars
        ],
    )
    def test_trailing_base64_padding_is_accepted(self, token):
        """RFC 6750 b64token allows trailing ``=`` (review P2)."""
        store = CredentialStore.from_json(json.dumps([self._entry(token=token)]))
        assert len(store) == 1
        assert store.resolve(token) is not None

    @pytest.mark.parametrize("token", ["====" + "0" * 12, "a=b" + "0" * 13])
    def test_padding_must_be_trailing_only(self, token):
        with pytest.raises(ValueError):
            CredentialStore.from_json(json.dumps([self._entry(token=token)]))

    @pytest.mark.parametrize(
        "token",
        [
            "a" + "=" * 15,  # 1 body character padded up to 16 total
            "a" * 15 + "=",  # 15 body characters
            "a" * 15 + "=" * 5,
        ],
        ids=["1-body-15-padding", "15-body", "15-body-5-padding"],
    )
    def test_padding_cannot_be_used_to_reach_the_minimum(self, token):
        """Review P1: the 16-character floor applies to the BODY."""
        with pytest.raises(ValueError) as excinfo:
            CredentialStore.from_json(json.dumps([self._entry(token=token)]))
        assert "excluding base64 padding" in str(excinfo.value)

    @pytest.mark.parametrize(
        "token",
        ["a" * 16, "a" * 16 + "=", "a" * 16 + "==", "a" * 16 + "=" * 8],
        ids=["16-body", "16-body-1-pad", "16-body-2-pad", "16-body-8-pad"],
    )
    def test_sixteen_body_characters_with_padding_are_accepted(self, token):
        store = CredentialStore.from_json(json.dumps([self._entry(token=token)]))
        assert len(store) == 1
        assert store.resolve(token) is not None

    def test_total_length_cap_still_applies_including_padding(self):
        with pytest.raises(ValueError) as excinfo:
            CredentialStore.from_json(
                json.dumps([self._entry(token="a" * MAX_CREDENTIAL_LENGTH + "=")])
            )
        assert "including padding" in str(excinfo.value)

    def test_pure_padding_is_still_rejected(self):
        with pytest.raises(ValueError):
            CredentialStore.from_json(
                json.dumps([self._entry(token="=" * (MIN_CREDENTIAL_LENGTH + 4))])
            )

    def test_a_padded_credential_authenticates_end_to_end(self):
        padded = "ZGV2LXRva2VuLXRlc3Q" + "=="  # ≥ MIN_CREDENTIAL_LENGTH
        entries = [{"tenant_id": "t1", "device_id": "d1", "token": padded}]
        with running_app(credentials=entries, default_credential=None) as h:
            res = h.client.post(
                "/api/v1/sessions",
                json={"channel": "text"},
                headers={"Authorization": f"Bearer {padded}"},
            )
            assert res.status_code == 201, res.text

    def test_over_long_token_is_rejected(self):
        with pytest.raises(ValueError):
            CredentialStore.from_json(
                json.dumps([self._entry(token="a" * (MAX_CREDENTIAL_LENGTH + 1))])
            )

    def test_an_invalid_credential_aborts_application_startup(self):
        """End-to-end: the application refuses to serve, it does not 500 later."""
        bad = [
            {
                "tenant_id": "t" * (MAX_TENANT_ID_LENGTH + 1),
                "device_id": "d1",
                "token": OWN,
            }
        ]
        with (  # startup must fail before the app can serve anything
            pytest.raises((TypeError, ValueError)),
            running_app(credentials=bad, default_credential=None),
        ):
            pass  # pragma: no cover - unreachable when startup fails


class TestSessionWriteOrdering:
    """Review P1: validate the response BEFORE storing the session."""

    def test_invalid_identity_leaves_no_session_behind(self):
        service = RunAdmissionService(repository=MemoryRunRepository())
        bad_principal = DevicePrincipal(
            tenant_id="t" * (MAX_TENANT_ID_LENGTH + 1), device_id="d1"
        )
        request = CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
        with pytest.raises(Exception):  # noqa: B017 - contract violation expected
            service.create_session(bad_principal, request)
        assert service.sessions == {}  # zero residue
        assert service.runs == {} and service.idempotency == {}

        # the service is still perfectly usable afterwards
        good = service.create_session(PRINCIPAL, request)
        assert good.session_id in service.sessions

    def test_invalid_device_leaves_no_session_behind(self):
        service = RunAdmissionService(repository=MemoryRunRepository())
        bad_principal = DevicePrincipal(
            tenant_id="t1", device_id="d" * (MAX_DEVICE_ID_LENGTH + 1)
        )
        with pytest.raises(Exception):  # noqa: B017 - contract violation expected
            service.create_session(
                bad_principal,
                CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN"),
            )
        assert service.sessions == {}

    def test_valid_identity_still_creates_a_session(self):
        service = RunAdmissionService(repository=MemoryRunRepository())
        session = service.create_session(
            PRINCIPAL, CreateSessionRequest(channel=Channel.TEXT, locale="zh-CN")
        )
        assert service.sessions[session.session_id].tenant_id == "t1"


class TestAuthenticationMatrix:
    @pytest.mark.parametrize(("method", "route"), ALL_ROUTES)
    def test_missing_credential_is_401(self, method, route):
        with running_app(credentials=CREDENTIALS, default_credential=None) as h:
            res = _call_with_headers(h, method, route, None)
            assert res.status_code == 401
            assert _envelope(res).code == "E_AUTH_MISSING_CREDENTIALS"
            assert set(res.json()) == ENVELOPE_FIELDS

    @pytest.mark.parametrize(("method", "route"), ALL_ROUTES)
    def test_malformed_scheme_is_401(self, method, route):
        with running_app(credentials=CREDENTIALS, default_credential=None) as h:
            res = _call_with_headers(h, method, route, {"Authorization": "Basic abc"})
            assert res.status_code == 401
            assert _envelope(res).code == "E_AUTH_MISSING_CREDENTIALS"

    @pytest.mark.parametrize(("method", "route"), ALL_ROUTES)
    def test_empty_bearer_is_401(self, method, route):
        with running_app(credentials=CREDENTIALS, default_credential=None) as h:
            res = _call_with_headers(h, method, route, {"Authorization": "Bearer  "})
            assert res.status_code == 401
            assert _envelope(res).code == "E_AUTH_MISSING_CREDENTIALS"

    @pytest.mark.parametrize(("method", "route"), ALL_ROUTES)
    def test_unknown_credential_is_401(self, method, route):
        with running_app(credentials=CREDENTIALS, default_credential=None) as h:
            res = _call_with_headers(h, method, route, _auth(UNKNOWN))
            assert res.status_code == 401
            assert _envelope(res).code == "E_AUTH_INVALID_CREDENTIALS"

    def test_health_endpoints_stay_public(self):
        with running_app(credentials=CREDENTIALS) as h:
            assert h.client.get("/api/v1/health/live").status_code == 200
            ready = h.client.get("/api/v1/health/ready")
            assert ready.status_code == 200
            assert ready.json()["checks"]["device_credentials"] == "configured"

    def test_success_path_with_real_credential(self):
        with running_app(credentials=CREDENTIALS) as h:
            session = h.client.post(
                "/api/v1/sessions", json={"channel": "text"}, headers=_auth(OWN)
            )
            assert session.status_code == 201
            body = session.json()
            assert body["tenant_id"] == "t1" and body["device_id"] == "d1"
            run = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": body["session_id"],
                    "input": {"type": "text", "text": "hello"},
                    "idempotency_key": "ok",
                },
                headers=_auth(OWN),
            )
            assert run.status_code == 200
            run_id = run.json()["run_id"]
            # a non-terminal stream never ends and TestClient buffers bodies, so
            # the run is cancelled first: the stream is then finite
            assert (
                h.client.delete(
                    f"/api/v1/agent/runs/{run_id}", headers=_auth(OWN)
                ).status_code
                == 200
            )
            stream = h.client.get(
                f"/api/v1/agent/runs/{run_id}/events", headers=_auth(OWN)
            )
            assert stream.status_code == 200
            assert "event: run.completed" in stream.text

    def test_credentials_never_appear_in_any_response(self):
        with running_app(credentials=CREDENTIALS) as h:
            responses = [
                h.client.get("/api/v1/health/ready"),
                h.client.get("/api/v1/agent/runs/ghost"),
                h.client.get("/api/v1/agent/runs/ghost", headers=_auth(OWN)),
                h.client.get("/api/v1/agent/runs/ghost", headers=_auth(UNKNOWN)),
                h.client.post("/api/v1/sessions", json={"channel": "nope"}),
            ]
            for response in responses:
                blob = response.text + json.dumps(dict(response.headers))
                for token in (OWN, OTHER_DEVICE, OTHER_TENANT, UNKNOWN):
                    assert token not in blob, response.request.url


class TestAclMatrix:
    """A valid credential must still not reach another device's resources."""

    @pytest.mark.parametrize(
        "credential", [OTHER_DEVICE, OTHER_TENANT], ids=["other-device", "other-tenant"]
    )
    def test_foreign_credential_cannot_read_or_cancel(self, credential):
        with running_app(credentials=CREDENTIALS) as h:
            session = h.client.post(
                "/api/v1/sessions", json={"channel": "text"}, headers=_auth(OWN)
            ).json()
            run = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "private"},
                    "idempotency_key": "private",
                },
                headers=_auth(OWN),
            ).json()

            headers = _auth(credential)
            run_url = f"/api/v1/agent/runs/{run['run_id']}"
            for res in (
                h.client.get(run_url, headers=headers),
                h.client.delete(run_url, headers=headers),
                h.client.get(f"{run_url}/events", headers=headers),
                h.client.delete(
                    f"/api/v1/sessions/{session['session_id']}", headers=headers
                ),
            ):
                assert res.status_code == 403
                assert _envelope(res).code == "E_AUTHZ_FORBIDDEN"
                assert "private" not in res.text
                assert "data:" not in res.text  # never a streamed byte

    def test_foreign_credential_cannot_create_a_run_in_a_foreign_session(self):
        with running_app(credentials=CREDENTIALS) as h:
            session = h.client.post(
                "/api/v1/sessions", json={"channel": "text"}, headers=_auth(OWN)
            ).json()
            res = h.client.post(
                "/api/v1/agent/runs",
                json={
                    "session_id": session["session_id"],
                    "input": {"type": "text", "text": "steal"},
                    "idempotency_key": "steal",
                },
                headers=_auth(OTHER_TENANT),
            )
            assert res.status_code == 403
            assert _envelope(res).code == "E_AUTHZ_FORBIDDEN"

    def test_foreign_credential_gets_403_not_404_leakage(self):
        """A foreign caller learns nothing about whether the run exists."""
        with running_app(credentials=CREDENTIALS) as h:
            res = h.client.get("/api/v1/agent/runs/ghost", headers=_auth(OTHER_TENANT))
            assert res.status_code == 404  # unknown run stays "not found"
            assert _envelope(res).code == "E_NOT_FOUND_RUN"

    def test_each_credential_only_sees_its_own_sessions(self):
        with running_app(credentials=CREDENTIALS) as h:
            mine = h.client.post(
                "/api/v1/sessions", json={"channel": "text"}, headers=_auth(OWN)
            ).json()
            theirs = h.client.post(
                "/api/v1/sessions",
                json={"channel": "text"},
                headers=_auth(OTHER_DEVICE),
            ).json()
            assert mine["session_id"] != theirs["session_id"]
            assert (
                h.client.delete(
                    f"/api/v1/sessions/{mine['session_id']}",
                    headers=_auth(OTHER_DEVICE),
                ).status_code
                == 403
            )
            assert (
                h.client.delete(
                    f"/api/v1/sessions/{theirs['session_id']}", headers=_auth(OWN)
                ).status_code
                == 403
            )
            assert (
                h.client.delete(
                    f"/api/v1/sessions/{mine['session_id']}", headers=_auth(OWN)
                ).status_code
                == 204
            )


class TestFailClosedWithoutCredentials:
    def test_empty_store_denies_every_route(self):
        with running_app(default_credential=None) as h:
            for method, route in ALL_ROUTES:
                res = _call_with_headers(h, method, route, _auth(OWN))
                assert res.status_code == 401
                assert _envelope(res).code == "E_AUTH_INVALID_CREDENTIALS"

    def test_repository_untouched_by_denied_requests(self):
        repository = MemoryRunRepository()
        with running_app(
            repository=repository, credentials=CREDENTIALS, default_credential=None
        ) as h:
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 401
            )
            assert h.app.state.agent_service.sessions == {}
            assert h.app.state.agent_service.runs == {}
