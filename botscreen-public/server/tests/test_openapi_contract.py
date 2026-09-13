"""The published contract must match the wire (#65B-2 B2-B review round).

The public SSE route used to advertise ``application/json`` for a 200 and only
a 422 for failures, while the server actually answers ``text/event-stream`` and
a unified ``ErrorEnvelope`` at 400/401/403/404. Both halves are asserted here:
the schema document, and the statuses the running app really returns.
"""

from __future__ import annotations

import pytest
from api_harness import (
    OTHER_DEVICE,
    PRINCIPAL,
    Harness,
    new_run,
    new_session,
    running_app,
)

from app.api.v1.auth import get_device_principal
from app.api.v1.errors import AppError
from app.config import Settings
from app.contracts.errors import ErrorCode
from app.main import create_app

EVENTS_PATH = "/api/v1/agent/runs/{run_id}/events"

#: statuses the public SSE route documents (and must actually return)
EVENTS_ERROR_STATUSES = {"400", "401", "403", "404", "429", "500", "503"}
ENVELOPE_FIELDS = {
    "code",
    "message",
    "request_id",
    "trace_id",
    "retryable",
    "retry_after_ms",
}


@pytest.fixture(scope="module")
def schema() -> dict:
    return create_app(settings=Settings(environment="test")).openapi()


@pytest.fixture
def harness() -> Harness:
    with running_app() as h:
        yield h


def _operation(schema: dict, path: str, method: str = "get") -> dict:
    return schema["paths"][path][method]


class TestDocumentedContract:
    def test_events_200_is_text_event_stream(self, schema):
        responses = _operation(schema, EVENTS_PATH)["responses"]
        assert list(responses["200"]["content"]) == ["text/event-stream"]

    def test_events_route_documents_its_resume_inputs(self, schema):
        params = {
            (p["name"], p["in"]) for p in _operation(schema, EVENTS_PATH)["parameters"]
        }
        assert ("after_seq", "query") in params
        assert ("Last-Event-ID", "header") in params
        header = next(
            p
            for p in _operation(schema, EVENTS_PATH)["parameters"]
            if p["name"] == "Last-Event-ID"
        )
        assert header["required"] is False

    def test_events_route_documents_the_real_error_statuses(self, schema):
        responses = _operation(schema, EVENTS_PATH)["responses"]
        assert set(responses) - {"200"} == EVENTS_ERROR_STATUSES
        for status, response in responses.items():
            content = response["content"]
            if status == "200":
                # exactly one media type: the spurious JSON default must be gone
                assert set(content) == {"text/event-stream"}
                continue
            # pre-stream failures are JSON envelopes, never SSE frames
            assert set(content) == {"application/json"}
            ref = content["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorEnvelope")

    def test_non_streaming_routes_document_json_only(self, schema):
        for path, path_item in schema["paths"].items():
            for method, operation in path_item.items():
                if path == EVENTS_PATH:
                    continue
                for status, response in operation.get("responses", {}).items():
                    content = response.get("content") or {}
                    assert "text/event-stream" not in content, (method, path, status)

    def test_no_route_documents_422(self, schema):
        """Validation failures are answered with 400 by the error boundary."""
        for path_item in schema["paths"].values():
            for operation in path_item.values():
                assert "422" not in operation.get("responses", {})

    def test_error_envelope_shape_is_published(self, schema):
        published = schema["components"]["schemas"]["ErrorEnvelope"]["properties"]
        assert set(published) == ENVELOPE_FIELDS

    def test_write_routes_document_their_conflicts_and_absences(self, schema):
        create = _operation(schema, "/api/v1/agent/runs", "post")["responses"]
        assert {"400", "401", "403", "404", "409", "500", "503"} <= set(create)
        cancel = _operation(schema, "/api/v1/agent/runs/{run_id}", "delete")[
            "responses"
        ]
        assert {"403", "404", "409", "503"} <= set(cancel)

    def test_bearer_security_scheme_is_published(self, schema):
        scheme = schema["components"]["securitySchemes"]["BearerAuth"]
        assert scheme["type"] == "http" and scheme["scheme"] == "bearer"

    def test_protected_routes_require_the_scheme(self, schema):
        for path, path_item in schema["paths"].items():
            for method, operation in path_item.items():
                if path in {"/api/v1/health/live", "/api/v1/health/ready"}:
                    assert "security" not in operation, (method, path)
                    continue
                assert operation["security"] == [{"BearerAuth": []}], (method, path)

    def test_health_probes_stay_public(self, schema):
        for path in ("/api/v1/health/live", "/api/v1/health/ready"):
            assert "security" not in schema["paths"][path]["get"]

    def test_rate_limited_routes_document_429_and_503(self, schema):
        """Every route that charges a window publishes BOTH codes it can return.

        The limiter raises 429 when a window is exhausted and 503 when the key
        table is at its hard cap — a route that documents only one of them has a
        contract that disagrees with the wire (review P2).
        """
        for path, path_item in schema["paths"].items():
            for method, operation in path_item.items():
                responses = operation.get("responses", {})
                if path.startswith("/api/v1/health"):
                    assert "429" not in responses, (method, path)
                    continue
                for code in ("429", "503"):
                    assert code in responses, (method, path, code)
                    ref = responses[code]["content"]["application/json"]["schema"][
                        "$ref"
                    ]
                    assert ref.endswith("/ErrorEnvelope"), (method, path, code)

    def test_key_table_overload_is_declared_on_every_limited_route(self, schema):
        """503 documents the OVERLOAD reason explicitly, not just any 503."""
        for path, path_item in schema["paths"].items():
            if path.startswith("/api/v1/health"):
                continue
            for method, operation in path_item.items():
                description = operation["responses"]["503"]["description"]
                assert "E_UNAVAILABLE_OVERLOADED" in description, (method, path)

    def test_ready_documentation_matches_the_published_schema(self, schema):
        """The readiness body shape is documented, not implied."""
        body = schema["paths"]["/api/v1/health/ready"]["get"]["responses"]["200"]
        assert "application/json" in body["content"]

    def test_readiness_documents_503(self, schema):
        responses = _operation(schema, "/api/v1/health/ready")["responses"]
        assert set(responses) == {"200", "503"}


class TestWireParity:
    """Every documented status for the SSE route is actually reachable."""

    def test_200_is_a_stream(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.client.delete(f"/api/v1/agent/runs/{run['run_id']}")
        res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")

    def test_400_for_an_invalid_cursor(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        res = harness.client.get(
            f"/api/v1/agent/runs/{run['run_id']}/events?after_seq=-1"
        )
        assert res.status_code == 400
        assert res.json()["code"] == "E_VALIDATION_INVALID_INPUT"

    def test_401_without_credentials(self):
        with running_app(default_credential=None) as h:
            res = h.client.get("/api/v1/agent/runs/whatever/events")
        assert res.status_code == 401
        assert res.json()["code"] == "E_AUTH_MISSING_CREDENTIALS"

    def test_403_for_a_foreign_device(self, harness):
        run = new_run(harness, new_session(harness)["session_id"])
        harness.app.dependency_overrides[get_device_principal] = lambda: OTHER_DEVICE
        try:
            res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        finally:
            harness.app.dependency_overrides[get_device_principal] = lambda: PRINCIPAL
        assert res.status_code == 403
        assert res.json()["code"] == "E_AUTHZ_FORBIDDEN"

    def test_404_for_an_unknown_run(self, harness):
        res = harness.client.get("/api/v1/agent/runs/ghost/events")
        assert res.status_code == 404
        assert res.json()["code"] == "E_NOT_FOUND_RUN"

    def test_429_is_reachable_and_carries_the_wait_hint(self):
        """The declared 429 really happens, with retry_after_ms and Retry-After."""
        from api_harness import running_app

        limits = {"rate_limit_tenant_per_minute": 1}
        with running_app(settings_kwargs=limits) as h:
            assert (
                h.client.post("/api/v1/sessions", json={"channel": "text"}).status_code
                == 201
            )
            res = h.client.get("/api/v1/agent/runs/ghost/events")
        assert res.status_code == 429
        body = res.json()
        assert body["code"] == "E_RATE_LIMIT_EXCEEDED"
        assert body["retry_after_ms"] > 0
        assert int(res.headers["retry-after"]) >= 1
        assert "data:" not in res.text  # never a streamed byte

    def test_500_pre_stream_is_a_json_envelope_without_sse_bytes(self, harness):
        """A pre-stream crash is answered like every other API failure."""
        run = new_run(harness, new_session(harness)["session_id"])

        async def exploding_authorize(principal, run_id):
            raise ValueError("sensitive detail /etc/passwd")

        harness.service.authorize_stream = exploding_authorize
        res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        assert res.status_code == 500
        assert res.headers["content-type"].startswith("application/json")
        assert "data:" not in res.text and "event:" not in res.text
        assert "sensitive" not in res.text
        assert harness.env(res).code == "E_INTERNAL_UNKNOWN"

    def test_503_pre_stream_is_a_json_envelope_without_sse_bytes(self, harness):
        """Storage unavailable BEFORE the stream starts is a plain 503."""
        run = new_run(harness, new_session(harness)["session_id"])

        async def unavailable(principal, run_id):
            raise AppError(ErrorCode.UNAVAILABLE_OVERLOADED)

        harness.service.authorize_stream = unavailable
        res = harness.client.get(f"/api/v1/agent/runs/{run['run_id']}/events")
        assert res.status_code == 503
        assert res.headers["content-type"].startswith("application/json")
        assert "data:" not in res.text and "event:" not in res.text
        envelope = harness.env(res)
        assert envelope.code == "E_UNAVAILABLE_OVERLOADED"
        assert envelope.retryable is True

    def test_every_documented_status_is_reachable(self, schema, harness):
        """Belt and braces: the parity tests above cover the whole declared set."""
        documented = set(_operation(schema, EVENTS_PATH)["responses"])
        covered = {"200", "400", "401", "403", "404", "429", "500", "503"}
        assert documented == covered
