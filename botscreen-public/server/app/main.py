"""FastAPI application assembly (issue #36).

- central exception handling: every failure leaves the process as an
  :class:`~app.contracts.errors.ErrorEnvelope` with a stable ErrorCode and
  safe message only (no raw exceptions, provider text, paths or stacks);
- request/trace ids are minted per request and echoed in headers and error
  envelopes;
- validation errors map to ``E_VALIDATION_INVALID_INPUT`` (HTTP 400 — the API
  never answers 422, and the published contract says so);
- anything uncaught maps to ``E_INTERNAL_UNKNOWN``;
- configuration comes from :class:`~app.config.Settings` (``GCMW_ENV`` and
  friends) and the run service lives in the **lifespan scope**: one service per
  application run, i.e. exactly one event loop owns its asyncio primitives;
- startup FAILS CLOSED in staging/production while the run repository or the
  session/idempotency admission store is in-memory, or no device credentials are
  configured (see :mod:`app.runtime`);
- device credentials come from the environment (``GCMW_DEVICE_CREDENTIALS`` by
  default) and are only ever kept as digests (see :mod:`app.api.v1.auth`);
- SSE connection leases (subscriber counting + reconnect grace) are created in
  the same lifespan scope and torn down with the application;
- per-tenant/device/session rate limits are created in the same scope, so a
  throttled caller is rejected before any work happens (#66).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.agents.registry import RegistryError
from app.api.v1.agent_api import AppError, RunAdmissionService
from app.api.v1.agent_api import router as agent_router
from app.api.v1.auth import CredentialStore
from app.api.v1.errors import request_ids
from app.api.v1.rate_limit import RateLimiter, rules_from_settings
from app.api.v1.stream_leases import DEFAULT_RECONNECT_GRACE_S, RunLeaseRegistry
from app.config import Settings
from app.contracts.errors import ErrorCode, ErrorEnvelope, http_status_for
from app.providers.model_gateway import ModelGatewayError
from app.runtime import build_run_repository, readiness_report

APP_TITLE = "gcmw agent api"
APP_VERSION = "0.1.0"

#: structured audit sink for security-relevant decisions (rate limiting today).
#: A durable audit store is a follow-up slice; operators can route this logger.
AUDIT_LOGGER_NAME = "gcmw.audit"

#: routes that are public by design (liveness/readiness probes)
PUBLIC_PATHS = frozenset({"/api/v1/health/live", "/api/v1/health/ready"})


def _envelope_response(
    request: Request,
    code: ErrorCode,
    status_override: int | None = None,
    retry_after_ms: int | None = None,
) -> JSONResponse:
    request_id, trace_id = request_ids(request)
    envelope = ErrorEnvelope.build(
        code=code,
        request_id=request_id,
        trace_id=trace_id,
        retry_after_ms=retry_after_ms,
    )
    headers = {"X-Request-ID": request_id, "X-Trace-ID": trace_id}
    if retry_after_ms is not None:
        # standard hint for intermediaries/clients, rounded up to whole seconds
        headers["Retry-After"] = str(max(1, -(-retry_after_ms // 1000)))
    return JSONResponse(
        status_code=status_override or http_status_for(code),
        content=envelope.model_dump(mode="json"),
        headers=headers,
    )


def _declare_bearer_auth(schema: dict) -> None:
    """Publish the credential scheme the API actually enforces (#66).

    Every route except the health probes requires a device credential, so the
    contract says so instead of leaving a client to discover the 401s.
    """
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "设备凭据（Authorization: Bearer <credential>）。缺失/非法凭据 → 401；"
            "凭据不属于目标 tenant/device → 403，均为统一错误信封。"
        ),
    }
    for path, path_item in schema.get("paths", {}).items():
        if path in PUBLIC_PATHS:
            continue
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation["security"] = [{"BearerAuth": []}]


def _log_audit_record(record) -> None:
    """Default audit sink: one structured JSON line per security decision.

    Kept as a logger (not a store) on purpose: it is operable today and can be
    routed to a file/SIEM, while a durable audit sink needs the persistent
    store that #65 gates.
    """
    logging.getLogger(AUDIT_LOGGER_NAME).warning(record.model_dump_json())


def _normalize_stream_media_types(responses: dict) -> None:
    """Keep exactly the media type each documented response really uses.

    FastAPI appends the route's ``response_class`` media type to EVERY response
    it documents. On the public SSE route that would advertise the pre-stream
    JSON error envelopes (400/401/403/404/500/503) as ``text/event-stream`` —
    exactly the kind of contract lie this API must not publish. A successful
    streaming response is SSE; every other response of that operation is the
    JSON ErrorEnvelope.
    """
    streaming = any(
        "text/event-stream" in response.get("content", {})
        for response in responses.values()
    )
    if not streaming:
        return
    for status, response in responses.items():
        content = response.get("content")
        if not content:
            continue
        keep = "text/event-stream" if status.startswith("2") else "application/json"
        for media_type in list(content):
            if media_type != keep:
                del content[media_type]


def create_app(
    settings: Settings | None = None,
    repository_factory: Callable[[Settings], object] | None = None,
    reconnect_grace_s: float = DEFAULT_RECONNECT_GRACE_S,
) -> FastAPI:
    """Compose the application.

    ``settings`` defaults to the environment (``GCMW_ENV`` …); the repository
    factory is the single composition seam (tests inject a repository, the
    runtime picks the backend for the environment).
    """
    settings = settings or Settings.from_env()
    factory = repository_factory or build_run_repository

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # the service is built ONCE per application run: a single event loop
        # owns its asyncio locks, and no module-level singleton can be shared
        # between loops (or between workers) by accident
        service = RunAdmissionService(repository=factory(settings))
        credentials = CredentialStore.from_env(settings.auth_credentials_env)
        rate_limiter = RateLimiter(
            rules_from_settings(settings), audit=_log_audit_record
        )
        report = readiness_report(settings, service.repository, credentials)
        if not report.ready:
            raise RuntimeError(
                f"refusing to start in environment {settings.environment!r}: "
                + "; ".join(report.problems)
            )
        # SSE connection leases live beside the service: the last subscriber of
        # a run leaving starts the reconnect grace, and only then is the run
        # cancelled (see ``RunLeaseRegistry``)
        leases = RunLeaseRegistry(
            on_expire=service.cancel_for_disconnect,
            grace_s=reconnect_grace_s,
        )
        app.state.agent_service = service
        app.state.credentials = credentials
        app.state.rate_limiter = rate_limiter
        app.state.readiness = report
        app.state.stream_leases = leases
        try:
            yield
        finally:
            await leases.shutdown()
            app.state.agent_service = None
            app.state.credentials = None
            app.state.rate_limiter = None
            app.state.readiness = None
            app.state.stream_leases = None

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.agent_service = None
    app.state.credentials = None
    app.state.rate_limiter = None
    app.state.readiness = None
    app.state.stream_leases = None

    @app.middleware("http")
    async def ids_middleware(request: Request, call_next):
        request_ids(request)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    # -- central error mapping ------------------------------------------------

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return _envelope_response(request, exc.code, retry_after_ms=exc.retry_after_ms)

    @app.exception_handler(RegistryError)
    async def registry_error_handler(request: Request, exc: RegistryError):
        return _envelope_response(request, exc.code)

    @app.exception_handler(ModelGatewayError)
    async def gateway_error_handler(request: Request, exc: ModelGatewayError):
        return _envelope_response(request, exc.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _envelope_response(
            request, ErrorCode.VALIDATION_INVALID_INPUT, status_override=400
        )

    @app.exception_handler(Exception)
    async def uncaught_error_handler(request: Request, exc: Exception):
        # never leak the exception; the audit/trace layer (observability) may
        # correlate via trace id
        return _envelope_response(request, ErrorCode.INTERNAL_UNKNOWN)

    app.include_router(agent_router)

    # -- published contract ----------------------------------------------------

    def openapi() -> dict:
        """Serve a contract that matches the wire, not the framework defaults.

        FastAPI advertises 422 for validated parameters, but this application
        maps EVERY validation failure onto a 400 ``ErrorEnvelope``; the entry is
        therefore removed so the published contract cannot disagree with the
        response a client actually receives.
        """
        if app.openapi_schema is None:
            schema = get_openapi(
                title=APP_TITLE, version=APP_VERSION, routes=app.routes
            )
            _declare_bearer_auth(schema)
            for path_item in schema.get("paths", {}).values():
                for operation in path_item.values():
                    if not isinstance(operation, dict):
                        continue
                    responses = operation.get("responses", {})
                    responses.pop("422", None)
                    _normalize_stream_media_types(responses)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


app = create_app()
