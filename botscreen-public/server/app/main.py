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

import asyncio
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
from app.api.v1.entry_guard import EntryGuardMiddleware
from app.api.v1.errors import envelope_response, request_ids
from app.api.v1.rate_limit import RateLimiter, rules_from_settings
from app.api.v1.stream_leases import DEFAULT_RECONNECT_GRACE_S, RunLeaseRegistry
from app.config import Settings
from app.contracts.errors import ErrorCode
from app.orchestration.assembly import build_agent_executor
from app.providers.model_gateway import ModelGatewayError
from app.runtime import build_run_repository, readiness_report
from app.tools.builtins import build_gateway
from app.tools.gateway import ToolGatewayError

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
    return envelope_response(request, code, status_override, retry_after_ms)


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
    agent_executor: bool = False,
    session_sweep_interval_s: float = 5.0,
) -> FastAPI:
    """Compose the application.

    ``settings`` defaults to the environment (``GCMW_ENV`` …); the repository
    factory is the single composition seam (tests inject a repository, the
    runtime picks the backend for the environment).

    ``agent_executor`` opts the app into the #55A vertical slice: admitted runs
    are then driven to a terminal state by the demo agent stack (synthetic
    knowledge + MockProvider + Manager/RAG/Verifier). It is OFF by default so
    admission-only tests keep their contract, and the assembly itself refuses
    to build anything outside development / test.

    ``session_sweep_interval_s`` is the fixed cadence of the ACTIVE session-TTL
    sweeper (P1-2): expiry must not depend on a session being touched again —
    a one-shot voice session never is. Tests may shrink the interval or drive
    ``RunAdmissionService.expire_due_sessions()`` directly.
    """
    settings = settings or Settings.from_env()
    factory = repository_factory or build_run_repository

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # the service is built ONCE per application run: a single event loop
        # owns its asyncio locks, and no module-level singleton can be shared
        # between loops (or between workers) by accident
        repository = factory(settings)
        executor = (
            build_agent_executor(repository=repository, settings=settings)
            if agent_executor
            else None
        )
        service = RunAdmissionService(repository=repository, executor=executor)
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
        # the tool gateway is the single tool entry point: whitelisted
        # read-only specs only, identity injected by the server (#57/#69)
        tool_gateway = build_gateway(audit_sink=_log_audit_record)
        app.state.tool_gateway = tool_gateway

        leases = RunLeaseRegistry(
            on_expire=service.cancel_for_disconnect,
            grace_s=reconnect_grace_s,
        )
        app.state.agent_service = service
        app.state.credentials = credentials
        app.state.rate_limiter = rate_limiter
        app.state.readiness = report
        app.state.stream_leases = leases
        app.state.run_executor = executor

        async def _session_sweeper() -> None:
            """Active TTL sweeper loop (P1-2). Faults are logged as a FIXED
            category with safe identifiers and NEVER kill the loop — a failed
            sweep simply retries at the next tick. No question text, no
            credentials, no raw exception text is ever logged."""
            while True:
                await asyncio.sleep(session_sweep_interval_s)
                try:
                    await service.expire_due_sessions()
                except Exception:  # noqa: BLE001 — the loop must survive
                    logging.getLogger("gcmw.session_expiry").warning(
                        "session_expiry_fault stage=sweeper_loop outcome=retry_next_sweep"
                    )

        sweeper_task = asyncio.create_task(
            _session_sweeper(), name="session-ttl-sweeper"
        )
        app.state.session_sweeper = sweeper_task
        try:
            yield
        finally:
            # ORDER MATTERS: stop the sweeper FIRST (cancel + await, so it can
            # never race the executor/lease teardown or touch state after
            # shutdown), then the existing teardown sequence.
            sweeper_task.cancel()
            try:
                await sweeper_task
            except asyncio.CancelledError:
                pass  # the expected way a healthy sweeper stops
            app.state.session_sweeper = None
            if executor is not None:
                await executor.shutdown()  # stop in-flight run tasks first
                if executor.tool_gateway is not None:
                    # the demo assembly's OWN gateway: release its worker pool
                    executor.tool_gateway.shutdown()
            await leases.shutdown()
            tool_gateway.shutdown()  # release the bounded tool worker pool
            app.state.agent_service = None
            app.state.credentials = None
            app.state.tool_gateway = None
            app.state.rate_limiter = None
            app.state.readiness = None
            app.state.stream_leases = None
            app.state.run_executor = None

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.agent_service = None
    app.state.credentials = None
    app.state.rate_limiter = None
    app.state.tool_gateway = None
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

    @app.exception_handler(ToolGatewayError)
    async def tool_error_handler(request: Request, exc: ToolGatewayError):
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

    # OUTERMOST middleware (added last on purpose): credentials and rate limits
    # are decided before routing, so an invalid body cannot buy a free attempt,
    # and the request body is replayed for the routes after being buffered here.
    app.add_middleware(
        EntryGuardMiddleware,
        public_paths=PUBLIC_PATHS,
        max_body_bytes=settings.max_request_body_bytes,
        body_timeout_s=settings.request_body_timeout_s,
    )

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


app = create_app(agent_executor=True)
