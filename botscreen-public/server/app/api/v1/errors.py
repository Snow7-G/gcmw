"""API-layer error type shared by routes, auth and app assembly."""

from __future__ import annotations

import re
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from app.contracts.errors import ErrorCode, ErrorEnvelope, http_status_for


class AppError(RuntimeError):
    """Application-level failure with a stable ErrorCode (#35)."""

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        retry_after_ms: int | None = None,
    ) -> None:
        self.code = code
        #: how long the caller should wait before retrying (rate limiting: #66)
        self.retry_after_ms = retry_after_ms
        super().__init__(message or code.value)


X_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def request_ids(request: Request) -> tuple[str, str]:
    """Mint request/trace ids exactly once per request (single authority).

    A client-supplied ``X-Request-ID`` is honoured only when it matches the
    restricted format; anything else is replaced and never echoed back raw.
    Error envelopes and mid-stream SSE error frames both use these ids, so a
    failure that happens after the response has started stays correlatable
    with the request that opened it.
    """
    existing = getattr(request.state, "request_id", None)
    if existing is not None:
        return existing, request.state.trace_id
    header = request.headers.get("x-request-id") or ""
    request_id = header if X_REQUEST_ID_RE.match(header) else uuid.uuid4().hex
    trace_id = uuid.uuid4().hex
    request.state.request_id = request_id
    request.state.trace_id = trace_id
    return request_id, trace_id


def error_responses(*codes: ErrorCode) -> dict[int, dict]:
    """OpenAPI ``responses`` entries for the EXACT ErrorCodes a route returns.

    The published contract must match the wire: every failure of the public API
    is the same :class:`~app.contracts.errors.ErrorEnvelope` (stable code + the
    registry message + request/trace ids), and the status comes from the same
    registry the runtime uses — never a framework default.
    """
    grouped: dict[int, list[str]] = {}
    for code in codes:
        grouped.setdefault(http_status_for(code), []).append(code.value)
    return {
        status: {
            "model": ErrorEnvelope,
            "description": "统一错误信封（" + " / ".join(sorted(values)) + "）",
        }
        for status, values in sorted(grouped.items())
    }


def envelope_response(
    request: Request,
    code: ErrorCode,
    status_override: int | None = None,
    retry_after_ms: int | None = None,
) -> JSONResponse:
    """The ONE error response shape of this API.

    Used by the exception handlers, the entry guard (which answers before any
    route runs) and anything else that must fail with a stable code — the
    message always comes from the error registry, never from the caller.
    """
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
