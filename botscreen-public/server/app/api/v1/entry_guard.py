"""Pre-route entry guard: authenticate + rate-limit BEFORE the route (#66).

The in-route limiter could not charge requests that fail request-body
validation — FastAPI only reaches a route AFTER validation — so a caller could
send unlimited malformed bodies for free. This ASGI middleware closes that gap
by running first:

1. resolve the credential through the SAME authority the routes use
   (``CredentialStore``), and expose the resulting principal on the request
   state so the dependency reuses this decision instead of verifying twice;
2. charge the tenant and device windows immediately (phase 1), then — once the
   body has been read — charge ONLY the session window (phase 2) for the session
   named by the PATH or by the JSON payload. Phase 2 never re-charges the first
   two scopes, so one request costs one unit per scope;
3. read the request body under a hard byte limit and a bounded timeout, and hand
   the application a REPLAYED stream, so reading it here is invisible to routes;
4. mint the request/trace ids BEFORE charging, so a rate-limit audit record and
   the response envelope can never disagree about which request was rejected.

An upload that disconnects mid-body is NOT forwarded: the partial payload is
never presented to the route as a complete request, because a write request must
not keep executing after its client is gone.

By design NOT charged: unauthenticated requests (they never yield a principal)
and the health probes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.requests import Request

from app.contracts.errors import ErrorCode

from .auth import CredentialStore, DevicePrincipal, presented_credential
from .errors import AppError, envelope_response, request_ids
from .rate_limit import SCOPE_DEVICE, SCOPE_SESSION, SCOPE_TENANT, RateLimiter

Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

#: methods whose body the guard buffers (and replays)
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_API_PREFIX = "/api/v1"


class _BodyTooLarge(Exception):
    """Raised internally when the buffered body exceeds the configured cap."""


class _ClientGone(Exception):
    """Raised internally when the client disconnects mid-body."""


class EntryGuardMiddleware:
    """Pure ASGI middleware: it must not buffer responses (SSE streams through)."""

    def __init__(
        self,
        app: Any,
        *,
        public_paths: frozenset[str],
        max_body_bytes: int,
        body_timeout_s: float,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if not body_timeout_s > 0:
            raise ValueError("body_timeout_s must be positive")
        self.app = app
        self._public_paths = public_paths
        self._max_body_bytes = max_body_bytes
        self._body_timeout_s = body_timeout_s

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith(_API_PREFIX) or path in self._public_paths:
            await self.app(scope, receive, send)
            return

        state = scope["app"].state
        credentials: CredentialStore | None = getattr(state, "credentials", None)
        limiter: RateLimiter | None = getattr(state, "rate_limiter", None)
        if credentials is None or limiter is None:
            # the application is not serving yet: fail closed, never open
            await self._reject(scope, receive, send, ErrorCode.UNAVAILABLE_MAINTENANCE)
            return

        # ids FIRST: the rate-limit audit record and the response envelope must
        # agree on which request was rejected (a malformed client id is replaced
        # here, never echoed)
        request = Request(scope, receive=receive)
        request_id, _trace_id = request_ids(request)

        credential = presented_credential(request)
        if credential is None:
            await self._reject(scope, receive, send, ErrorCode.AUTH_MISSING_CREDENTIALS)
            return
        resolved = credentials.resolve(credential)
        if resolved is None:
            await self._reject(scope, receive, send, ErrorCode.AUTH_INVALID_CREDENTIALS)
            return

        principal = DevicePrincipal(
            tenant_id=resolved.tenant_id, device_id=resolved.device_id
        )
        # PHASE 1 — tenant + device, before any body is read
        try:
            limiter.enforce(
                principal,
                request_id=request_id,
                scopes=(SCOPE_TENANT, SCOPE_DEVICE),
            )
        except AppError as exc:
            await self._reject(
                scope, receive, send, exc.code, retry_after_ms=exc.retry_after_ms
            )
            return

        scope.setdefault("state", {})["principal"] = principal
        replay = receive
        session_id = self._path_session(scope)
        if scope.get("method") in _BODY_METHODS:
            try:
                body = await self._read_bounded(receive)
            except _BodyTooLarge:
                # the attempt was already charged: it really happened
                await self._reject(
                    scope, receive, send, ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE
                )
                return
            except _ClientGone:
                # never hand a partial upload to the route as a complete request
                return
            except TimeoutError:
                await self._reject(
                    scope, receive, send, ErrorCode.UNAVAILABLE_CLIENT_TIMEOUT
                )
                return
            replay = _replay(body, receive)
            if session_id is None:
                session_id = _body_session(body)

        # PHASE 2 — only the session window, now that the session is known
        if session_id is not None:
            try:
                limiter.enforce(
                    principal,
                    session_id=session_id,
                    request_id=request_id,
                    scopes=(SCOPE_SESSION,),
                )
            except AppError as exc:
                await self._reject(
                    scope, receive, send, exc.code, retry_after_ms=exc.retry_after_ms
                )
                return
        await self.app(scope, replay, send)

    # -- helpers ----------------------------------------------------------------

    async def _read_bounded(self, receive: Receive) -> bytes:
        """Buffer the body under a byte cap and a wall-clock bound.

        Raises :class:`_BodyTooLarge` for an oversized payload,
        :class:`_ClientGone` when the client disconnects mid-body (the partial
        bytes are discarded — never forwarded as if complete) and
        ``TimeoutError`` when a slow upload exceeds the configured window.
        """
        async with asyncio.timeout(self._body_timeout_s):
            chunks: list[bytes] = []
            total = 0
            while True:
                message = await receive()
                if message.get("type") == "http.disconnect":
                    raise _ClientGone
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self._max_body_bytes:
                    raise _BodyTooLarge
                chunks.append(chunk)
                if not message.get("more_body", False):
                    return b"".join(chunks)

    @staticmethod
    def _path_session(scope: Message) -> str | None:
        """Session named by the PATH, when the path names one.

        Routing has not happened yet, so ``scope["path_params"]`` is empty and
        the path is parsed here. A run id is resolved through the in-memory
        admission lookup (no storage read); an unknown run yields ``None`` and
        the request is only tenant/device charged before it 404s.
        """
        segments = [segment for segment in scope.get("path", "").split("/") if segment]
        if len(segments) >= 4 and segments[:3] == ["api", "v1", "sessions"]:
            return segments[3]
        if len(segments) >= 5 and segments[:4] == ["api", "v1", "agent", "runs"]:
            service = getattr(scope["app"].state, "agent_service", None)
            if service is None:
                return None
            return service.session_id_for(segments[4])
        return None

    async def _reject(
        self,
        scope: Message,
        receive: Receive,
        send: Send,
        code: ErrorCode,
        *,
        retry_after_ms: int | None = None,
    ) -> None:
        response = envelope_response(
            Request(scope, receive=receive), code, retry_after_ms=retry_after_ms
        )
        await response(scope, receive, send)


def _replay(body: bytes, receive: Receive) -> Receive:
    """A receive that yields the buffered body once, then the real stream."""
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _body_session(body: bytes) -> str | None:
    """Session named by a JSON payload, when it is cheaply identifiable.

    A minimal, defensive peek: a payload that is not a JSON object, has no
    string ``session_id``, or carries an absurdly long one simply yields
    ``None`` (the request is then only tenant/device charged). The route still
    performs full validation — this only decides the session rate-limit key.

    There is deliberately NO size shortcut here: the body is already bounded by
    ``max_request_body_bytes``, and skipping big-but-legal payloads would let a
    caller pad a request past a threshold to escape the session window.
    """
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    if len(session_id) > 128 or session_id != session_id.strip():
        return None
    return session_id


__all__ = ["EntryGuardMiddleware"]
