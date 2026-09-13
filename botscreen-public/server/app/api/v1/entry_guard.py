"""Pre-route entry guard: authenticate + rate-limit BEFORE the route (#66).

The in-route limiter could not charge requests that fail request-body
validation — FastAPI only reaches a route AFTER validation — so a caller could
send unlimited malformed bodies for free. This ASGI middleware closes that gap
by running first:

1. resolve the credential through the SAME authority the routes use
   (``CredentialStore``), and expose the resulting principal on the request
   state so the dependency reuses this decision instead of verifying twice;
2. charge the tenant/device windows, plus the session window when the PATH
   determines one, before any work happens (no storage access, no stream, no
   route execution);
3. read the request body under a hard byte limit and hand the application a
   REPLAYED stream, so reading it here is invisible to the routes.

By design NOT charged: unauthenticated requests (they never yield a principal)
and the health probes. Session scope stays PATH-derived only — a session id that
arrives inside a JSON payload is not charged (documented limitation, #66).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.requests import Request

from app.contracts.errors import ErrorCode

from .auth import CredentialStore, DevicePrincipal, presented_credential
from .errors import AppError, envelope_response
from .rate_limit import RateLimiter

Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

#: methods whose body the guard buffers (and replays)
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_API_PREFIX = "/api/v1"


class _BodyTooLarge(Exception):
    """Raised internally when the buffered body exceeds the configured cap."""


class EntryGuardMiddleware:
    """Pure ASGI middleware: it must not buffer responses (SSE streams through)."""

    def __init__(
        self,
        app: Any,
        *,
        public_paths: frozenset[str],
        max_body_bytes: int,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self._public_paths = public_paths
        self._max_body_bytes = max_body_bytes

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

        request = Request(scope, receive=receive)
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
        try:
            limiter.enforce(
                principal,
                session_id=self._path_session(scope),
                request_id=_request_id(scope),
            )
        except AppError as exc:
            await self._reject(
                scope, receive, send, exc.code, retry_after_ms=exc.retry_after_ms
            )
            return

        scope.setdefault("state", {})["principal"] = principal
        replay = receive
        if scope.get("method") in _BODY_METHODS:
            try:
                body = await self._read_bounded(receive)
            except _BodyTooLarge:
                # the attempt was already charged: it really happened
                await self._reject(
                    scope, receive, send, ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE
                )
                return
            replay = _replay(body, receive)
        await self.app(scope, replay, send)

    # -- helpers ----------------------------------------------------------------

    async def _read_bounded(self, receive: Receive) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                break  # the client left: hand the route what we have
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self._max_body_bytes:
                raise _BodyTooLarge
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
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


def _request_id(scope: Message) -> str:
    state = scope.get("state") or {}
    return str(state.get("request_id") or "")


__all__ = ["EntryGuardMiddleware"]
