"""Per-scope request rate limiting (#66 remainder — minimal slice).

Three independent fixed windows, all keyed from the AUTHENTICATED principal:

- ``tenant``  — every request from a tenant;
- ``device``  — every request from one device;
- ``session`` — requests that name one session (explicitly, or through a run).

Counting happens after authentication and before any work, so a throttled
caller cannot create runs, open streams or touch storage. Within one request,
every configured scope is charged even when a narrower scope rejects it: a
tenant or device budget limits attempts, so a hammered session cannot hide
behind its own window.

Enforcement happens in the PRE-ROUTE entry guard (``app.api.v1.entry_guard``),
in two phases:
1. right after authentication — the tenant and device windows are charged, so a
   request that later fails body validation has already been paid for;
2. after the body has been read (bounded) — ONLY the session window is charged,
   because the session may only become identifiable once the payload is parsed
   (``{"session_id": …}``) or the path names one. Phase 2 never re-charges the
   tenant/device windows, so one request costs exactly one unit per scope.

Keys are structured tuples ``(scope, tenant_id, device_id, session_id)`` — never
delimiter-joined strings — so two different identities can never collide, and a
session window belongs to the AUTHENTICATED principal: a foreign caller who
merely knows a session id charges its own window, not the victim's. Exceeding a window
raises ``E_RATE_LIMIT_EXCEEDED`` (429) carrying ``retry_after_ms`` — the only
error whose envelope carries a wait hint — and emits one structured
:class:`~app.contracts.audit.AuditRecord`.

Deliberately NOT here: distributed/shared counters (a multi-worker deployment
needs the persistent store that #65 gates), credential issuance/rotation, and a
durable audit sink (records go to the ``gcmw.audit`` logger today).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

from app.contracts.audit import AuditRecord
from app.contracts.errors import ErrorCode

from .auth import DevicePrincipal
from .errors import AppError

#: scopes are independent: exhausting one never charges another
SCOPE_TENANT = "tenant"
SCOPE_DEVICE = "device"
SCOPE_SESSION = "session"

#: how many distinct keys are tracked before expired windows are pruned; a cap
#: (not a policy) that keeps a long-running robot's memory bounded
DEFAULT_MAX_KEYS = 10_000

AuditSink = Callable[[AuditRecord], None]


@dataclass(frozen=True)
class RateLimitRule:
    """One fixed window: ``limit`` requests per ``window_s`` seconds."""

    scope: str
    limit: int
    window_s: float

    def __post_init__(self) -> None:
        if self.scope not in {SCOPE_TENANT, SCOPE_DEVICE, SCOPE_SESSION}:
            raise ValueError(f"unknown rate-limit scope {self.scope!r}")
        if self.limit <= 0:
            raise ValueError("rate-limit limit must be positive")
        if not self.window_s > 0:
            raise ValueError("rate-limit window must be positive")


#: (scope, tenant_id, device_id, session_id) — never a joined string
RateKey = tuple[str, str, str, str]


class _KeyTableFull(RuntimeError):
    """Internal signal: the key table is at its hard cap (see ``_charge``)."""


@dataclass
class _Window:
    started_at: float
    count: int = 0


class RateLimiter:
    """Fixed-window counters per ``(scope, key)`` (process-local)."""

    def __init__(
        self,
        rules: tuple[RateLimitRule, ...],
        *,
        clock: Callable[[], float] = time.monotonic,
        audit: AuditSink | None = None,
        max_keys: int = DEFAULT_MAX_KEYS,
    ) -> None:
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        # Pruning reclaims keys by age against ONE horizon; with mixed window
        # lengths an expired short window could survive next to a long one and
        # turn a full table into a spurious 503. Mixing is therefore refused
        # until the pruner tracks per-window expiries.
        windows = {rule.window_s for rule in rules}
        if len(windows) > 1:
            raise ValueError(
                "all rate-limit rules must share one window length "
                f"(got {sorted(windows)})"
            )
        self._rules = rules
        self._clock = clock
        self._audit = audit
        self._max_keys = max_keys
        self._windows: dict[RateKey, _Window] = {}

    # -- enforcement -----------------------------------------------------------

    def enforce(
        self,
        principal: DevicePrincipal,
        *,
        session_id: str | None = None,
        request_id: str = "",
        scopes: Collection[str] | None = None,
    ) -> None:
        """Charge the configured scopes; raise on the first exhausted window.

        ``scopes`` narrows the charge to specific scopes — the entry guard uses
        it for its two phases (tenant/device after authentication, session once
        the session id is known) so a single request is never charged twice for
        the same scope.
        """
        selected = (
            self._rules
            if scopes is None
            else tuple(rule for rule in self._rules if rule.scope in scopes)
        )
        now = self._clock()
        for rule in selected:
            key = self._key_for(rule.scope, principal, session_id)
            if key is None:
                continue
            try:
                retry_after_ms = self._charge(rule, key, now)
            except _KeyTableFull:
                # an overloaded limiter must not silently stop limiting, and it
                # must never evict a live counter to make room
                self._record_audit(
                    principal,
                    rule,
                    session_id,
                    request_id,
                    result="overloaded",
                    code=ErrorCode.UNAVAILABLE_OVERLOADED,
                )
                raise AppError(ErrorCode.UNAVAILABLE_OVERLOADED) from None
            if retry_after_ms is not None:
                self._record_audit(principal, rule, session_id, request_id)
                raise AppError(
                    ErrorCode.RATE_LIMIT_EXCEEDED, retry_after_ms=retry_after_ms
                )

    def _key_for(
        self, scope: str, principal: DevicePrincipal, session_id: str | None
    ) -> RateKey | None:
        """Structured key: ``(scope, tenant, device, session)``.

        Components are kept as separate tuple fields, so identifiers containing
        separators cannot collide — and the SESSION window is scoped to the
        authenticated tenant+device, so a foreign caller that guesses a session
        id only ever charges its own budget.
        """
        if scope == SCOPE_TENANT:
            return (SCOPE_TENANT, principal.tenant_id, "", "")
        if scope == SCOPE_DEVICE:
            return (SCOPE_DEVICE, principal.tenant_id, principal.device_id, "")
        if not session_id:
            return None  # a request naming no session is not session-charged
        return (SCOPE_SESSION, principal.tenant_id, principal.device_id, session_id)

    def _charge(self, rule: RateLimitRule, key: RateKey, now: float) -> int | None:
        """Return ``retry_after_ms`` when the window is exhausted, else ``None``.

        Raises :class:`_KeyTableFull` when a NEW key would exceed ``max_keys``
        after pruning: the cap is hard, so the counter table cannot grow without
        bound and no live window is evicted to make room.
        """
        entry_key = key  # the structured key already carries the scope
        window = self._windows.get(entry_key)
        if window is None or now - window.started_at >= rule.window_s:
            self._prune(now)
            if entry_key not in self._windows and len(self._windows) >= self._max_keys:
                raise _KeyTableFull
            window = _Window(started_at=now)
            self._windows[entry_key] = window
        window.count += 1
        if window.count <= rule.limit:
            return None
        remaining = window.started_at + rule.window_s - now
        return max(1, int(remaining * 1000) + 1)

    def _prune(self, now: float) -> None:
        """Drop expired windows once the table grows past its cap."""
        if len(self._windows) < self._max_keys:
            return
        horizon = max(rule.window_s for rule in self._rules)
        self._windows = {
            key: window
            for key, window in self._windows.items()
            if now - window.started_at < horizon
        }

    # -- inspection / audit ------------------------------------------------------

    def tracked_keys(self) -> int:
        return len(self._windows)

    def _record_audit(
        self,
        principal: DevicePrincipal,
        rule: RateLimitRule,
        session_id: str | None,
        request_id: str,
        *,
        result: str = "throttled",
        code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
    ) -> None:
        if self._audit is None:
            return
        self._audit(
            AuditRecord(
                tenant_id=principal.tenant_id,
                actor_type="device",
                actor_id_hash=_digest(principal.device_id),
                session_id_hash=_digest(session_id) if session_id else "",
                request_id=request_id or "unknown",
                action=f"rate_limit.{rule.scope}",
                result=result,
                error_code=code,
            )
        )


def _digest(value: str) -> str:
    """Stable, non-reversible identifier for audit records."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rules_from_settings(settings: object) -> tuple[RateLimitRule, ...]:
    """Build the configured windows (``0`` disables one scope)."""
    rules: list[RateLimitRule] = []
    for scope, attr in (
        (SCOPE_TENANT, "rate_limit_tenant_per_minute"),
        (SCOPE_DEVICE, "rate_limit_device_per_minute"),
        (SCOPE_SESSION, "rate_limit_session_per_minute"),
    ):
        per_minute = int(getattr(settings, attr, 0) or 0)
        if per_minute > 0:
            rules.append(RateLimitRule(scope=scope, limit=per_minute, window_s=60.0))
    return tuple(rules)


__all__ = [
    "DEFAULT_MAX_KEYS",
    "SCOPE_DEVICE",
    "SCOPE_SESSION",
    "SCOPE_TENANT",
    "RateKey",
    "RateLimitRule",
    "RateLimiter",
    "rules_from_settings",
]
