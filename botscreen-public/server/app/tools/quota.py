"""Run-scoped tool quota, enforced INSIDE the ToolGateway (#55A-B).

Why this exists: an integer ``grant`` handed to a sub-agent is advisory — a
rogue or buggy runner can ignore it, call the gateway any number of times, and
report ``tool_calls=0``. The only place a tool call can be made impossible is
the gateway's own submission boundary, so the quota lives there:

* the Manager creates ONE :class:`RunToolQuota` per ``execute()`` — first
  execution and every revision share the SAME state (counts accumulate across
  revisions, never reset);
* the quota object travels in a :class:`~contextvars.ContextVar`, so it needs
  no global run-id registry;
* ``ToolGateway._prepare`` calls :meth:`RunToolQuota.acquire` as its LAST gate
  — atomically, BEFORE ``_submit()`` — so an exhausted quota means zero
  executions, and the refusal is audited like any other gate denial;
* a unit is consumed when the attempt STARTS: a timeout, a tool fault or a
  cancellation afterwards never refunds it.

LIFETIME — explicit close, not context teardown: ``ContextVar.reset()`` only
unbinds the CURRENT task. ``asyncio.create_task()`` copies the context, so a
background task spawned by a runner would keep holding a live quota object and
could make LATE tool calls after the Manager returned, timed out or was
cancelled. The Manager therefore calls :meth:`RunToolQuota.close` FIRST in its
``finally``: close is atomic under the same lock as acquire, so every context
holding the object — current or copied — sees the quota as closed and every
late acquire is refused (audited as ``rejected:tool_quota_closed``). Calls
that acquired BEFORE the close are allowed to finish; nothing is refunded and
the quota never reopens.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar

#: outcomes of :meth:`RunToolQuota.acquire`
GRANTED = "granted"
EXHAUSTED = "exhausted"
CLOSED = "closed"


class RunToolQuota:
    """One run's tool-call budget, Manager-owned and gateway-enforced."""

    def __init__(self, limit: int) -> None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("tool quota limit must be >= 0")
        self._lock = threading.Lock()
        self._remaining = limit
        self._consumed = 0
        self._refused_exhausted = 0
        self._refused_closed = 0
        self._closed = False

    def acquire(self) -> str:
        """Atomically reserve ONE tool call; returns ``GRANTED``,
        ``EXHAUSTED`` or ``CLOSED``.

        Synchronous on purpose: check-and-decrement happen without an await,
        and the lock also covers pool-thread callers. Any non-granted outcome
        means the gateway must refuse BEFORE submission, with zero side
        effects. A close always wins over a later acquire — the two are
        serialized by the same lock, so after :meth:`close` returns, no
        context holding this object can ever acquire again."""
        with self._lock:
            if self._closed:
                self._refused_closed += 1
                return CLOSED
            if self._remaining <= 0:
                self._refused_exhausted += 1
                return EXHAUSTED
            self._remaining -= 1
            self._consumed += 1
            return GRANTED

    def try_acquire(self) -> bool:
        """Convenience wrapper: ``True`` iff :meth:`acquire` granted."""
        return self.acquire() == GRANTED

    def close(self) -> None:
        """Atomically close the quota: every later acquire — in ANY context
        that copied this object — is refused as ``CLOSED``. Irreversible: no
        refund, no reopen. Calls that already acquired may finish."""
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def consumed(self) -> int:
        """The AUTHORITATIVE number of tool calls started this run."""
        with self._lock:
            return self._consumed

    @property
    def refused_exhausted(self) -> int:
        """Calls refused because the budget ran out."""
        with self._lock:
            return self._refused_exhausted

    @property
    def refused_closed(self) -> int:
        """LATE calls refused because the run had already ended."""
        with self._lock:
            return self._refused_closed


_quota_var: ContextVar[RunToolQuota | None] = ContextVar(
    "gcmw_run_tool_quota", default=None
)


def set_run_quota(quota: RunToolQuota) -> object:
    """Bind the quota to the current task context; returns a reset token."""
    return _quota_var.set(quota)


def reset_run_quota(token: object) -> None:
    """Unbind the CURRENT task only — NOT enough to stop copied contexts.
    Always pair with :meth:`RunToolQuota.close` (called first)."""
    _quota_var.reset(token)  # type: ignore[arg-type]


def current_run_quota() -> RunToolQuota | None:
    """The quota governing THIS context, or ``None`` (legacy/unrestricted)."""
    return _quota_var.get()
