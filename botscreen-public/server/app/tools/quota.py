"""Run-scoped tool quota, enforced INSIDE the ToolGateway (#55A-B).

Why this exists: an integer ``grant`` handed to a sub-agent is advisory — a
rogue or buggy runner can ignore it, call the gateway any number of times, and
report ``tool_calls=0``. The only place a tool call can be made impossible is
the gateway's own submission boundary, so the quota lives there:

* the Manager creates ONE :class:`RunToolQuota` per ``execute()`` — first
  execution and every revision share the SAME state (counts accumulate across
  revisions, never reset);
* the quota object travels in a :class:`~contextvars.ContextVar`, so it needs
  no global run-id registry and is released when the Manager's task context
  ends;
* ``ToolGateway._prepare`` calls :meth:`RunToolQuota.try_acquire` as its LAST
  gate — atomically, BEFORE ``_submit`` — so an exhausted quota means zero
  executions, and the refusal is audited like any other gate denial;
* a unit is consumed when the attempt STARTS: a timeout, a tool fault or a
  cancellation afterwards never refunds it.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar


class RunToolQuota:
    """One run's tool-call budget, Manager-owned and gateway-enforced."""

    def __init__(self, limit: int) -> None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("tool quota limit must be >= 0")
        self._lock = threading.Lock()
        self._remaining = limit
        self._consumed = 0
        self._refused = 0

    def try_acquire(self) -> bool:
        """Atomically reserve ONE tool call.

        Synchronous on purpose: check-and-decrement happen without an await,
        and the lock also covers pool-thread callers. ``False`` means the
        budget is exhausted — the gateway must refuse BEFORE submission, with
        zero side effects."""
        with self._lock:
            if self._remaining <= 0:
                self._refused += 1
                return False
            self._remaining -= 1
            self._consumed += 1
            return True

    @property
    def consumed(self) -> int:
        """The AUTHORITATIVE number of tool calls started this run."""
        with self._lock:
            return self._consumed

    @property
    def refused(self) -> int:
        """How many calls the boundary refused (audit/ops probe)."""
        with self._lock:
            return self._refused


_quota_var: ContextVar[RunToolQuota | None] = ContextVar(
    "gcmw_run_tool_quota", default=None
)


def set_run_quota(quota: RunToolQuota) -> object:
    """Bind the quota to the current task context; returns a reset token."""
    return _quota_var.set(quota)


def reset_run_quota(token: object) -> None:
    """Release the binding (the Manager's ``finally``)."""
    _quota_var.reset(token)  # type: ignore[arg-type]


def current_run_quota() -> RunToolQuota | None:
    """The quota governing THIS context, or ``None`` (legacy/unrestricted)."""
    return _quota_var.get()
