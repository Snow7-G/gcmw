"""ToolGateway: the only tool entry point for every Agent (issue #57).

Hard constraints (V2.3 §6.2):
- Agents never touch files/DB/shell/network/other agents directly — every
  access path is a whitelisted read-only tool behind this gateway;
- only tools declared in the sealed read-only whitelist (``specs.py``) can be
  registered; write tools cannot even be constructed;
- calls are authorized against the caller's declared capability set
  (``AgentManifest.allowed_tools``), then parameters are Schema-validated,
  then the executor runs under a timeout; the result is Schema-validated and
  size-capped before it is returned;
- arguments and results never reach logs or audit text verbatim — audit
  records carry identifiers, tool names, outcome markers and sizes only;
- every gate failure raises :class:`ToolGatewayError` with a stable ErrorCode
  (mapped to envelopes by the #36 boundary) and writes an audit record;
- IDENTITY IS SERVER-INJECTED: ``invoke``/``ainvoke`` take a trusted
  :class:`~app.contracts.common.TenantContext` as their first argument (the
  same authority the knowledge store uses) and derive tenant/session/run from
  it. Tool ARGUMENTS may never carry an identity — such keys are rejected
  structurally before any executor can run (``IDENTITY_ARGUMENT_KEYS``), so a
  model can neither widen nor redirect its own scope;
- ``ainvoke`` runs the gate + executor OFF the event loop with a hard
  ``asyncio.wait_for`` bound, so a slow tool cannot stall the server.

Executor side effects are strictly read-only by construction: specs carry no
write capability. On timeout the call returns TOOL_TIMEOUT immediately and the
timeout is FINAL for that call: audit is written once, and a late completion of
abandoned work can never append a second, contradictory "success" record.

Execution is bounded: every call runs on a fixed-size worker pool
(``max_concurrency``), so slow tools cannot spawn unbounded work. A timeout
stops WAITING, not the worker — Python threads cannot be cancelled, and this
module never claims otherwise: the abandoned worker finishes on its own, its
result is discarded, and its audit slot is already closed.

Audit policy: a gateway without an ``audit_sink`` still executes calls but runs
in a DEGRADED, UNAUDITED mode and logs a warning at construction. Deployment
requires a sink; a durable audit store remains a production gate (#65) and is
not claimed here.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.contracts.agent import ToolRequest, ToolResult
from app.contracts.audit import AuditRecord
from app.contracts.common import TenantContext
from app.contracts.errors import ErrorCode, lookup
from app.tools.quota import EXHAUSTED, GRANTED, current_run_quota
from app.tools.specs import READONLY_TOOL_NAMES, ToolSpec
from app.tools.validation import validate

_LOGGER = logging.getLogger(__name__)

#: argument keys a model must never supply: identity belongs to the injected
#: context. Checked RECURSIVELY (nested objects are exactly how a smuggled
#: identity would try to ride through), and only the key NAME is ever reported.
IDENTITY_ARGUMENT_KEYS: frozenset[str] = frozenset(
    {
        "tenant",
        "tenant_id",
        "device",
        "device_id",
        "session",
        "session_id",
        "run",
        "run_id",
        "request_id",
        "principal",
        "actor",
        "user",
        "user_id",
        "reviewer",
        "reviewer_id",
        "tenant_context",
    }
)


#: The complete set of violation types ``app.tools.validation`` can emit. A
#: violation string is ``"<path>: <kind>"``, and the PATH may contain a
#: model-supplied key (even one containing ``": "``), so the kind is taken from
#: the END of the string and MUST be one of these — anything else is reported as
#: ``other`` rather than echoed.
SCHEMA_VIOLATION_KINDS: frozenset[str] = frozenset(
    {
        "schema-false",
        "invalid-schema",
        "type",
        "enum",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "required",
        "additionalProperties",
        "minItems",
        "maxItems",
    }
)


def _violation_kinds(violations: list[str]) -> str:
    """Fixed classification of schema violations, WITHOUT path or value.

    Operators keep a useful signal (``required``/``additionalProperties`` …);
    no untrusted text — not even a fragment of a model-supplied key — reaches a
    durable audit record.
    """
    kinds = set()
    for item in violations:
        candidate = item.rsplit(": ", 1)[-1]  # take the type from the END
        kinds.add(candidate if candidate in SCHEMA_VIOLATION_KINDS else "other")
    return "+".join(sorted(kinds)) if kinds else "unspecified"


def _validate_timeout_s(timeout_s: float) -> float:
    """A caller-supplied budget must be a positive finite number."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError(f"timeout_s must be a positive finite number: {timeout_s!r}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive finite number: {timeout_s!r}")
    return float(timeout_s)


def _audit_tool_label(tool_name: str) -> str:
    """Fixed label for anything outside the sealed whitelist.

    A model-controlled tool name is attacker-supplied text: it is never written
    to the audit trail verbatim. Unknown names collapse to ``unregistered`` plus
    a short digest, which keeps records correlatable without echoing payload.
    """
    if isinstance(tool_name, str) and tool_name in READONLY_TOOL_NAMES:
        return tool_name
    digest = hashlib.sha256(str(tool_name).encode("utf-8")).hexdigest()[:12]
    return f"unregistered:{digest}"


def find_identity_argument(node: Any) -> str | None:
    """Return the first identity key found at ANY depth (values never echoed)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in IDENTITY_ARGUMENT_KEYS:
                return key
            found = find_identity_argument(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            found = find_identity_argument(item)
            if found is not None:
                return found
    return None


@dataclass
class _Prepared:
    """One call's post-gate state (what the pool worker needs)."""

    spec: ToolSpec
    tool_name: str
    arguments: dict[str, Any]
    timeout_s: float
    audit: Callable[..., None]
    started: float
    size_bytes: int = 0


class _RunnerFuture:
    """Adapter giving an injected runner the ``result``/``cancel`` contract."""

    def __init__(self, call: Callable[[], Any]) -> None:
        self._call = call

    def result(self, timeout: float | None = None) -> Any:
        return self._call()

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def done(self) -> bool:
        return False

    def add_done_callback(self, callback: Callable[[Any], None]) -> None:
        callback(self)


class _ToolFault(Exception):
    """Internal fault marker: a FIXED audit label plus a stable ErrorCode.

    It deliberately stores NO message, NO schema path and NO executor text: it
    lives only long enough for the awaiting path to decide, and is dropped
    before the public error is raised, so nothing it saw can be reached later
    through ``__context__`` or any attribute.
    """

    def __init__(self, label: str, code: ErrorCode) -> None:
        super().__init__(label)
        self.label = label
        self.code = code


@dataclass
class _CallState:
    """One call's audit slot: closed exactly once, late writes are dropped."""

    closed: bool = False


@dataclass(frozen=True)
class TrustedIdentity:
    """Identity derived ONLY from the server-injected context."""

    tenant_id: str
    session_id_hash: str = ""
    run_id: str = ""
    request_id: str = ""


def identity_from_context(context: Any) -> TrustedIdentity:
    """Extract the identity the server vouches for (duck-typed on purpose)."""

    def _field(name: str) -> str:
        value = getattr(context, name, "")
        return value if isinstance(value, str) else ""

    session_id = _field("session_id")
    return TrustedIdentity(
        tenant_id=_field("tenant_id"),
        session_id_hash=(
            hashlib.sha256(session_id.encode("utf-8")).hexdigest() if session_id else ""
        ),
        run_id=_field("run_id"),
        request_id=_field("request_id"),
    )


class ToolGatewayError(RuntimeError):
    """Tool failure carrying a stable ErrorCode (mapped to envelopes by #36).

    The text presented to callers is ALWAYS the registry message: this is the
    single construction boundary that enforces it, so no ``raise`` site — the
    whitelist gate, the permission gate, the schema gates or an executor — can
    leak a tool name, an argument/result key or payload text into a string that
    a future Agent or log handler might record.

    Detailed, VALUE-FREE reason markers live in the audit record's ``result``
    field (``denied:not_whitelisted``, ``rejected:input_schema:required`` …) and
    nowhere else: the constructor accepts a ``message`` purely so raise sites
    read well, and deliberately does NOT retain it — a raw path could embed a
    model-supplied key and would then be readable through the exception object.
    """

    def __init__(self, code: ErrorCode, message: str = "") -> None:
        del message  # accepted for readability at raise sites; never stored
        self.code = code
        super().__init__(lookup(code).message)


def _default_runner(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` on a daemon thread with a wall-clock timeout.

    On expiry ``TimeoutError`` is raised
    and the runaway thread is left to finish on its own (daemon): Python cannot
    cancel a thread, and nothing here pretends it can.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            box["error"] = exc

    thread = threading.Thread(target=target, name="tool-executor", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"tool exceeded {timeout_seconds:g}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class ToolGateway:
    """Sealed whitelist + permission/schema/timeout/size enforcement + audit."""

    def __init__(
        self,
        audit_sink: Callable[[AuditRecord], None] | None = None,
        clock: Callable[[], Any] | None = None,
        runner: Callable[[Callable[[], Any], float], Any] | None = None,
        *,
        default_timeout_ms: int = 5_000,
        default_max_result_bytes: int = 64 * 1024,
        max_concurrency: int = 4,
        executor: concurrent.futures.ThreadPoolExecutor | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._audit_sink = audit_sink
        if audit_sink is None:
            _LOGGER.warning(
                "tool gateway constructed WITHOUT an audit sink: calls run "
                "unaudited (degraded mode; deployment requires a sink)"
            )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # bounded execution: slow tools share one fixed-size pool instead of
        # spawning a thread per call
        self._pool = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="tool-executor"
        )
        self._owns_pool = executor is None
        # an INJECTED runner (tests, sync-only) replaces the pool for `invoke`;
        # the async path always uses the bounded pool so concurrency stays capped
        self._runner = runner
        self._default_timeout_ms = default_timeout_ms
        self._default_max_result_bytes = default_max_result_bytes
        self._lock = threading.RLock()
        self._in_flight = 0
        self._peak_in_flight = 0
        self._specs: dict[str, ToolSpec] = {}
        self._order: list[str] = []
        self._disabled: set[str] = set()

    # -- registration ---------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register a whitelisted tool. Duplicate names are rejected; the
        whitelist itself is sealed (see ``ToolSpec``)."""
        if spec.executor is None:
            raise ValueError(f"tool {spec.name!r} needs an executor")
        with self._lock:
            if spec.name in self._specs:
                raise ToolGatewayError(
                    ErrorCode.CONFLICT_IDEMPOTENCY,
                    f"tool {spec.name!r} already registered",
                )
            self._specs[spec.name] = copy.deepcopy(spec)
            self._order.append(spec.name)

    def tools(self) -> list[str]:
        """Registered whitelist tool names in registration order."""
        with self._lock:
            return list(self._order)

    def spec(self, name: str) -> ToolSpec | None:
        with self._lock:
            spec = self._specs.get(name)
            return copy.deepcopy(spec) if spec is not None else None

    def is_enabled(self, name: str) -> bool:
        with self._lock:
            return name in self._specs and name not in self._disabled

    def enable(self, name: str) -> None:
        with self._lock:
            self._require_registered(name)
            self._disabled.discard(name)

    def disable(self, name: str) -> None:
        """Runtime-disable a whitelisted tool (maintenance path); calls are
        rejected with TOOL_DISABLED while disabled."""
        with self._lock:
            self._require_registered(name)
            self._disabled.add(name)

    def _require_registered(self, name: str) -> None:
        if name not in self._specs:
            raise ValueError(f"tool {name!r} is not registered")

    # -- invocation ------------------------------------------------------------

    def _prepare(
        self,
        context: TenantContext | Any,
        request: ToolRequest,
        *,
        allowed_tools: Sequence[str] | None,
        agent_id: str,
        state: _CallState,
    ) -> _Prepared:
        """Every gate that needs no worker: identity, whitelist, permission,
        input schema, deadline. Cheap, CPU-only, and it NEVER touches the pool —
        that is what keeps ``max_concurrency=1`` from deadlocking the async path.
        """
        identity = identity_from_context(context)
        name = request.tool_name
        with self._lock:
            spec = self._specs.get(name)
            disabled = name in self._disabled
        audit = self._audit(
            tool_name=name,
            agent_id=agent_id,
            tenant_id=identity.tenant_id,
            session_id_hash=identity.session_id_hash,
            run_id=identity.run_id,
            request_id=identity.request_id,
            state=state,
        )

        # 0. identity gate: the model may never supply tenant/session/reviewer.
        smuggled = find_identity_argument(request.arguments)
        if smuggled is not None:
            audit(
                result=f"denied:identity_argument={smuggled}",
                code=ErrorCode.TOOL_SCHEMA_REJECTED,
            )
            raise ToolGatewayError(
                ErrorCode.TOOL_SCHEMA_REJECTED,
                f"tool {name!r} may not receive an identity argument ({smuggled!r})",
            )

        # 1. whitelist gate: unknown/disabled tools never reach an executor.
        if spec is None or disabled:
            audit(result="denied:not_whitelisted", code=ErrorCode.TOOL_DISABLED)
            raise ToolGatewayError(
                ErrorCode.TOOL_DISABLED, f"tool {name!r} is not enabled"
            )

        # 2. permission gate: the caller's declared capability set decides.
        declared = (
            {allowed_tools}
            if isinstance(allowed_tools, str)
            else set(allowed_tools or ())
        )
        if name not in declared:
            audit(result="denied:not_allowed", code=ErrorCode.AUTHZ_FORBIDDEN)
            raise ToolGatewayError(
                ErrorCode.AUTHZ_FORBIDDEN,
                f"agent {agent_id!r} is not allowed to call {name!r}",
            )

        # 3. input schema gate (no executor side effects before this point).
        violations = validate(spec.input_schema, request.arguments)
        if violations:
            # the audit carries the DETAILED, value-free reason (schema paths and
            # expected types only), while the exception text stays registry-only
            audit(
                result=f"rejected:input_schema:{_violation_kinds(violations)}",
                code=ErrorCode.TOOL_SCHEMA_REJECTED,
            )
            raise ToolGatewayError(
                ErrorCode.TOOL_SCHEMA_REJECTED,
                f"input schema violations at {', '.join(violations)}",
            )

        # 4. deadline + per-tool timeout.
        timeout_ms = self._effective_timeout_ms(request)
        if timeout_ms <= 0:
            audit(result="rejected:deadline_expired", code=ErrorCode.TOOL_TIMEOUT)
            raise ToolGatewayError(ErrorCode.TOOL_TIMEOUT, "deadline already expired")

        # 5. #55A-B RUN QUOTA — the HARD, call-time enforcement point. The quota
        # object is Manager-owned and travels in a ContextVar; the unit is taken
        # atomically BEFORE _submit, so an exhausted budget makes the execution
        # impossible (zero side effects), and the refusal is audited exactly
        # like every other gate denial. A started attempt (timeout / tool fault
        # / cancellation) is never refunded.
        quota = current_run_quota()
        if quota is not None:
            outcome = quota.acquire()
            if outcome != GRANTED:
                # DISTINGUISHABLE refusals: the budget ran out vs the run has
                # already ended (a LATE call from a context that copied the
                # quota). Same public error code, different audit markers.
                result = (
                    "rejected:tool_quota_exhausted"
                    if outcome == EXHAUSTED
                    else "rejected:tool_quota_closed"
                )
                message = (
                    "run tool quota is exhausted"
                    if outcome == EXHAUSTED
                    else "run has ended: the tool quota is closed"
                )
                audit(result=result, code=ErrorCode.TOOL_OVER_LIMIT)
                raise ToolGatewayError(ErrorCode.TOOL_OVER_LIMIT, message)

        return _Prepared(
            spec=spec,
            tool_name=name,
            arguments=request.arguments,
            timeout_s=timeout_ms / 1000,
            audit=audit,
            started=time.perf_counter(),
        )

    def _run_tool(self, prepared: _Prepared, context: Any) -> Any:
        """Executor + output gates. Runs ON a pool worker; audits nothing.

        It reports failures by raising :class:`_ToolFault` (label + code), so the
        DECIDING side — whichever coroutine/thread awaited this call — writes the
        one and only terminal audit record.
        """
        try:
            payload = prepared.spec.executor(context, prepared.arguments)
        except ToolGatewayError as exc:
            # NEVER trust executor-provided text — not even from a builtin: the
            # boundary keeps the CODE and the registry's audited message only, so
            # an error embedding model input can never reach a caller's log
            raise _ToolFault("executor:" + exc.code.value, exc.code) from None
        except Exception:  # noqa: BLE001 - executor internals NEVER leak
            # the exception text may carry provider payloads/paths/model output:
            # only the code and a fixed sentence reach the caller
            raise _ToolFault("executor:internal", ErrorCode.INTERNAL_UNKNOWN) from None

        violations = validate(prepared.spec.output_schema, payload)
        if violations:
            raise _ToolFault(
                f"rejected:output_schema:{_violation_kinds(violations)}",
                ErrorCode.TOOL_SCHEMA_REJECTED,
            )
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError):
            raise _ToolFault(
                "rejected:output_json", ErrorCode.TOOL_SCHEMA_REJECTED
            ) from None
        size_bytes = len(encoded.encode("utf-8"))
        cap = prepared.spec.max_result_bytes or self._default_max_result_bytes
        if size_bytes > cap:
            raise _ToolFault(
                f"rejected:result_bytes={size_bytes}", ErrorCode.TOOL_OVER_LIMIT
            )
        prepared.size_bytes = size_bytes
        return payload

    def _submit(
        self, prepared: _Prepared, context: Any, *, allow_injected: bool
    ) -> Any:
        """The ONE execution submission per call (bounded concurrency lives here)."""

        def counted() -> Any:
            with self._lock:
                self._in_flight += 1
                self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
            try:
                return self._run_tool(prepared, context)
            finally:
                with self._lock:
                    self._in_flight -= 1

        if allow_injected and self._runner is not None:
            return _RunnerFuture(lambda: self._runner(counted, prepared.timeout_s))
        return self._pool.submit(counted)

    def _finalize_success(self, prepared: _Prepared, payload: Any) -> ToolResult:
        latency_ms = int((time.perf_counter() - prepared.started) * 1000)
        prepared.audit(
            result=f"ok:result_bytes={prepared.size_bytes}", latency_ms=latency_ms
        )
        return ToolResult(tool_name=prepared.tool_name, ok=True, data=payload)

    def _finalize_failure(self, prepared: _Prepared, fault: _ToolFault) -> None:
        prepared.audit(result=fault.label, code=fault.code)
        raise ToolGatewayError(fault.code)

    def _finalize_cancelled(self, prepared: _Prepared) -> None:
        """The CALLER cancelled: one terminal record, then cancel propagates.

        No error code is attached — a cancelled call is not a tool failure —
        and the abandoned worker still cannot append a success record, because
        only the awaiting side ever writes audit.
        """
        prepared.audit(result="cancelled", latency_ms=None)

    def invoke(
        self,
        context: TenantContext | Any,
        request: ToolRequest,
        *,
        allowed_tools: Sequence[str] | None = (),
        agent_id: str = "",
        _state: _CallState | None = None,
    ) -> ToolResult:
        """Authorized, schema-checked, timed and capped tool execution.

        ``context`` is the TRUSTED identity injected by the server (never by the
        model). Gates run on the CALLER's thread, the executor runs on the
        bounded pool exactly once, and the caller writes the single terminal
        audit record — so one call can never produce contradictory records.
        """
        prepared = self._prepare(
            context,
            request,
            allowed_tools=allowed_tools,
            agent_id=agent_id,
            state=_state if _state is not None else _CallState(),
        )
        future = self._submit(prepared, context, allow_injected=True)
        fault: _ToolFault | None = None
        payload: Any = None
        try:
            payload = future.result(timeout=prepared.timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()  # stops a QUEUED task; a running thread finishes
            fault = _ToolFault("rejected:timeout", ErrorCode.TOOL_TIMEOUT)
        except _ToolFault as exc:
            fault = exc
        if fault is not None:  # raised OUTSIDE the except: no __context__
            self._finalize_failure(prepared, fault)
        return self._finalize_success(prepared, payload)

    async def ainvoke(
        self,
        context: TenantContext | Any,
        request: ToolRequest,
        *,
        allowed_tools: Sequence[str] | None = (),
        agent_id: str = "",
        timeout_s: float | None = None,
    ) -> ToolResult:
        """Async entry point: gates here, execution on the bounded pool.

        Crucially the gates do NOT consume a worker, so ``max_concurrency=1``
        still serves calls (this used to deadlock: the async path submitted
        ``invoke`` — which itself waits for a worker — to the same pool).
        A timeout is FINAL: the awaiting side writes the timeout record and the
        abandoned work can never append a second, contradictory "success"
        record, because auditing happens ONLY on the awaiting side.
        """
        # validated FIRST: an invalid budget fails before ANY gate runs
        caller_budget = None if timeout_s is None else _validate_timeout_s(timeout_s)
        prepared = self._prepare(
            context,
            request,
            allowed_tools=allowed_tools,
            agent_id=agent_id,
            state=_CallState(),
        )
        # an explicit caller cap may only TIGHTEN the tool budget; it can never
        # extend it (that would let a caller outrun the deadline/timeout gates)
        budget = prepared.timeout_s
        if caller_budget is not None:
            budget = min(budget, caller_budget)
        future = self._submit(prepared, context, allow_injected=False)
        wrapped = asyncio.wrap_future(future, loop=asyncio.get_running_loop())
        fault: _ToolFault | None = None
        payload: Any = None
        try:
            payload = await asyncio.wait_for(wrapped, timeout=budget)
        except asyncio.CancelledError:
            # the abandoned worker must not append a "success" record later
            future.cancel()
            prepared.audit(result="cancelled")
            raise
        except TimeoutError:
            fault = _ToolFault(
                f"rejected:async_timeout={budget:g}s", ErrorCode.TOOL_TIMEOUT
            )
        except _ToolFault as exc:
            fault = exc
        if fault is not None:  # raised OUTSIDE the except: no __context__
            self._finalize_failure(prepared, fault)
        return self._finalize_success(prepared, payload)

    def peak_concurrency(self) -> int:
        """Highest simultaneous execution count observed (test/ops probe)."""
        with self._lock:
            return self._peak_in_flight

    def running_calls(self) -> int:
        """Executions in flight right now (never above ``max_concurrency``)."""
        with self._lock:
            return self._in_flight

    def shutdown(self, *, wait: bool = False) -> None:
        """Release the bounded worker pool (called when the app shuts down)."""
        if self._owns_pool:
            self._pool.shutdown(wait=wait, cancel_futures=True)

    # -- helpers ----------------------------------------------------------------

    def _effective_timeout_ms(self, request: ToolRequest) -> int:
        """Remaining budget: per-tool default capped by an explicit deadline."""
        timeout_ms = self._default_timeout_ms
        if request.deadline is not None:
            remaining_ms = int(
                (request.deadline - self._clock()).total_seconds() * 1000
            )
            timeout_ms = min(timeout_ms, remaining_ms)
        return max(timeout_ms, 0)

    def _audit(
        self,
        *,
        tool_name: str,
        agent_id: str,
        tenant_id: str,
        session_id_hash: str,
        run_id: str,
        request_id: str,
        state: _CallState | None = None,
    ) -> Callable[[str, ErrorCode | None, int], None]:
        """Build a per-invocation audit closer.

        Records carry outcome markers/sizes only — never arguments or result
        content (no raw text in logs or audit).
        """

        call_state = state if state is not None else _CallState()
        label = _audit_tool_label(tool_name)

        def close(
            result: str = "",
            code: ErrorCode | None = None,
            latency_ms: int | None = 0,
        ) -> None:
            if call_state.closed:
                return  # a late completion must never contradict a timeout
            call_state.closed = True
            if self._audit_sink is None:
                return
            try:
                self._audit_sink(
                    AuditRecord(
                        actor_type="agent",
                        actor_id_hash=hashlib.sha256(
                            agent_id.encode("utf-8")
                        ).hexdigest(),
                        session_id_hash=session_id_hash,
                        # a bare TenantContext carries no request id; the field
                        # is required by the audit contract, so an explicit
                        # placeholder is used rather than fabricating an id
                        request_id=request_id or "unattributed",
                        run_id=run_id,
                        tenant_id=tenant_id or "unknown",
                        action="tool.invoke",
                        # never the raw model-supplied name: an unknown tool
                        # gets a fixed label (with a short digest to correlate)
                        tool_names=[label],
                        error_code=code,
                        latency_ms=latency_ms or 0,
                        result=result,
                    )
                )
            except Exception:  # noqa: BLE001 - audit must never break the call
                # …but a broken sink is a SECURITY fault, so it is loud — with
                # NO exception text: a sink may fail on untrusted content and
                # the message must not carry any of it into the log.
                _LOGGER.warning(
                    "tool audit record dropped (sink raised; details withheld)"
                )

        return close
