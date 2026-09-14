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
write capability. On timeout the call returns TOOL_TIMEOUT immediately; a
runaway executor thread is daemon and cannot block process shutdown.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.contracts.agent import ToolRequest, ToolResult
from app.contracts.audit import AuditRecord
from app.contracts.common import TenantContext
from app.contracts.errors import ErrorCode
from app.tools.specs import ToolSpec
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
    """Tool failure carrying a stable ErrorCode (mapped to envelopes by #36)."""

    def __init__(self, code: ErrorCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


def _default_runner(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` in a daemon thread with a wall-clock timeout.

    On expiry ``TimeoutError`` is raised and the runaway thread is left to die
    on its own (daemon) — the caller never waits for it.
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
    ) -> None:
        self._audit_sink = audit_sink
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._runner = runner or _default_runner
        self._default_timeout_ms = default_timeout_ms
        self._default_max_result_bytes = default_max_result_bytes
        self._lock = threading.RLock()
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

    def invoke(
        self,
        context: TenantContext | Any,
        request: ToolRequest,
        *,
        allowed_tools: Sequence[str] | None = (),
        agent_id: str = "",
    ) -> ToolResult:
        """Authorized, schema-checked, timed and capped tool execution.

        ``context`` is the TRUSTED identity injected by the server (never by the
        model); every failure raises :class:`ToolGatewayError` after an audit
        record, and the executor never runs for rejected calls.
        """
        identity = identity_from_context(context)
        name = request.tool_name
        with self._lock:
            spec = self._specs.get(name)
            disabled = name in self._disabled
        started = time.perf_counter()
        audit = self._audit(
            tool_name=name,
            agent_id=agent_id,
            tenant_id=identity.tenant_id,
            session_id_hash=identity.session_id_hash,
            run_id=identity.run_id,
            request_id=identity.request_id,
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
            audit(result="rejected:input_schema", code=ErrorCode.TOOL_SCHEMA_REJECTED)
            raise ToolGatewayError(
                ErrorCode.TOOL_SCHEMA_REJECTED,
                f"input schema violations at {', '.join(violations)}",
            )

        # 4. deadline + per-tool timeout.
        timeout_ms = self._effective_timeout_ms(request)
        if timeout_ms <= 0:
            audit(result="rejected:deadline_expired", code=ErrorCode.TOOL_TIMEOUT)
            raise ToolGatewayError(ErrorCode.TOOL_TIMEOUT, "deadline already expired")

        # 5. executor under timeout.
        try:
            payload = self._runner(
                lambda: spec.executor(context, request.arguments), timeout_ms / 1000
            )
        except ToolGatewayError as exc:
            audit(result="executor:" + exc.code.value, code=exc.code)
            raise
        except TimeoutError:
            audit(result="rejected:timeout", code=ErrorCode.TOOL_TIMEOUT)
            raise ToolGatewayError(
                ErrorCode.TOOL_TIMEOUT, f"tool {name!r} timed out"
            ) from None
        except Exception:  # noqa: BLE001 - executor internals never leak
            audit(result="executor:internal", code=ErrorCode.INTERNAL_UNKNOWN)
            raise ToolGatewayError(
                ErrorCode.INTERNAL_UNKNOWN, f"tool {name!r} failed internally"
            ) from None

        # 6. result schema + size gates.
        violations = validate(spec.output_schema, payload)
        if violations:
            audit(result="rejected:output_schema", code=ErrorCode.TOOL_SCHEMA_REJECTED)
            raise ToolGatewayError(
                ErrorCode.TOOL_SCHEMA_REJECTED,
                f"output schema violations at {', '.join(violations)}",
            )
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError):
            audit(result="rejected:output_json", code=ErrorCode.TOOL_SCHEMA_REJECTED)
            raise ToolGatewayError(
                ErrorCode.TOOL_SCHEMA_REJECTED, f"tool {name!r} returned non-JSON data"
            ) from None
        size_bytes = len(encoded.encode("utf-8"))
        cap = spec.max_result_bytes or self._default_max_result_bytes
        if size_bytes > cap:
            audit(
                result=f"rejected:result_bytes={size_bytes}",
                code=ErrorCode.TOOL_OVER_LIMIT,
            )
            raise ToolGatewayError(
                ErrorCode.TOOL_OVER_LIMIT,
                f"tool {name!r} result exceeded {cap} bytes",
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        audit(result=f"ok:result_bytes={size_bytes}", latency_ms=latency_ms)
        return ToolResult(tool_name=name, ok=True, data=payload)

    async def ainvoke(
        self,
        context: TenantContext | Any,
        request: ToolRequest,
        *,
        allowed_tools: Sequence[str] | None = (),
        agent_id: str = "",
        timeout_s: float | None = None,
    ) -> ToolResult:
        """Async entry point: the whole gated call runs OFF the event loop.

        The same gates as :meth:`invoke` apply (the sync path runs in a worker
        thread), and ``asyncio.wait_for`` adds a hard upper bound so a slow or
        hung tool cannot stall the loop. Cancellation propagates unchanged —
        a cancelled call applies no side effects (executors are read-only, and
        the gateway itself never writes).
        """
        budget = (
            self._default_timeout_ms / 1000 if timeout_s is None else float(timeout_s)
        )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.invoke,
                    context,
                    request,
                    allowed_tools=allowed_tools,
                    agent_id=agent_id,
                ),
                timeout=budget,
            )
        except TimeoutError:
            identity = identity_from_context(context)
            self._audit(
                tool_name=request.tool_name,
                agent_id=agent_id,
                tenant_id=identity.tenant_id,
                session_id_hash=identity.session_id_hash,
                run_id=identity.run_id,
                request_id=identity.request_id,
            )(
                result=f"rejected:async_timeout={budget:g}s",
                code=ErrorCode.TOOL_TIMEOUT,
            )
            raise ToolGatewayError(
                ErrorCode.TOOL_TIMEOUT,
                f"tool {request.tool_name!r} exceeded the async budget",
            ) from None

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
    ) -> Callable[[str, ErrorCode | None, int], None]:
        """Build a per-invocation audit closer.

        Records carry outcome markers/sizes only — never arguments or result
        content (no raw text in logs or audit).
        """

        def close(
            result: str = "", code: ErrorCode | None = None, latency_ms: int = 0
        ) -> None:
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
                        tool_names=[tool_name],
                        error_code=code,
                        latency_ms=latency_ms,
                        result=result,
                    )
                )
            except Exception:  # audit must never break the call…
                # …but a broken audit sink is a SECURITY fault: make it loud
                # instead of silently dropping records.
                _LOGGER.warning("tool audit record dropped", exc_info=True)

        return close
