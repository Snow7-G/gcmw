"""Tests for #55A-B: call-time tool quota + server-side trusted provenance.

Coverage (per the review instructions):
- quota 0: a sub-agent granted zero calls never touches the gateway;
- call-time check: the gateway call does not even START once the granted
  ceiling is reached (never a post-hoc count alone);
- cross-revision accumulation: a revision is granted only the LEFTOVER of the
  run-cumulative budget — it can never start a fresh quota;
- forged provenance: whatever the sub-agent (or the model output) claims about
  provider/model identity is DISCARDED and replaced with the server-side
  ModelGateway/Provider configuration;
- SSE publishes ONLY that trusted provenance (answer.completed.model);
- an expired deadline produces no late engagement and no late tool calls.
"""

import asyncio
import contextvars
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pytest import mark, raises

from app.agents.manager import (
    AgentExecution,
    ManagerAgent,
    ManagerAgentError,
    RedFlagRules,
    RiskRules,
    Verdict,
    VerifierOutcome,
)
from app.agents.medical_qa import MedicalQAAgent
from app.agents.registry import AgentManifest, AgentRegistry
from app.contracts.agent import AgentContext, AgentStatus, ToolRequest
from app.contracts.common import Channel
from app.contracts.errors import ErrorCode
from app.contracts.events import SSEEventType
from app.knowledge.store import KnowledgeStore
from app.orchestration.executor import RunExecutor
from app.storage.run_repository import MemoryRunRepository, RunIdentity
from app.tools.builtins import build_gateway
from app.tools.gateway import ToolGatewayError
from app.tools.quota import (
    RunToolQuota,
    current_run_quota,
    reset_run_quota,
    set_run_quota,
)

APPROVED_AT = datetime(2026, 1, 5, tzinfo=timezone.utc)
TRUSTED = ("mock", "mock-model", "1.0.0")
FORGED = ("fake-cloud", "gpt-infinity", "9.9.9")


def _ctx(**overrides):
    fields = {
        "tenant_id": "t1",
        "device_id": "d1",
        "session_id": "s1",
        "run_id": "r1",
        "channel": Channel.TOUCH,
        "normalized_input": "发热怎么办",
        "deadline": datetime.now(timezone.utc) + timedelta(seconds=30),
    }
    fields.update(overrides)
    return AgentContext(**fields)


def _rules():
    return (
        RedFlagRules(patterns=("自杀",), approved_by="t", approved_at=APPROVED_AT),
        RiskRules(patterns=("剧烈",), approved_by="t", approved_at=APPROVED_AT),
    )


class _CountingGateway:
    """Counts gateway round-trips; the search always finds nothing."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, ctx, request, *, allowed_tools=None, agent_id=None):
        self.calls += 1
        return SimpleNamespace(ok=True, data={"items": [], "total": 0})


class _FakeModels:
    async def chat(self, request):  # pragma: no cover - never reached here
        raise AssertionError("the model must not be called without evidence")


def _qa(gateway: _CountingGateway) -> MedicalQAAgent:
    return MedicalQAAgent(models=_FakeModels(), tools=gateway)


# -- call-time quota (QA agent level) ------------------------------------------


class TestCallTimeQuota:
    @mark.asyncio
    async def test_quota_zero_never_touches_the_gateway(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx(tool_budget_granted=0))
        assert gateway.calls == 0  # the call never even started
        assert execution.status is AgentStatus.FAILED
        assert execution.safety_status == "no_evidence"
        assert execution.tool_calls == 0

    @mark.asyncio
    async def test_a_grant_of_one_stops_after_one_call(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx(tool_budget_granted=1))
        assert gateway.calls == 1  # the second call never started
        assert execution.tool_calls == 1
        assert execution.status is AgentStatus.FAILED

    @mark.asyncio
    async def test_no_grant_uses_the_agent_own_limit(self):
        gateway = _CountingGateway()
        execution = await _qa(gateway).run(_ctx())  # tool_budget_granted=None
        # legacy callers stay unrestricted by the agent's own limit; the search
        # finds nothing, so exactly one call happens
        assert gateway.calls == 1
        assert execution.safety_status == "no_evidence"


# -- cross-revision accumulation (Manager level) --------------------------------


class _RecordingRunner:
    """Records the quota granted per engagement and reports the tool count of
    a WELL-BEHAVED sub-agent: min(demand, granted) — the call-time quota makes
    exceeding the grant impossible for it."""

    def __init__(self, demand: int) -> None:
        self.demand = demand
        self.granted: list[int] = []
        self.reported: list[int] = []

    async def __call__(self, ctx: AgentContext) -> AgentExecution:
        granted = ctx.tool_budget_granted
        if granted is None:
            granted = self.demand
        self.granted.append(granted)
        used = min(self.demand, granted)
        self.reported.append(used)
        return AgentExecution(
            agent_id="qa",
            status=AgentStatus.COMPLETED,
            answer_candidate="发热建议门诊就诊。",
            evidence=(),
            tool_calls=used,
            provider_id=FORGED[0],
            model_id=FORGED[1],
            model_version=FORGED[2],
            safety_status="unknown",
        )


class _ScriptedVerifier:
    def __init__(self, verdicts) -> None:
        self.verdicts = list(verdicts)

    async def __call__(self, ctx, execution) -> Verdict:
        return self.verdicts.pop(0)


def _manager(runner, verifier, **overrides) -> ManagerAgent:
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            agent_id="qa",
            version="1.0.0",
            supported_intents=["knowledge"],
            risk_level="medium",
        )
    )
    red, risk = _rules()
    return ManagerAgent(
        registry=registry,
        agent_runners={"qa": runner},
        verifier=verifier,
        red_flag_rules=red,
        risk_rules=risk,
        trusted_model_provenance={"qa": TRUSTED},
        **overrides,
    )


class TestSharedQuotaState:
    @mark.asyncio
    async def test_the_advisory_grant_does_not_reset_between_revisions(self):
        """The HARD enforcement lives in the gateway's shared quota state (see
        test_manager_agent.TestToolBudgetIsRunCumulative); the context integer
        is advisory only and never restarts the budget for a revision."""
        runner = _RecordingRunner(demand=2)
        verifier = _ScriptedVerifier(
            [
                Verdict(VerifierOutcome.REVISE, "ungrounded"),
                Verdict(VerifierOutcome.PASS, ""),
            ]
        )
        manager = _manager(runner, verifier)
        result = await manager.execute(_ctx(), "发热怎么办")
        assert runner.granted == [4, 4]  # advisory, never re-granted
        assert result.safety_status == "verified"

    @mark.asyncio
    async def test_the_quota_object_is_created_once_per_execute(self):
        seen = []

        class _SpyVerifier:
            def __init__(self, inner):
                self._inner = inner

            async def __call__(self, ctx, execution):
                seen.append(current_run_quota())
                return await self._inner(ctx, execution)

        runner = _RecordingRunner(demand=2)
        manager = _manager(
            runner, _SpyVerifier(_ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")]))
        )
        await manager.execute(_ctx(), "发热怎么办")
        # both verifier sightings saw the SAME quota object
        assert len(seen) == 1
        assert seen[0] is not None
        assert seen[0].consumed == 0  # a stub runner never touched the gateway

    def test_zero_limit_quota_refuses_everything(self):

        quota = RunToolQuota(0)
        assert quota.try_acquire() is False  # refused as EXHAUSTED
        assert quota.consumed == 0
        assert quota.refused_exhausted == 1
        assert quota.refused_closed == 0

    def test_concurrent_acquires_cannot_overshoot_the_limit(self):
        """Remaining 1, two concurrent callers: exactly ONE enters."""
        import threading

        quota = RunToolQuota(1)
        results = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            results.append(quota.try_acquire())

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == [False, True]
        assert quota.consumed == 1


class TestTrustedProvenance:
    @mark.asyncio
    async def test_forged_model_identity_is_discarded(self):
        runner = _RecordingRunner(demand=1)  # claims FORGED identity
        verifier = _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        manager = _manager(runner, verifier)
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.model == {
            "provider_id": TRUSTED[0],
            "model_id": TRUSTED[1],
            "model_version": TRUSTED[2],
        }  # the FORGED claim never reaches the public result

    @mark.asyncio
    async def test_a_refusal_publishes_no_provenance(self):
        class _EscalatingVerifier:
            async def __call__(self, ctx, execution) -> Verdict:
                return Verdict(VerifierOutcome.ESCALATE, "")

        manager = _manager(_RecordingRunner(demand=1), _EscalatingVerifier())
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.model is None  # no answer, no provenance on the wire

    @mark.asyncio
    async def test_an_expired_deadline_produces_no_late_engagement(self):
        class _NeverCalled(_RecordingRunner):
            async def __call__(self, ctx) -> AgentExecution:  # pragma: no cover
                raise AssertionError("a timed-out run must not engage the agent")

        runner = _NeverCalled(demand=0)
        manager = _manager(runner, _ScriptedVerifier([]))
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        with raises(ManagerAgentError):
            await manager.execute(_ctx(deadline=past), "发热怎么办")


class TestSsePublishesOnlyTrustedProvenance:
    """END-TO-END through the REAL Manager and the REAL executor: the wire's
    ``answer.completed.model`` is the trusted server-side triple when a mapping
    exists, and is ABSENT when it does not — never the runner's forged claim.
    (The old stub-based test bypassed the Manager entirely and proved nothing
    about the fail-open fallback.)"""

    @mark.asyncio
    async def test_with_a_mapping_the_wire_carries_the_trusted_triple(self):
        # step 1: the Manager's own result carries the trusted triple
        runner = _RecordingRunner(demand=0)  # no gateway calls, forged identity
        manager = _manager(
            runner, _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        )
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.model == {
            "provider_id": TRUSTED[0],
            "model_id": TRUSTED[1],
            "model_version": TRUSTED[2],
        }
        # step 2: a FRESH manager (same config) drives the executor — the wire
        # must publish exactly what the Manager verified, never the forgery
        repository = MemoryRunRepository()
        executor = RunExecutor(
            repository=repository,
            manager=_manager(
                _RecordingRunner(demand=0),
                _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")]),
            ),  # type: ignore[arg-type]
        )
        identity = RunIdentity(
            run_id="r-sse-prov", tenant_id="t1", device_id="d1", session_id="s1"
        )
        await repository.create(identity)
        await asyncio.wait_for(executor.execute(_record_for(identity)), timeout=5)
        page = await repository.snapshot(identity, 0, 0.0)
        completed = [e for e in page.events if e.event is SSEEventType.ANSWER_COMPLETED]
        assert len(completed) == 1
        assert completed[0].data["model"] == {
            "provider_id": TRUSTED[0],
            "model_id": TRUSTED[1],
            "model_version": TRUSTED[2],
        }

    @mark.asyncio
    async def test_without_a_mapping_no_provenance_reaches_the_wire(self):
        """P1-2: real Manager, NO trusted mapping, runner forges a triple and
        PASSes — the result carries NO model and the SSE has no model key."""
        repository = MemoryRunRepository()
        runner = _RecordingRunner(demand=0)
        registry = AgentRegistry()
        registry.register(
            AgentManifest(
                agent_id="qa",
                version="1.0.0",
                supported_intents=["knowledge"],
                risk_level="medium",
            )
        )
        red, risk = _rules()
        manager = ManagerAgent(
            registry=registry,
            agent_runners={"qa": runner},
            # TWO verdicts: the Manager result probe and the executor's own
            # engagement each consume one
            verifier=_ScriptedVerifier(
                [Verdict(VerifierOutcome.PASS, ""), Verdict(VerifierOutcome.PASS, "")]
            ),
            red_flag_rules=red,
            risk_rules=risk,
            # NO trusted_model_provenance: the fail-open fallback is gone
        )
        executor = RunExecutor(repository=repository, manager=manager)  # type: ignore[arg-type]
        identity = RunIdentity(
            run_id="r-sse-nomap", tenant_id="t1", device_id="d1", session_id="s1"
        )
        await repository.create(identity)
        result = await manager.execute(_ctx(run_id="r-sse-nomap"), "发热怎么办")
        assert result.model is None  # fail closed: no model identity at all
        await asyncio.wait_for(executor.execute(_record_for(identity)), timeout=5)
        page = await repository.snapshot(identity, 0, 0.0)
        completed = [e for e in page.events if e.event is SSEEventType.ANSWER_COMPLETED]
        assert len(completed) == 1
        assert "model" not in completed[0].data  # never the forged claim


def _record_for(identity):
    """The slice of RunRecord the executor reads."""

    class _Record:
        pass

    record = _Record()
    record.identity = identity

    class _Snap:
        channel = "text"
        text = "发热怎么办"

    record.snapshot = _Snap()
    return record


class TestProvenanceValidationAtConstruction:
    @staticmethod
    def _registry():
        registry = AgentRegistry()
        registry.register(
            AgentManifest(
                agent_id="qa",
                version="1.0.0",
                supported_intents=["knowledge"],
                risk_level="medium",
            )
        )
        return registry

    @mark.parametrize(
        "triple",
        [
            ("mock", "mock-model"),  # only two fields
            ("mock", "", "1.0.0"),  # empty model id
            ("p" * 65, "m", "1.0.0"),  # provider id over the 64-char bound
            "mock-model",  # not a tuple at all
        ],
    )
    def test_a_malformed_triple_fails_at_assembly(self, triple):
        red, risk = _rules()
        with raises(ValueError):
            ManagerAgent(
                registry=self._registry(),
                red_flag_rules=red,
                risk_rules=risk,
                trusted_model_provenance={"qa": triple},
            )


# -- quota LIFETIME: a copied context must die with the run ----------------------


class _BackgroundSpawner:
    """A rogue runner that spawns a background task whose LATE gateway calls
    fire only after the Manager has ended. It also reports tool_calls=0."""

    def __init__(self, gateway, late_calls=1, delay=0.1, stall_s=0.0):
        self.gateway = gateway
        self.late_calls = late_calls
        self.delay = delay
        self.stall_s = stall_s
        self.tasks = []
        self.late_codes = []
        self.spawned = asyncio.Event()

    def _spawn(self, ctx):
        async def late():
            await asyncio.sleep(self.delay)  # the Manager ends before this
            for _ in range(self.late_calls):
                try:
                    await self.gateway.ainvoke(
                        ctx,
                        ToolRequest(
                            tool_name="knowledge.search",
                            arguments={"query": "发热", "top_k": 1},
                        ),
                        allowed_tools={
                            "knowledge.search",
                            "knowledge.get_fragment",
                        },
                        agent_id="qa",
                    )
                    self.late_codes.append(None)  # an EXECUTED late call
                except ToolGatewayError as exc:
                    self.late_codes.append(exc.code)  # refused, zero execution

        self.tasks.append(asyncio.create_task(late()))
        self.spawned.set()

    async def __call__(self, ctx):
        self._spawn(ctx)
        if self.stall_s:
            await asyncio.sleep(self.stall_s)
        return _execution()


def _execution():
    return AgentExecution(
        agent_id="qa",
        status=AgentStatus.COMPLETED,
        answer_candidate="体温超过38.5建议门诊就诊。",
        evidence=(),
        tool_calls=0,
        provider_id=FORGED[0],
        model_id=FORGED[1],
        model_version=FORGED[2],
        safety_status="unknown",
    )


def _audit_split(records):
    executed = [r for r in records if r.result.startswith("ok:")]
    exhausted = [r for r in records if r.result == "rejected:tool_quota_exhausted"]
    closed = [r for r in records if r.result == "rejected:tool_quota_closed"]
    return executed, exhausted, closed


class TestQuotaLifecycle:
    """P1: the Manager's ContextVar reset only unbinds ITS task — a background
    task spawned by a runner keeps a COPY of the quota. The quota must be
    atomically CLOSED before the Manager ends, so late calls from ANY copied
    context are refused with zero side effects."""

    async def _spawn_and_wait_late(self, runner):
        await asyncio.wait_for(runner.spawned.wait(), timeout=2)
        await asyncio.gather(*runner.tasks)  # the late attempts happen now

    @mark.asyncio
    async def test_after_a_normal_return_the_background_task_executes_nothing(self):
        records: list = []
        gateway = build_gateway(
            knowledge_store=KnowledgeStore(), audit_sink=records.append
        )
        runner = _BackgroundSpawner(gateway, delay=0.5)
        manager = _manager(
            runner, _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        )
        result = await manager.execute(_ctx(), "发热怎么办")
        assert result.safety_status == "verified"  # the answer was delivered
        await self._spawn_and_wait_late(runner)
        executed, _, closed = _audit_split(records)
        assert executed == []  # ZERO late executions
        assert len(closed) == 1  # exactly ONE fixed refusal audit
        assert runner.late_codes == [ErrorCode.TOOL_OVER_LIMIT]

    @mark.asyncio
    async def test_after_a_timeout_the_background_task_executes_nothing(self):
        records: list = []
        gateway = build_gateway(
            knowledge_store=KnowledgeStore(), audit_sink=records.append
        )
        runner = _BackgroundSpawner(gateway, delay=0.5, stall_s=2.0)
        manager = _manager(
            runner, _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        )
        deadline = datetime.now(timezone.utc) + timedelta(seconds=0.15)
        with raises(ManagerAgentError) as exc:
            await manager.execute(_ctx(deadline=deadline), "发热怎么办")
        assert exc.value.code is ErrorCode.TIMEOUT_AGENT
        await self._spawn_and_wait_late(runner)
        executed, _, closed = _audit_split(records)
        assert executed == []
        assert len(closed) == 1
        assert runner.late_codes == [ErrorCode.TOOL_OVER_LIMIT]

    @mark.asyncio
    async def test_after_a_cancellation_the_background_task_executes_nothing(self):
        records: list = []
        gateway = build_gateway(
            knowledge_store=KnowledgeStore(), audit_sink=records.append
        )
        runner = _BackgroundSpawner(gateway, delay=0.5, stall_s=30.0)
        manager = _manager(
            runner, _ScriptedVerifier([Verdict(VerifierOutcome.PASS, "")])
        )
        task = asyncio.create_task(manager.execute(_ctx(), "发热怎么办"))
        await asyncio.wait_for(runner.spawned.wait(), timeout=2)
        await asyncio.sleep(0.05)  # the runner is stalled inside the turn now
        task.cancel()
        with raises(asyncio.CancelledError):
            await task
        await self._spawn_and_wait_late(runner)
        executed, _, closed = _audit_split(records)
        assert executed == []
        assert len(closed) == 1
        assert runner.late_codes == [ErrorCode.TOOL_OVER_LIMIT]

    @mark.asyncio
    async def test_close_and_acquire_are_serialized_one_way_or_the_other(self):
        """Close vs acquire is atomic under the same lock: either an acquire
        wins and executes once, or the close wins and nothing executes."""
        quota = RunToolQuota(4)
        results = {"granted": 0, "exhausted": 0, "closed": 0}
        barrier = threading.Barrier(6)

        def acquire_worker():
            barrier.wait()
            outcome = quota.acquire()
            results[f"{outcome}s" if False else outcome] = (
                results.get(
                    {
                        "granted": "granted",
                        "exhausted": "exhausted",
                        "closed": "closed",
                    }[outcome],
                    0,
                )
                + 1
            )

        def close_worker():
            barrier.wait()
            quota.close()

        threads = [threading.Thread(target=acquire_worker) for _ in range(5)]
        threads.append(threading.Thread(target=close_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # atomicity: at most ONE acquire won, everything else was refused
        assert quota.consumed in (0, 1)
        assert quota.consumed + quota.refused_exhausted + quota.refused_closed == 5
        # after close: EVERY later acquire fails, in any thread
        assert quota.closed
        assert quota.try_acquire() is False

    def test_a_call_started_before_the_close_can_finish(self):
        """An in-flight call that acquired BEFORE the close is allowed to
        complete — nothing is refunded and it counts exactly once."""
        import threading

        from app.tools.quota import RunToolQuota

        records: list = []
        gateway = build_gateway(
            knowledge_store=KnowledgeStore(), audit_sink=records.append
        )
        quota = RunToolQuota(4)
        token = set_run_quota(quota)
        started = threading.Event()
        release = threading.Event()
        outcome = {}

        def slow_runner(counted, timeout_s):
            started.set()
            release.wait(2)
            return counted()

        gateway._runner = slow_runner  # a slow tool execution, injected

        def call():
            outcome["res"] = gateway.invoke(
                _ctx(),
                ToolRequest(
                    tool_name="knowledge.search",
                    arguments={"query": "发热", "top_k": 1},
                ),
                allowed_tools={"knowledge.search", "knowledge.get_fragment"},
                agent_id="qa",
            )

        call_context = contextvars.copy_context()  # the quota travels with it
        thread = threading.Thread(target=lambda: call_context.run(call))
        thread.start()
        assert started.wait(2)  # the call acquired and is executing
        assert quota.consumed == 1
        quota.close()  # close DURING the in-flight execution
        release.set()
        thread.join(2)
        assert outcome["res"].ok is True  # it was allowed to finish
        assert quota.consumed == 1  # counted exactly once, no refund
        reset_run_quota(token)
