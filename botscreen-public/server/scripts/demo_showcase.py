"""后端 MVP 一键五场景验收（路线图"演示收口"）：一条命令的后端演示 +
五个合成场景验收。

用法::

    python3 scripts/demo_showcase.py            # 跑全部场景，打印 PASS/FAIL
    python3 scripts/demo_showcase.py --serve    # 场景全过后保持服务运行，供人工演示

这条命令做什么：

* 在本进程内启动真实 uvicorn（127.0.0.1，随机端口），装配 demo 栈——
  合成知识走真实 #56 治理生命周期（候选→审核→已批准，仅 DEMO_TENANT_ID）、
  MockProvider（罐头回复，零出网）、RAG + SafetyEvidenceVerifier + ManagerAgent；
* 对真实 TCP 跑五个场景，任一断言失败即收集并在最后以**非零退出码**结束：
    1. 有引用回答（双层 SSE + 引用三元组 + 可信模型溯源 + 工具配额审计）
    2. 无证据拒答（不编造答案）
    3. 红旗转人工
    4. 取消 + Last-Event-ID 断线重连
    5. 租户/设备隔离
* 全程 MockProvider、合成数据、不出网（无任何云 Provider 配置）。

诚实边界（与 app.orchestration.assembly 的声明一致）：

* 场景 4 需要一次可见的"运行中"窗口，脚本注入了 demo 专用的仓储延迟
  （``_DemoStallRepository``，每次运行在 RETRIEVING 停留 ``DEMO_STALL_S``
  秒）。这是**唯一**的演示支架，其余全部走真实栈；
* 规则集（红旗/风险）只带已记录的 attestation，真实临床签核仍是流程门禁。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

# 该脚本位于 server/scripts/ 下：把 server/ 加入导入路径，app.* 才可见
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.main import create_app
from app.storage.run_repository import MemoryRunRepository

# -- demo 常量 ------------------------------------------------------------------

DEMO_TENANT = "t1"
OTHER_TENANT = "t2"

#: 三个演示设备凭据（确定性低熵，仅演示环境；真实凭据永不入库/入日志）
TOKEN_PRIMARY = "demo-primary-device-000001"  # t1/d1 — 主设备
TOKEN_SAME_TENANT = "demo-second-device-00002"  # t1/d2 — 同租户异设备
TOKEN_OTHER_TENANT = "demo-other-tenant-00003"  # t2/d9 — 异租户

DEMO_STALL_S = 1.0  # 每次运行在 RETRIEVING 的可见停留窗口
POLL_DEADLINE_S = 15.0
TERMINAL_STATES = {"COMPLETED", "DEGRADED", "HANDOFF", "FAILED", "CANCELLED"}

#: MockProvider 的服务端可信三元组（assembly 用 models.provenance() 注入）
TRUSTED_MODEL = {
    "provider_id": "mock",
    "model_id": "mock-model",
    "model_version": "1.0.0",
}


class DemoFailure(AssertionError):
    """一个场景失败（消息即断言差异）。"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DemoFailure(message)


# -- demo 支架：可见的运行中窗口 -------------------------------------------------


class _DemoStallRepository(MemoryRunRepository):
    """演示专用：每次运行在进入 DRAFTING 前停留一次，让人眼能看到
    RETRIEVING/运行中状态（否则 MockProvider 毫秒级完成，取消与重连无法
    展示）。除此之外与真实仓储逐字节一致。"""

    def __init__(self, stall_s: float) -> None:
        super().__init__()
        self._stall_s = stall_s
        self._stalled: set[str] = set()

    async def commit_transition(
        self, identity, *, expected_state, next_state, data=None
    ):
        if next_state.value == "DRAFTING" and identity.run_id not in self._stalled:
            self._stalled.add(identity.run_id)
            import asyncio

            await asyncio.sleep(self._stall_s)
        return await super().commit_transition(
            identity, expected_state=expected_state, next_state=next_state, data=data
        )


class _AuditCapture(logging.Handler):
    """收集 gcmw.audit 通道的结构化工具审计 JSON 行（网关审计核验用）。"""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(json.loads(record.getMessage()))
        except (ValueError, TypeError):
            pass


# -- SSE 帧解析（自包含，不依赖 tests/） -------------------------------------------


def parse_frames(body: str) -> list[dict[str, Any]]:
    """SSE 正文 → 帧。注释帧（`: keep-alive`）→ {"comment": …}；协议帧 →
    id/event/data。"""
    frames: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        if block.startswith(":"):
            frames.append({"comment": block.lstrip(":").strip()})
            continue
        frame: dict[str, Any] = {}
        for line in block.split("\n"):
            if line.startswith("id: "):
                frame["id"] = int(line[4:])
            elif line.startswith("event: "):
                frame["event"] = line[7:]
            elif line.startswith("data: "):
                frame["data"] = json.loads(line[6:])
        if frame:
            frames.append(frame)
    return frames


def protocol_frames(body: str) -> list[dict[str, Any]]:
    """只保留携带运行事件的帧（丢弃注释心跳）。"""
    return [f for f in parse_frames(body) if "event" in f and "comment" not in f]


# -- HTTP 助手 ------------------------------------------------------------------


class DemoClient:
    def __init__(self, base: str, token: str) -> None:
        self.base = base
        self.headers = {"Authorization": f"Bearer {token}"}
        self._client = httpx.Client(timeout=30, headers=self.headers)

    def close(self) -> None:
        self._client.close()

    def create_session(self) -> str:
        res = self._client.post(
            f"{self.base}/api/v1/sessions", json={"channel": "text"}
        )
        _check(
            res.status_code == 201, f"create_session -> {res.status_code}: {res.text}"
        )
        return res.json()["session_id"]

    def create_run(self, session_id: str, text: str, key: str) -> dict:
        res = self._client.post(
            f"{self.base}/api/v1/agent/runs",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": text},
                "idempotency_key": key,
            },
        )
        _check(res.status_code == 200, f"create_run -> {res.status_code}: {res.text}")
        return res.json()

    def state(self, run_id: str) -> str:
        res = self._client.get(f"{self.base}/api/v1/agent/runs/{run_id}")
        _check(res.status_code == 200, f"get_run -> {res.status_code}: {res.text}")
        return res.json()["state"]

    def wait_terminal(self, run_id: str, expect: str | None = None) -> str:
        deadline = time.time() + POLL_DEADLINE_S
        state = self.state(run_id)
        while state not in TERMINAL_STATES:
            _check(time.time() < deadline, f"run {run_id} 卡在 {state}")
            time.sleep(0.05)
            state = self.state(run_id)
        if expect is not None:
            _check(state == expect, f"终态 {state} != 期望 {expect}")
        return state

    def wait_state(self, run_id: str, target: str) -> None:
        deadline = time.time() + POLL_DEADLINE_S
        while True:
            state = self.state(run_id)
            if state == target:
                return
            _check(
                state not in TERMINAL_STATES,
                f"等 {target} 时运行已终态：{state}",
            )
            _check(time.time() < deadline, f"run {run_id} 一直没到 {target}")
            time.sleep(0.02)

    def events(self, run_id: str, last_event_id: str | None = None) -> str:
        headers = dict(self.headers)
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        res = self._client.get(
            f"{self.base}/api/v1/agent/runs/{run_id}/events", headers=headers
        )
        _check(res.status_code == 200, f"events -> {res.status_code}: {res.text}")
        return res.text

    def cancel(self, run_id: str) -> dict:
        res = self._client.delete(f"{self.base}/api/v1/agent/runs/{run_id}")
        _check(res.status_code == 200, f"cancel -> {res.status_code}: {res.text}")
        return res.json()

    def stream_frames(self, run_id: str, max_events: int) -> tuple[str, list[dict]]:
        """真实流式读取 max_events 个事件帧后主动断开（模拟掉线）。"""
        seen: list[str] = []
        with httpx.stream(
            "GET",
            f"{self.base}/api/v1/agent/runs/{run_id}/events",
            timeout=30,
            headers=self.headers,
        ) as res:
            _check(res.status_code == 200, f"stream -> {res.status_code}")
            for line in res.iter_lines():
                seen.append(line)
                if len([x for x in seen if x.startswith("event:")]) >= max_events:
                    break
            res.close()
        return "\n".join(seen), parse_frames("\n".join(seen))


def _frames(body: str) -> list[dict]:
    return protocol_frames(body)


# -- 五个场景 --------------------------------------------------------------------


def scenario_1_cited_answer(client: DemoClient, session_id: str) -> str:
    """场景 1：有引用回答 — 双层 SSE + 引用三元组 + 可信模型溯源。"""
    run = client.create_run(session_id, "发热怎么办", key="demo-1-cited")
    run_id = run["run_id"]
    _check(client.wait_terminal(run_id) == "COMPLETED", "有证据运行应 COMPLETED")

    frames = _frames(client.events(run_id))
    ids = [f["id"] for f in frames]
    names = [f["event"] for f in frames]
    _check(ids == list(range(1, len(ids) + 1)), f"事件序号有缺口: {ids}")
    _check(names[0] == "run.accepted", f"首帧应是 run.accepted: {names[0]}")
    _check(names[-1] == "run.completed", f"末帧应是 run.completed: {names[-1]}")
    _check(names.count("run.completed") == 1, "run.completed 必须恰好一次")

    # 双层 SSE：process 与 answer 两层都在线上
    layers = {f["data"]["layer"] for f in frames}
    _check({"process", "answer"} <= layers, f"双层 SSE 缺层: {layers}")

    # 隔离字段：每个帧的 envelope 身份与运行一致
    for f in frames:
        env = f["data"]
        _check(
            env["tenant_id"] == DEMO_TENANT
            and env["device_id"] == "d1"
            and env["session_id"] == session_id
            and env["run_id"] == run_id,
            f"帧身份字段不一致: {env}",
        )

    completed = next(f for f in frames if f["event"] == "answer.completed")
    payload = completed["data"]["data"]
    citations = payload["citations"]
    _check(len(citations) >= 1, "answer.completed 缺引用")
    # 可核验引用三元组：source_id + knowledge_version + content_hash 三者
    # 非空缺一不可（content_hash 必须是 64 位十六进制 SHA-256）；PR 描述
    # 声称的 title/source_uri 也分别断言存在，防止描述与脚本漂移假绿
    for c in citations:
        _check(
            c.get("source_id") and c.get("knowledge_version") and c.get("content_hash"),
            f"引用三元组缺字段（source_id/knowledge_version/content_hash）: {c}",
        )
        _check(
            isinstance(c.get("content_hash"), str)
            and len(c["content_hash"]) == 64
            and all(ch in "0123456789abcdef" for ch in c["content_hash"]),
            f"content_hash 应为 64 位十六进制: {c.get('content_hash')!r}",
        )
        _check(
            c.get("title") and c.get("source_uri"),
            f"引用展示字段缺失（title/source_uri）: {c}",
        )
    _check(
        payload["content_origin"] == "approved_faq",
        f"content_origin 应为 approved_faq: {payload['content_origin']}",
    )
    # 可信模型溯源：线上只允许服务端三元组
    _check(
        payload.get("model") == TRUSTED_MODEL,
        f"模型溯源不是服务端可信三元组: {payload.get('model')}",
    )

    # 答案增量与引用标记
    deltas = [
        f["data"]["data"]["delta"] for f in frames if f["event"] == "answer.delta"
    ]
    _check(bool(deltas), "answer.delta 缺失")
    _check("资料[1]" in "".join(deltas), "答案文本缺引用标记 资料[1]")
    return run_id


def scenario_2_no_evidence_refusal(client: DemoClient, session_id: str) -> str:
    """场景 2：无证据拒答 — 绝不编造答案。"""
    run = client.create_run(session_id, "空调病怎么预防", key="demo-2-refusal")
    run_id = run["run_id"]
    _check(client.wait_terminal(run_id) == "FAILED", "无证据运行应 FAILED")

    frames = _frames(client.events(run_id))
    names = [f["event"] for f in frames]
    completed = next(f for f in frames if f["event"] == "run.completed")
    result = completed["data"]["data"].get("result")
    _check(
        result == "refused_no_answer",
        f"拒绝标记应为 refused_no_answer: {result}",
    )
    _check(
        "answer.completed" not in names and "answer.delta" not in names,
        f"无证据运行不得产出答案帧: {names}",
    )
    return run_id


def scenario_3_red_flag_handoff(client: DemoClient, session_id: str) -> str:
    """场景 3：红旗转人工 — 确定性升级，无答案上站。"""
    run = client.create_run(session_id, "我最近有自杀的念头", key="demo-3-redflag")
    run_id = run["run_id"]
    _check(client.wait_terminal(run_id) == "HANDOFF", "红旗输入应转 HANDOFF")

    frames = _frames(client.events(run_id))
    names = [f["event"] for f in frames]
    completed = next(f for f in frames if f["event"] == "run.completed")
    result = completed["data"]["data"].get("result")
    _check(
        result == "escalated_to_human",
        f"升级标记应为 escalated_to_human: {result}",
    )
    _check(
        "answer.completed" not in names,
        "红旗升级不得有答案帧上站",
    )
    return run_id


def scenario_4_cancel_and_resume(client: DemoClient, session_id: str) -> None:
    """场景 4：取消 + Last-Event-ID 断线重连（无缺口、终态恰好一次）。"""
    # -- 4a. 断线重连 --
    run = client.create_run(session_id, "眼睛不适", key="demo-4a-reconnect")
    run_id = run["run_id"]
    client.wait_state(run_id, "RETRIEVING")  # 演示支架窗口：可见的运行中
    body, _partial = client.stream_frames(run_id, max_events=3)
    cursor_ids = [f["id"] for f in _frames(body) if "id" in f]
    _check(bool(cursor_ids), "流式阶段未读到任何事件帧")
    cursor = max(cursor_ids)
    _check(client.wait_terminal(run_id) == "COMPLETED", "重连场景应 COMPLETED")

    replayed = _frames(client.events(run_id, last_event_id=str(cursor)))
    replay_ids = [f["id"] for f in replayed]
    _check(
        replay_ids == list(range(cursor + 1, cursor + 1 + len(replay_ids))),
        f"重连后续传序号不连续: cursor={cursor}, ids={replay_ids}",
    )
    replay_names = [f["event"] for f in replayed]
    _check(replay_names.count("run.completed") == 1, "重连后 run.completed 恰好一次")
    _check("answer.completed" in replay_names, "重连后答案事件丢失")

    # -- 4b. 运行中取消 --
    run = client.create_run(session_id, "发热怎么办", key="demo-4b-cancel")
    cancel_run_id = run["run_id"]
    client.wait_state(cancel_run_id, "RETRIEVING")
    cancelled = client.cancel(cancel_run_id)
    _check(
        cancelled["state"] == "CANCELLED",
        f"取消后状态应 CANCELLED: {cancelled['state']}",
    )
    _check(client.wait_terminal(cancel_run_id) == "CANCELLED", "终态应保持 CANCELLED")
    frames = _frames(client.events(cancel_run_id))
    names = [f["event"] for f in frames]
    _check(
        names[-1] == "run.completed", f"取消运行的末帧应是 run.completed: {names[-1]}"
    )
    _check("answer.completed" not in names, "已取消运行不得有答案帧")


def _assert_acl_denied(client: DemoClient, target_run_id: str, who: str) -> None:
    """凭据对目标运行的 GET 与 events GET 都必须 403，且响应零流字节。"""
    res = client._client.get(f"{client.base}/api/v1/agent/runs/{target_run_id}")
    _check(res.status_code == 403, f"{who}读运行应 403: {res.status_code}")
    res = client._client.get(f"{client.base}/api/v1/agent/runs/{target_run_id}/events")
    _check(res.status_code == 403, f"{who}读事件流应 403: {res.status_code}")
    _check("data:" not in res.text, f"{who}的 403 响应不得携带任何流字节")


def scenario_5_isolation(client: DemoClient, session_id: str) -> None:
    """场景 5：租户/设备隔离 — 知识按租户发布、运行按设备属主。"""
    # 自建一个 t1/d1 的目标运行（隔离探测的对象）
    target = client.create_run(session_id, "眼睛不适", key="demo-5-target")
    client.wait_terminal(target["run_id"])
    target_run_id = target["run_id"]

    # 5a. 异租户：知识未对 t2 发布 → 无证据拒答（锁定具体拒绝标记，部署
    # 故障之类的普通 FAILED 不能冒充隔离成功）
    other = DemoClient(client.base, TOKEN_OTHER_TENANT)
    try:
        session_t2 = other.create_session()
        run = other.create_run(session_t2, "发热怎么办", key="demo-5a-tenant")
        _check(
            other.wait_terminal(run["run_id"]) == "FAILED",
            "异租户有知识问题应 FAILED（隔离）",
        )
        frames = _frames(other.events(run["run_id"]))
        names = [f["event"] for f in frames]
        _check("answer.completed" not in names, "异租户不得拿到 t1 的知识答案")
        result = next(f for f in frames if f["event"] == "run.completed")["data"][
            "data"
        ].get("result")
        _check(
            result == "refused_no_answer",
            f"异租户拒绝标记应为 refused_no_answer: {result}",
        )
        # 5b. 异租户凭据直接读 t1/d1 的运行与事件流：必须 403 且零流字节
        _assert_acl_denied(other, target_run_id, "异租户")
    finally:
        other.close()

    # 5c. 同租户异设备：访问 d1 的运行必须 403 且零流字节
    same_tenant = DemoClient(client.base, TOKEN_SAME_TENANT)
    try:
        _assert_acl_denied(same_tenant, target_run_id, "异设备")
    finally:
        same_tenant.close()


def verify_tool_quota_audit(audit: _AuditCapture, run_id: str) -> None:
    """工具配额审计：真实网关审计 sink 记录了结构化 tool.invoke 记录。"""
    records = [r for r in audit.lines if r.get("action") == "tool.invoke"]
    _check(bool(records), "gcmw.audit 无 tool.invoke 审计记录")
    _check(
        any(r.get("result", "").startswith("ok:") for r in records),
        "无成功的工具调用审计",
    )
    mine = [r for r in records if r.get("run_id") == run_id]
    _check(bool(mine), "场景 1 的运行无工具审计记录")
    for r in mine:
        _check(
            r.get("tenant_id") == DEMO_TENANT
            and r.get("actor_type") == "agent"
            and r.get("session_id_hash")
            and r.get("actor_id_hash"),
            f"工具审计记录字段不全: {sorted(r)}",
        )


# -- 启动与编排 ------------------------------------------------------------------


def start_server() -> SimpleNamespace:
    """进程内启动真实 uvicorn + demo 装配（合成知识、MockProvider、零出网）。"""
    import uvicorn

    settings = Settings(
        environment="test",
        active_provider="mock",
        rate_limit_tenant_per_minute=10_000,
        rate_limit_device_per_minute=10_000,
        rate_limit_session_per_minute=10_000,
    )
    import os

    env_name = settings.auth_credentials_env
    previous = os.environ.get(env_name)
    os.environ[env_name] = json.dumps(
        [
            {"tenant_id": "t1", "device_id": "d1", "token": TOKEN_PRIMARY},
            {"tenant_id": "t1", "device_id": "d2", "token": TOKEN_SAME_TENANT},
            {"tenant_id": "t2", "device_id": "d9", "token": TOKEN_OTHER_TENANT},
        ]
    )
    repository = _DemoStallRepository(DEMO_STALL_S)
    app = create_app(
        settings=settings,
        repository_factory=lambda _settings: repository,
        agent_executor=True,  # demo 栈：合成已审核知识 + MockProvider
    )

    # 认证走真实 Bearer 头（entry guard 先于路由生效），不再覆盖依赖
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not instance.started and time.time() < deadline:
        time.sleep(0.05)
    if not instance.started:
        raise RuntimeError("uvicorn 未能在 30s 内启动")
    port = instance.servers[0].sockets[0].getsockname()[1]
    return SimpleNamespace(
        app=app,
        instance=instance,
        thread=thread,
        base=f"http://127.0.0.1:{port}",
        _env_name=env_name,
        _env_previous=previous,
    )


def stop_server(server: SimpleNamespace) -> None:
    import os

    server.instance.should_exit = True
    server.thread.join(timeout=15)
    if server._env_previous is None:
        os.environ.pop(server._env_name, None)
    else:
        os.environ[server._env_name] = server._env_previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--serve",
        action="store_true",
        help="场景全过后保持服务运行（Ctrl+C 结束），供人工演示",
    )
    args = parser.parse_args()

    logging.getLogger("gcmw.audit").setLevel(logging.WARNING)
    audit = _AuditCapture()
    logging.getLogger("gcmw.audit").addHandler(audit)
    logging.getLogger("gcmw.audit").propagate = False
    # 取消场景会按设计产生一条 executor 固定类别日志（category=cancelled），
    # 属预期行为 — 演示输出里静默它
    logging.getLogger("app.orchestration.executor").setLevel(logging.ERROR)

    server = start_server()
    print(f"[demo] 服务已启动: {server.base}  (MockProvider · 合成数据 · 零出网)")

    results: list[tuple[str, str]] = []

    def record(name: str, fn) -> None:
        try:
            fn()
            results.append((name, "PASS"))
            print(f"[demo] 场景 {name}: PASS")
        except Exception as exc:  # noqa: BLE001 — 收集后统一非零退出
            results.append((name, "FAIL"))
            print(f"[demo] 场景 {name}: FAIL — {exc}")

    primary = DemoClient(server.base, TOKEN_PRIMARY)
    try:
        session_id = primary.create_session()

        run_1_id: str = ""

        def _s1() -> None:
            nonlocal run_1_id
            run_1_id = scenario_1_cited_answer(primary, session_id)
            verify_tool_quota_audit(audit, run_1_id)

        record("1 有引用回答（双层SSE/引用/溯源/审计）", _s1)
        record(
            "2 无证据拒答", lambda: scenario_2_no_evidence_refusal(primary, session_id)
        )
        record("3 红旗转人工", lambda: scenario_3_red_flag_handoff(primary, session_id))
        record(
            "4 取消+断线重连", lambda: scenario_4_cancel_and_resume(primary, session_id)
        )
        record("5 租户/设备隔离", lambda: scenario_5_isolation(primary, session_id))
    finally:
        primary.close()
        if not args.serve:
            stop_server(server)

    print("\n===== 后端 MVP 五场景验收 =====")
    failed = [r for r in results if r[1] != "PASS"]
    for name, status in results:
        print(f"  {status:<6} {name}")
    if failed:
        print(
            f"\n结果: {len(results) - len(failed)}/{len(results)} 通过 — 存在失败，非零退出"
        )
        return 1
    print(f"\n结果: {len(results)}/{len(results)} 全部通过")

    if args.serve:
        print(f"[demo] 服务保持运行: {server.base} （Ctrl+C 结束）")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
