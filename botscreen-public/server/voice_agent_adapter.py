"""Voice Agent compatibility adapter — the fastest voice-to-Agent slice.

Bridges the legacy voice flow (M260C + 讯飞 AIUI → ROS → ``POST /chat`` on
this 8000 service) to the NEW verified Agent backend (8001):
Manager → MedicalQA/RAG → Verifier → SSE.

Design contract (do not weaken):

* the legacy ``/chat`` request/response shape is unchanged; only WHERE the
  answer comes from switches, behind an explicit ``GCMW_VOICE_ANSWER_BACKEND``
  switch (default ``legacy`` = byte-for-byte old behaviour);
* Agent mode FAILS CLOSED: any fault returns a fixed safe copy with
  ``source="error"`` — there is NO silent fallback to local KB or DeepSeek;
* an answer is only delivered when the full delivery gate passes: non-empty
  accumulated deltas + a legal ``answer.completed`` (non-empty citations, each
  with source_id / knowledge_version / 64-hex lowercase content_hash) +
  ``run.completed`` with ``result == "answered"``. Anything else discards the
  partial answer;
* the credential lives ONLY in the ``Authorization: Bearer`` header — never in
  URLs, logs, exception texts, or anything this module returns.

Timeout model (fourth review round — replaces the abandoned threadpool):

* ONE absolute monotonic deadline covers the whole voice turn. It is generated
  at ``ask()`` entry and enforced as a REAL cancellation domain: the turn runs
  inside ``asyncio.timeout_at(deadline)`` over an ``httpx.AsyncClient``. When
  the deadline fires, the in-flight request coroutine is actually cancelled
  and its connection is closed on context exit — there is NO worker thread
  whose execution keeps running past the caller's return.
* the client-side business deadline is a WAITING hard bound only. A
  distributed POST that hits it has an UNKNOWN outcome: the server may have
  committed. Client cancellation can never PROVE "nothing was written", so
  result-unknown is reconciled explicitly:
  - the Session POST carries a per-turn ``idempotency_key`` (minted BEFORE the
    request). During cleanup, the creation is replayed with the SAME key — the
    server's session idempotency contract returns the already-committed (or
    freshly created) session, which is then deleted along its cascade.
    Retrying with a NEW key is forbidden: it would create a second session.
  - the Run POST key is likewise minted before the request and reused by any
    retry of the same turn; if the Session id is known, deleting the Session
    cascades to any possibly-committed Run.
* cleanup is BOUNDED and genuinely cancellable, under its own
  ``_CLEANUP_TOTAL_GRACE_S`` deadline. Semantics are EXPLICIT: on SUCCESS
  nothing is deleted — the temporary Session (and its Run/events) is left for
  the server's ACTIVE TTL sweeper to reclaim (the audit record survives until
  TTL); on FAILURE the Session is deleted, which CASCADES to the Run and
  events (transient voice turns are not meant to be auditable). A cleanup that
  hits its own deadline is cancelled mid-flight; the server-side TTL remains
  the final backstop. A cleanup failure never changes the already-decided safe
  return;
* one throwaway Agent Session per voice question (no multi-turn voice memory
  in this slice);
* configuration errors report FIELD NAMES and fixed reasons only — the
  rejected raw value is never echoed (it may be a misplaced secret);
* logging: this module logs nothing — no response bodies, no question text,
  no credentials.

Lifecycle: every turn creates its own AsyncClient inside an ``async with`` and
closes it before ``ask()`` returns; no module-level executor, thread, task, or
connection pool survives a turn. The sync surface is preserved on purpose:
the legacy ``/chat`` endpoint runs in FastAPI's threadpool and the ROS nodes
are plain scripts — ``ask()`` bridges to async internally via ``asyncio.run``.

Still NOT included: multi-turn voice memory, streaming ASR/TTS, barge-in
playback interruption, real knowledge base, production-durable auditing.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

# ============ 固定契约常量 ============

#: 第一版演示契约：语音只允许打这个精确地址（与前端 CSP/契约一致）
DEMO_CONTRACT_BASE_URL = "http://127.0.0.1:8001/api/v1"
PLACEHOLDER_CREDENTIAL = "YOUR_DEMO_CREDENTIAL_HERE"

_CONNECT_TIMEOUT_S = 5.0
_READ_TIMEOUT_S = 30.0
#: 业务预算：整个语音问答（Session + Run + SSE 流）从 ask() 入口起算的
#: 墙钟硬上界——以真正的 asyncio 取消域执行，不是"停止等待"
_TOTAL_DEADLINE_S = 60.0
#: 清理总 grace：失败路径上"幂等对账 + 取消 Run + 删除 Session"共享的
#: 墙钟上界；到点即真取消（可中断），服务端 TTL 是最终兜底
_CLEANUP_TOTAL_GRACE_S = 4.0
#: 清理单请求预算（≤ 清理总 grace 的一半，保证两个请求都能被调度）
_CLEANUP_GRACE_S = 2.0
#: ROS1/ROS2 调用方的端到端超时 = 业务预算 + 清理总 grace + 传输余量。
#: 两节点必须共用本常量（调用方超时若小于适配器预算，旧请求未结束就允许
#: 下一次唤醒，会产生并发与答案次序问题）。勿在节点里另行写死。
VOICE_TURN_TIMEOUT_S = _TOTAL_DEADLINE_S + _CLEANUP_TOTAL_GRACE_S + 5.0

#: 64 位小写十六进制（引用三元组的 content_hash 门槛）
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

#: 固定文案 —— 绝不携带服务端异常原文或凭据
COPY_REFUSED = "目前没有足够可靠的资料回答这个问题，建议咨询专业人员。"
COPY_ESCALATED = "这个问题需要人工或专业人员进一步协助。"
COPY_CANCELLED = "本次问答已取消。"
COPY_TIMEOUT = "本次处理超时，请稍后重试。"
COPY_UNAVAILABLE = "问答服务暂时不可用，请稍后再试。"

_TERMINAL_COPY = {
    "refused_no_answer": COPY_REFUSED,
    "escalated_to_human": COPY_ESCALATED,
    "cancelled": COPY_CANCELLED,
    "deadline_exceeded": COPY_TIMEOUT,
}


class VoiceAgentConfigError(RuntimeError):
    """启动期配置错误：只报告字段名与固定原因，绝不回显原始值
    （错误值可能是误填进配置的机密串）。"""


class VoiceAgentBackendError(RuntimeError):
    """运行期故障（网络/HTTP/畸形流/未知终态/超时）：失败关闭。

    文本中绝不包含凭据或服务端响应正文。
    """


# ============ 配置 ============


@dataclass(frozen=True)
class VoiceAgentConfig:
    """解析后的语音回答后端配置（mode=legacy 时 base_url/credential 为空）。"""

    mode: str
    base_url: str
    credential: str


def resolve_voice_agent_config(env: dict[str, str] | None = None) -> VoiceAgentConfig:
    """解析 GCMW_VOICE_* 配置。

    ``env`` 缺省读真实环境变量；测试可传 dict。非法值一律 raise
    :class:`VoiceAgentConfigError`，异常文本只含字段名与固定原因——
    **绝不回显非法 mode/base/credential 的原始值**。调用方（qa_server）
    在启动期把它变成显式退出，绝不静默回退 legacy。
    """
    env = os.environ if env is None else env
    mode = (env.get("GCMW_VOICE_ANSWER_BACKEND", "legacy") or "legacy").strip().lower()
    if mode not in ("legacy", "agent"):
        raise VoiceAgentConfigError(
            "GCMW_VOICE_ANSWER_BACKEND 配置非法：只允许 legacy 或 agent"
        )
    if mode == "legacy":
        return VoiceAgentConfig(mode="legacy", base_url="", credential="")
    base = (env.get("GCMW_VOICE_AGENT_API_BASE", "") or "").strip().rstrip("/")
    if base != DEMO_CONTRACT_BASE_URL:
        raise VoiceAgentConfigError(
            f"GCMW_VOICE_AGENT_API_BASE 配置非法：必须精确等于 {DEMO_CONTRACT_BASE_URL}"
        )
    credential = (env.get("GCMW_VOICE_AGENT_CREDENTIAL", "") or "").strip()
    if not credential or credential == PLACEHOLDER_CREDENTIAL:
        raise VoiceAgentConfigError(
            "GCMW_VOICE_AGENT_CREDENTIAL 配置非法：缺失、为空或仍是占位符，"
            "agent 模式拒绝启动"
        )
    return VoiceAgentConfig(mode="agent", base_url=base, credential=credential)


# ============ SSE 帧解析（流式 UTF-8 安全） ============


class SseFrameParser:
    """增量 SSE 解析：跨 chunk 的多字节 UTF-8、\\n\\n 与 \\r\\n\\r\\n、
    ``:`` 注释行、多行 ``data:``、EOF flush 全部覆盖。

    ``feed(chunk)`` / ``flush()`` 返回完整帧列表（dict，键 event/data）。
    终态后同一 chunk 内的迟到帧由调用方丢弃——本解析器只负责成帧，不做
    语义拦截。
    """

    def __init__(self) -> None:
        # 单一增量解码器跨 chunk 复用：中文字节被 TCP 拆开也不会产生 U+FFFD
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[dict[str, str]]:
        return self._consume(self._decoder.decode(chunk))

    def flush(self) -> list[dict[str, str]]:
        # EOF：冲掉解码器里未完成的多字节序列，再解析残留帧
        tail = self._decoder.decode(b"", final=True)
        frames = self._consume(tail)
        if self._buffer.strip():
            last = self._parse_block(self._buffer)
            if last is not None:
                frames.append(last)
        self._buffer = ""
        return frames

    def _consume(self, text: str) -> list[dict[str, str]]:
        # CRLF 归一化必须在拼接之后做：\r 与 \n 可能分属两次 feed
        # （residue 体积受限于未完结的单个帧，重新扫描成本可忽略）
        self._buffer = (self._buffer + text).replace("\r\n", "\n")
        *complete, self._buffer = self._buffer.split("\n\n")
        frames = [self._parse_block(block) for block in complete]
        return [f for f in frames if f is not None]

    @staticmethod
    def _parse_block(block: str) -> dict[str, str] | None:
        block = block.lstrip("\n")
        if not block.strip():
            return None
        data_lines: list[str] = []
        event = ""
        is_comment = True
        for line in block.split("\n"):
            if line.startswith(":"):
                continue  # keep-alive 注释
            is_comment = False
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip(" "))
            # 其余字段（id:/retry:）与交付无关，忽略
        if is_comment:
            return None
        return {"event": event, "data": "\n".join(data_lines)}


# ============ 客户端 ============


@dataclass
class _TurnState:
    """One voice turn's mutable bookkeeping (idempotency + result-unknown)."""

    session_id: str | None = None
    run_id: str | None = None
    #: the Session POST was ISSUED — the server may have committed it even if
    #: the client never saw the response
    session_sent: bool = False
    #: the Run POST was issued — cascaded by the Session delete
    run_sent: bool = False


class VoiceAgentClient:
    """同步外观的 Agent 客户端：一次语音问题 = 一个临时 Session + 一个 Run。

    ``ask()`` 保持同步（FastAPI sync 路由线程池 + ROS 纯脚本调用方），
    内部经 ``asyncio.run()`` 进入真正的取消域：每轮一个 ``httpx.AsyncClient``、
    一条贯穿 Session/Run/SSE 的绝对 deadline（``asyncio.timeout_at``）。
    ``http`` 可注入异步测试替身（零网络）；返回值只有
    ``(固定文案或已核验答案, "agent"|"error")`` —— 不向外暴露响应正文、
    异常原文或凭据。
    """

    def __init__(self, config: VoiceAgentConfig, http: Any = None) -> None:
        self._cfg = config
        # None → 每轮真实 httpx.AsyncClient；否则为异步测试替身
        # （须提供 async post/get/delete 与 stream() 异步上下文管理器）
        self._http = http

    # ---------- 公共入口 ----------

    def ask(self, question: str) -> tuple[str, str]:
        """返回 (robot_answer, source)；source ∈ {"agent", "error"}。

        绝对墙钟 deadline 在本入口生成，贯穿 Session/Run/SSE 全部阶段，
        由 asyncio 取消域真实执行：到点即取消在途请求并关闭连接。
        """
        try:
            return asyncio.run(self._ask_async(question))
        except VoiceAgentBackendError:
            return COPY_UNAVAILABLE, "error"
        except Exception:  # noqa: BLE001 — 兜底失败关闭是安全要求，不是疏忽
            # 任何未预期异常（含事件循环层故障）都失败关闭，
            # 绝不携带原始错误文本
            return COPY_UNAVAILABLE, "error"

    # ---------- 异步主体 ----------

    async def _ask_async(self, question: str) -> tuple[str, str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TOTAL_DEADLINE_S
        # 幂等键在任何受保护请求发出之前生成；同一轮的任何重试必须复用
        session_key = f"voice-session-{uuid.uuid4().hex}"
        run_key = f"voice-run-{uuid.uuid4().hex}"
        state = _TurnState()

        own_client = self._http is None
        client = (
            httpx.AsyncClient(timeout=self._httpx_timeout(_TOTAL_DEADLINE_S))
            if own_client
            else self._http
        )
        reached_terminal = False
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    session_id = await self._create_session(
                        client, state, session_key, deadline
                    )
                    state.session_id = session_id
                    run_id = await self._create_run(
                        client, state, session_id, run_key, question, deadline
                    )
                    state.run_id = run_id
                    answer = await self._collect(client, run_id, deadline)
                reached_terminal = True
                return answer, "agent"
            except TimeoutError:
                # 取消域到点：在途请求已被真实取消。交给 finally 清理
                # （结果未知的 POST 用幂等键对账）
                raise VoiceAgentBackendError("absolute deadline exceeded") from None
            except VoiceAgentBackendError:
                raise
            except Exception as exc:
                raise VoiceAgentBackendError("unexpected client failure") from exc
            finally:
                # 结果已确定的异常路径也要走有界清理；此处的 return 不受
                # 清理结果影响（清理异常在 _bounded_cleanup 内部吞掉）
                if not reached_terminal:
                    await self._bounded_cleanup(client, state, session_key)
        finally:
            if own_client:
                await client.aclose()  # 每轮关闭，退出后零残留连接

    # ---------- 内部工具 ----------

    @staticmethod
    def _httpx_timeout(remaining: float) -> httpx.Timeout:
        """Per-phase HTTP timeouts; the connect budget also shrinks with the
        remaining wall clock (a sub-second budget must not spend a full 5s
        connect). The absolute deadline on top is the REAL hard bound."""
        connect = min(_CONNECT_TIMEOUT_S, max(remaining, 0.05))
        read = min(_READ_TIMEOUT_S, max(remaining, 0.05))
        return httpx.Timeout(connect=connect, read=read, write=read, pool=read)

    @staticmethod
    def _remaining_timeout(deadline: float) -> httpx.Timeout:
        # the deadline lives on the loop clock, which on the default event loop
        # IS time.monotonic() — same epoch, so the subtraction is sound
        return VoiceAgentClient._httpx_timeout(deadline - time.monotonic())

    def _headers(self) -> dict[str, str]:
        # 凭据只出现在这里
        return {
            "Authorization": f"Bearer {self._cfg.credential}",
            "Content-Type": "application/json",
        }

    # ---------- Agent 调用序列 ----------

    async def _create_session(
        self, client: Any, state: _TurnState, session_key: str, deadline: float
    ) -> str:
        state.session_sent = True  # 先置位：请求在途即视为"结果可能已提交"
        try:
            resp = await client.post(
                f"{self._cfg.base_url}/sessions",
                json={
                    "channel": "text",
                    "locale": "zh-CN",
                    "idempotency_key": session_key,
                },
                headers=self._headers(),
                timeout=self._remaining_timeout(deadline),
            )
        except VoiceAgentBackendError:
            raise
        except Exception as exc:
            raise VoiceAgentBackendError("session request failed") from exc
        if resp.status_code != 201:  # 契约严格锁定：session 创建返回 201
            raise VoiceAgentBackendError(f"session http {resp.status_code}")
        try:
            session_id = resp.json().get("session_id")
        except ValueError as exc:
            raise VoiceAgentBackendError("session malformed json") from exc
        if not isinstance(session_id, str) or not session_id:
            raise VoiceAgentBackendError("session malformed payload")
        return session_id

    async def _create_run(
        self,
        client: Any,
        state: _TurnState,
        session_id: str,
        run_key: str,
        question: str,
        deadline: float,
    ) -> str:
        state.run_sent = True  # 先置位：同轮重试必须复用同一 key
        try:
            resp = await client.post(
                f"{self._cfg.base_url}/agent/runs",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": question},
                    "idempotency_key": run_key,
                },
                headers=self._headers(),
                timeout=self._remaining_timeout(deadline),
            )
        except VoiceAgentBackendError:
            raise
        except Exception as exc:
            raise VoiceAgentBackendError("run request failed") from exc
        if resp.status_code != 200:
            raise VoiceAgentBackendError(f"run http {resp.status_code}")
        try:
            run_id = resp.json().get("run_id")
        except ValueError as exc:
            raise VoiceAgentBackendError("run malformed json") from exc
        if not isinstance(run_id, str) or not run_id:
            raise VoiceAgentBackendError("run malformed payload")
        return run_id

    # ---------- 有界清理（可真取消） ----------

    async def _bounded_cleanup(
        self, client: Any, state: _TurnState, session_key: str
    ) -> None:
        """失败路径的有界 best-effort 清理，共享一个总期限。

        到点后**真取消**在途清理请求（asyncio 取消域，不是"停止等待"）；
        服务端主动 TTL sweeper 是未回收残留的最终兜底。清理失败/超时
        绝不改变已确定的返回。
        """
        loop = asyncio.get_running_loop()
        cleanup_deadline = loop.time() + _CLEANUP_TOTAL_GRACE_S
        try:
            async with asyncio.timeout_at(cleanup_deadline):
                session_id = state.session_id
                # (1) Session 结果未知 → 用同一 idempotency_key 重放对账。
                #     禁止换新 key：那会创建第二个 Session。
                if session_id is None and state.session_sent:
                    session_id = await self._reconcile_session(client, session_key)
                    state.session_id = session_id
                # (2) 已知 Run：显式有界取消
                if state.run_id is not None:
                    await self._best_effort_delete(
                        client, f"/agent/runs/{state.run_id}"
                    )
                # (3) 删除 Session：级联回收一切可能已提交的 Run 与事件
                if session_id is not None:
                    await self._best_effort_delete(client, f"/sessions/{session_id}")
        except (TimeoutError, asyncio.CancelledError):
            pass  # 清理到点被真取消：有界放弃（TTL 兜底），设计如此
        except Exception:  # noqa: BLE001, S110 — best-effort：静默是设计
            pass  # 非 2xx / 网络错误：不影响已确定的返回

    async def _reconcile_session(self, client: Any, session_key: str) -> str | None:
        """结果未知时按同一 idempotency_key 重放 Session 创建。

        服务端幂等契约保证：已提交 → 返回原 Session（不建第二个）；
        未提交 → 这次创建后随即被删除。返回拿到的 session_id 或 None。
        """
        try:
            resp = await client.post(
                f"{self._cfg.base_url}/sessions",
                json={
                    "channel": "text",
                    "locale": "zh-CN",
                    "idempotency_key": session_key,
                },
                headers=self._headers(),
                timeout=self._httpx_timeout(_CLEANUP_GRACE_S),
            )
        except Exception:  # noqa: BLE001 — 对账失败：TTL 兜底
            return None
        if resp.status_code != 201:
            return None
        try:
            session_id = resp.json().get("session_id")
        except ValueError:
            return None
        return session_id if isinstance(session_id, str) and session_id else None

    async def _best_effort_delete(self, client: Any, path: str) -> None:
        try:
            await client.delete(
                f"{self._cfg.base_url}{path}",
                headers=self._headers(),
                timeout=self._httpx_timeout(_CLEANUP_GRACE_S),
            )
        except Exception:  # noqa: BLE001, S110 — best-effort：静默是设计
            pass  # 非 2xx / 网络错误 / 到点取消：不影响已确定的返回

    # ---------- SSE 收集与交付门槛 ----------

    async def _collect(self, client: Any, run_id: str, deadline: float) -> str:
        state = _CollectState()
        try:
            async with client.stream(
                "GET",
                f"{self._cfg.base_url}/agent/runs/{run_id}/events",
                headers={**self._headers(), "Accept": "text/event-stream"},
                timeout=self._remaining_timeout(deadline),
            ) as resp:
                if resp.status_code != 200:
                    raise VoiceAgentBackendError(f"events http {resp.status_code}")
                async for chunk in resp.aiter_bytes():
                    for frame in state.parser.feed(chunk):
                        state.apply(frame)
                        if state.finished():
                            break
                    if state.finished():
                        break
                else:
                    # 流正常结束（无 break）：先 EOF flush，再判定是否缺失终态
                    for frame in state.parser.flush():
                        state.apply(frame)
                        if state.finished():
                            break
        except VoiceAgentBackendError:
            raise
        except (TimeoutError, asyncio.CancelledError):
            # 取消域到点（流读取被真实取消）：按超时失败关闭
            raise VoiceAgentBackendError("absolute deadline exceeded") from None
        except Exception as exc:
            raise VoiceAgentBackendError("stream read failed") from exc

        if state.failed:
            raise VoiceAgentBackendError("stream failed (stream.error/malformed)")
        if state.terminal is None:
            # EOF 前没有终态：失败关闭（含"响应提前结束"）
            raise VoiceAgentBackendError("stream ended without terminal")

        # ---- 终态映射 ----
        if state.terminal == "answered":
            if not state.verified or not "".join(state.parts).strip():
                raise VoiceAgentBackendError("answered without verified gate")
            return "".join(state.parts)
        copy = _TERMINAL_COPY.get(state.terminal)
        if copy is None:
            raise VoiceAgentBackendError("unknown terminal result")
        return copy


class _CollectState:
    """一次 SSE 收集的可变状态机（frame → 状态回写；终态后停止派发）。"""

    __slots__ = ("failed", "parser", "parts", "terminal", "verified")

    def __init__(self) -> None:
        self.parser = SseFrameParser()
        self.parts: list[str] = []
        self.verified = False
        self.terminal: str | None = None
        self.failed = False

    def finished(self) -> bool:
        return self.terminal is not None or self.failed

    def apply(self, frame: dict[str, str]) -> None:
        if self.finished():
            return  # 终态后同 chunk 的迟到帧：一律丢弃
        event = frame.get("event", "")
        raw = frame.get("data", "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except ValueError:
            self.failed = True  # malformed JSON → 失败关闭
            return
        if not isinstance(payload, dict):
            self.failed = True
            return
        business = payload.get("data")
        if not isinstance(business, dict):
            # SSEEvent 信封：业务载荷必须位于内层 data
            self.failed = True
            return

        if event == "answer.delta":
            delta = business.get("delta")
            if isinstance(delta, str) and delta:
                self.parts.append(delta)
        elif event == "answer.completed":
            self.verified = _citations_valid(business.get("citations"))
        elif event == "run.completed":
            result = business.get("result")
            if isinstance(result, str) and result:
                self.terminal = result
            else:
                self.failed = True
        elif event == "stream.error":
            self.failed = True


def _citations_valid(citations: Any) -> bool:
    """引用三元组门槛：非空数组，每条 source_id/knowledge_version 非空、
    content_hash 为 64 位小写十六进制。"""
    if not isinstance(citations, list) or not citations:
        return False
    for c in citations:
        if not isinstance(c, dict):
            return False
        if not isinstance(c.get("source_id"), str) or not c["source_id"]:
            return False
        if (
            not isinstance(c.get("knowledge_version"), str)
            or not c["knowledge_version"]
        ):
            return False
        if not isinstance(c.get("content_hash"), str) or not _HASH_RE.match(
            c["content_hash"]
        ):
            return False
    return True


# ============ 便捷入口 ============


def ask(question: str, config: VoiceAgentConfig) -> tuple[str, str]:
    """模块级便捷入口：qa_server 的 /chat 在 agent 模式下调这一个函数。"""
    return VoiceAgentClient(config).ask(question)
