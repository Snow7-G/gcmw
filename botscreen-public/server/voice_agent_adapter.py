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
  URLs, logs, exception texts, or anything this module returns;
* ONE absolute deadline covers the whole voice turn: it is generated at
  ``ask()`` entry and propagated as the remaining budget into session
  creation, run creation, and the SSE stream; a blocking read is actively
  interrupted by a watchdog that closes the connection at the deadline —
  a silent server cannot stretch the turn past the budget;
* cleanup is BOUNDED best-effort: each cleanup request gets its own short
  grace (independent of the business budget), success terminal does not
  delete the Run (the server already finished it), and a cleanup failure
  never changes the already-decided safe return;
* one throwaway Agent Session per voice question (no multi-turn voice memory
  in this slice);
* configuration errors report FIELD NAMES and fixed reasons only — the
  rejected raw value is never echoed (it may be a misplaced secret).

This module is deliberately SYNC (``requests``): the legacy ``/chat`` endpoint
runs in FastAPI's threadpool and the ROS nodes are plain scripts.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

# ============ 固定契约常量 ============

#: 第一版演示契约：语音只允许打这个精确地址（与前端 CSP/契约一致）
DEMO_CONTRACT_BASE_URL = "http://127.0.0.1:8001/api/v1"
PLACEHOLDER_CREDENTIAL = "YOUR_DEMO_CREDENTIAL_HERE"

_CONNECT_TIMEOUT_S = 5.0
_READ_TIMEOUT_S = 30.0
#: 业务预算：整个语音问答（Session + Run + SSE 流）从 ask() 入口起算
_TOTAL_DEADLINE_S = 60.0
#: 清理预算：每个清理请求独立、短且总量有界（与业务预算无关）
_CLEANUP_GRACE_S = 2.0
#: ROS1/ROS2 调用方的端到端超时 = 业务预算 + 清理 grace + 传输余量。
#: 两节点必须共用本常量（调用方超时若小于适配器预算，旧请求未结束就允许
#: 下一次唤醒，会产生并发与答案次序问题）。勿在节点里另行写死。
VOICE_TURN_TIMEOUT_S = _TOTAL_DEADLINE_S + _CLEANUP_GRACE_S + 5.0

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


class VoiceAgentClient:
    """同步 Agent 客户端：一次语音问题 = 一个临时 Session + 一个 Run。

    ``http`` 可注入（测试零网络）；缺省用 :mod:`requests`。返回值只有
    ``(固定文案或已核验答案, "agent"|"error")`` —— 不向外暴露响应正文、
    异常原文或凭据。
    """

    def __init__(self, config: VoiceAgentConfig, http: Any = None) -> None:
        self._cfg = config
        self._http = http if http is not None else requests

    # ---------- 公共入口 ----------

    def ask(self, question: str) -> tuple[str, str]:
        """返回 (robot_answer, source)；source ∈ {"agent", "error"}。

        绝对 deadline 在本入口生成，贯穿 Session/Run/SSE 全部阶段。
        """
        deadline = time.monotonic() + _TOTAL_DEADLINE_S
        session_id: str | None = None
        run_id: str | None = None
        reached_terminal = False
        try:
            session_id = self._create_session(deadline)
            run_id = self._create_run(session_id, question, deadline)
            answer = self._collect(run_id, deadline)
            reached_terminal = True
            return answer, "agent"
        except VoiceAgentBackendError:
            return COPY_UNAVAILABLE, "error"
        except Exception:  # noqa: BLE001 — 兜底失败关闭是安全要求，不是疏忽
            # 任何未预期异常都失败关闭，绝不携带原始错误文本
            return COPY_UNAVAILABLE, "error"
        finally:
            # 有界 best-effort 清理（每个请求独立短 grace，总量有界；
            # 清理失败不改变已确定的返回）：
            # - 成功终态不删 Run：服务端已完成该 Run，删除没有意义；
            # - 失败路径取消 Run（可能仍在执行）；Session 总是删除。
            if run_id is not None and not reached_terminal:
                self._best_effort_delete(f"/agent/runs/{run_id}")
            if session_id is not None:
                self._best_effort_delete(f"/sessions/{session_id}")

    # ---------- 内部工具 ----------

    @staticmethod
    def _remaining_or_raise(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VoiceAgentBackendError("absolute deadline exceeded")
        return remaining

    def _headers(self) -> dict[str, str]:
        # 凭据只出现在这里
        return {
            "Authorization": f"Bearer {self._cfg.credential}",
            "Content-Type": "application/json",
        }

    # ---------- Agent 调用序列 ----------

    def _create_session(self, deadline: float) -> str:
        remaining = self._remaining_or_raise(deadline)
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/sessions",
                json={"channel": "text", "locale": "zh-CN"},
                headers=self._headers(),
                timeout=(_CONNECT_TIMEOUT_S, min(_READ_TIMEOUT_S, remaining)),
            )
        except requests.RequestException as exc:
            raise VoiceAgentBackendError("session connect failed") from exc
        if resp.status_code != 201:  # 契约严格锁定：session 创建返回 201
            raise VoiceAgentBackendError(f"session http {resp.status_code}")
        try:
            session_id = resp.json().get("session_id")
        except ValueError as exc:
            raise VoiceAgentBackendError("session malformed json") from exc
        if not isinstance(session_id, str) or not session_id:
            raise VoiceAgentBackendError("session malformed payload")
        return session_id

    def _create_run(self, session_id: str, question: str, deadline: float) -> str:
        remaining = self._remaining_or_raise(deadline)
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/agent/runs",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": question},
                    "idempotency_key": f"voice-{uuid.uuid4()}",
                },
                headers=self._headers(),
                timeout=(_CONNECT_TIMEOUT_S, min(_READ_TIMEOUT_S, remaining)),
            )
        except requests.RequestException as exc:
            raise VoiceAgentBackendError("run connect failed") from exc
        if resp.status_code != 200:
            raise VoiceAgentBackendError(f"run http {resp.status_code}")
        try:
            run_id = resp.json().get("run_id")
        except ValueError as exc:
            raise VoiceAgentBackendError("run malformed json") from exc
        if not isinstance(run_id, str) or not run_id:
            raise VoiceAgentBackendError("run malformed payload")
        return run_id

    def _best_effort_delete(self, path: str) -> None:
        # 独立短 grace：清理失败/悬挂都不改变已确定的返回，也不追加业务预算
        try:
            self._http.delete(
                f"{self._cfg.base_url}{path}",
                headers=self._headers(),
                timeout=(1.0, _CLEANUP_GRACE_S),
            )
        except Exception:  # noqa: BLE001, S110 — best-effort 清理：静默是设计
            pass

    # ---------- SSE 收集与交付门槛 ----------

    def _collect(self, run_id: str, deadline: float) -> str:
        remaining = self._remaining_or_raise(deadline)
        try:
            resp = self._http.get(
                f"{self._cfg.base_url}/agent/runs/{run_id}/events",
                headers={**self._headers(), "Accept": "text/event-stream"},
                stream=True,
                timeout=(_CONNECT_TIMEOUT_S, min(_READ_TIMEOUT_S, remaining)),
            )
        except requests.RequestException as exc:
            raise VoiceAgentBackendError("events connect failed") from exc
        if resp.status_code != 200:
            raise VoiceAgentBackendError(f"events http {resp.status_code}")

        # 看门狗：deadline 一到就主动 close 连接，主动打断阻塞中的
        # iter_content —— 不能指望"等下一个 chunk 再检查"
        watchdog = threading.Timer(max(0.05, deadline - time.monotonic()), resp.close)
        watchdog.daemon = True
        watchdog.start()

        state = _CollectState()
        try:
            for chunk in resp.iter_content(chunk_size=1024):
                if time.monotonic() > deadline:
                    raise VoiceAgentBackendError("absolute deadline exceeded")
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
        except Exception as exc:
            # 看门狗 close 或网络层中断：越过 deadline 记为超时，否则记为流故障
            if time.monotonic() >= deadline:
                raise VoiceAgentBackendError("absolute deadline exceeded") from exc
            raise VoiceAgentBackendError("stream read failed") from exc
        finally:
            watchdog.cancel()
            resp.close()

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
