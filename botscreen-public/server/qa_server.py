"""
互动问答后端服务
- 启动时从 docx 加载知识库，jieba 分词建索引
- 提问先搜知识库 → 命中直接返回 → 未命中调用 DeepSeek
- 支持 session 会话历史（LLM 兜底时带上下文）

启动: python qa_server.py
端口: 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
import zipfile

# ========== jieba 分词 ==========
import jieba
import jieba.analyse as jieba_analyse
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ========== 语音回答后端（legacy | agent 兼容桥，#58 快速切片） ==========
import voice_agent_adapter

# ========== 全局配置 ==========
KB_MATCH_THRESHOLD = 0.22  # 知识库匹配阈值（0~1，低于此分数走 LLM）

STOP_WORDS = {
    "的",
    "了",
    "是",
    "吗",
    "呢",
    "啊",
    "吧",
    "呀",
    "哦",
    "么",
    "嘛",
    "我",
    "你",
    "他",
    "她",
    "它",
    "这",
    "那",
    "什么",
    "怎么",
    "怎样",
    "一个",
    "一下",
    "一些",
    "一点",
    "不",
    "很",
    "都",
    "就",
    "也",
    "还",
    "要",
    "会",
    "能",
    "可以",
    "应该",
    "在",
    "有",
    "和",
    "与",
    "或",
    "对",
    "把",
    "被",
    "让",
    "给",
    "从",
    "到",
    "向",
    "跟",
    "用",
}

# ========== FastAPI 初始化 ==========

#: 只监听回环地址。语音桥（ROS/ROS2 节点）、打包后的 Electron 页面与麦克风
#: 轮询全部发生在这台机器上，没有任何一条链路需要 LAN 可达——把 8000 暴露到
#: 局域网只会平白放大攻击面（这条服务带固定低熵演示凭据）。
HOST = "127.0.0.1"
PORT = 8000

#: 精确 CORS 白名单：打包后的 Electron 渲染进程发 ``Origin: null``，开发前端
#: 走 Vite 的两个本机地址。**不使用通配**，也不允许凭据——页面与本服务的交互
#: 只有 GET/POST 加 ``Content-Type``（ROS/ROS2 直连不带 Origin，不受 CORS 约束）。
ALLOWED_ORIGINS = [
    "null",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
ALLOWED_METHODS = ["GET", "POST"]
ALLOWED_HEADERS = ["Content-Type"]

app = FastAPI(title="互动问答后端", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)

# ================================================
# 第一部分：docx → 知识库解析
# ================================================

# 文档路径（优先环境变量，否则默认 D 盘）
DOCX_PATH = os.environ.get("KB_DOCX_PATH", r"D:\机器人-科普问答.docx")


def extract_paragraphs(docx_path: str) -> list[str]:
    """从 docx 提取所有段落文本"""
    paragraphs: list[str] = []
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            for p in tree.iter(f"{{{ns}}}p"):
                texts: list[str] = []
                for t in p.iter(f"{{{ns}}}t"):
                    if t.text:
                        texts.append(t.text)
                line = "".join(texts).strip()
                if line:
                    paragraphs.append(line)
    except FileNotFoundError:
        print(f"[WARN]  文档不存在: {docx_path}，使用内置知识库")
    except Exception as e:  # noqa: BLE001 — 历史容错：解析失败回退内置知识库
        print(f"[WARN]  文档解析失败: {e}，使用内置知识库")
    return paragraphs


def parse_qa_pairs(paragraphs: list[str]) -> list[dict]:
    """
    将段落解析为 Q&A 列表。

    支持两种格式：
    1. 新格式：「问：xxx\n答：xxx」
    2. 旧格式（对话脚本）：提取「机器人：xxx」作为知识点
    """
    full_text = "\n".join(paragraphs)
    qa_list: list[dict] = []

    # ── 尝试新格式：问 / 答 配对 ──
    qa_pattern = re.compile(
        r"问[：:]\s*(.+?)\s*\n\s*答[：:]\s*(.+?)(?=\n(?:问[：:]|\Z))", re.DOTALL
    )
    matches = qa_pattern.findall(full_text)
    if matches:
        for i, (q, a) in enumerate(matches):
            qa_list.append(
                {
                    "id": f"qa_{i}",
                    "question": q.strip(),
                    "answer": a.strip(),
                }
            )
        if qa_list:
            print(f"   [OK] 解析到 {len(qa_list)} 组 Q&A（新格式）")
            return qa_list

    # ── 旧格式：从对话脚本中提取知识点 ──
    # 匹配「机器人：xxx」作为答案
    # 匹配标题/主题行
    topics: list[dict] = [
        {
            "question": "眼轴长度是多少",
            "answer": (
                "每个小朋友的眼睛里都有一个叫“眼轴”的东西，"
                "它就像眼睛的“身高”一样，会随着你长大而变长～\n\n"
                "不同年龄小朋友的平均眼轴长度：\n"
                "6岁→22.46mm | 7岁→22.56mm | 8岁→22.78mm | 9岁→22.95mm\n"
                "10岁→23.13mm | 11岁→23.26mm | 12岁→23.32mm\n\n"
                "记得每三个月检查一次，建立视觉健康档案哦！📋💕"
            ),
        },
        {
            "question": "近视是怎么分级的",
            "answer": (
                "近视按度数分成三个等级哦～👓\n\n"
                "🟢 轻度近视：-3.00D 以下（不到300度）\n"
                "🟡 中度近视：-3.00D 到 -6.00D（300-600度）\n"
                "🔴 高度近视：-6.00D 以上（超过600度）\n\n"
                "度数越高，视网膜就越脆弱，要好好保护眼睛！💪"
            ),
        },
        {
            "question": "什么是近视",
            "answer": (
                "眼睛就像一台超级精密的照相机📷，帮我们看清美丽的世界～\n\n"
                "但如果这台“照相机”的镜头度数太高了，"
                "我们就看不清远处的东西了，这就是近视。\n\n"
                "科学家发现：度数涨得很高的话，"
                "眼睛里叫“视网膜”的重要零件（就像相机的底片）"
                "就会变得脆弱，更容易生病！\n\n"
                "所以保护眼睛从现在开始吧～🌟"
            ),
        },
        {
            "question": "看多久要休息眼睛",
            "answer": (
                "记住「20-20-20」魔法法则！🪄\n\n"
                "📖 看书、做作业或看电视的时候——\n"
                "⏰ 每 20 分钟左右\n"
                "👀 望向 20 英尺（大约6米）远的地方\n"
                "🕐 至少看 20 秒钟\n\n"
                "就像给眼睛做“课间操”一样，让它放松一下～"
            ),
        },
        {
            "question": "正确读写姿势是什么",
            "answer": (
                "读写姿势记住「一拳一尺一寸」！✊📏\n\n"
                "✊ 一拳：胸口离桌子一个拳头距离\n"
                "📏 一尺：眼睛离书本一尺（约33厘米）\n"
                "☝️ 一寸：手指离笔尖一寸（约3厘米）\n\n"
                "坐直了，眼睛和身体都会感谢你的！😊"
            ),
        },
        {
            "question": "每天要在外面玩多久",
            "answer": (
                "每天去户外运动是保护眼睛的“超级武器”！🌞\n\n"
                "⏰ 中小学生每天至少需要 2 小时的户外运动～\n\n"
                "因为阳光会让身体产生「多巴胺」，"
                "能有效预防近视！\n\n"
                "下课后去操场上跑一跑，看看远处的绿色植物，"
                "眼睛和身体都会棒棒的！🏃‍♂️🌳✨"
            ),
        },
        {
            "question": "看书要开灯吗",
            "answer": (
                "用眼光线很重要！💡\n\n"
                "☀️ 光线太强 → 刺激眼睛不舒服\n"
                "🌙 光线太暗 → 眼睛看不清容易累\n\n"
                "正确做法：\n"
                "[OK] 自然光充足时直接利用\n"
                "[OK] 用台灯时亮度以不刺眼为宜\n"
                "[OK] 台灯 + 房间顶灯同时开\n"
                "[OK] 台灯光线从左前方照过来\n\n"
                "这样书本就不会被阴影挡住啦～📖✨"
            ),
        },
        {
            "question": "吃什么对眼睛好",
            "answer": (
                "好好睡觉 + 好好吃饭 = 眼睛健康！😴🥕\n\n"
                "😴 充足睡眠：晚上是眼睛休息的宝贵时间\n\n"
                "🥗 均衡饮食：\n"
                "[OK] 多吃：鱼肉🐟、水果🍎、蔬菜🥬、豆制品🫘、鸡蛋🥚\n"
                "❌ 少吃：甜食🍰、辛辣食物🌶️、含糖饮料🥤\n\n"
                "维生素A、叶黄素、Omega-3 都是眼睛的好朋友哦～"
            ),
        },
        {
            "question": "为什么要检查眼睛",
            "answer": (
                "每个小朋友都应该有一份“视觉健康档案”！📋\n\n"
                "就像每年量身高体重一样，眼睛也需要定期检查～\n\n"
                "⏰ 建议每 3 个月检查一次\n"
                "📝 记录：视力、眼轴长度、屈光度\n\n"
                "早发现问题早处理，让眼睛健健康康陪你长大！💕"
            ),
        },
    ]
    print(
        f"   [OK] 使用内置知识库（{len(topics)} 条），如需自定义请更新 docx 为 Q&A 格式"
    )
    return topics


# ================================================
# 第二部分：jieba 搜索索引
# ================================================


class KBSearcher:
    """基于 jieba 分词的轻量知识库搜索"""

    def __init__(self, qa_list: list[dict]):
        self.qa_list = qa_list
        # 为每条 Q&A 预处理：分词 + 关键词提取
        self.entries: list[dict] = []
        for entry in qa_list:
            q_words = set(jieba.cut_for_search(entry["question"])) - STOP_WORDS
            # 也从答案中提取关键词增加匹配面
            a_keywords = (
                set(
                    jieba_analyse.extract_tags(
                        entry["answer"], topK=10, withWeight=False
                    )
                )
                - STOP_WORDS
            )
            self.entries.append(
                {
                    **entry,
                    "q_words": q_words,
                    "a_keywords": set(a_keywords),
                    "_all_words": q_words | set(a_keywords),
                }
            )
        print(f"   [IDX] 索引已构建：{len(self.entries)} 条")

    def search(self, question: str) -> tuple[dict | None, float]:
        """
        搜索最佳匹配。

        1. jieba 分词后计算词重叠度
        2. 同时用字符 2-gram 做补充（处理 jieba 对专有名词分不准的情况）
        3. 取两种方式中较高的分数
        """
        if not self.entries:
            return None, 0.0

        # --- 方式1：jieba 分词匹配 ---
        q_words_full = set(jieba.lcut(question))
        q_words = q_words_full - STOP_WORDS
        if not q_words:
            q_words = q_words_full

        # --- 方式2：字符 2-gram（处理 jieba 分不准的专有名词，如「眼轴」）---
        q_chars = question.strip()
        char_grams = set()
        for i in range(len(q_chars) - 1):
            bigram = q_chars[i : i + 2]
            # 只保留纯中文 bigram
            if all("一" <= c <= "鿿" for c in bigram):
                char_grams.add(bigram)

        best_entry = None
        best_score = 0.0

        for entry in self.entries:
            kb_words = entry["_all_words"]
            kb_q = entry["question"]

            # --- 方式1 分数 ---
            if q_words and kb_words:
                hit1 = len(q_words & kb_words)
                score1 = hit1 / len(q_words)
            else:
                score1 = 0.0

            # --- 方式2 分数（字符 2-gram）---
            if char_grams:
                # 把 KB 问题也转成 2-gram
                kb_char_grams = set()
                for i in range(len(kb_q) - 1):
                    bg = kb_q[i : i + 2]
                    if all("一" <= c <= "鿿" for c in bg):
                        kb_char_grams.add(bg)
                hit2 = len(char_grams & kb_char_grams)
                score2 = hit2 / len(char_grams) if char_grams else 0.0
            else:
                score2 = 0.0

            # 取较高分
            score = max(score1, score2)

            # 完整子串命中 → 满分
            if len(kb_q) >= 2 and (kb_q in question or question in kb_q):
                score = 1.0

            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry, best_score


# ================================================
# 第三部分：Session 会话管理
# ================================================

SESSION_MAX_MESSAGES = 10  # 每个 session 最多保留多少轮对话
SESSION_TTL = 1800  # 30 分钟无活动则清理


class SessionManager:
    """管理多个对话 session，用于 LLM 兜底时带上下文"""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._last_access: dict[str, float] = {}

    def get_history(self, session_id: str) -> list[dict]:
        """获取指定 session 的对话历史"""
        self._cleanup()
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, user_msg: str, robot_msg: str):
        """添加一轮对话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        hist = self._sessions[session_id]
        hist.append({"role": "user", "content": user_msg})
        hist.append({"role": "assistant", "content": robot_msg})
        # 保留最近 N 轮
        max_messages = SESSION_MAX_MESSAGES * 2  # 每轮 user+assistant
        if len(hist) > max_messages:
            self._sessions[session_id] = hist[-max_messages:]
        self._last_access[session_id] = time.time()

    def _cleanup(self):
        """清理过期 session"""
        now = time.time()
        expired = [sid for sid, t in self._last_access.items() if now - t > SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]
            del self._last_access[sid]


sessions = SessionManager()

# ================================================
# 第四部分：DeepSeek API
# ================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

KID_SYSTEM_PROMPT = (
    '你是天津市眼科医院视光中心的智能导诊小伙伴"小视"，主要对话对象是小朋友。'
    "你的回答：\n"
    "1. 用小朋友能听懂的语言，像大哥哥大姐姐一样\n"
    "2. 简短有趣，每段不超过2句话\n"
    "3. 多用可爱的emoji和拟声词\n"
    "4. 把眼健康知识变成有趣的小故事\n"
    '5. 医疗建议要温柔提醒"问医生叔叔阿姨"\n'
    "6. 多夸奖小朋友\n"
    "7. 控制在150字以内"
)


def call_deepseek(question: str, history: list[dict] | None = None) -> tuple[str, str]:
    """调用 DeepSeek，可选带对话历史"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "system", "content": KID_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 400,
    }
    try:
        resp = requests.post(
            DEEPSEEK_API_URL, json=payload, headers=headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], "deepseek"
    except requests.exceptions.Timeout:
        return "哎呀，我想了好久都没想出来～要不我们去问问医生叔叔阿姨吧？😊", "error"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status in (401, 403):
            return f"DeepSeek API Key 认证失败 (HTTP {status})，请检查 Key。", "error"
        return f"DeepSeek 返回错误 (HTTP {status})，请稍后再试。", "error"
    except Exception as e:  # noqa: BLE001 — 历史容错：DeepSeek 任意故障转 error
        return f"服务异常：{e!s}", "error"


# ================================================
# 第五部分：启动 — 加载知识库并建索引
# ================================================

print("=" * 50)
print("  天津市眼科医院视光中心 — 智能导诊 v3.0")
print("=" * 50)

paragraphs = extract_paragraphs(DOCX_PATH)
qa_list = parse_qa_pairs(paragraphs)
kb_searcher = KBSearcher(qa_list)

print(f"  匹配阈值: {KB_MATCH_THRESHOLD}")
print(f"  DeepSeek: {'[OK]' if len(DEEPSEEK_API_KEY) > 20 else '❌'}")
print("=" * 50)

# ── 语音回答后端开关：解析失败（非法 mode / 非契约 URL / 缺凭据）显式退出，
#    绝不静默回退 legacy —— 配置错误必须被人看见 ──
try:
    VOICE_AGENT_CONFIG = voice_agent_adapter.resolve_voice_agent_config()
except voice_agent_adapter.VoiceAgentConfigError as _exc:
    print(f"[FATAL] 语音回答后端配置错误: {_exc}")
    raise SystemExit(1)
print(f"  语音回答后端: {VOICE_AGENT_CONFIG.mode}")


# ================================================
# 第六部分：API 模型
# ================================================


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    user_question: str
    robot_answer: str
    source: str  # "kb" | "deepseek" | "error" | "agent"


class SuggestionsResponse(BaseModel):
    suggestions: list[str]


# ================================================
# 第七部分：SSE 推送（答案 + 麦克风状态事件）
# ================================================

_latest_answer: dict | None = None
_answer_id = 0
_answer_lock = threading.Lock()

# ── 麦克风状态事件队列（供 SSE 推送前端 UI 状态变化）──
_mic_events: list[dict] = []
_mic_event_lock = threading.Lock()
_mic_event_id = 0


def _push_mic_event(event_type: str, text: str = ""):
    """向 SSE 推送麦克风状态事件（硬件唤醒 / ASR 收到 / 停止等）"""
    global _mic_event_id
    with _mic_event_lock:
        _mic_event_id += 1
        _mic_events.append(
            {
                "id": _mic_event_id,
                "event": event_type,
                "text": text,
                "timestamp": time.time(),
            }
        )
        # 最多保留 20 个未消费事件
        if len(_mic_events) > 20:
            _mic_events.pop(0)


# ================================================
# 第八部分：API 端点
# ================================================


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="提交问题并获取回答（回答引擎双模式）",
    description=(
        "请求/响应结构与旧版逐字一致，只有**回答来自哪里**由 "
        "`GCMW_VOICE_ANSWER_BACKEND` 决定：\n\n"
        "- `legacy`（默认）：先检索本地 docx 知识库，未命中该库时调用 DeepSeek；\n"
        "- `agent`：只走新 Agent（Manager → RAG → Verifier，端口 8001），"
        "回答必须原句引用已审核资料并带引用编号；任何故障（不可达/超时/"
        "证据不足/协议违规）都失败关闭为固定安全文案，**绝不回退** KB 或 DeepSeek。\n\n"
        "配置非法时服务拒绝启动，不会静默回退 legacy。"
    ),
)
def chat_api(req: ChatRequest):
    global _latest_answer, _answer_id
    q = req.question.strip()
    if not q:
        return ChatResponse(
            user_question="",
            robot_answer="跟我说说话吧～你想问什么呢？😊",
            source="error",
        )

    if VOICE_AGENT_CONFIG.mode == "agent":
        # ── Agent 模式：只走新 Agent（Manager → RAG → Verifier），任何故障
        #    都由适配器失败关闭为固定安全文案，严禁回退 KB/DeepSeek ──
        answer, source = voice_agent_adapter.ask(q, VOICE_AGENT_CONFIG)
        print(f"   [AGENT] source={source}")
        result = ChatResponse(user_question=q, robot_answer=answer, source=source)
        with _answer_lock:
            _answer_id += 1
            _latest_answer = result.model_dump()
            _latest_answer["id"] = _answer_id
        return result

    # ── legacy 模式：行为逐字不变 ──
    best_entry, score = kb_searcher.search(q)

    if best_entry is not None and score >= KB_MATCH_THRESHOLD:
        # [OK] 知识库命中 → 直接返回，不调 DeepSeek
        answer = best_entry["answer"]
        source = "kb"
        print(f"   [KB] KB 命中 (score={score:.2f}): {best_entry['question'][:30]}...")
    else:
        # ❌ 未命中 → 调 DeepSeek，带对话历史
        history = sessions.get_history(req.session_id)
        answer, source = call_deepseek(q, history if history else None)
        print(f"   [LLM] LLM 兜底 (score={score:.2f}): {q[:30]}...")

    result = ChatResponse(user_question=q, robot_answer=answer, source=source)

    # ── 保存会话历史 ──
    sessions.add_turn(req.session_id, q, answer)

    # ── 广播 SSE ──
    with _answer_lock:
        _answer_id += 1
        _latest_answer = result.model_dump()
        _latest_answer["id"] = _answer_id

    return result


@app.get("/suggestions", response_model=SuggestionsResponse)
def get_suggestions():
    """返回知识库中的前 6 个问题作为建议"""
    qs = [entry["question"] for entry in qa_list[:6]]
    return SuggestionsResponse(suggestions=qs)


# ================================================
# 麦克风硬件唤醒控制
# ================================================

_mic_state = {
    "triggered": False,
    "start_time": 0.0,
    "last_asr_text": "",
    "hw_triggered": False,  # 是否由硬件唤醒触发（喊"小微小微"）
}


@app.post("/mic/wakeup")
def mic_wakeup():
    """前端点击麦克风 → 设置唤醒标记"""
    _mic_state["triggered"] = True
    _mic_state["start_time"] = time.time()
    _mic_state["last_asr_text"] = ""
    _mic_state["hw_triggered"] = False
    _push_mic_event("manual_wakeup")
    print("   [MIC] 手动唤醒触发")
    return {"status": "ok"}


@app.post("/mic/hw_wakeup")
def mic_hw_wakeup():
    """硬件语音唤醒（喊"小微小微"） → 通知前端显示聆听状态"""
    if _mic_state["triggered"]:
        return {"status": "already_triggered"}
    _mic_state["triggered"] = True
    _mic_state["start_time"] = time.time()
    _mic_state["last_asr_text"] = ""
    _mic_state["hw_triggered"] = True
    _push_mic_event("hw_wakeup")
    print("   [MIC] 硬件语音唤醒（小微小微）")
    return {"status": "ok"}


@app.post("/mic/stop")
def mic_stop():
    """前端关闭 → 清除唤醒标记"""
    _mic_state["triggered"] = False
    _mic_state["last_asr_text"] = ""
    _mic_state["hw_triggered"] = False
    _push_mic_event("mic_stop")
    print("   [MIC] 已停止")
    return {"status": "ok"}


@app.get("/mic/status")
def mic_status():
    """轮询接口：voice_transfer_node 检测唤醒；前端获取 ASR 结果"""
    elapsed = time.time() - _mic_state["start_time"] if _mic_state["triggered"] else 0.0
    return {
        "triggered": _mic_state["triggered"],
        "elapsed_seconds": round(elapsed, 1),
        "asr_text": _mic_state["last_asr_text"],
    }


class AsrNotifyBody(BaseModel):
    text: str


@app.post("/mic/notify_asr")
def mic_notify_asr(body: AsrNotifyBody):
    """voice_transfer_node 收到 ASR 文本后回传，供前端轮询获取"""
    _mic_state["last_asr_text"] = body.text
    _mic_state["triggered"] = False
    _mic_state["hw_triggered"] = False
    _push_mic_event("asr_received", body.text)
    return {"status": "ok"}


@app.get("/sse")
async def sse_endpoint():
    """SSE 端点：语音答案 + 麦克风状态实时推送到屏幕"""
    last_seen = _answer_id
    last_mic_event_id = _mic_event_id

    async def generate():
        nonlocal last_seen, last_mic_event_id
        while True:
            # 1. 推送麦克风状态事件
            with _mic_event_lock:
                if _mic_events:
                    new_events = [e for e in _mic_events if e["id"] > last_mic_event_id]
                    for evt in new_events:
                        last_mic_event_id = evt["id"]
                        yield f"event: mic_status\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"

            # 2. 推送问答答案
            with _answer_lock:
                if _latest_answer is not None and _answer_id > last_seen:
                    last_seen = _answer_id
                    data = json.dumps(_latest_answer, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "3.0.0",
        "kb_entries": len(qa_list),
        "docx_path": DOCX_PATH,
        "match_threshold": KB_MATCH_THRESHOLD,
        "answer_backend": VOICE_AGENT_CONFIG.mode,
    }


if __name__ == "__main__":
    # 默认只绑回环（见 HOST）：部署侧的 systemd unit 也显式写 --host 127.0.0.1，
    # 代码层与部署层双保险，任何一层漏了都不会把服务摊到局域网。
    uvicorn.run("qa_server:app", host=HOST, port=PORT, reload=False)
