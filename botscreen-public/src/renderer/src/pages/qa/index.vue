<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import LiquidBar from '@renderer/components/LiquidBar.vue'
import qaMascotUrl from '@renderer/assets/qa-mascot-cutout.png'
import {
  AgentRunView,
  AgentStreamError,
  cancelRun,
  createRun,
  createSession,
  resolveAgentConfig,
  streamRun,
  type AgentClientOptions,
  type Citation
} from './agent-client'

// 旧服务：仅语音/硬件链路继续使用（本切片不迁移语音）
const API_BASE = 'http://127.0.0.1:8000'

// ===== 新 Agent 演示链路（development/test 合成数据演示） =====
// 配置来自 .env.local（占位符见 .env.example），缺配置时文字问答失败关闭并
// 显示固定提示；绝不回显、不提交凭据值。
const agentOptions = resolveAgentConfig(
  import.meta.env as unknown as Record<string, string | undefined>
)
const agentReady = agentOptions !== null

const agentSessionId = ref('')
const activeRunId = ref('')
let agentAbortController: AbortController | null = null
let agentRunView: AgentRunView | null = null
// 旧语音 SSE（legacyVoiceSse）与新 Agent SSE 各用各的变量，互不关闭
let legacyVoiceSse: EventSource | null = null
let legacySseReconnectTimer: ReturnType<typeof setTimeout> | null = null
let pageAlive = true
let agentSessionAbort: AbortController | null = null
let suggestionsAbort: AbortController | null = null

/** 演示确定性：前三个问题固定对应三条后端路径（有引用 / 无证据 / 红旗） */
const DEMO_QUESTIONS = ['眼部不适怎么办', '色盲能治好吗', '眼睛突然看不见了']

const AGENT_COPY = {
  configMissing: '演示服务还没配置好，请先在 .env.local 里填写演示配置～',
  sessionFailed: '暂时连不上问答服务，请稍后再试～',
  refused: '目前没有足够可靠的资料回答这个问题，建议咨询专业人员。',
  escalated: '这个问题需要人工或专业人员进一步协助。',
  cancelled: '本次问答已取消',
  timeout: '本次处理超时，请稍后重试。',
  genericError: '哎呀，服务暂时不可用，请稍后再试～',
  sourcePrefix: '\n\n资料来源：\n'
} as const

/** 统一失败出口：清空部分答案、把固定安全文案放到可见的答案面板 */
function showAgentFailure(copy: string): void {
  answerText.value = copy
  showAnswer.value = true
  mascotState.value = 'idle'
}

function agentFailCopy(err: unknown): string {
  if (err instanceof AgentStreamError) {
    switch (err.kind) {
      case 'unauthorized':
      case 'forbidden':
        return AGENT_COPY.sessionFailed
      case 'rate_limited':
        return '问得有点太快啦，休息一下再试试～'
      case 'aborted':
        return AGENT_COPY.cancelled
      default:
        return AGENT_COPY.genericError
    }
  }
  return AGENT_COPY.genericError
}

type MascotState = 'idle' | 'listening' | 'processing' | 'answering' | 'encourage' | 'rest'

interface ChatMessage {
  role: 'user' | 'robot'
  text: string
  source?: 'kb' | 'deepseek' | 'error' | 'agent'
}

const mascotState = ref<MascotState>('idle')
const messages = ref<ChatMessage[]>([])
const suggestions = ref<string[]>([])
const welcomeText = ref('')
const isSending = ref(false)
const answerText = ref('')
const showAnswer = ref(false)

let lastManualQuestion = ''

function goBack(): void {
  // 主动取消：中止 Agent SSE 流并对未终态的 Run best-effort 取消
  cancelActiveRun()
  showAnswer.value = false
  answerText.value = ''
  messages.value = []
  mascotState.value = 'idle'
  // 恢复默认欢迎语：不能残留"正在思考中..."
  welcomeText.value = defaultWelcome.value
}

/** 终态后把引用追加到现有答案文本（第一版不做引用卡片） */
function appendCitations(citations: Citation[]): void {
  const lines = citations.map(
    (c, i) => `${i + 1}. ${c.title ?? c.source_id} · ${c.knowledge_version}`
  )
  if (lines.length > 0) {
    answerText.value += `${AGENT_COPY.sourcePrefix}${lines.join('\n')}`
  }
}

/** run.completed 的 result → 页面固定文案（终态只能进来一次） */
function applyTerminalResult(result: string, view: AgentRunView | null): void {
  switch (result) {
    case 'answered':
      // 交付门槛：非空答案 + 合法 answer.completed + ≥1 条有效引用三元组，
      // 否则未核验的医疗回答一律清掉并显示固定文案
      if (!view || !view.canDeliverAnswer()) {
        answerText.value = AGENT_COPY.genericError
        showAnswer.value = true
        mascotState.value = 'idle'
        return
      }
      // 保留已累积答案；吉祥物走鼓励再回 idle
      mascotState.value = 'encourage'
      setTimeout(() => {
        if (mascotState.value === 'encourage') mascotState.value = 'idle'
      }, 2500)
      break
    case 'refused_no_answer':
      answerText.value = AGENT_COPY.refused
      showAnswer.value = true
      mascotState.value = 'idle'
      break
    case 'escalated_to_human':
      answerText.value = AGENT_COPY.escalated
      showAnswer.value = true
      mascotState.value = 'idle'
      break
    case 'cancelled':
      answerText.value = AGENT_COPY.cancelled
      showAnswer.value = true
      mascotState.value = 'idle'
      break
    case 'deadline_exceeded':
      answerText.value = AGENT_COPY.timeout
      showAnswer.value = true
      mascotState.value = 'idle'
      break
    default:
      answerText.value = AGENT_COPY.genericError
      showAnswer.value = true
      mascotState.value = 'idle'
      break
  }
}

async function sendQuestion(question: string): Promise<void> {
  if (!question.trim() || isSending.value) return
  if (!agentReady || !agentSessionId.value) {
    messages.value.push({ role: 'user', text: question })
    showAgentFailure(agentReady ? AGENT_COPY.sessionFailed : AGENT_COPY.configMissing)
    return
  }

  isSending.value = true
  mascotState.value = 'processing'
  welcomeText.value = '正在思考中...'
  answerText.value = ''
  showAnswer.value = false
  messages.value.push({ role: 'user', text: question })

  // 新问题生成新 key；同一次创建的重试必须复用同一个 key
  const idempotencyKey = `qa-${Date.now()}-${crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`
  agentRunView = new AgentRunView()
  agentAbortController = new AbortController()

  try {
    let runId = ''
    try {
      const run = await createRun(
        agentOptions as AgentClientOptions,
        agentSessionId.value,
        question,
        idempotencyKey,
        agentAbortController.signal
      )
      runId = run.run_id
    } catch (err) {
      // 网络故障：同 key 重试一次（服务端幂等去重）
      if (err instanceof AgentStreamError && err.kind === 'network') {
        const run = await createRun(
          agentOptions as AgentClientOptions,
          agentSessionId.value,
          question,
          idempotencyKey,
          agentAbortController.signal
        )
        runId = run.run_id
      } else {
        throw err
      }
    }

    activeRunId.value = runId
    await streamRun(agentOptions as AgentClientOptions, runId, {
      lastEventId: 0,
      signal: agentAbortController.signal,
      onEvent: handleAgentEvent
    })
  } catch (err) {
    if (!(err instanceof AgentStreamError && err.kind === 'aborted')) {
      // 统一失败出口：清空部分答案/引用，固定安全文案可见，绝不显示
      // 服务端异常原文
      showAgentFailure(agentFailCopy(err))
      messages.value.push({ role: 'robot', text: answerText.value, source: 'agent' })
    }
  } finally {
    isSending.value = false
    activeRunId.value = ''
    agentRunView = null
    agentAbortController = null
  }
}

/** Agent SSE 事件 → 现有 UI 状态（业务逻辑在 AgentRunView，页面只做映射） */
function handleAgentEvent(event: { id?: number; event: string; data: unknown }): void {
  const view = agentRunView
  if (!view) return
  const action = view.apply(event)
  if (!action) return

  if (action.stage) {
    mascotState.value = action.stage
  }
  if (action.appendDelta) {
    answerText.value = view.answer
    showAnswer.value = true
  }
  if (action.citations) {
    appendCitations(action.citations)
  }
  if (action.terminalResult !== undefined) {
    applyTerminalResult(action.terminalResult, view)
  }
  if (action.streamError) {
    answerText.value = AGENT_COPY.genericError
    showAnswer.value = true
    mascotState.value = 'idle'
  }
}

/** best-effort 取消当前 Run（页面离开 / 返回按钮）；不阻塞、不泄漏错误 */
function cancelActiveRun(): void {
  agentAbortController?.abort()
  agentAbortController = null
  if (activeRunId.value && agentOptions) {
    const runId = activeRunId.value
    cancelRun(agentOptions, runId).catch(() => {
      /* best-effort: the server also reaps unfinished runs */
    })
    activeRunId.value = ''
  }
}

// ========== 硬件麦克风控制（测试模式）==========
const micActive = ref(false)
const defaultWelcome = ref('')
let micPollTimer: ReturnType<typeof setInterval> | null = null

function toggleMic(): void {
  if (micActive.value) {
    stopMic()
  } else {
    startMic()
  }
}

async function startMic(): Promise<void> {
  try {
    const res = await fetch(API_BASE + '/mic/wakeup', { method: 'POST' })
    if (!res.ok) throw new Error('HTTP ' + res.status)
    micActive.value = true
    mascotState.value = 'listening'
    welcomeText.value = '正在聆听，请说话...'
    startMicPolling()
  } catch {
    welcomeText.value = '无法连接后端，请检查服务～'
    setTimeout(() => {
      welcomeText.value = defaultWelcome.value
    }, 3000)
  }
}

async function stopMic(): Promise<void> {
  try {
    await fetch(API_BASE + '/mic/stop', { method: 'POST' })
  } catch {
    /* ignore */
  }
  micActive.value = false
  mascotState.value = 'idle'
  welcomeText.value = defaultWelcome.value
  stopMicPolling()
}

function startMicPolling(): void {
  stopMicPolling()
  micPollTimer = setInterval(async () => {
    try {
      const res = await fetch(API_BASE + '/mic/status')
      const data = await res.json()
      // 收到 ASR 文字 → 等待 SSE 推送（voice_transfer_node 已调 /chat）
      if (data.asr_text && micActive.value) {
        stopMicPolling()
        micActive.value = false
        mascotState.value = 'processing'
        welcomeText.value = '正在思考中...'
      }
      // 超时 15 秒
      if (data.elapsed_seconds > 15 && micActive.value) {
        stopMic()
      }
    } catch {
      /* ignore */
    }
  }, 500)
}

function stopMicPolling(): void {
  if (micPollTimer) {
    clearInterval(micPollTimer)
    micPollTimer = null
  }
}

/** 演示固定问题放最前，其余旧问题去重后继续保留 */
function withDemoQuestionsFirst(questions: string[]): string[] {
  return [...DEMO_QUESTIONS, ...questions.filter((q) => !DEMO_QUESTIONS.includes(q))]
}

onMounted(() => {
  welcomeText.value =
    '你好呀～我是小视，你的眼睛健康小伙伴！\n有什么想知道的，点点按钮或者按一下麦克风跟我说吧～'
  defaultWelcome.value = welcomeText.value
  // 旧语音链路 FIRST：同步启动，不依赖任何 await，8001 卡住绝不阻塞 8000
  connectSSE()
  // 两个相互独立的异步任务，各自带组件级 AbortController，卸载统一中止
  void loadSuggestions()
  void initAgentSession()
})

/** 任务一：建议问题（独立 abort；每次状态写入前检查 pageAlive） */
async function loadSuggestions(): Promise<void> {
  suggestionsAbort = new AbortController()
  try {
    const res = await fetch(`${API_BASE}/suggestions`, {
      signal: suggestionsAbort.signal
    })
    if (!pageAlive) return // 卸载后丢弃结果，不写状态
    if (res.ok) {
      const data = await res.json()
      if (pageAlive) suggestions.value = withDemoQuestionsFirst(data.suggestions || [])
    } else throw new Error('fallback')
  } catch {
    if (pageAlive) {
      suggestions.value = withDemoQuestionsFirst([
        '眼轴正常长度',
        '近视小科普',
        '近视分级标准',
        '用眼休息时长',
        '正确读写姿势',
        '每日户外时长',
        '用眼光线环境',
        '护眼饮食推荐',
        '定期检查眼睛'
      ])
    }
  }
}

/** 任务二：Agent Session（组件级 abort + 5s 超时；失败则文字问答失败关闭） */
async function initAgentSession(): Promise<void> {
  if (!agentReady || !pageAlive) return
  agentSessionAbort = new AbortController()
  const sessionTimeout = setTimeout(() => agentSessionAbort?.abort(), 5000)
  try {
    const id = await createSession(agentOptions as AgentClientOptions, agentSessionAbort.signal)
    if (!pageAlive) return // 卸载后丢弃结果，不再写状态
    agentSessionId.value = id
  } catch {
    // 失败关闭：session 拿不到时 sendQuestion 会给出固定提示，语音链路不受影响
  } finally {
    clearTimeout(sessionTimeout)
    agentSessionAbort = null
  }
}

onBeforeUnmount(() => {
  pageAlive = false
  stopMicPolling()
  if (hwTimeoutTimer) clearTimeout(hwTimeoutTimer)
  cancelActiveRun()
  suggestionsAbort?.abort()
  suggestionsAbort = null
  agentSessionAbort?.abort()
  agentSessionAbort = null
  if (legacySseReconnectTimer) {
    clearTimeout(legacySseReconnectTimer)
    legacySseReconnectTimer = null
  }
  if (legacyVoiceSse) {
    legacyVoiceSse.close()
    legacyVoiceSse = null
  }
})

// ========== 硬件语音唤醒超时定时器 ==========
let hwTimeoutTimer: ReturnType<typeof setTimeout> | null = null

function connectSSE(): void {
  if (!pageAlive) return // 卸载后禁止任何续连
  legacyVoiceSse = new EventSource(`${API_BASE}/sse`)

  // ── 问答答案事件 ──
  legacyVoiceSse.onmessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data)
      if (data.user_question === lastManualQuestion) {
        lastManualQuestion = ''
        return
      }
      messages.value.push({ role: 'user', text: data.user_question })
      mascotState.value = 'answering'
      answerText.value = data.robot_answer
      showAnswer.value = true
      messages.value.push({
        role: 'robot',
        text: data.robot_answer,
        source: data.source || 'deepseek'
      })
      // 硬件唤醒模式：收到答案后自动结束麦克风状态
      stopMicHw()
      setTimeout(() => {
        mascotState.value = 'encourage'
        setTimeout(() => {
          if (mascotState.value === 'encourage') mascotState.value = 'idle'
        }, 2500)
      }, 2000)
    } catch {
      /* ignore */
    }
  }

  // ── 麦克风状态事件（硬件唤醒 / ASR 收到等）──
  legacyVoiceSse.addEventListener('mic_status', (event: MessageEvent) => {
    try {
      const evt = JSON.parse(event.data)
      switch (evt.event) {
        case 'hw_wakeup':
          // 硬件语音唤醒 → 前端显示"聆听"动画
          if (!micActive.value) {
            micActive.value = true
            mascotState.value = 'listening'
            welcomeText.value = '正在聆听，请说话...'
            // 15 秒超时自动停止
            hwTimeoutTimer = setTimeout(() => {
              stopMicHw()
            }, 15000)
          }
          break
        case 'asr_received':
          // ASR 识别到了文字 → 切换到"思考中"
          if (micActive.value) {
            mascotState.value = 'processing'
            welcomeText.value = '正在思考中...'
          }
          break
        case 'mic_stop':
          // 停止
          stopMicHw()
          break
        case 'manual_wakeup':
          // 手动点击触发，不需要额外处理（前端已通过 startMic 设置状态）
          break
      }
    } catch {
      /* ignore */
    }
  })

  legacyVoiceSse.onerror = () => {
    if (legacyVoiceSse) {
      legacyVoiceSse.close()
      legacyVoiceSse = null
    }
    if (pageAlive) {
      // timer 必须可清理：卸载时 clear，绝不复活旧语音流
      legacySseReconnectTimer = setTimeout(connectSSE, 3000)
    }
  }
}

/** 停止硬件唤醒的麦克风状态（不调后端 /mic/stop，因为是硬件触发的） */
function stopMicHw(): void {
  if (hwTimeoutTimer) {
    clearTimeout(hwTimeoutTimer)
    hwTimeoutTimer = null
  }
  micActive.value = false
  welcomeText.value = defaultWelcome.value
  if (mascotState.value === 'listening' || mascotState.value === 'processing') {
    mascotState.value = 'idle'
  }
}

// ========== TTS ==========
function speakText(text: string): void {
  window.speechSynthesis.cancel()
  const u = new SpeechSynthesisUtterance(text)
  u.lang = 'zh-CN'
  u.rate = 0.9
  u.pitch = 1.1
  window.speechSynthesis.speak(u)
}
</script>

<template>
  <div class="qa-kid-root relative h-screen w-full flex flex-col overflow-hidden">
    <div class="absolute inset-0 bg-black/5 backdrop-blur-[3px]"></div>

    <!-- Header：深阴影，浮在页面上方 -->
    <div class="relative z-50 w-full mt-[1.5em] flex flex-col shrink-0" style="filter:drop-shadow(0 4px 16px rgba(0,60,60,0.2));">
      <div class="max-w-3xl mx-auto w-full flex justify-center px-[2em]">
        <LiquidBar title="互动问答" back="/" />
      </div>
    </div>

    <!-- 主体：左右布局 -->
    <div class="relative z-10 flex-1 flex items-center justify-center min-h-0 px-[2em] py-[1em] gap-[1.5em]">

      <!-- ====== 左侧：表情 + 文字 + 麦克风 ====== -->
      <div class="flex flex-col items-center justify-center gap-8 shrink-0 w-[36%] max-w-[400px]">
        <div class="mascot-container relative">
          <div
            class="absolute inset-0 rounded-full transition-all duration-500"
            :class="{
              'bg-[#168378]/6 scale-115': mascotState === 'listening',
              'bg-[#168378]/5 scale-110': mascotState === 'processing',
              'bg-[#168378]/8 scale-115': mascotState === 'answering',
              'bg-transparent scale-100': mascotState === 'idle',
            }"
          ></div>
          <img
            :src="qaMascotUrl"
            alt="小视"
            class="mascot-image relative z-10 transition-all duration-500"
            :class="{ 'scale-105': mascotState === 'answering', 'scale-100': mascotState !== 'answering' }"
            draggable="false"
          />
        </div>

        <!-- 欢迎文字 -->
        <div v-if="!showAnswer && !messages.length" class="text-center max-w-[320px]">
          <p class="whitespace-pre-wrap text-[#1a3a2a]" style="line-height:1.6;">
            <span class="text-lg font-semibold">{{ welcomeText.split('\n')[0] }}</span>
            <br v-if="welcomeText.includes('\n')" />
            <span class="text-sm">{{ welcomeText.split('\n').slice(1).join('\n') }}</span>
          </p>
        </div>

        <!-- 麦克风 -->
        <button
          class="relative w-[72px] h-[72px] rounded-full flex items-center justify-center transition-all duration-300 cursor-pointer border-none outline-none shrink-0"
          :class="{
            'text-white shadow-md shadow-[#168378]/25 hover:scale-110 hover:shadow-lg hover:shadow-[#168378]/35': !micActive,
            'bg-[#ef4444] text-white shadow-md shadow-[#ef4444]/25 scale-110': micActive,
          }"
          :style="!micActive ? { background: 'radial-gradient(circle at 40% 40%, #2dd4bf, #168378)' } : {}"
          @click="toggleMic"
        >
          <span v-if="micActive" class="absolute inset-0 rounded-full bg-[#ef4444] animate-ping opacity-25"></span>
          <svg viewBox="0 0 36 36" width="28" height="28" fill="none" class="relative z-10">
            <rect x="12" y="4" width="12" height="18" rx="6" fill="currentColor" />
            <path d="M8 16 Q8 22 18 22 Q28 22 28 16" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" />
            <line x1="18" y1="22" x2="18" y2="29" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
            <line x1="11" y1="29" x2="25" y2="29" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
          </svg>
        </button>
      </div>

      <!-- ====== 右侧：问题卡片 / 答案面板 ====== -->
      <div class="flex-1 max-w-[580px] flex flex-col justify-center gap-4">
        <template v-if="showAnswer">
          <div class="question-card px-6 py-5 overflow-y-auto max-h-[50vh]">
            <p class="text-[15px] whitespace-pre-wrap text-[#1a3a2a] animate-fade-in" style="line-height:1.6;">
              {{ answerText }}
            </p>
            <button class="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium cursor-pointer transition-all active:scale-95 hover:bg-[#168378]/15" style="background:rgba(22,131,120,0.08);color:#168378;" @click="speakText(answerText)">
              🔊 听播报
            </button>
          </div>
          <button class="self-center px-5 py-2 rounded-xl text-sm font-medium cursor-pointer transition-all duration-200 active:scale-95" style="background:rgba(255,255,255,0.09);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid rgba(22,131,120,0.13);color:#168378;" @click="goBack">
            ← 返回
          </button>
        </template>
        <template v-else>
          <div class="relative mb-3 px-5 py-2.5 mx-auto" style="background:rgba(255,255,255,0.1);backdrop-filter:blur(12px);border-radius:16px 16px 16px 4px;border:1px solid rgba(22,131,120,0.15);">
            <h3 class="text-lg font-extrabold text-[#168378] text-center" style="letter-spacing:0.02em;">
              💬 试试问我这些问题吧～
            </h3>
          </div>
          <div class="grid grid-cols-2 auto-rows-fr gap-4 max-h-[60vh] overflow-y-auto pr-2">
            <button
              v-for="(q, idx) in suggestions" :key="idx"
              class="question-card flex items-center gap-2 px-4 py-4 transition-all duration-200 cursor-pointer active:scale-[0.97]"
              style="min-height:56px;"
              :disabled="isSending" @click="sendQuestion(q)"
            >
              <span class="text-[15px] font-medium leading-snug text-[#1a3a2a] dark:text-white">{{ q }}</span>
            </button>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.qa-kid-root { background: url(rc://bg.png) center / cover no-repeat fixed; zoom: 1.15; }

.mascot-image {
  display: block;
  width: 300px;
  height: auto;
  max-width: 100%;
  user-select: none;
  filter: drop-shadow(0 8px 14px rgba(15, 32, 30, 0.16));
}

@keyframes fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.animate-fade-in { animation: fade-in 0.4s ease-out; }

div.overflow-y-auto::-webkit-scrollbar { width: 0; }

/* 问题卡片/答案面板：低白色透明度 + 同色系细边框 */
.question-card {
  background: rgba(255,255,255,0.09); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(22,131,120,0.13); border-radius: 16px; transition: all 0.25s ease;
}
.question-card:hover:not(:disabled) {
  background: rgba(255,255,255,0.16); backdrop-filter: blur(16px);
  border-color: rgba(22,131,120,0.25);
  box-shadow: 0 6px 20px rgba(0,0,0,0.08); transform: translateY(-1px);
}
</style>
