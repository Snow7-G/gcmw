/**
 * Page wiring regressions for the /qa Agent demo slice (P1-1/P1-2 round).
 *
 * Mounts the REAL index.vue in jsdom with a stubbed EventSource and a
 * scripted fetch — no @vue/test-utils, no network. Covers the acceptance
 * list: visible fixed failure copy, unmount-during-hang semantics, and
 * "no new Agent requests / no legacy voice reconnect after unmount".
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, type App } from 'vue'

import QaPage from './index.vue'

const DEMO_BASE = 'http://127.0.0.1:8001/api/v1'

class FakeEventSource {
  static instances: FakeEventSource[] = []
  url: string
  closed = false
  onmessage: ((ev: unknown) => void) | null = null
  onerror: (() => void) | null = null
  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }
  close(): void {
    this.closed = true
  }
  addEventListener(): void {
    /* no-op */
  }
}

interface FetchLogEntry {
  url: string
  signal?: AbortSignal
}

type FetchHandler = (url: string, signal?: AbortSignal) => Response | Promise<never>

let fetchLog: FetchLogEntry[] = []
let handler: FetchHandler = () => {
  throw new Error('no route')
}

function stubFetch(fn: FetchHandler): void {
  handler = fn
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      const signal = init?.signal ?? undefined
      fetchLog.push({ url, signal })
      return handler(url, signal)
    })
  )
}

const okJson = (obj: unknown): Response =>
  ({ ok: true, status: 200, json: async () => obj }) as unknown as Response
const httpError = (status: number): Response =>
  ({ ok: false, status, json: async () => ({}) }) as unknown as Response
/** 字节级 SSE 响应（answer 流测试用；帧可跨 chunk） */
function sseBody(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    }
  })
}

function okResponse(body: ReadableStream<Uint8Array>): Response {
  return { ok: true, status: 200, body } as unknown as Response
}

const hang = (_url: string, signal: AbortSignal | undefined): Promise<never> =>
  new Promise((_resolve, reject) => {
    signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
  })

const flush = async (rounds = 6): Promise<void> => {
  for (let i = 0; i < rounds; i++) await new Promise((r) => setTimeout(r, 0))
}

let app: App | null = null
let root: HTMLElement | null = null

function mountPage(): void {
  root = document.createElement('div')
  document.body.appendChild(root)
  app = createApp(QaPage)
  app.mount(root)
}

function unmountPage(): void {
  app?.unmount()
  app = null
  root?.remove()
  root = null
}

const text = (): string => root?.textContent ?? ''

function clickSuggestion(label: string): void {
  const buttons = [...(root?.querySelectorAll('button') ?? [])]
  const target = buttons.find((b) => b.textContent?.includes(label))
  if (!target) throw new Error(`suggestion button not found: ${label}`)
  target.click()
}

beforeEach(() => {
  fetchLog = []
  vi.stubGlobal('EventSource', FakeEventSource)
  // jsdom 没有 ResizeObserver（LiquidBar/布局组件用到）
  vi.stubGlobal(
    'ResizeObserver',
    class {
      /* eslint-disable @typescript-eslint/no-empty-function */
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
      /* eslint-enable @typescript-eslint/no-empty-function */
    }
  )
  FakeEventSource.instances = []
  vi.stubEnv('VITE_GCMW_AGENT_API_BASE', DEMO_BASE)
  vi.stubEnv('VITE_GCMW_DEMO_CREDENTIAL', 'vitest-demo-token-00001')
})

afterEach(() => {
  unmountPage()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('qa page agent wiring', () => {
  it('renders the transparent mascot image instead of the LED panel', async () => {
    stubFetch((url) => {
      if (url.includes('/suggestions')) return okJson({ suggestions: ['眼部不适怎么办'] })
      if (url.includes('/api/v1/sessions')) return okJson({ session_id: 's-1' })
      throw new Error('unexpected ' + url)
    })
    mountPage()
    await flush()

    const mascot = root?.querySelector<HTMLImageElement>('img[alt="小视"]')
    expect(mascot?.getAttribute('src')).toContain('qa-mascot-cutout')
    expect(root?.querySelector('.led-panel')).toBeNull()
  })

  it('shows the fixed config error VISIBLY when the demo config is missing', async () => {
    vi.stubEnv('VITE_GCMW_DEMO_CREDENTIAL', '') // 缺配置
    stubFetch((url) => {
      if (url.includes('/suggestions')) return okJson({ suggestions: ['眼部不适怎么办'] })
      throw new Error('agent APIs must not be called without config')
    })
    mountPage()
    await flush()
    clickSuggestion('眼部不适怎么办')
    await flush()
    // 可见面板出现固定文案（而不是只写进未渲染的 messages）
    expect(text()).toContain('演示服务还没配置好')
    expect(fetchLog.some((c) => c.url.includes('/api/v1/'))).toBe(false) // 零 Agent 请求
  })

  it('shows a fixed visible error when the Agent session creation fails', async () => {
    stubFetch((url) => {
      if (url.includes('/suggestions')) return okJson({ suggestions: ['眼部不适怎么办'] })
      if (url.includes('/api/v1/sessions')) return httpError(500)
      throw new Error('unexpected ' + url)
    })
    mountPage()
    await flush()
    expect(agentSessionCount()).toBe(1)
    clickSuggestion('眼部不适怎么办')
    await flush()
    expect(text()).toContain('暂时连不上问答服务') // 固定安全文案，可见
    expect(text()).not.toContain('HTTP 500') // 不泄漏服务端细节
  })

  it('unmount during hangs: session request aborted, no new requests', async () => {
    const signals: Record<string, AbortSignal | null | undefined> = {}
    stubFetch((url, signal) => {
      if (url.includes('/suggestions')) {
        signals.suggestions = signal
        return hang(url, signal) // suggestions 挂起
      }
      if (url.includes('/api/v1/sessions')) {
        signals.session = signal
        return hang(url, signal) // session 挂起
      }
      throw new Error('unexpected ' + url)
    })
    mountPage()
    await flush()
    expect(agentSessionCount()).toBe(1) // session 请求恰好一次

    unmountPage() // suggestions 与 session 都还挂着时卸载
    await flush()

    expect(signals.session?.aborted).toBe(true) // Session 请求被 abort
    expect(signals.suggestions?.aborted).toBe(true)
    expect(agentSessionCount()).toBe(1) // 卸载后零新增 Agent 请求
  })

  it('cancels the armed legacy reconnect timer when unmounted within 3s', async () => {
    vi.useFakeTimers()
    try {
      stubFetch((url, signal) => {
        if (url.includes('/suggestions')) return hang(url, signal)
        if (url.includes('/api/v1/sessions')) return hang(url, signal)
        throw new Error('unexpected ' + url)
      })
      mountPage()
      // 挂载即创建旧语音 EventSource（同步 FIRST）
      expect(FakeEventSource.instances.length).toBe(1)
      const legacy = FakeEventSource.instances[0]

      // 主动触发服务端断开：进入重连等待期
      legacy.onerror?.()
      expect(legacy.closed).toBe(true)

      // 3 秒到期前卸载
      unmountPage()

      // 推进虚拟时间跨过 3 秒重连窗口：绝不允许创建新 EventSource
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
      expect(FakeEventSource.instances.length).toBe(1)
      // 卸载后零新增 Agent 请求
      expect(agentSessionCount()).toBe(1)
    } finally {
      vi.useRealTimers() // 恢复真实计时器，不污染其他测试
    }
  })

  it('keeps the legacy voice SSE independent of a slow Agent backend', async () => {
    stubFetch((url, signal) => {
      if (url.includes('/suggestions')) return hang(url, signal)
      if (url.includes('/api/v1/sessions')) return hang(url, signal)
      throw new Error('unexpected ' + url)
    })
    mountPage()
    await flush(2)
    // 旧语音流必须已经启动——不等 suggestions/Agent Session
    expect(FakeEventSource.instances.length).toBe(1)
    expect(FakeEventSource.instances[0].url).toBe('http://127.0.0.1:8000/sse')
  })

  it('restores the default welcome text after returning from an answer', async () => {
    stubFetch((url) => {
      if (url.includes('/suggestions')) return okJson({ suggestions: ['眼部不适怎么办'] })
      if (url.includes('/api/v1/sessions')) return okJson({ session_id: 's-1' })
      if (url.includes('/events')) {
        const body = [
          'id: 1\nevent: run.accepted\ndata: {"data":{}}\n\n',
          'id: 2\nevent: process.status\ndata: {"data":{"stage":"retrieving"}}\n\n',
          'id: 3\nevent: answer.delta\ndata: {"data":{"delta":"体温超过38.5建议门诊就诊（资料[1]）。"}}\n\n',
          'id: 4\nevent: answer.completed\ndata: {"data":{"citations":[{"source_id":"faq-fever","knowledge_version":"1","content_hash":"' + 'a'.repeat(64) + '","title":"发热护理须知","source_uri":"kbase://faq-fever"}],"content_origin":"approved_faq"}}\n\n',
          'id: 5\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n'
        ]
        return okResponse(sseBody([body.join('')]))
      }
      if (url.includes('/agent/runs')) {
        // 注意：events 路由必须在前（/agent/runs/{id}/events 也含此前缀）
        return okJson({ run_id: 'r-1', state: 'ACCEPTED' })
      }
      throw new Error('unexpected ' + url)
    })
    mountPage()
    await flush()
    clickSuggestion('眼部不适怎么办')
    await flush()
    // 已进入答案态：回答与引用必须真实展示过，防止请求意外走进固定失败
    // 页面时测试假绿（失败页同样有"返回"按钮）
    expect(text()).toContain('体温超过38.5建议门诊就诊')
    expect(text()).toContain('资料来源')
    expect(text()).toContain('发热护理须知')
    expect(text()).not.toContain('暂时连不上问答服务')

    // 点击"返回"
    const back = [...(root?.querySelectorAll('button') ?? [])].find((b) =>
      b.textContent?.includes('返回')
    )
    expect(back).toBeDefined()
    back!.click()
    await flush()

    // 欢迎语恢复；不残留"正在思考中..."；答案/引用已清空；问题卡片回来
    expect(text()).toContain('眼睛健康小伙伴')
    expect(text()).not.toContain('正在思考中...')
    expect(text()).not.toContain('资料来源')
    expect(text()).toContain('眼部不适怎么办') // 建议问题重新可见
  })

  function agentSessionCount(): number {
    return fetchLog.filter((c) => c.url.includes('/api/v1/sessions')).length
  }
})
