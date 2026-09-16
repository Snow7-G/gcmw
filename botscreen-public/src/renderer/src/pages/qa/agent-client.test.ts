/**
 * Unit tests for the /qa Agent client (development/test demo slice).
 *
 * Every test runs against a stubbed global fetch; NO real network, NO real
 * credential (the token used here is a fixed fake — it must never appear in
 * a URL, which is itself one of the assertions).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  AgentRunView,
  SseFrameParser,
  createSession,
  resolveAgentConfig,
  streamRun,
  validateCitations,
  type AgentSseEvent
} from './agent-client'

const BASE = 'http://127.0.0.1:8001/api/v1'
const CREDENTIAL = 'fake-demo-credential-000001'

const OPTS = { baseUrl: BASE, credential: CREDENTIAL }

const encoder = new TextEncoder()

function sseBody(chunks: string[], opts?: { errorAfter?: boolean }): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      if (opts?.errorAfter) controller.error(new Error('simulated network drop'))
      else controller.close()
    }
  })
}

function okResponse(body: ReadableStream<Uint8Array>): Response {
  return { ok: true, status: 200, body } as unknown as Response
}

function errorResponse(status: number): Response {
  return { ok: false, status, body: null } as unknown as Response
}

function jsonResponse(obj: unknown): Response {
  return { ok: true, status: 200, body: null, json: async () => obj } as unknown as Response
}

interface FetchCall {
  url: string
  headers: Record<string, string>
}

/** Stub fetch; returns the calls and a way to queue responses. */
function stubFetch(responses: Array<Response | Error>): { calls: FetchCall[] } {
  const calls: FetchCall[] = []
  const queue = [...responses]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const headers: Record<string, string> = {}
      if (init?.headers) {
        for (const [k, v] of Object.entries(init.headers as Record<string, string>)) {
          headers[k.toLowerCase()] = v
        }
      }
      calls.push({ url: String(url), headers })
      const next = queue.shift()
      if (next instanceof Error) throw next
      if (!next) throw new Error('no queued response')
      return next
    })
  )
  return { calls }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

// -- 1/2/3: 帧解析（跨 chunk / 单 chunk 多帧 / keep-alive） ------------------

describe('SseFrameParser', () => {
  it('reassembles a frame split across arbitrary chunk boundaries', () => {
    const parser = new SseFrameParser()
    expect(parser.push('id: 1\neve')).toEqual([])
    expect(parser.push('nt: process.status\ndata: {"d')).toEqual([])
    const frames = parser.push('ata": {"stage": "drafting"}}\n\n')
    expect(frames).toHaveLength(1)
    expect(frames[0]).toMatchObject({ id: 1, event: 'process.status' })
    expect(JSON.parse(frames[0].data)).toEqual({ data: { stage: 'drafting' } })
  })

  it('handles CRLF terminators and multi-line data', () => {
    const parser = new SseFrameParser()
    const frames = parser.push('id: 1\r\nevent: answer.delta\r\ndata: line1\r\ndata: line2\r\n\r\n')
    expect(frames).toHaveLength(1)
    expect(frames[0].data).toBe('line1\nline2')
  })

  it('emits several frames from one chunk and ignores keep-alive comments', () => {
    const parser = new SseFrameParser()
    const frames = parser.push(
      ': keep-alive\n\nid: 1\nevent: a\ndata: {}\n\nid: 2\nevent: b\ndata: {}\n\n'
    )
    expect(frames).toHaveLength(2)
    expect(frames.map((f) => f.event)).toEqual(['a', 'b'])
  })

  it('never advances the cursor for a malformed id', () => {
    const parser = new SseFrameParser()
    const frames = parser.push('id: not-a-number\nevent: x\ndata: {}\n\n')
    expect(frames).toHaveLength(1)
    expect(frames[0].id).toBeUndefined()
  })

  it('flushes a trailing buffer frame without terminator', () => {
    const parser = new SseFrameParser()
    expect(parser.push('id: 9\nevent: run.completed\ndata: {}')).toEqual([])
    const frames = parser.flush()
    expect(frames).toHaveLength(1)
    expect(frames[0].event).toBe('run.completed')
  })
})

// -- 4/7: 重连携带 Last-Event-ID；非数字 id 不推进 ---------------------------

describe('streamRun reconnect', () => {
  it('reconnects with Last-Event-ID of the last numeric id', async () => {
    const first = okResponse(
      sseBody([
        'id: 1\nevent: process.status\ndata: {"data":{"stage":"retrieving"}}\n\n',
        'id: 2\nevent: process.status\ndata: {"data":{"stage":"verifying"}}\n\n'
      ])
    )
    const second = okResponse(
      sseBody(['id: 3\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n'])
    )
    const { calls } = stubFetch([first, second])

    const events: AgentSseEvent[] = []
    await streamRun(OPTS, 'r-1', { onEvent: (e) => events.push(e), retryDelayMs: 0 })

    expect(calls).toHaveLength(2)
    expect(calls[1].headers['last-event-id']).toBe('2')
    expect(events.map((e) => e.id)).toEqual([1, 2, 3])
  })

  it('does not advance the cursor for malformed ids', async () => {
    const first = okResponse(
      sseBody([
        'id: bogus\nevent: process.status\ndata: {}\n\n',
        'id: 5\nevent: process.status\ndata: {}\n\n'
      ])
    )
    const second = okResponse(sseBody(['id: 6\nevent: run.completed\ndata: {}\n\n']))
    const { calls } = stubFetch([first, second])

    await streamRun(OPTS, 'r-1', { onEvent: () => {}, retryDelayMs: 0 })
    expect(calls[1].headers['last-event-id']).toBe('5')
  })

  it('gives up after the bounded retries and reports network failure', async () => {
    stubFetch([new Error('down'), new Error('down'), new Error('down'), new Error('down')])
    await expect(
      streamRun(OPTS, 'r-1', { onEvent: () => {}, retryDelayMs: 0 })
    ).rejects.toMatchObject({ kind: 'network' })
  })
})

it('keeps multi-byte characters intact when bytes split across chunks', async () => {
  // 回归：中文 UTF-8 字节被 TCP 任意拆分（这里每 7 字节一切，必然切开
  // '体温' 等多字节字符）时，decoder 必须跨 chunk 复用，否则出现 U+FFFD
  const frame =
    'id: 1\nevent: answer.delta\ndata: {"data":{"delta":"体温超过38.5建议门诊就诊"}}\n\n' +
    'id: 2\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n'
  const bytes = new TextEncoder().encode(frame)
  const chunks: Uint8Array[] = []
  for (let i = 0; i < bytes.length; i += 7) chunks.push(bytes.subarray(i, i + 7))
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(ch)
      c.close()
    }
  })
  stubFetch([okResponse(stream)])

  const events: AgentSseEvent[] = []
  await streamRun(OPTS, 'r-1', { onEvent: (e) => events.push(e) })
  expect(events.map((e) => e.event)).toEqual(['answer.delta', 'run.completed'])
  const delta = (events[0].data as { data: { delta: string } }).data.delta
  expect(delta).toBe('体温超过38.5建议门诊就诊')
  expect(delta).not.toContain('\uFFFD')
})

// -- 8/9/10/11: 终态唯一、迟到事件、stream.error、abort -----------------------

describe('streamRun terminal & failure handling', () => {
  it('stops immediately after run.completed and dispatches nothing later', async () => {
    const res = okResponse(
      sseBody([
        'id: 1\nevent: run.completed\ndata: {"data":{"result":"answered"}}\n\n',
        'id: 2\nevent: answer.delta\ndata: {"data":{"delta":"LATE"}}\n\n'
      ])
    )
    const { calls } = stubFetch([res])

    const events: AgentSseEvent[] = []
    await streamRun(OPTS, 'r-1', { onEvent: (e) => events.push(e) })
    expect(calls).toHaveLength(1)
    expect(events.map((e) => e.event)).toEqual(['run.completed']) // late delta never dispatched
  })

  it('dispatches stream.error once and never reconnects', async () => {
    const res = okResponse(
      sseBody([
        'event: stream.error\ndata: {"detail":"internal whatever"}\n\n',
        'id: 9\nevent: answer.delta\ndata: {}\n\n'
      ])
    )
    const { calls } = stubFetch([res])

    const events: AgentSseEvent[] = []
    await streamRun(OPTS, 'r-1', { onEvent: (e) => events.push(e) })
    expect(calls).toHaveLength(1)
    expect(events).toHaveLength(1)
    expect(events[0].event).toBe('stream.error')
  })

  it('does not reconnect after abort and reports aborted', async () => {
    const controller = new AbortController()
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(encoder.encode('id: 1\nevent: process.status\ndata: {}\n\n'))
        setTimeout(() => {
          controller.abort()
          c.error(new Error('aborted'))
        }, 10)
      }
    })
    const { calls } = stubFetch([okResponse(stream)])

    await expect(
      streamRun(OPTS, 'r-1', { onEvent: () => {}, signal: controller.signal, retryDelayMs: 0 })
    ).rejects.toMatchObject({ kind: 'aborted' })
    expect(calls).toHaveLength(1)
  })

  it('maps HTTP failures to fixed safe kinds', async () => {
    for (const [status, kind] of [
      [401, 'unauthorized'],
      [403, 'forbidden'],
      [429, 'rate_limited'],
      [503, 'unavailable']
    ] as const) {
      stubFetch([errorResponse(status)])
      await expect(streamRun(OPTS, 'r-1', { onEvent: () => {} })).rejects.toMatchObject({ kind })
      vi.unstubAllGlobals()
    }
  })
})

// -- 12: 凭据只进 Authorization 头，不进 URL ----------------------------------

describe('credential hygiene', () => {
  it('sends the credential only in the Authorization header, never in the URL', async () => {
    const { calls } = stubFetch([
      okResponse(sseBody(['id: 1\nevent: run.completed\ndata: {}\n\n']))
    ])
    await streamRun(OPTS, 'r-1', { onEvent: () => {} })

    expect(calls).toHaveLength(1)
    expect(calls[0].url).not.toContain(CREDENTIAL)
    expect(calls[0].headers.authorization).toBe(`Bearer ${CREDENTIAL}`)
    expect(new URL(calls[0].url).search).toBe('')
  })

  it('createSession posts the credential in the header only', async () => {
    const { calls } = stubFetch([jsonResponse({ session_id: 's-1' })])
    const sessionId = await createSession(OPTS)
    expect(sessionId).toBe('s-1')
    expect(calls[0].url).not.toContain(CREDENTIAL)
    expect(calls[0].headers.authorization).toBe(`Bearer ${CREDENTIAL}`)
  })
})

// -- 13/14: 配置缺失失败关闭 ---------------------------------------------------

describe('resolveAgentConfig fails closed', () => {
  it('returns null when either value is missing or the placeholder', () => {
    expect(resolveAgentConfig({})).toBeNull()
    expect(resolveAgentConfig({ VITE_GCMW_AGENT_API_BASE: BASE })).toBeNull()
    expect(
      resolveAgentConfig({
        VITE_GCMW_AGENT_API_BASE: BASE,
        VITE_GCMW_DEMO_CREDENTIAL: 'YOUR_DEMO_CREDENTIAL_HERE'
      })
    ).toBeNull()
    expect(
      resolveAgentConfig({
        VITE_GCMW_AGENT_API_BASE: '   ',
        VITE_GCMW_DEMO_CREDENTIAL: 'x'
      })
    ).toBeNull()
  })

  it('returns options and strips trailing slashes when configured', () => {
    expect(
      resolveAgentConfig({
        VITE_GCMW_AGENT_API_BASE: `${BASE}/`,
        VITE_GCMW_DEMO_CREDENTIAL: 'real-looking-demo-token'
      })
    ).toEqual({ baseUrl: BASE, credential: 'real-looking-demo-token' })
  })

  it('rejects non-loopback demo addresses at config time (zero fetch)', () => {
    for (const bad of [
      'http://192.168.1.10:8001/api/v1',
      'http://0.0.0.0:8001',
      'https://demo.example.com/api/v1',
      'ws://127.0.0.1:8001',
      'not a url'
    ]) {
      expect(
        resolveAgentConfig({
          VITE_GCMW_AGENT_API_BASE: bad,
          VITE_GCMW_DEMO_CREDENTIAL: 'real-looking-demo-token'
        })
      ).toBeNull()
    }
    // loopback HTTP hosts stay allowed
    for (const good of ['http://localhost:8001/api/v1', 'http://127.0.0.1:8001/api/v1']) {
      expect(
        resolveAgentConfig({
          VITE_GCMW_AGENT_API_BASE: good,
          VITE_GCMW_DEMO_CREDENTIAL: 'real-looking-demo-token'
        })
      ).not.toBeNull()
    }
  })
})

// -- 6/8/11: AgentRunView 顺序拼接 / 重放去重 / 终态后忽略 --------------------

describe('AgentRunView reducer', () => {
  // wire envelope: the SSE `data:` JSON carries the business payload under `.data`
  const envelope = (inner: object): { data: object } => ({ data: inner })

  it('concatenates answer.delta in id order', () => {
    const view = new AgentRunView()
    const a = view.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: '体温' }) })
    const b = view.apply({ id: 2, event: 'answer.delta', data: envelope({ delta: '超过' }) })
    const c = view.apply({ id: 3, event: 'answer.delta', data: envelope({ delta: '38.5' }) })
    expect([a?.appendDelta, b?.appendDelta, c?.appendDelta]).toEqual(['体温', '超过', '38.5'])
    expect(view.answer).toBe('体温超过38.5')
  })

  it('drops replayed events so deltas never duplicate', () => {
    const view = new AgentRunView()
    view.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: 'A' }) })
    expect(view.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: 'A' }) })).toBeNull()
    expect(
      view.apply({ id: 0, event: 'answer.delta', data: envelope({ delta: 'OLD' }) })
    ).toBeNull()
    expect(view.answer).toBe('A')
  })

  it('ignores every event after the terminal one', () => {
    const view = new AgentRunView()
    const terminal = view.apply({
      id: 9,
      event: 'run.completed',
      data: envelope({ result: 'refused_no_answer' })
    })
    expect(terminal?.terminalResult).toBe('refused_no_answer')
    expect(view.terminalResult).toBe('refused_no_answer')
    expect(
      view.apply({ id: 10, event: 'answer.delta', data: envelope({ delta: 'LATE' }) })
    ).toBeNull()
    expect(
      view.apply({ id: 11, event: 'run.completed', data: envelope({ result: 'answered' }) })
    ).toBeNull()
    expect(view.terminalResult).toBe('refused_no_answer')
  })

  it('maps process stages and accepts only valid citations', () => {
    const view = new AgentRunView()
    expect(
      view.apply({ id: 1, event: 'process.status', data: envelope({ stage: 'retrieving' }) })
    ).toEqual({ stage: 'processing' })
    expect(
      view.apply({ id: 2, event: 'process.status', data: envelope({ stage: 'streaming' }) })
    ).toEqual({ stage: 'answering' })

    const good = {
      source_id: 'faq-fever',
      knowledge_version: '1',
      content_hash: 'a'.repeat(64),
      title: '发热护理须知'
    }
    const applied = view.apply({
      id: 3,
      event: 'answer.completed',
      data: envelope({ citations: [good] })
    })
    expect(applied?.citations?.[0].source_id).toBe('faq-fever')

    const bad = { ...good, content_hash: 'zz' }
    expect(
      view.apply({ id: 4, event: 'answer.completed', data: envelope({ citations: [bad] }) })
    ).toBeNull()
    expect(view.citations).toHaveLength(1)
  })

  it('gates the answered terminal behind verified completion', () => {
    const view = new AgentRunView()
    // delta 已累积但还没有合法 answer.completed：answered 不得交付
    view.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: '部分答案' }) })
    expect(
      view.apply({ id: 2, event: 'run.completed', data: envelope({ result: 'answered' }) })
        ?.terminalResult
    ).toBe('answered')
    expect(view.canDeliverAnswer()).toBe(false) // 缺 answer.completed

    // answer.completed 缺引用/引用非法：同样不交付
    const view2 = new AgentRunView()
    view2.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: '答案' }) })
    view2.apply({ id: 2, event: 'answer.completed', data: envelope({ citations: [] }) })
    view2.apply({ id: 3, event: 'run.completed', data: envelope({ result: 'answered' }) })
    expect(view2.canDeliverAnswer()).toBe(false)

    // 非空答案 + 合法 completed + 有效引用 → 交付
    const view3 = new AgentRunView()
    view3.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: '体温超过38.5' }) })
    view3.apply({
      id: 2,
      event: 'answer.completed',
      data: envelope({
        citations: [{ source_id: 's', knowledge_version: 'v', content_hash: 'a'.repeat(64) }]
      })
    })
    view3.apply({ id: 3, event: 'run.completed', data: envelope({ result: 'answered' }) })
    expect(view3.canDeliverAnswer()).toBe(true)
  })

  it('rejects non-string deltas', () => {
    const view = new AgentRunView()
    expect(view.apply({ id: 1, event: 'answer.delta', data: envelope({ delta: 42 }) })).toBeNull()
    expect(view.answer).toBe('')
  })
})

// -- 引用校验辅助 -------------------------------------------------------------

describe('validateCitations', () => {
  it('accepts a valid triple and rejects broken ones', () => {
    const good = {
      source_id: 's',
      knowledge_version: 'v',
      content_hash: 'b'.repeat(64)
    }
    expect(validateCitations({ citations: [good] })).not.toBeNull()
    expect(validateCitations({ citations: [] })).toBeNull()
    expect(validateCitations({ citations: [{ ...good, source_id: '' }] })).toBeNull()
    expect(validateCitations({ citations: [{ ...good, knowledge_version: '' }] })).toBeNull()
    expect(validateCitations({ citations: [{ ...good, content_hash: 'a'.repeat(63) }] })).toBeNull()
    expect(validateCitations(null)).toBeNull()
    expect(validateCitations({ nope: true })).toBeNull()
  })
})
