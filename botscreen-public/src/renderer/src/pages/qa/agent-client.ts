/**
 * Minimal Agent API client for the /qa demo page (development/test only).
 *
 * Wire-up for: Session → Run → Manager → MedicalQA/RAG → Verifier → SSE.
 *
 * Why not native EventSource: the new Agent API requires
 * `Authorization: Bearer <credential>` on EVERY call, and EventSource cannot
 * set request headers. So the SSE stream is read with fetch + ReadableStream
 * and parsed here. The legacy voice/hardware SSE keeps using EventSource —
 * do NOT unify them in this slice.
 *
 * Security contract (must hold):
 * - the credential travels ONLY in the Authorization header — never in the
 *   URL, never logged, never thrown in an exception message;
 * - no wildcard CORS on the server side for this demo;
 * - server error bodies are NEVER surfaced: failures map to fixed
 *   `AgentFailureKind` values and the page renders fixed copy.
 *
 * The demo backend: `server/scripts/demo_showcase.py --serve` (synthetic
 * data, MockProvider, zero outbound calls). This client is NOT a production
 * front end.
 */

/** Fixed, safe failure kinds the page can map to copy — no server text. */
export type AgentFailureKind =
  | 'unauthorized'
  | 'forbidden'
  | 'rate_limited'
  | 'unavailable'
  | 'network'
  | 'aborted'

/** Thrown for every fatal stream/HTTP failure (kind is safe to render). */
export class AgentStreamError extends Error {
  constructor(readonly kind: AgentFailureKind) {
    super(`agent stream failed: ${kind}`)
    this.name = 'AgentStreamError'
  }
}

/** The single allowed demo backend URL — must equal the CSP connect-src entry. */
export const DEMO_BASE_URL = 'http://127.0.0.1:8001/api/v1'

/** Thrown when the demo env config is missing or still a placeholder. */
export class AgentConfigError extends Error {
  constructor() {
    super('agent demo configuration missing')
    this.name = 'AgentConfigError'
  }
}

/**
 * Read the demo config from import.meta.env-ish keys. Returns `null` when
 * either value is missing, blank, or still the committed placeholder — the
 * page then fails CLOSED with fixed copy (and never renders the values).
 */
export function resolveAgentConfig(env: {
  VITE_GCMW_AGENT_API_BASE?: string
  VITE_GCMW_DEMO_CREDENTIAL?: string
}): AgentClientOptions | null {
  const base = (env.VITE_GCMW_AGENT_API_BASE ?? '').trim()
  const credential = (env.VITE_GCMW_DEMO_CREDENTIAL ?? '').trim()
  if (!base || !credential || credential === 'YOUR_DEMO_CREDENTIAL_HERE') {
    return null
  }
  // Demo contract is FIXED (development/test only): exactly one loopback
  // base URL, matching the renderer CSP connect-src entry one-to-one. Any
  // other value — other loopback ports, LAN IPs, https remotes, garbage — is
  // rejected AT CONFIG TIME with zero fetches, so a "valid" config can never
  // be silently blocked by the CSP afterwards.
  const normalized = base.replace(/\/+$/, '')
  if (normalized !== DEMO_BASE_URL) return null
  return { baseUrl: normalized, credential }
}

export interface AgentClientOptions {
  baseUrl: string
  credential: string
}

export interface AgentSseEvent {
  /** numeric SSE id; undefined for frames without one (e.g. stream.error) */
  id?: number
  event: string
  data: unknown
}

export interface StreamOptions {
  lastEventId?: number
  signal?: AbortSignal
  onEvent: (event: AgentSseEvent) => void
  /** delay between reconnect attempts (ms); tests pass 0 */
  retryDelayMs?: number
}

export interface Citation {
  source_id: string
  knowledge_version: string
  content_hash: string
  title?: string
  source_uri?: string
}

/** Fixed hex alphabet for the 64-char SHA-256 content hash check. */
const HEX = '0123456789abcdef'

/**
 * Validate the citation triple (source_id + knowledge_version + content_hash
 * all non-empty, content_hash 64-hex). Returns the citation array as-is when
 * every entry passes, or null when the payload is missing/invalid — the page
 * then renders the answer WITHOUT the 资料来源 block instead of garbage.
 */
export function validateCitations(payload: unknown): Citation[] | null {
  if (!payload || typeof payload !== 'object') return null
  const raw = (payload as { citations?: unknown }).citations
  if (!Array.isArray(raw) || raw.length === 0) return null
  for (const item of raw) {
    if (!item || typeof item !== 'object') return null
    const c = item as Record<string, unknown>
    const hash = c.content_hash
    if (
      typeof c.source_id !== 'string' ||
      c.source_id.length === 0 ||
      typeof c.knowledge_version !== 'string' ||
      c.knowledge_version.length === 0 ||
      typeof hash !== 'string' ||
      hash.length !== 64 ||
      ![...hash].every((ch) => HEX.includes(ch))
    ) {
      return null
    }
  }
  return raw as Citation[]
}

// ---------------------------------------------------------------------------
// SSE frame parsing (kept pure so chunk-splitting is directly unit-testable)
// ---------------------------------------------------------------------------

export interface SseFrame {
  id?: number
  event: string
  data: string
}

/**
 * Incremental SSE parser. Feed chunks of any size (TCP split anywhere),
 * frames come out complete. Handles `\n\n` AND `\r\n\r\n` terminators,
 * `: keep-alive` comment frames, `id:` / `event:` / `data:` fields,
 * multi-line data, and a trailing buffer via `flush()`.
 */
export class SseFrameParser {
  private buffer = ''

  /** Feed one text chunk; returns every complete frame inside it. */
  push(chunk: string): SseFrame[] {
    this.buffer += chunk
    const frames: SseFrame[] = []
    for (;;) {
      const cut = this.findTerminator()
      if (!cut) break
      const block = this.buffer.slice(0, cut.index)
      this.buffer = this.buffer.slice(cut.index + cut.length)
      const frame = parseFrameBlock(block)
      if (frame) frames.push(frame)
    }
    return frames
  }

  /** End of stream: emit a frame out of any trailing buffer residue. */
  flush(): SseFrame[] {
    const rest = this.buffer
    this.buffer = ''
    const frame = parseFrameBlock(rest)
    return frame ? [frame] : []
  }

  /** Earliest of `\n\n` / `\r\n\r\n`, or null when none is complete yet. */
  private findTerminator(): { index: number; length: number } | null {
    const lf = this.buffer.indexOf('\n\n')
    const crlf = this.buffer.indexOf('\r\n\r\n')
    if (lf === -1 && crlf === -1) return null
    if (lf === -1) return { index: crlf, length: 4 }
    if (crlf === -1) return { index: lf, length: 2 }
    return crlf < lf ? { index: crlf, length: 4 } : { index: lf, length: 2 }
  }
}

function parseFrameBlock(block: string): SseFrame | null {
  let sawField = false
  let event = 'message'
  let id: number | undefined
  const dataLines: string[] = []

  for (const rawLine of block.split('\n')) {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
    if (line === '' || line.startsWith(':')) continue // blank / keep-alive
    if (line.startsWith('id:')) {
      sawField = true
      const value = line.slice(3).trim()
      // a NON-numeric id is ignored per the SSE spec and must NOT advance
      // the replay cursor
      if (/^\d+$/.test(value)) id = Number.parseInt(value, 10)
    } else if (line.startsWith('event:')) {
      sawField = true
      const value = line.slice(6)
      event = value.startsWith(' ') ? value.slice(1) : value
      if (event === '') event = 'message'
    } else if (line.startsWith('data:')) {
      sawField = true
      const value = line.slice(5)
      dataLines.push(value.startsWith(' ') ? value.slice(1) : value)
    }
    // unknown fields are ignored per the SSE spec
  }
  if (!sawField) return null
  return { id, event, data: dataLines.join('\n') }
}

// ---------------------------------------------------------------------------
// Typed HTTP helpers
// ---------------------------------------------------------------------------

function kindForStatus(status: number): AgentFailureKind {
  if (status === 401) return 'unauthorized'
  if (status === 403) return 'forbidden'
  if (status === 429) return 'rate_limited'
  return 'unavailable' // 503 and everything else: fixed safe copy
}

function authHeaders(credential: string): Record<string, string> {
  return { Authorization: `Bearer ${credential}` }
}

/** fetch wrapper: maps HTTP + transport failures to safe AgentStreamError. */
async function request(
  opts: AgentClientOptions,
  path: string,
  init: RequestInit
): Promise<unknown> {
  let res: Response
  try {
    res = await fetch(`${opts.baseUrl}${path}`, init)
  } catch {
    if (init.signal?.aborted) throw new AgentStreamError('aborted')
    throw new AgentStreamError('network')
  }
  if (!res.ok) throw new AgentStreamError(kindForStatus(res.status))
  try {
    return await res.json()
  } catch {
    throw new AgentStreamError('unavailable')
  }
}

/** Create one Agent session; returns the SERVER-issued session id. */
export async function createSession(
  opts: AgentClientOptions,
  signal?: AbortSignal
): Promise<string> {
  const data = (await request(opts, '/sessions', {
    method: 'POST',
    headers: { ...authHeaders(opts.credential), 'Content-Type': 'application/json' },
    body: JSON.stringify({ channel: 'text', locale: 'zh-CN' }),
    signal
  })) as { session_id?: unknown }
  if (typeof data.session_id !== 'string' || data.session_id.length === 0) {
    throw new AgentStreamError('unavailable')
  }
  return data.session_id
}

export interface AgentRun {
  run_id: string
  state: string
}

/** Create one run; `idempotencyKey` MUST be reused across retries of the
 * SAME question and regenerated for the next one. */
export async function createRun(
  opts: AgentClientOptions,
  sessionId: string,
  question: string,
  idempotencyKey: string,
  signal?: AbortSignal
): Promise<AgentRun> {
  const data = (await request(opts, '/agent/runs', {
    method: 'POST',
    headers: { ...authHeaders(opts.credential), 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      input: { type: 'text', text: question },
      idempotency_key: idempotencyKey
    }),
    signal
  })) as { run_id?: unknown; state?: unknown }
  if (typeof data.run_id !== 'string' || data.run_id.length === 0) {
    throw new AgentStreamError('unavailable')
  }
  return { run_id: data.run_id, state: String(data.state ?? '') }
}

/** Best-effort cancel (page unload / leaving the answer view). */
export async function cancelRun(
  opts: AgentClientOptions,
  runId: string,
  signal?: AbortSignal
): Promise<void> {
  await request(opts, `/agent/runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
    headers: authHeaders(opts.credential),
    signal
  })
}

// ---------------------------------------------------------------------------
// Stream reader (fetch + ReadableStream, bounded reconnect, Last-Event-ID)
// ---------------------------------------------------------------------------

const MAX_RECONNECTS = 3

/**
 * Read one run's SSE stream to its terminal event.
 *
 * - reconnects ONLY on network interruptions while the run is not terminal,
 *   at most MAX_RECONNECTS times, always carrying `Last-Event-ID: <cursor>`;
 * - the cursor advances ONLY for numeric ids;
 * - dispatches every frame to `onEvent` (already replay-deduplicated? NO —
 *   dedupe is the caller's job via `AgentRunView`, because only it knows
 *   which deltas were already rendered);
 * - `event: stream.error` is a SERVER decision: dispatched once, then the
 *   stream stops — never retried;
 * - abort never reconnects and throws `AgentStreamError('aborted')`.
 */
export async function streamRun(
  opts: AgentClientOptions,
  runId: string,
  options: StreamOptions
): Promise<void> {
  const { onEvent, retryDelayMs = 300, signal } = options
  let cursor = options.lastEventId ?? 0
  let reconnects = 0
  let done = false

  while (!done) {
    let res: Response
    try {
      res = await fetch(`${opts.baseUrl}/agent/runs/${encodeURIComponent(runId)}/events`, {
        headers: {
          ...authHeaders(opts.credential),
          Accept: 'text/event-stream',
          'Last-Event-ID': String(cursor)
        },
        signal
      })
    } catch {
      if (signal?.aborted) throw new AgentStreamError('aborted')
      if (reconnects < MAX_RECONNECTS) {
        reconnects += 1
        await delay(retryDelayMs, signal)
        continue
      }
      throw new AgentStreamError('network')
    }

    if (!res.ok) throw new AgentStreamError(kindForStatus(res.status))
    if (!res.body) throw new AgentStreamError('unavailable')

    const parser = new SseFrameParser()
    // ONE decoder per response, reused across chunks with streaming mode:
    // a fresh decoder per chunk would drop the incomplete trailing UTF-8
    // sequence and turn a Chinese character split across TCP chunks into U+FFFD
    const decoder = new TextDecoder()
    const reader = res.body.getReader()

    const dispatch = (frame: SseFrame): void => {
      if (typeof frame.id === 'number') cursor = frame.id
      onEvent({ id: frame.id, event: frame.event, data: parseData(frame.data) })
      if (frame.event === 'run.completed' || frame.event === 'stream.error') {
        done = true
      }
    }

    const dispatchBatch = (frames: SseFrame[]): void => {
      for (const frame of frames) {
        // a terminal frame stops dispatch INSIDE the same chunk: late frames
        // that share one Uint8Array with run.completed are never emitted
        if (done) return
        dispatch(frame)
      }
    }

    try {
      for (;;) {
        const { done: eof, value } = await reader.read()
        if (eof) {
          // flush any final incomplete multi-byte sequence, then any trailing
          // frame residue left in the parser buffer
          const tail = decoder.decode()
          if (tail.length > 0) dispatchBatch(parser.push(tail))
          dispatchBatch(parser.flush())
          break
        }
        dispatchBatch(parser.push(decoder.decode(value, { stream: true })))
        if (done) break
      }
    } catch {
      if (signal?.aborted) throw new AgentStreamError('aborted')
      // a mid-stream read fault falls through to the bounded reconnect below
    }

    if (done) return
    if (reconnects < MAX_RECONNECTS) {
      // unfinished stream (server closed early or mid-stream network fault):
      // bounded reconnect, always carrying the last confirmed cursor
      reconnects += 1
      await delay(retryDelayMs, signal)
      continue
    }
    throw new AgentStreamError('network')
  }
}

function parseData(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

async function delay(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(new AgentStreamError('aborted'))
      },
      { once: true }
    )
  })
}

// ---------------------------------------------------------------------------
// Run view: the stateful reducer the page renders from (pure & testable)
// ---------------------------------------------------------------------------

export type RunStage = 'processing' | 'answering'

export interface RunViewAction {
  /** process layer progressed (run.accepted / process.status) */
  stage?: RunStage
  /** an answer.delta string to append (order- and replay-safe) */
  appendDelta?: string
  /** validated citations from answer.completed */
  citations?: Citation[]
  /** run.completed result (exactly once) */
  terminalResult?: string
  /** server-sent stream.error: stop, show fixed copy */
  streamError?: boolean
}

/**
 * Reducer for ONE run's events. Guarantees:
 * - replayed events (numeric id <= applied cursor) are dropped entirely;
 * - deltas arrive in id order and are never duplicated;
 * - after the terminal event NOTHING is accepted any more (late deltas
 *   cannot rewrite the page);
 * - citation validation happens here so invalid triples never reach the UI.
 */
export class AgentRunView {
  private appliedId = 0
  private terminal = false
  answer = ''
  citations: Citation[] = []
  terminalResult: string | null = null
  /**
   * True only when `answer.completed` arrived AND its citations validated
   * (at least one valid triple). A medical answer is deliverable ONLY with
   * non-empty accumulated deltas + this flag — `run.completed/answered`
   * alone can never keep unverified text on the page.
   */
  answerVerified = false

  /** Delivery gate for the `answered` terminal result. */
  canDeliverAnswer(): boolean {
    return this.answer.length > 0 && this.answerVerified
  }

  apply(event: AgentSseEvent): RunViewAction | null {
    if (typeof event.id === 'number') {
      if (event.id <= this.appliedId) return null // replay: already shown
      this.appliedId = event.id
    }
    if (this.terminal) return null // late events after the terminal are void

    const payload = event.data as { data?: unknown } | null
    const inner =
      payload && typeof payload === 'object' && payload.data && typeof payload.data === 'object'
        ? (payload.data as Record<string, unknown>)
        : undefined

    switch (event.event) {
      case 'run.accepted':
      case 'process.status': {
        const stage = inner?.stage
        if (stage === 'streaming') return { stage: 'answering' }
        return { stage: 'processing' }
      }
      case 'answer.delta': {
        const delta = inner?.delta
        if (typeof delta !== 'string' || delta.length === 0) return null
        this.answer += delta
        return { appendDelta: delta }
      }
      case 'answer.completed': {
        const citations = validateCitations(inner)
        if (citations) {
          this.citations = citations
          // only a completed event WITH at least one valid triple verifies
          // the answer; an invalid/missing triple leaves answerVerified=false
          // so a later run.completed/answered cannot deliver unverified text
          this.answerVerified = true
          return { citations }
        }
        return null
      }
      case 'run.completed': {
        this.terminal = true
        this.terminalResult = typeof inner?.result === 'string' ? inner.result : 'unknown'
        return { terminalResult: this.terminalResult }
      }
      case 'stream.error': {
        this.terminal = true
        return { streamError: true }
      }
      default:
        return null
    }
  }
}
