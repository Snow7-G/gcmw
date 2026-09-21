import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Hardening contract tests: pin the security-relevant wiring in the Electron
 * main/preload sources and the renderer CSP so that regressions (sandbox off,
 * unsafe-eval, shell.openExternal reintroduced, third-party preload imports,
 * dev URL trusted in production, …) fail CI.
 */

function readProjectFile(relativePath: string): string {
  const file = join(process.cwd(), relativePath)
  if (!existsSync(file)) {
    throw new Error(`project file not found at ${file} (run vitest from the package root)`)
  }
  return readFileSync(file, 'utf-8')
}

const mainSource = readProjectFile('src/main/index.ts')
const indexHtml = readProjectFile('src/renderer/index.html')
const preloadSource = readProjectFile('src/preload/index.ts')
const rendererMainSource = readProjectFile('src/renderer/src/main.ts')

function csp(): Map<string, string[]> {
  const meta = /http-equiv="Content-Security-Policy"[\s\S]*?content="([\s\S]*?)"/.exec(indexHtml)
  if (!meta) throw new Error('renderer CSP meta not found')
  const directives = new Map<string, string[]>()
  for (const part of meta[1].split(';')) {
    const [name, ...sources] = part.trim().split(/\s+/)
    if (name) directives.set(name, sources)
  }
  return directives
}

function directive(name: string): string[] {
  const sources = csp().get(name)
  if (!sources) throw new Error(`CSP directive ${name} missing`)
  return sources
}

describe('renderer CSP contract', () => {
  it('defaults and scripts are strictly self-only', () => {
    expect(directive('default-src')).toEqual(["'self'"])
    expect(directive('script-src')).toEqual(["'self'"])
    expect(indexHtml).not.toContain("'unsafe-eval'")
  })

  it('rc:/data:/blob: stay confined to media and images', () => {
    expect(directive('img-src').sort()).toEqual(["'self'", 'blob:', 'data:', 'rc:'])
    expect(directive('media-src').sort()).toEqual(["'self'", 'blob:', 'data:', 'rc:'])
  })

  it('locks down frames, objects and base URL manipulation', () => {
    expect(directive('frame-src')).toEqual(["'none'"])
    expect(directive('object-src')).toEqual(["'none'"])
    expect(directive('base-uri')).toEqual(["'none'"])
  })

  it('pins connect-src to the two local QA backends instead of all localhost ports', () => {
    // legacy voice service (8000) + Agent demo service (8001), exact hosts,
    // no wildcards and no other localhost ports
    expect(directive('connect-src')).toEqual([
      "'self'",
      'http://127.0.0.1:8000',
      'http://127.0.0.1:8001'
    ])
  })

  it('keeps inline styles for MathJax/UI but grants no worker or font sources', () => {
    // style attributes are set by MathJax v3 SVG output and UI components;
    // verified via the Electron smoke that blocking them breaks math layout.
    expect(directive('style-src')).toEqual(["'self'", "'unsafe-inline'"])
    // MathJax v3 (mathjax-full) typesets on the main thread: no worker-src and
    // no font-src grants needed (v4 would have needed worker handling).
    expect(csp().has('worker-src')).toBe(false)
    expect(csp().has('font-src')).toBe(false)
  })
})

describe('MathJax renderer contract', () => {
  it('loads the worker-free v3 component from mathjax-full', () => {
    expect(rendererMainSource).toContain("import 'mathjax-full/es5/tex-svg.js'")
    expect(rendererMainSource).not.toContain("import 'mathjax/es5/tex-svg.js'")
    expect(rendererMainSource).not.toContain('enableSpeech')
  })
})

describe('Electron webPreferences contract (sandbox)', () => {
  it('enables sandbox, context isolation and keeps node out of the renderer', () => {
    expect(mainSource).toMatch(/sandbox:\s*true/)
    expect(mainSource).toMatch(/contextIsolation:\s*true/)
    expect(mainSource).toMatch(/nodeIntegration:\s*false/)
    expect(mainSource).toMatch(/webviewTag:\s*false/)
    expect(mainSource).toMatch(/devTools:\s*!app\.isPackaged/)
  })
})

describe('preload contract (sandbox-compatible, no third-party requires)', () => {
  it('only exposes a frozen minimal window.api through contextBridge', () => {
    expect(preloadSource).toContain("contextBridge.exposeInMainWorld('api'")
    expect(preloadSource).toContain('Object.freeze(api)')
    expect(preloadSource).not.toContain("exposeInMainWorld('electron'")
    expect(preloadSource).not.toContain('window.electron')
    expect(preloadSource).not.toContain('@electron-toolkit/preload')
    expect(preloadSource).not.toMatch(/from\s+'node:/)
    expect(preloadSource).not.toMatch(/require\(/)
  })
})

describe('trusted-origin wiring contract', () => {
  it('trusts ELECTRON_RENDERER_URL only in dev (is.dev gate)', () => {
    expect(mainSource).toMatch(
      /devUrl:\s*is\.dev\s*\?\s*process\.env\.ELECTRON_RENDERER_URL\s*:\s*undefined/
    )
    expect(mainSource).toContain('rendererDir')
  })

  it('denies all new windows and never opens external URLs through shell', () => {
    expect(mainSource).toMatch(/setWindowOpenHandler/)
    expect(mainSource).toMatch(/action:\s*'deny'/)
    expect(mainSource).not.toMatch(/openExternal/)
  })

  it('guards will-navigate with the trusted renderer origin', () => {
    expect(mainSource).toMatch(/will-navigate/)
    expect(mainSource).toMatch(/isTrustedUrl/)
  })

  it('registers every resource IPC channel through the trusted wrapper', () => {
    for (const channel of ['getyml', 'getmd', 'hasyml', 'hasmd']) {
      expect(mainSource).toContain(`registerTrustedIpc(ipcMain, '${channel}'`)
    }
    expect(mainSource).not.toMatch(/ipcMain\.handle\(/)
  })

  it('denies permission requests in kiosk mode', () => {
    expect(mainSource).toMatch(/setPermissionRequestHandler/)
    expect(mainSource).toMatch(/setPermissionCheckHandler/)
  })
})

/**
 * DEBUG=1 must produce a normal window: framed, closable, not pinned, not
 * refocusing. The failure mode these tests exist for is precisely the one we
 * hit in deployment — a lock applied *outside* the kiosk branch (the old
 * `setAlwaysOnTop` ran unconditionally), or a lock that quietly lost its guard.
 */
describe('window mode contract (kiosk lock vs closable developer window)', () => {
  it('derives every lock from the single resolved mode', () => {
    expect(mainSource).toContain("import { resolveWindowMode } from './windowMode'")
    expect(mainSource).toMatch(/const windowMode = resolveWindowMode\(process\.env\)/)
    expect(mainSource).toMatch(/windowMode\.kiosk\s*\?/)
  })

  it('keeps always-on-top behind the mode flag', () => {
    expect(mainSource).toMatch(
      /if \(windowMode\.alwaysOnTop\) \{\s*\n\s*mainWindow\.setAlwaysOnTop\(/
    )
  })

  it('keeps blur-refocus and close-blocking behind the mode flags', () => {
    const blurGuard = /windowMode\.refocusOnBlur\) \{\s*\n\s*mainWindow\.on\('blur'/
    const closeGuard = /windowMode\.blockClose\) \{\s*\n\s*mainWindow\.on\('close'/
    expect(mainSource).toMatch(blurGuard)
    expect(mainSource).toMatch(closeGuard)
  })

  it('no window lock is decided by an inline DEBUG check any more', () => {
    expect(mainSource).not.toMatch(/process\.env\.DEBUG\s*\?/)
    expect(mainSource).not.toMatch(/if \(!process\.env\.DEBUG\)/)
  })

  it('keeps packaged devTools disabled', () => {
    expect(mainSource).toMatch(/devTools:\s*!app\.isPackaged/)
  })
})

describe('smoke probe contract', () => {
  it('is gated to dev only — packaged builds can never enable it', () => {
    expect(mainSource).toMatch(
      /const isSmokeRun = !app\.isPackaged && process\.env\.ELECTRON_SMOKE === '1'/
    )
    expect(mainSource).toMatch(/if \(isSmokeRun\)/)
  })

  it('never relaunches on a crashed renderer during smoke runs', () => {
    const goneIndex = mainSource.indexOf("app.on('render-process-gone'")
    expect(goneIndex).toBeGreaterThan(-1)
    const handler = mainSource.slice(goneIndex, goneIndex + 300)
    expect(handler).toContain('smokeFail(')
    expect(handler).toContain('app.relaunch()')
    expect(handler.indexOf('smokeFail(')).toBeLessThan(handler.indexOf('app.relaunch()'))
    expect(handler).toContain('if (isSmokeRun)')
  })

  it('requires real MathJax output (mjx-container) and probes all four APIs', () => {
    expect(mainSource).toContain("el.textContent = '$x^2$'")
    expect(mainSource).toContain("el.querySelectorAll('mjx-container')")
    expect(mainSource).toContain("typeof window.api.hasYaml === 'function'")
    expect(mainSource).toContain("result.mathjax === 'typeset-ok'")
  })
})

describe('CI gate contract', () => {
  const ciYml = readProjectFile('../.github/workflows/ci.yml')

  it('runs the electron smoke with the real Chromium sandbox (no --no-sandbox)', () => {
    expect(ciYml).not.toContain('--no-sandbox')
    expect(ciYml).toContain('xvfb-run')
    expect(ciYml).toContain('ELECTRON_SMOKE')
    expect(ciYml).toMatch(/unprivileged user namespaces/)
  })

  it('keeps the required gates wired: build, preload purity, smoke job name', () => {
    expect(ciYml).toMatch(/name: Frontend electron smoke/)
    expect(ciYml).toContain('check-preload-purity.mjs')
    expect(ciYml).toContain('npm run build')
  })
})
