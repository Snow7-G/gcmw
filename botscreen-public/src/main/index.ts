import { app, BrowserWindow, ipcMain } from 'electron'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { join } from 'node:path'
import { getMarkdown, getYaml, hasMarkdown, hasYaml } from './api'
import { RC_ALLOWED_EXTENSIONS, RC_MIME, resolveAllowedPath } from '../shared/pathSecurity'
import { protocol } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import { Readable } from 'node:stream'
import { isTrustedAppUrl, registerTrustedIpc } from './securityGuards'
import { resolveRange } from './range'
import { resolveWindowMode } from './windowMode'

let exitArmed = false
let exitInputBuffer = ''
let exitArmedAt = 0
const EXIT_PASSWORD = process.env.GCMW_EXIT_PASSWORD || ''

// Trusted renderer origin, used by the navigation guard and the IPC sender checks.
// ELECTRON_RENDERER_URL is only ever trusted in dev — the packaged build never loads it.
const rendererDir = join(__dirname, '../renderer')
const isTrustedUrl = (url: string): boolean =>
  isTrustedAppUrl(url, {
    devUrl: is.dev ? process.env.ELECTRON_RENDERER_URL : undefined,
    rendererDir
  })

// CI smoke mode: dev-only (never in packaged builds, even with the env set).
const isSmokeRun = !app.isPackaged && process.env.ELECTRON_SMOKE === '1'

// Window locks (kiosk / always-on-top / refocus / block-close) all derive from
// ONE decision — see windowMode.ts. DEBUG=1 must yield a normal, framed,
// closable window; every lock is applied only in kiosk mode.
const windowMode = resolveWindowMode(process.env)

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    ...(windowMode.kiosk
      ? {
          fullscreen: true,
          frame: false,
          kiosk: true,
          skipTaskbar: true
        }
      : {}),
    show: true,
    autoHideMenuBar: true,
    // ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webviewTag: false,
      devTools: !app.isPackaged
    }
  })

  mainWindow.on('ready-to-show', () => {
    if (windowMode.alwaysOnTop) {
      mainWindow.setAlwaysOnTop(true, 'screen-saver')
    }
    mainWindow.show()
    mainWindow.focus()
  })

  if (windowMode.refocusOnBlur) {
    mainWindow.on('blur', () => {
      mainWindow.focus()
    })
  }

  if (windowMode.blockClose) {
    mainWindow.on('close', (e) => {
      e.preventDefault()
    })
  }

  mainWindow.webContents.on('before-input-event', (_event, input) => {
    if (input.type !== 'keyDown') return
    if (input.isAutoRepeat || input.alt || input.control || input.meta || input.shift) {
      return
    }
    if (input.location !== 0) return

    if (input.key === 'Escape') {
      exitArmed = !exitArmed
      exitInputBuffer = ''
      exitArmedAt = Date.now()
      return
    }

    if (!exitArmed) return

    if (Date.now() - exitArmedAt > 1000) {
      exitArmed = false
      exitInputBuffer = ''
      return
    }

    if (input.key.length !== 1) {
      exitArmed = false
      exitInputBuffer = ''
      return
    }

    exitInputBuffer += input.key
    exitArmedAt = Date.now()

    if (!EXIT_PASSWORD.startsWith(exitInputBuffer)) {
      exitArmed = false
      exitInputBuffer = ''
      return
    }

    if (exitInputBuffer === EXIT_PASSWORD) {
      app.exit(0)
    }
  })

  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders
      }
    })
  })

  // Kiosk: never grant permission requests (camera, microphone, geolocation,
  // notifications, clipboard-read, …).
  mainWindow.webContents.session.setPermissionRequestHandler(
    (_webContents, _permission, callback) => callback(false)
  )
  mainWindow.webContents.session.setPermissionCheckHandler(() => false)

  app.on('render-process-gone', () => {
    // Smoke runs must fail fast: relaunching could loop forever on a crash.
    if (isSmokeRun) {
      smokeFail('render-process-gone')
      return
    }
    app.relaunch()
    app.exit()
  })

  // Deny all new windows by default. External opening must go through an approved IPC flow in future.
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  // The app never navigates away from its own document. Block everything else
  // (external links, drag-and-dropped files, …) as defense in depth.
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!isTrustedUrl(url)) {
      event.preventDefault()
    }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // CI smoke gate: verify preload/api/IPC and fail on preload/CSP/uncaught errors.
  // Never active in packaged builds (see isSmokeRun).
  if (isSmokeRun) {
    attachSmokeProbe(mainWindow)
  }
}

// ===== CI smoke probe (dev only, gated by isSmokeRun above) =====
const SMOKE_ERROR_PATTERN = /preload|Content Security Policy|Refused to|Unable to load|Uncaught/i
const smokeProblems: string[] = []

function smokeFail(reason: string): void {
  smokeProblems.push(reason)
  console.log(`SMOKE_RESULT ${JSON.stringify({ problems: smokeProblems })}`)
  app.exit(1)
}

function attachSmokeProbe(win: BrowserWindow): void {
  // Watchdog: never let CI hang if the renderer stalls.
  setTimeout(() => smokeFail('watchdog timeout'), 60_000)

  win.webContents.on('console-message', (details) => {
    const message = details.message
    const severity = details.level
    if ((severity === 'error' || severity === 'warning') && SMOKE_ERROR_PATTERN.test(message)) {
      smokeProblems.push(`console[${severity}]: ${message}`)
    }
  })
  win.webContents.on('preload-error', (_event, preloadPath, error) => {
    smokeFail(`preload-error ${preloadPath}: ${error.message ?? String(error)}`)
  })
  // Renderer crash: fail immediately instead of relying on the watchdog.
  win.webContents.on('render-process-gone', (_event, details) => {
    smokeFail(`render-process-gone: ${details.reason}`)
  })
  win.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
    if (isMainFrame) smokeFail(`did-fail-load ${code} ${description} ${url}`)
  })
  win.webContents.once('did-finish-load', () => {
    void runSmokeProbe(win).then(
      (code) => app.exit(code),
      (error) => smokeFail(String(error))
    )
  })
}

async function runSmokeProbe(win: BrowserWindow): Promise<number> {
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
  // Let first paint, fonts, lazy chunks and any MathJax startup settle.
  await sleep(2000)

  const apiShape = await win.webContents.executeJavaScript(
    `typeof window.api === 'object' &&
     typeof window.api.getYaml === 'function' &&
     typeof window.api.getMarkdown === 'function' &&
     typeof window.api.hasYaml === 'function' &&
     typeof window.api.hasMarkdown === 'function'`
  )
  const ipcOk = await win.webContents.executeJavaScript(
    `Promise.resolve(window.api.hasMarkdown('/__smoke_absent__.md')).then((v) => v === false)`
  )
  // Real math rendering: uses the project's own `$...$` delimiters (MJ.config.tex)
  // and requires an mjx-container to actually exist — typesetPromise resolving
  // without producing output must fail the smoke.
  const mathjaxProbe = await win.webContents.executeJavaScript(
    `(async () => {
      const mj = window.MathJax
      if (!mj || typeof mj.typesetPromise !== 'function') return 'no-mathjax-global'
      const el = document.createElement('div')
      el.textContent = '$x^2$'
      document.body.appendChild(el)
      try {
        await mj.typesetPromise([el])
        return el.querySelectorAll('mjx-container').length > 0
          ? 'typeset-ok'
          : 'typeset-no-output'
      } catch (e) {
        return 'typeset-failed: ' + (e && e.message ? e.message : String(e))
      }
    })()`
  )
  await sleep(1500) // flush late CSP/worker console messages

  const result = {
    apiShape: Boolean(apiShape),
    ipcRoundtrip: Boolean(ipcOk),
    mathjax: mathjaxProbe,
    problems: smokeProblems
  }
  const ok =
    result.apiShape &&
    result.ipcRoundtrip &&
    result.mathjax === 'typeset-ok' &&
    result.problems.length === 0
  console.log(`SMOKE_RESULT ${JSON.stringify(result)}`)
  return ok ? 0 : 1
}

function getResourceDir(): string {
  if (!process.env.RCPATH) return process.cwd()
  let rcPath = process.env.RCPATH
  // 在 Windows 上，将 Unix 风格绝对路径（如 /d/config）转换为 Windows 路径（D:/config）
  // bash (Git Bash/MSYS2) 的 /d/config 在 Node.js 的 path.resolve() 中不会被正确解析
  if (process.platform === 'win32' && /^\/[a-zA-Z]\//.test(rcPath)) {
    rcPath = rcPath[1] + ':' + rcPath.substring(2)
  }
  return path.resolve(rcPath)
}

const resourceDir = getResourceDir()

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'rc',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true
    }
  }
])

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  protocol.handle('rc', async (request) => {
    try {
      const rawPath = request.url.replace(/^rc:\/\//, '').split(/[?#]/)[0]
      const urlPath = decodeURIComponent(rawPath)
      const canonical = resolveAllowedPath(resourceDir, urlPath, RC_ALLOWED_EXTENSIONS)

      if (!canonical) {
        return new Response(null, { status: 404 })
      }

      const stat = fs.statSync(canonical)
      const size = stat.size
      const range = request.headers.get('range')

      const ext = path.extname(canonical).toLowerCase()
      const mimeType = RC_MIME[ext] || 'application/octet-stream'
      const commonHeaders = {
        'Content-Type': mimeType,
        'X-Content-Type-Options': 'nosniff'
      }

      // ===== Range 请求（视频播放关键）=====
      if (range) {
        const resolved = resolveRange(range, size)
        if (resolved.status === 416) {
          return new Response(null, {
            status: 416,
            headers: {
              'Content-Range': `bytes */${size}`,
              'Accept-Ranges': 'bytes'
            }
          })
        }

        // 显式 end 超出文件大小时 resolveRange 已裁剪到 size-1；
        // Content-Range / Content-Length 必须按裁剪后的实际范围声明。
        const nodeStream = fs.createReadStream(canonical, {
          start: resolved.start,
          end: resolved.end
        })
        const webStream = Readable.toWeb(nodeStream)

        return new Response(webStream as BodyInit, {
          status: 206,
          headers: {
            ...commonHeaders,
            'Accept-Ranges': 'bytes',
            'Content-Range': `bytes ${resolved.start}-${resolved.end}/${size}`,
            'Content-Length': String(resolved.length)
          }
        })
      }

      // ===== 普通请求 =====
      const nodeStream = fs.createReadStream(canonical)
      const webStream = Readable.toWeb(nodeStream)

      return new Response(webStream as BodyInit, {
        headers: {
          ...commonHeaders,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(size)
        }
      })
    } catch {
      return new Response(null, { status: 500 })
    }
  })

  registerTrustedIpc(ipcMain, 'getyml', getYaml, isTrustedUrl)
  registerTrustedIpc(ipcMain, 'getmd', getMarkdown, isTrustedUrl)
  registerTrustedIpc(ipcMain, 'hasyml', hasYaml, isTrustedUrl)
  registerTrustedIpc(ipcMain, 'hasmd', hasMarkdown, isTrustedUrl)

  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
