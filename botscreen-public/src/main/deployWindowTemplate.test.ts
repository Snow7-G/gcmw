import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { resolveWindowMode } from './windowMode'

/**
 * The deployment bundle must not be able to disagree with the window policy.
 *
 * `resolveWindowMode` decides whether the packaged app is a locked kiosk or a
 * framed, closable window; the systemd unit is what actually runs in the field.
 * If the unit ever stops setting `DEBUG`, or goes back to `Restart=always`, the
 * deployed behaviour silently reverts to "fullscreen lock that pops back up
 * after you close it" — so this reads the real unit file and feeds its real
 * value through the real policy function instead of reminding anyone in prose.
 */

function readProjectFile(relativePath: string): string {
  const file = join(process.cwd(), relativePath)
  if (!existsSync(file)) {
    throw new Error(`project file not found at ${file} (run vitest from the package root)`)
  }
  return readFileSync(file, 'utf-8')
}

const uiUnit = readProjectFile('deploy/systemd/botscreen.service')
const launcher = readProjectFile('deploy/bin/start-botscreen-ui')

function directives(unit: string, key: string): string[] {
  return unit
    .split('\n')
    .filter((line) => line.startsWith(`${key}=`))
    .map((line) => line.slice(key.length + 1).trim())
}

describe('botscreen.service window contract', () => {
  it('sets DEBUG to a non-empty value, which means a normal window', () => {
    const debug = directives(uiUnit, 'Environment')
      .filter((entry) => entry.startsWith('DEBUG='))
      .map((entry) => entry.slice('DEBUG='.length))

    expect(debug).toHaveLength(1)
    expect(debug[0]).not.toBe('')

    // The unit's ACTUAL value, through the ACTUAL policy function: a closable,
    // unframed-lock-free window is the requirement, not a comment about it.
    expect(resolveWindowMode({ DEBUG: debug[0] })).toEqual({
      kiosk: false,
      alwaysOnTop: false,
      refocusOnBlur: false,
      blockClose: false
    })
  })

  it('never restarts the window after a clean user-initiated close', () => {
    expect(directives(uiUnit, 'Restart')).toEqual(['on-failure'])
    expect(uiUnit).not.toMatch(/^Restart=always$/m)
  })

  it('does not hardcode the kiosk display', () => {
    expect(uiUnit).not.toMatch(/Environment=DISPLAY=/)
    expect(uiUnit).toContain('bin/start-botscreen-ui')
  })
})

describe('start-botscreen-ui launcher', () => {
  it('defaults DEBUG on, so a manual launch matches the unit', () => {
    const match = /export DEBUG="\$\{DEBUG:-([^}]*)\}"/.exec(launcher)
    expect(match).not.toBeNull()
    expect(resolveWindowMode({ DEBUG: match![1] })).toEqual({
      kiosk: false,
      alwaysOnTop: false,
      refocusOnBlur: false,
      blockClose: false
    })
  })

  it('hands the app its exit code through exec', () => {
    expect(launcher).toMatch(/\nexec /)
    expect(launcher).not.toContain('nohup')
    expect(launcher).not.toContain('setsid')
  })

  it('resolves the display from the session that owns this uid', () => {
    expect(launcher).toContain('stat -c %u')
    expect(launcher).toContain('id -u')
    // The kiosk display cannot decorate windows, so it must never be a fallback.
    expect(launcher).not.toContain('DISPLAY=":0"')
  })
})
