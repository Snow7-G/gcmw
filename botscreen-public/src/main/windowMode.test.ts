import { describe, expect, it } from 'vitest'
import { resolveWindowMode } from './windowMode'

/**
 * The deployed kiosk screen and the developer/acceptance window must never be
 * confused: the kiosk window is locked (frameless, fullscreen, pinned, refuses
 * to close) and the developer window is a normal, closable, framed window.
 * Every one of those locks is decided here.
 */
describe('resolveWindowMode', () => {
  it('locks the window down when DEBUG is unset (deployed kiosk)', () => {
    expect(resolveWindowMode({})).toEqual({
      kiosk: true,
      alwaysOnTop: true,
      refocusOnBlur: true,
      blockClose: true
    })
  })

  it('returns a normal closable window whenever DEBUG is set', () => {
    for (const value of ['1', 'true', 'yes', '0', 'DEBUG']) {
      const mode = resolveWindowMode({ DEBUG: value })
      expect(mode).toEqual({
        kiosk: false,
        alwaysOnTop: false,
        refocusOnBlur: false,
        blockClose: false
      })
    }
  })

  it('treats an empty DEBUG as unset (kiosk stays locked)', () => {
    expect(resolveWindowMode({ DEBUG: '' }).kiosk).toBe(true)
  })

  it('never half-applies the locks: all four flags move together', () => {
    for (const env of [{}, { DEBUG: '1' }]) {
      const mode = resolveWindowMode(env)
      const flags = [mode.kiosk, mode.alwaysOnTop, mode.refocusOnBlur, mode.blockClose]
      expect(new Set(flags).size).toBe(1)
    }
  })
})
