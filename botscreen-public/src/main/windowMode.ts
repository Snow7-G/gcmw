/**
 * Window mode policy — one pure decision behind two very different windows.
 *
 * The DEPLOYED kiosk screen wants a locked window: fullscreen, frameless, pinned
 * above everything, refocusing on blur and refusing to close. Local development,
 * rehearsal, acceptance and the "framed window must stay closable" requirement
 * want the exact opposite.
 *
 * `DEBUG` selects between them. Any NON-EMPTY value (including `"0"`) counts as
 * the developer window — that is this project's historical behaviour
 * (`!process.env.DEBUG`) and it is pinned here so nobody has to guess: to get the
 * kiosk window you must leave `DEBUG` UNSET.
 *
 * Keeping the decision in one pure function lets the contract be asserted
 * without booting Electron, and makes "every lock comes from the same knob"
 * checkable in CI.
 */

export interface WindowMode {
  /** frameless + fullscreen + skipTaskbar + `kiosk: true` */
  readonly kiosk: boolean
  /** pinned above other windows (`setAlwaysOnTop`) */
  readonly alwaysOnTop: boolean
  /** grab focus back whenever the window loses it */
  readonly refocusOnBlur: boolean
  /** swallow user-initiated close */
  readonly blockClose: boolean
}

const KIOSK: WindowMode = {
  kiosk: true,
  alwaysOnTop: true,
  refocusOnBlur: true,
  blockClose: true
}

const DEVELOPER: WindowMode = {
  kiosk: false,
  alwaysOnTop: false,
  refocusOnBlur: false,
  blockClose: false
}

export function resolveWindowMode(env: NodeJS.ProcessEnv = process.env): WindowMode {
  return env.DEBUG ? DEVELOPER : KIOSK
}
