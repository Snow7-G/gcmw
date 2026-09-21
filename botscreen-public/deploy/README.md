# Deployment bundle — packaged Electron UI + voice bridge + demo Agent API

Three processes on one host, all on loopback:

| Unit | Scope | Listens | What it is |
|---|---|---|---|
| `gcmw-agent-demo.service` | **user** unit (operator session) | `127.0.0.1:8001` | Pinned demo Agent API (`/api/v1`) |
| `qa-server.service` | system | `127.0.0.1:8000` | Voice bridge (agent mode) — mic channel, `/chat`, SSE |
| `botscreen.service` | system | none | Packaged Electron UI (framed, closable window) |

Nothing here listens on `0.0.0.0`. The Agent API carries fixed low-entropy
credentials and the bridge carries a device credential, so both are deliberately
loopback-only — and that is enforced twice: in the unit file and in the code.

## What is NOT in this directory (on purpose)

This bundle was reconstructed from a live deployment by reviewing each file
individually. The following were deliberately **not** migrated:

- SSH private/public keys and anything under an operator's `.ssh`;
- real `EnvironmentFile` contents (only `*.env.example` placeholders are here);
- journal/log dumps and crash metadata;
- machine-specific values: hostnames, LAN addresses, user home paths, release SHAs.

If a value is machine-specific it belongs in the rendered output, not here.

## Layout

```
deploy/
├── README.md                      ← this file
├── bin/
│   ├── render-assets.sh           substitute @@PLACEHOLDER@@ + verify
│   ├── run-demo-agent-api.py      Agent API entry (loopback, packaged-renderer CORS)
│   └── start-botscreen-ui         X-display resolver + window launcher
├── config/
│   ├── demo-agent-api.env.example
│   └── voice-bridge.env.example
└── systemd/
    ├── gcmw-agent-demo.service
    ├── qa-server.service
    └── botscreen.service
```

## Render (do this first)

Every path, user and interpreter is a placeholder, so the tracked files carry no
host specifics:

| Placeholder | Meaning | Example |
|---|---|---|
| `@@GCMW_USER@@` | user that runs the two system units | `dev` |
| `@@GCMW_HOME@@` | that user's home | `/home/dev` |
| `@@GCMW_SERVER_DIR@@` | server sources (`qa_server.py` lives here) | `/opt/gcmw/current/botscreen-public/server` |
| `@@GCMW_RELEASE_DIR@@` | packaged Electron release | `/opt/gcmw/current` |
| `@@GCMW_DEPLOY_DIR@@` | this bundle, installed | `/opt/gcmw/deploy/current` |
| `@@GCMW_CONFIG_DIR@@` | `EnvironmentFile` directory | `/opt/gcmw/config` |
| `@@GCMW_VENV_PYTHON@@` | interpreter | `/opt/gcmw/venv/bin/python` |
| `@@GCMW_RCPATH@@` | app resource path passed to Electron | `/opt/gcmw/src` |

```bash
deploy/bin/render-assets.sh --out /tmp/gcmw-rendered \
  --root /opt/gcmw --user dev --home /home/dev
```

The renderer substitutes, then asserts **no `@@GCMW_` survives**, then runs
`bash -n` on the shell files, an AST parse on the Python entry, and
`systemd-analyze verify` where available. Nothing is installed by it.

Use a stable path for the deploy bundle (`<root>/deploy/current`) so the unit
files do not have to change when the release SHA changes: the release is reached
through `<root>/current`, which is swapped atomically.

## Startup order

The Agent API is a **user** unit, so the two system units cannot order against
it. Bring it up first, in the operator's own session:

```bash
# 1) Agent API (user unit, operator session)
systemctl --user enable --now gcmw-agent-demo.service
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/api/v1/health/ready

# 2) voice bridge (system)
sudo install -m 0600 config/voice-bridge.env  /opt/gcmw/config/voice-bridge.env   # after filling it
sudo systemctl enable --now qa-server.service
curl -s http://127.0.0.1:8000/health

# 3) packaged UI (system)
sudo systemctl enable --now botscreen.service
```

Starting the bridge before the Agent API is *safe* (it answers with the fixed
fail-closed copy and never falls back to the legacy model) but the demo will not
answer properly.

## Health checks

```bash
curl -s -o /dev/null -w '8001 ready: %{http_code}\n' http://127.0.0.1:8001/api/v1/health/ready
curl -s http://127.0.0.1:8000/health                     # expect "answer_backend":"agent"
ss -ltn | grep -E ':8000|:8001'                          # both must be 127.0.0.1 only
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' -d '{"question":"发热怎么办"}'
```

Acceptance on this bundle also covers: `Origin: null` preflight/response for the
packaged renderer, unknown origins receiving **no** `Access-Control-Allow-Origin`,
and a missing/unknown device credential still being rejected by the API's own
entry guard (CORS never bypasses authentication).

## Rollback

The release tree and the units are independent knobs.

```bash
# A) back to the previous release (atomic symlink swap)
sudo ln -sfn /opt/gcmw/releases/<PREVIOUS_SHA> /opt/gcmw/current.new
sudo mv -Tf /opt/gcmw/current.new /opt/gcmw/current
sudo systemctl restart qa-server.service botscreen.service

# B) back to the previous unit files
sudo install -m 0644 /var/backups/gcmw/qa-server.service.<STAMP>  /etc/systemd/system/qa-server.service
sudo install -m 0644 /var/backups/gcmw/botscreen.service.<STAMP>  /etc/systemd/system/botscreen.service
sudo systemctl daemon-reload && sudo systemctl restart qa-server.service botscreen.service

# C) Agent API back to the pre-migration interpreter/entry
systemctl --user daemon-reload && systemctl --user restart gcmw-agent-demo.service
```

Always back up the unit files **before** installing new ones
(`install -d -m 0755 /var/backups/gcmw && cp -a <unit> /var/backups/gcmw/<unit>.<timestamp>`).

## The framed-window requirement

`DEBUG` selects between two very different windows (`src/main/windowMode.ts`):

- **set** (the unit pins `Environment=DEBUG=1`) → framed, closable, not pinned
  above other windows, no refocus-on-blur, close is honoured;
- **unset** → kiosk: fullscreen, frameless, `skipTaskbar`, always-on-top,
  refocus on blur, close swallowed.

Two consequences this bundle encodes:

1. `botscreen.service` uses `Restart=on-failure`, **not** `always`. A user
   clicking close is a clean exit (code 0), so the window must stay closed. The
   launcher `exec`s the app so that exit code reaches systemd unchanged.
2. `DISPLAY` is **resolved, not hardcoded**. On the host this bundle came from,
   `:0` is the kiosk session and its window manager does not reparent client
   windows at all: a plain `xclock` stays an undecorated 300x300 window on `:0`
   while the operator's own session wraps the same client in
   `mutter-x11-frames` (title bar + borders). A packaged Electron window is not
   even mapped on `:0` — the main process hangs. So "framed and closable" is only
   achievable on a display the operator's session decorates;
   `bin/start-botscreen-ui` picks that display (or honours an existing `DISPLAY`
   whose socket exists) and **fails loudly** if it cannot find one.

## Operational notes (observed, not theoretical)

- **A hung Electron instance ignores SIGTERM.** If the UI ever hangs, `restart`
  blocks until systemd's `TimeoutStopSec` (default 90 s) and then SIGKILLs it;
  those SIGKILLs leave apport crash dialogs on the desktop (`/var/crash`). Either
  wait it out or `systemctl kill -s KILL botscreen.service` deliberately.
- **Never start GUI processes with `nohup`/`setsid` for diagnosis** on this host:
  a detached Electron survives the shell that spawned it and becomes an orphan
  (observed: two ~370 MB orphans that had to be SIGKILLed). Use
  `systemd-run --user --unit=<name> --collect …` instead — it stops cleanly.
- **Demo knowledge is seeded by the code, not by hand.** The synthetic corpus in
  `app/knowledge/demo_seed.py` is re-applied on every start through the existing
  review lifecycle and is idempotent (identical content ⇒ zero writes, no new
  versions). Do not patch the corpus on the host: that produced a drift between
  the release tree and git on the previous deployment.

## Not yet verified on hardware

The following were **not** machine-verified when this bundle was added, and must
be signed off on the target host:

1. `Origin: null` CORS for the packaged renderer (preflight *and* actual response);
2. window has a title bar / close button, and closing it does not respawn it;
3. the new QA page loads;
4. **a real spoken utterance through the physical microphone** — an API smoke
   test cannot stand in for this;
5. active TTL reclaim (short-TTL environment) and the idempotent-reconcile path
   after an uncertain Session create.
