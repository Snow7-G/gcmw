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
`systemd-analyze verify` where available. **Nothing is installed by it.**

`--deploy-dir` defaults to `<root>/deploy/current`, so omitting it is safe: the
rendered units point at the **installed** bundle, never at this checkout.
(Defaulting to the source directory produced units that exec'd unrendered
templates in the repository — and the render still exited 0.) Override it only for
a rehearsal layout.

## Install

**Install the rendered files, never the templates.** The units exec
`<root>/deploy/current/bin/…`, so that directory must exist and hold the rendered
`bin/`; a unit pointing back into the checkout is exactly the bug this section
prevents.

| Rendered | Goes to | Mode | Owner |
|---|---|---|---|
| `bin/render-assets.sh` | `<root>/deploy/current/bin/` | `0755` | `root:root` |
| `bin/start-botscreen-ui` | `<root>/deploy/current/bin/` | `0755` | `root:root` |
| `bin/run-demo-agent-api.py` | `<root>/deploy/current/bin/` | `0755` | `root:root` |
| `systemd/qa-server.service` | `/etc/systemd/system/` | `0644` | `root:root` |
| `systemd/botscreen.service` | `/etc/systemd/system/` | `0644` | `root:root` |
| `systemd/gcmw-agent-demo.service` | `~/.config/systemd/user/` | `0644` | the operator |
| `config/voice-bridge.env` | `<root>/config/` | `0600` | `root:root` |
| `config/demo-agent-api.env` | `<root>/config/` | `0600` | the operator |
| `README.md` (**not rendered**) | `<root>/deploy/current/README.md` | `0644` | `root:root` |

This README is copied straight from the bundle, not rendered — its placeholder
table documents the token **names**, so substituting them would destroy it.

```bash
sudo install -d -m 0755 /opt/gcmw/deploy/current/bin /opt/gcmw/config
sudo install -m 0755 <rendered>/bin/start-botscreen-ui     /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/run-demo-agent-api.py  /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/render-assets.sh       /opt/gcmw/deploy/current/bin/
sudo install -m 0644 <bundle>/README.md                    /opt/gcmw/deploy/current/README.md

# back up the units BEFORE overwriting them
sudo install -d -m 0755 /var/backups/gcmw
sudo cp -a /etc/systemd/system/qa-server.service \
  /var/backups/gcmw/qa-server.service.$(date +%Y%m%d-%H%M%S)
sudo install -m 0644 <rendered>/systemd/qa-server.service /etc/systemd/system/
sudo install -m 0644 <rendered>/systemd/botscreen.service /etc/systemd/system/
sudo systemctl daemon-reload

# the Agent API unit belongs to the OPERATOR's session: install it as that user
install -m 0644 <rendered>/systemd/gcmw-agent-demo.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

Two permission traps worth stating out loud:

- `voice-bridge.env` is read by **PID 1** on behalf of a system unit, so
  `0600 root:root` is correct — the service user never needs to open it.
- `demo-agent-api.env` is read by the **user** manager. A `root:root 0600` file
  there is silently ignored (`EnvironmentFile=-…` is optional), so the offline
  MockProvider would be used even after you configure a cloud provider. Keep it
  readable by that user (`0600`, owner = operator), or omit it entirely for mock mode.

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

Two independent things can be wrong — the **release** the units point at, and the
**deployment assets** (rendered `bin/`, unit files). Fix them in that order, then
bring the stack back in *startup* order. **Never restart the UI first.**

The release is reached through `<root>/current`, which is swapped atomically, so
the unit files do not change when the release SHA changes. But **a swap restarts
nothing**: a running process keeps the image it started with, so pointing
`<root>/current` at an older release has no effect on the Agent API or the UI until
they are restarted. That is the trap in doing only step 2.

### 1. Decide what to restore

| Symptom | Restore |
|---|---|
| the new release behaves badly | step 2 (release) |
| a unit was overwritten, or a launcher is broken | step 3 (assets + units) |
| both | step 2, then step 3 |

### 2. Release (atomic symlink swap)

```bash
sudo ln -sfn /opt/gcmw/releases/<PREVIOUS_SHA> /opt/gcmw/current.new
sudo mv -Tf /opt/gcmw/current.new /opt/gcmw/current
readlink -f /opt/gcmw/current          # confirm before restarting anything
```

### 3. Deployment assets and units

The units exec `<root>/deploy/current/bin/…`, so the **rendered** launcher and
entry point must be restored as well — restoring the symlink alone is not enough.

```bash
sudo install -d -m 0755 /opt/gcmw/deploy/current/bin
sudo install -m 0755 <rendered>/bin/start-botscreen-ui     /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/run-demo-agent-api.py  /opt/gcmw/deploy/current/bin/
sudo install -m 0644 <bundle>/README.md                    /opt/gcmw/deploy/current/README.md

# prefer the timestamped copies taken during install
sudo install -m 0644 /var/backups/gcmw/qa-server.service.<STAMP>  /etc/systemd/system/qa-server.service
sudo install -m 0644 /var/backups/gcmw/botscreen.service.<STAMP>  /etc/systemd/system/botscreen.service
sudo systemctl daemon-reload
```

### 4. Bring the stack back — startup order, matching scope

```bash
# a) Agent API first. It is a USER unit: run it as its OWNING user, without sudo.
systemctl --user restart gcmw-agent-demo.service
curl -sf -o /dev/null -w '8001 ready: %{http_code}\n' http://127.0.0.1:8001/api/v1/health/ready

# b) only once 8001 answers: the voice bridge (system)
sudo systemctl restart qa-server.service
curl -s http://127.0.0.1:8000/health          # expect "answer_backend":"agent"

# c) and only then the UI (system)
sudo systemctl restart botscreen.service
```

Scopes are not interchangeable: `sudo systemctl` cannot manage the user unit, and
`systemctl --user` cannot manage the system ones. Running a restart in the wrong
scope fails (or applies to nothing), which is how a "rolled back" host ends up
still serving the old code.

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
