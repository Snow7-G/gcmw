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
│   ├── backup-assets.sh           snapshot what an install overwrites (one stamp)
│   ├── rollback-assets.sh         restore a snapshot (file operations only)
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
| `bin/backup-assets.sh` | `<root>/deploy/current/bin/` | `0755` | `root:root` |
| `bin/rollback-assets.sh` | `<root>/deploy/current/bin/` | `0755` | `root:root` |
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
# 0) back up EVERY file this install overwrites, under ONE stamp.
#    A per-file timestamp is how the old procedure backed up qa-server.service and
#    then overwrote botscreen.service and the user unit with no copy at all.
STAMP="$(date -u +%Y%m%d-%H%M%S)"
sudo install -d -m 0755 /var/backups/gcmw
sudo bin/backup-assets.sh --stamp "${STAMP}" --backup-dir /var/backups/gcmw \
  --user-unit-dir /home/<operator>/.config/systemd/user   # explicit: this runs as root
#    Anything that does not exist yet is recorded ABSENT, so rolling back a
#    first-time install removes it again instead of leaving it behind.
#    The snapshot is BUILT under a temporary .<stamp>.partial.xxxxxx directory and
#    published under the stamp only once all five files are in it. A run that
#    dies halfway therefore leaves no stamp anyone can name — no half-snapshot
#    that a restore would read as "some files old, some new".

# 1) rendered sources
sudo install -d -m 0755 /opt/gcmw/deploy/current/bin /opt/gcmw/config
sudo install -m 0755 <rendered>/bin/start-botscreen-ui     /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/run-demo-agent-api.py  /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/backup-assets.sh       /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/rollback-assets.sh     /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered>/bin/render-assets.sh       /opt/gcmw/deploy/current/bin/
sudo install -m 0644 <bundle>/README.md                    /opt/gcmw/deploy/current/README.md

# 2) system units
sudo install -m 0644 <rendered>/systemd/qa-server.service /etc/systemd/system/
sudo install -m 0644 <rendered>/systemd/botscreen.service /etc/systemd/system/
sudo systemctl daemon-reload

# 3) the Agent API unit belongs to the OPERATOR's session: install it as that user
install -m 0644 <rendered>/systemd/gcmw-agent-demo.service ~/.config/systemd/user/
systemctl --user daemon-reload

# 4) start in startup order — 8001, health, 8000, UI (see below)
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

Everything the install overwrote comes back from the snapshot taken in the
`## Install` step — the three units, the rendered launcher, the Agent entry point.
Restoring the symlink alone is not enough, because the units exec
`<root>/deploy/current/bin/…`.

```bash
# what snapshots exist, and exactly what this one would do
sudo bin/rollback-assets.sh --list
sudo bin/rollback-assets.sh --stamp <STAMP> --dry-run \
  --user-unit-dir /home/<operator>/.config/systemd/user

# then apply it: every file at its recorded mode
sudo bin/rollback-assets.sh --stamp <STAMP> \
  --user-unit-dir /home/<operator>/.config/systemd/user

sudo systemctl daemon-reload          # system scope
systemctl --user daemon-reload        # user scope — the Agent unit is the operator's
```

**`--stamp` is required and there is no "newest wins" default.** The snapshot you
least want to restore is the one left by an install that died halfway, and sorting
directory names is exactly what picks that one. `--list` shows the stamps that
were actually published; a leftover `.<stamp>.partial.xxxxxx` directory is never
listed and cannot be named.

**Nothing is written until the whole snapshot has been checked.** The restore
first proves *all* of the following, and refuses with a non-zero exit — changing
nothing at all — if any one fails:

| Checked before the first write | Why it matters |
|---|---|
| the manifest's header | a truncated file must not be read as data |
| exactly five entries, no repeats | a short manifest used to restore one file and **exit 0** |
| every status and every mode | `COPIED`/`ABSENT`, and a real octal mode |
| every `COPIED` file is present | the old code restored the first four, kept the fifth new, and exited 1 |
| stored paths stay inside the snapshot | a manifest is not allowed to name `/etc/hosts` |
| targets == the five files under **the directories you passed** | otherwise the manifest decides where `cp`/`rm` goes, and a temp `--deploy-dir` still writes to the live install |

That last row is why the directory arguments are not decoration: they are the
constraint the manifest is checked against, and the destination is always taken
from them. `--dry-run` runs the identical checks.

This is not a transaction, and it is not claimed to be one: a real disk error
between two writes can still leave the tree half restored. Two things keep a
half-done restore from being mistaken for a finished one. A defect discoverable
*in advance* — a partial snapshot, a truncated manifest, a redirected target — is
rejected before anything is touched. And if a write fails anyway, the restore
stops there, exits non-zero, and prints `THE TREE IS PARTIALLY RESTORED` followed
by the exact list of targets it had already changed; the snapshot is untouched, so
re-running the same command finishes the job.

`MANIFEST.tsv` records one of two things per file, and both are handled:

| Entry | Meaning | Rollback does |
|---|---|---|
| `COPIED` | the file existed before that install | puts the old content back, at the recorded mode |
| `ABSENT` | the file did not exist before that install | **removes** it — restoring the previous state for a first-time install means removing it |

`--dry-run` prints every removal before it happens. The snapshot's `VERSION` file
(`current` symlink target, server `HEAD`, timestamp) is how you tell **which**
revision the backed-up assets belonged to.

One thing the snapshot deliberately does *not* store: the release tree itself.
`<rendered-PREVIOUS>` below means the render output from the **older** revision —
re-rendering the current (broken) revision and installing that would undo the
rollback:

```bash
# only needed if the rendered entry points themselves changed
sudo install -d -m 0755 /opt/gcmw/deploy/current/bin
sudo install -m 0755 <rendered-PREVIOUS>/bin/start-botscreen-ui     /opt/gcmw/deploy/current/bin/
sudo install -m 0755 <rendered-PREVIOUS>/bin/run-demo-agent-api.py  /opt/gcmw/deploy/current/bin/
sudo install -m 0644 <bundle>/README.md                             /opt/gcmw/deploy/current/README.md
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

A rollback is only possible if `bin/backup-assets.sh` ran **before** the install —
that is step 0 above, and it is the only step whose absence cannot be repaired
afterwards.

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
   test cannot stand in for this. *Partially observed* on the host: the physical
   device produced a ROS ASR line (`麦克风识别: …`), which covers **capture +
   recognition**. This item stays open: a full voice acceptance needs one round to
   show recognition → `/chat` → an Agent answer → **actual playback**, and the ASR
   line alone does not close that loop;
5. active TTL reclaim (short-TTL environment) and the idempotent-reconcile path
   after an uncertain Session create.
