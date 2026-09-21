#!/usr/bin/env bash
#
# Render the deployment bundle: substitute @@PLACEHOLDER@@ values and verify the
# result. Nothing in this bundle is machine-specific on disk — every path, user
# and interpreter comes from here, so the tracked files carry no credentials, no
# hostnames and no release SHA.
#
# Usage:
#   render-assets.sh --out DIR [--root /opt/gcmw] [--user dev] [--home /home/dev]
#                    [--server-dir DIR] [--release-dir DIR] [--deploy-dir DIR]
#                    [--config-dir DIR] [--venv-python PATH] [--rcpath DIR]
#
# Defaults follow one conventional layout under --root:
#   <root>/current                     release tree (symlink, swapped atomically)
#   <root>/current/botscreen-public/server     server sources
#   <root>/venv/bin/python             interpreter
#   <root>/config                      EnvironmentFile directory
#   <root>/deploy/current              this bundle once INSTALLED (see README)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle="${here%/bin}"

root="/opt/gcmw"
user="$(id -un)"
home="${HOME}"
server_dir=""
release_dir=""
deploy_dir=""
config_dir=""
venv_python=""
rcpath=""
out=""

while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --user) user="$2"; shift 2 ;;
    --home) home="$2"; shift 2 ;;
    --server-dir) server_dir="$2"; shift 2 ;;
    --release-dir) release_dir="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --config-dir) config_dir="$2"; shift 2 ;;
    --venv-python) venv_python="$2"; shift 2 ;;
    --rcpath) rcpath="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$out" ] || { echo "--out DIR is required" >&2; exit 2; }
: "${server_dir:=${root}/current/botscreen-public/server}"
: "${release_dir:=${root}/current}"
# NOT $bundle: the default must be the path the bundle is INSTALLED to, because
# the rendered units exec `$deploy_dir/bin/...`. Defaulting to the source checkout
# produced units that pointed at unrendered templates in the repository, and the
# render still exited 0. Override with --deploy-dir for a rehearsal layout.
: "${deploy_dir:=${root}/deploy/current}"
: "${config_dir:=${root}/config}"
: "${venv_python:=${root}/venv/bin/python}"
: "${rcpath:=${root}/src}"

# Copy FILES only, and only the three asset directories: a build artefact that
# happens to sit in bin/ (a __pycache__ from importing the entry point, say) is
# not part of the bundle and must not be rendered, scanned or installed.
mkdir -p "$out/systemd" "$out/config" "$out/bin"
cp "$bundle/systemd/"*.service "$out/systemd/"
cp "$bundle/config/"*.example "$out/config/"
for source in "$bundle/bin/"*; do
  [ -f "$source" ] || continue
  cp "$source" "$out/bin/"
done

echo "== 1) substitute placeholders =="
python3 - "$out" "$user" "$home" "$server_dir" "$release_dir" \
  "$deploy_dir" "$config_dir" "$venv_python" "$rcpath" <<'PYEOF'
import pathlib
import sys

# Names are composed rather than written out in full so this renderer never
# contains a complete placeholder token itself.
NAMES = (
    "USER",
    "HOME",
    "SERVER_DIR",
    "RELEASE_DIR",
    "DEPLOY_DIR",
    "CONFIG_DIR",
    "VENV_PYTHON",
    "RCPATH",
)

root = pathlib.Path(sys.argv[1])
values = dict(zip(NAMES, sys.argv[2:]))

rendered: list[str] = []
skipped: list[str] = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        skipped.append(str(path.relative_to(root)))  # not a template; leave alone
        continue
    replaced = text
    for name, value in values.items():
        replaced = replaced.replace("@@GCMW_" + name + "@@", value)
    # A placeholder that is not in NAMES stays in place on purpose: step 2 below
    # reports it as a template bug instead of silently substituting nothing.
    if replaced != text:
        path.write_text(replaced, encoding="utf-8")
        rendered.append(str(path.relative_to(root)))

for name in rendered:
    print(f"  rendered {name}")
for name in skipped:
    print(f"  skipped (not a text template): {name}")
print(f"  {len(rendered)} rendered, {len(skipped)} skipped")
PYEOF

echo "== 2) no placeholder may survive =="
python3 - "$out" <<'PYEOF'
import pathlib
import re
import sys

COMPLETE = re.compile(r"@@GCMW_[A-Z_]+@@")
root = pathlib.Path(sys.argv[1])
offenders: list[str] = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for number, line in enumerate(text.splitlines(), 1):
        if COMPLETE.search(line):
            offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
if offenders:
    print("\n".join(f"  {entry}" for entry in offenders))
    print("ERROR: unresolved placeholders above", file=sys.stderr)
    raise SystemExit(1)
print("  none left")
PYEOF

echo "== 3) syntax check =="
for file in "$out"/bin/*; do
  case "$file" in
    *.py) python3 -c 'import ast,sys; ast.parse(open(sys.argv[1], encoding="utf-8").read())' "$file"
          echo "  python OK  $(basename "$file")" ;;
    *)    bash -n "$file"
          echo "  bash -n OK $(basename "$file")" ;;
  esac
done
chmod 0755 "$out"/bin/*

echo "== 4) systemd verify (best effort, needs systemd-analyze) =="
if command -v systemd-analyze >/dev/null 2>&1; then
  for unit in "$out"/systemd/*.service; do
    # Messages here are advisory: an unknown user or an interpreter path that
    # does not exist yet on this machine is expected before installation.
    systemd-analyze verify "$unit" 2>&1 | sed "s|^|  |" || true
  done
else
  echo "  systemd-analyze not available on this host — verify on the target"
fi

echo
echo "RENDER DONE -> $out"
echo "Next: install the RENDERED files (the ones under $out), never the templates:"
echo "  0) back up FIRST: bin/backup-assets.sh --stamp \$(date -u +%Y%m%d-%H%M%S)"
echo "     it snapshots every file the install overwrites; without it there is no"
echo "     rollback (bin/rollback-assets.sh restores from that snapshot)"
echo "  bin/          -> $deploy_dir/bin          (0755)  <- the units exec these"
echo "  systemd/*     -> /etc/systemd/system     (0644)  or ~/.config/systemd/user for the user unit"
echo "  config/*.env  -> $config_dir              (0600)"
echo "Also copy README.md (NOT rendered: its table documents the token names) to"
echo "  $deploy_dir/README.md   (0644)"
echo "Then daemon-reload and follow the startup order in README.md."
