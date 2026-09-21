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
#   <deploy dir>                       this bundle (self-locating)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

root="/opt/gcmw"
user="$(id -un)"
home="${HOME}"
server_dir=""
release_dir=""
deploy_dir="${here%/bin}"
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
: "${config_dir:=${root}/config}"
: "${venv_python:=${root}/venv/bin/python}"
: "${rcpath:=${root}/src}"

mkdir -p "$out"
cp -R "${here%/bin}/systemd" "${here%/bin}/bin" "${here%/bin}/config" "$out/"

substitute() {
  local file="$1"
  python3 - "$file" "$user" "$home" "$server_dir" "$release_dir" \
    "$deploy_dir" "$config_dir" "$venv_python" "$rcpath" <<'PYEOF'
import pathlib
import sys

# Names are composed rather than written out in full so that this renderer itself
# never contains a complete placeholder token — otherwise the "no placeholder
# may survive" check below would trip over the tool that does the substituting.
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
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for name, value in zip(NAMES, sys.argv[2:]):
    text = text.replace("@@GCMW_" + name + "@@", value)
path.write_text(text, encoding="utf-8")
PYEOF
}

echo "== 1) substitute placeholders =="
while IFS= read -r file; do
  substitute "$file"
  echo "  rendered ${file#"$out"/}"
done < <(grep -rlE '@@GCMW_[A-Z_]+@@' "$out" || true)

echo "== 2) no placeholder may survive =="
if grep -rnE '@@GCMW_[A-Z_]+@@' "$out"; then
  echo "ERROR: unresolved placeholders above" >&2
  exit 1
fi
echo "  none left"

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
    systemd-analyze verify "$unit" 2>&1 | sed "s|^|  |" || true
  done
else
  echo "  systemd-analyze not available on this host — verify on the target"
fi

echo
echo "RENDER DONE -> $out"
echo "Next: install systemd/*.service, place config/*.env (mode 600), then follow"
echo "the startup order in README.md."
