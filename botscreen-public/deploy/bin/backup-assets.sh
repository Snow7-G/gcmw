#!/usr/bin/env bash
#
# Snapshot everything an install overwrites, under ONE stamp.
#
# Usage:
#   backup-assets.sh --stamp STAMP [--backup-dir /var/backups/gcmw]
#                     [--system-unit-dir /etc/systemd/system]
#                     [--user-unit-dir /home/<operator>/.config/systemd/user]
#                     [--deploy-dir <root>/deploy/current] [--root /opt/gcmw]
#                     [--server-dir DIR]
#
# Why a single stamp: the previous procedure timestamped per file, and that is
# how it ended up backing up only qa-server.service and then overwriting
# botscreen.service and the user unit with no copy at all — leaving the rollback
# to reference files that never existed.
#
# A target that is not there yet is recorded ABSENT rather than skipped, so the
# rollback knows a first-time install has to be undone by REMOVING the file.
#
# File operations only. This never calls systemctl: a restore needs a
# daemon-reload per scope, and those need the matching privileges.
set -euo pipefail

stamp=""
backup_dir="/var/backups/gcmw"
system_unit_dir="/etc/systemd/system"
user_unit_dir="${HOME}/.config/systemd/user"
root="/opt/gcmw"
deploy_dir=""
server_dir=""

while [ $# -gt 0 ]; do
  case "$1" in
    --stamp) stamp="$2"; shift 2 ;;
    --backup-dir) backup_dir="$2"; shift 2 ;;
    --system-unit-dir) system_unit_dir="$2"; shift 2 ;;
    --user-unit-dir) user_unit_dir="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --server-dir) server_dir="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$stamp" ] || {
  echo "--stamp is required: one stamp must cover every file" >&2
  exit 2
}
: "${deploy_dir:=${root}/deploy/current}"
: "${server_dir:=${root}/current/botscreen-public/server}"

file_mode() {
  # GNU stat, then BSD/macOS stat.
  stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1"
}

resolve_link() {
  readlink -f "$1" 2>/dev/null || readlink "$1" 2>/dev/null || echo "(none)"
}

snapshot="${backup_dir}/${stamp}"
[ ! -e "$snapshot" ] || {
  echo "refusing to overwrite an existing snapshot: $snapshot" >&2
  exit 2
}
mkdir -p "$snapshot/files"

# The SAME list, in the SAME order, drives rollback-assets.sh. Keep them in step.
paths=(
  "${system_unit_dir}/qa-server.service"
  "${system_unit_dir}/botscreen.service"
  "${user_unit_dir}/gcmw-agent-demo.service"
  "${deploy_dir}/bin/start-botscreen-ui"
  "${deploy_dir}/bin/run-demo-agent-api.py"
)

printf 'status\tmode\toriginal\tstored\n' > "$snapshot/MANIFEST.tsv"

index=0
copied=0
absent=0
for path in "${paths[@]}"; do
  index=$((index + 1))
  stored="files/${index}_$(basename "$path")"
  if [ -f "$path" ]; then
    cp -p "$path" "$snapshot/$stored"
    printf 'COPIED\t%s\t%s\t%s\n' "$(file_mode "$path")" "$path" "$stored" \
      >> "$snapshot/MANIFEST.tsv"
    copied=$((copied + 1))
    echo "  backed up $path"
  else
    printf 'ABSENT\t-\t%s\t-\n' "$path" >> "$snapshot/MANIFEST.tsv"
    absent=$((absent + 1))
    echo "  ABSENT    $path (did not exist)"
  fi
done

# Version information: enough to tell which release and source revision the
# backed-up units belonged to.
{
  echo "stamp=${stamp}"
  echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=${root}"
  echo "current=$(resolve_link "${root}/current")"
  echo "server_head=$(git -C "$server_dir" rev-parse HEAD 2>/dev/null || echo '(unknown)')"
} > "$snapshot/VERSION"

echo
echo "SNAPSHOT -> $snapshot"
echo "  ${copied} file(s) copied, ${absent} recorded absent"
echo "Next: install the new assets, then ROLLBACK with"
echo "  rollback-assets.sh --stamp ${stamp} --backup-dir ${backup_dir}"
