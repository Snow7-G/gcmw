#!/usr/bin/env bash
#
# Restore what backup-assets.sh snapshotted.
#
# Usage:
#   rollback-assets.sh [--stamp STAMP] [--backup-dir /var/backups/gcmw]
#                       [--system-unit-dir /etc/systemd/system]
#                       [--user-unit-dir /home/<operator>/.config/systemd/user]
#                       [--deploy-dir <root>/deploy/current]
#                       [--list] [--dry-run]
#
# With no --stamp the newest snapshot under --backup-dir is used (stamps are
# timestamps, so the names sort chronologically).
#
# An entry recorded ABSENT means the file did not exist before that install, so
# restoring the previous state REMOVES it. Every removal is printed; use
# --dry-run to see the whole plan first.
#
# File operations only. daemon-reload and the restart ORDER
# (8001 -> health -> 8000 -> UI) stay with the operator: the Agent API is a USER
# unit and the bridge/UI are system units, so they need different privileges.
set -euo pipefail

stamp=""
backup_dir="/var/backups/gcmw"
system_unit_dir="/etc/systemd/system"
user_unit_dir="${HOME}/.config/systemd/user"
root="/opt/gcmw"
deploy_dir=""
list_only=0
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stamp) stamp="$2"; shift 2 ;;
    --backup-dir) backup_dir="$2"; shift 2 ;;
    --system-unit-dir) system_unit_dir="$2"; shift 2 ;;
    --user-unit-dir) user_unit_dir="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --list) list_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${deploy_dir:=${root}/deploy/current}"

newest_stamp() {
  # Portable: no GNU-only find -printf.
  ls -1 "$backup_dir" 2>/dev/null | sort | tail -1
}

if [ "$list_only" = "1" ]; then
  echo "snapshots in ${backup_dir}:"
  ls -1 "$backup_dir" 2>/dev/null | sort | sed 's/^/  /' || true
  exit 0
fi

if [ -z "$stamp" ]; then
  stamp="$(newest_stamp)"
  [ -n "$stamp" ] || { echo "no snapshots under ${backup_dir}" >&2; exit 2; }
  echo "using the newest snapshot: ${stamp}"
fi

snapshot="${backup_dir}/${stamp}"
manifest="${snapshot}/MANIFEST.tsv"
[ -f "$manifest" ] || { echo "no MANIFEST.tsv in ${snapshot}" >&2; exit 2; }

do_or_print() {
  if [ "$dry_run" = "1" ]; then
    echo "  would: $*"
  else
    "$@"
  fi
}

echo "== rollback from ${snapshot} =="
if [ -f "${snapshot}/VERSION" ]; then
  sed 's/^/  /' "${snapshot}/VERSION"
fi
echo

while IFS=$'\t' read -r status mode original stored; do
  case "$status" in
    COPIED)
      source_path="${snapshot}/${stored}"
      [ -f "$source_path" ] || {
        echo "backup is incomplete: ${stored} is missing" >&2
        exit 1
      }
      do_or_print cp -f "$source_path" "$original"
      do_or_print chmod "$mode" "$original"
      echo "  restored ${original} (mode ${mode})"
      ;;
    ABSENT)
      do_or_print rm -f "$original"
      echo "  removed  ${original} (absent before that install)"
      ;;
    *)
      echo "unknown status '${status}' for ${original}" >&2
      exit 1
      ;;
  esac
done < <(tail -n +2 "$manifest")

echo
echo "Files restored. Now reload and restart, each in its own scope and order:"
echo "  sudo systemctl daemon-reload"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user restart gcmw-agent-demo.service        # 8001 first"
echo "  curl -sf -o /dev/null -w '8001 ready: %{http_code}\\n' http://127.0.0.1:8001/api/v1/health/ready"
echo "  sudo systemctl restart qa-server.service                # 8000, after 8001 answers"
echo "  sudo systemctl restart botscreen.service                # UI last"
