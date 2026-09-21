#!/usr/bin/env bash
#
# Restore what backup-assets.sh snapshotted.
#
# Usage:
#   rollback-assets.sh --stamp STAMP [--backup-dir /var/backups/gcmw]
#                       [--system-unit-dir /etc/systemd/system]
#                       [--user-unit-dir /home/<operator>/.config/systemd/user]
#                       [--deploy-dir <root>/deploy/current]
#                       [--list] [--dry-run]
#
# --stamp is REQUIRED and names one snapshot explicitly. There is deliberately no
# "newest wins" shortcut: the snapshot least safe to restore is the one left by an
# install that died halfway, and name ordering is exactly what picks that one.
#
# The five targets come from the DIRECTORIES THIS CALLER PASSES, and the manifest
# must agree with them entry by entry. The manifest never decides where cp or rm
# writes: it is evidence to be checked, not a source of authority.
#
# Every check runs BEFORE the first write. A half-restored tree is the worst
# outcome on offer - some files old, some new, and no record of which - so a
# snapshot that cannot be proven complete changes nothing at all and exits
# non-zero. This is not a transaction: a real disk error between two writes can
# still leave the tree half restored, and that is reported rather than hidden.
#
# An entry recorded ABSENT means the file did not exist before that install, so
# restoring the previous state REMOVES it. Every removal is printed; use
# --dry-run to see the whole plan first (it validates the same way).
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
    -h|--help) sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${deploy_dir:=${root}/deploy/current}"

# The fixed set of files an install overwrites, in the order backup-assets.sh
# writes them.
targets=(
  "${system_unit_dir}/qa-server.service"
  "${system_unit_dir}/botscreen.service"
  "${user_unit_dir}/gcmw-agent-demo.service"
  "${deploy_dir}/bin/start-botscreen-ui"
  "${deploy_dir}/bin/run-demo-agent-api.py"
)
target_count=${#targets[@]}

refuse() {
  echo "REFUSING to restore: $*" >&2
  echo "Nothing was modified." >&2
  exit 1
}

if [ "$list_only" = "1" ]; then
  echo "published snapshots in ${backup_dir}:"
  found=0
  if [ -d "$backup_dir" ]; then
    for entry in "$backup_dir"/*; do
      [ -d "$entry" ] || continue
      [ -f "${entry}/MANIFEST.tsv" ] || continue
      printf '  %s\n' "$(basename "$entry")"
      found=1
    done
  fi
  [ "$found" = "1" ] || echo "  (none)"
  echo
  echo "A leftover .<stamp>.partial.<xxxxxx> directory is an unfinished backup."
  echo "It is never listed here and never selectable: only a published stamp is."
  exit 0
fi

[ -n "$stamp" ] || {
  echo "--stamp is required: name the snapshot explicitly." >&2
  echo "There is no newest-wins default, because an interrupted backup must not" >&2
  echo "become selectable by a name-ordering guess. --list shows the stamps." >&2
  exit 2
}
case "$stamp" in
  */*|.|..) echo "--stamp must be a single directory name, not a path: $stamp" >&2; exit 2 ;;
  .*) echo "refusing a staging directory as a snapshot: ${stamp}" >&2; exit 2 ;;
esac

snapshot="${backup_dir}/${stamp}"
[ -d "$snapshot" ] || refuse "no such snapshot directory: ${snapshot} (see --list)"
manifest="${snapshot}/MANIFEST.tsv"
[ -f "$manifest" ] || refuse "no MANIFEST.tsv in ${snapshot}: this snapshot was never published"

# ---------------------------------------------------------------------------
# Validate the snapshot COMPLETELY. Nothing below this block writes anything.
# ---------------------------------------------------------------------------
snapshot_real="$(cd "$snapshot" && pwd -P)"

header=""
read -r header < "$manifest" || true
expected_header="$(printf 'status\tmode\toriginal\tstored')"
[ "$header" = "$expected_header" ] ||
  refuse "unexpected manifest header in ${manifest}: '${header}'"

m_status=()
m_mode=()
m_orig=()
m_stored=()
rows=0
lineno=1
while IFS= read -r line; do
  lineno=$((lineno + 1))
  [ -n "$line" ] || refuse "manifest line ${lineno} is empty"
  IFS=$'\t' read -r -a fields <<< "$line"
  [ "${#fields[@]}" -eq 4 ] ||
    refuse "manifest line ${lineno} does not hold four tab-separated fields: ${line}"
  m_status+=("${fields[0]}")
  m_mode+=("${fields[1]}")
  m_orig+=("${fields[2]}")
  m_stored+=("${fields[3]}")
  rows=$((rows + 1))
done < <(tail -n +2 "$manifest")

[ "$rows" -eq "$target_count" ] ||
  refuse "the manifest lists ${rows} target(s); exactly ${target_count} are expected"

i=0
while [ "$i" -lt "$rows" ]; do
  entry=$((i + 1))
  status="${m_status[$i]}"
  mode="${m_mode[$i]}"
  stored="${m_stored[$i]}"
  case "$status" in
    COPIED)
      case "$mode" in
        [0-7][0-7][0-7]|[0-7][0-7][0-7][0-7]) ;;
        *) refuse "entry ${entry} records mode '${mode}', which is not an octal file mode" ;;
      esac
      if [ -z "$stored" ] || [ "$stored" = "-" ]; then
        refuse "entry ${entry} is COPIED but records no stored file"
      fi
      case "$stored" in
        /*) refuse "entry ${entry} stores an absolute path: ${stored}" ;;
        *..*) refuse "entry ${entry} stores a path containing '..': ${stored}" ;;
      esac
      stored_path="${snapshot}/${stored}"
      [ -f "$stored_path" ] ||
        refuse "the snapshot is incomplete: ${stored} is missing"
      [ ! -L "$stored_path" ] ||
        refuse "the stored file ${stored} is a symlink, so it may point outside"
      stored_dir="$(cd "$(dirname "$stored_path")" 2>/dev/null && pwd -P)" ||
        refuse "cannot resolve the directory holding ${stored}"
      case "${stored_dir}/" in
        "${snapshot_real}/"*) ;;
        *) refuse "the stored path ${stored} resolves outside the snapshot" ;;
      esac
      ;;
    ABSENT)
      [ "$mode" = "-" ] || refuse "entry ${entry} is ABSENT but records mode '${mode}'"
      [ "$stored" = "-" ] || refuse "entry ${entry} is ABSENT but records a stored file"
      ;;
    *) refuse "entry ${entry} has an unknown status '${status}'" ;;
  esac
  i=$((i + 1))
done

# Nothing the caller did not name may appear, then every caller-supplied target
# exactly once. The first test is what stops the manifest from steering a write:
# an entry pointing anywhere else has no destination to write to.
i=0
while [ "$i" -lt "$rows" ]; do
  covered=0
  for want in "${targets[@]}"; do
    if [ "${m_orig[$i]}" = "$want" ]; then
      covered=1
    fi
  done
  [ "$covered" = "1" ] ||
    refuse "the manifest targets ${m_orig[$i]}, which the directories given to me do not cover"
  i=$((i + 1))
done

for want in "${targets[@]}"; do
  seen=0
  i=0
  while [ "$i" -lt "$rows" ]; do
    if [ "${m_orig[$i]}" = "$want" ]; then
      seen=$((seen + 1))
    fi
    i=$((i + 1))
  done
  [ "$seen" -eq 1 ] ||
    refuse "the manifest names ${want} ${seen} time(s); each target must appear exactly once"
done

# ---------------------------------------------------------------------------
# The snapshot is provably complete. Only now may anything be written.
# ---------------------------------------------------------------------------
echo "== rollback from ${snapshot} =="
if [ -f "${snapshot}/VERSION" ]; then
  sed 's/^/  /' "${snapshot}/VERSION"
fi
echo "  ${rows} target(s) validated before the first write"
echo

do_or_print() {
  if [ "$dry_run" = "1" ]; then
    echo "  would: $*"
  else
    "$@"
  fi
}

apply_entry() { # status index destination
  case "$1" in
    COPIED)
      cp -f "${snapshot}/${m_stored[$2]}" "$3" || return 1
      chmod "${m_mode[$2]}" "$3" || return 1
      ;;
    ABSENT)
      rm -f "$3" || return 1
      ;;
  esac
  return 0
}

i=0
changed=()
while [ "$i" -lt "$rows" ]; do
  status="${m_status[$i]}"
  # The destination comes from THIS CALLER's target list, never from the
  # manifest: the manifest was only allowed to agree with it.
  dest=""
  for want in "${targets[@]}"; do
    if [ "${m_orig[$i]}" = "$want" ]; then
      dest="$want"
    fi
  done

  if [ "$dry_run" = "1" ]; then
    case "$status" in
      COPIED)
        do_or_print cp -f "${snapshot}/${m_stored[$i]}" "$dest"
        do_or_print chmod "${m_mode[$i]}" "$dest"
        ;;
      ABSENT)
        do_or_print rm -f "$dest"
        ;;
    esac
    i=$((i + 1))
    continue
  fi

  # A bare `set -e` abort here would leave the operator with a half-restored tree
  # and no account of which half. Stop deliberately instead, and say so: an
  # unreported partial restore is indistinguishable from a successful one.
  if apply_entry "$status" "$i" "$dest"; then
    changed+=("$dest")
    case "$status" in
      COPIED) echo "  restored ${dest} (mode ${m_mode[$i]})" ;;
      ABSENT) echo "  removed  ${dest} (absent before that install)" ;;
    esac
  else
    echo >&2
    echo "STOPPED after ${#changed[@]} of ${rows} target(s): the step for ${dest} failed." >&2
    echo "THE TREE IS PARTIALLY RESTORED. Targets already changed:" >&2
    if [ "${#changed[@]}" -gt 0 ]; then
      for already in "${changed[@]}"; do
        echo "  ${already}" >&2
      done
    fi
    echo "Nothing further was attempted. The snapshot is unchanged, so after fixing" >&2
    echo "the cause, re-running this exact command finishes the job." >&2
    exit 1
  fi
  i=$((i + 1))
done

echo
if [ "$dry_run" = "1" ]; then
  echo "Dry run: the snapshot validated and NOTHING was modified."
  echo "Re-run without --dry-run to apply, then reload and restart in order."
else
  echo "Files restored. Now reload and restart, each in its own scope and order:"
fi
echo "  sudo systemctl daemon-reload"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user restart gcmw-agent-demo.service        # 8001 first"
echo "  curl -sf -o /dev/null -w '8001 ready: %{http_code}\\n' http://127.0.0.1:8001/api/v1/health/ready"
echo "  sudo systemctl restart qa-server.service                # 8000, after 8001 answers"
echo "  sudo systemctl restart botscreen.service                # UI last"
