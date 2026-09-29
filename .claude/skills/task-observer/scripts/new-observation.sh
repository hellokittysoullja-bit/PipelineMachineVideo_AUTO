#!/usr/bin/env bash
# new-observation.sh — derive the next observation id AND create the file,
# as one command that cannot be split.
#
# Usage:
#   scripts/new-observation.sh <slug> [workspace-root]
#
#   <slug>            kebab-case, no id prefix, no extension ("short-slug")
#   [workspace-root]  the pinned absolute workspace path — the directory that
#                     holds skill-observations/. Defaults to the environment
#                     variable TASK_OBSERVER_WORKSPACE. One of the two is
#                     required; the log is never resolved from the cwd.
#
# Prints the created file's absolute path on stdout (one line) and exits 0.
# Any guard firing prints its message on stderr and exits 1 with NO file
# created. Write the observation body into the printed path, with the id
# from its NNNN- prefix in the `id:` field — copied, never retyped.
#
# This is the id-derivation snippet from SKILL.md "How to Log", unchanged in
# substance, packaged so that the moment of derivation cannot drift away
# from the moment of writing: an id derived at session start and carried in
# memory to a later write is the collision the snippet's rules forbid, and
# a rule that has failed twice while loaded needs the write path itself to
# enforce it. Where this script can run, it is the only write path.
#
# Steps, in order: archival sweep of stale resolved files (a side effect of
# deriving the id, never a separate duty) → id = max(active prefixes,
# archive prefixes, .id-floor) + 1 → floor write → PREFIX collision guard
# across active and archive (two slugs sharing one number is the failure;
# a path check cannot see it) → noclobber create → print the path.
#
# bash, not sh: `read -r -d ''` is a bash extension. Runs on bash 3.2 (stock
# macOS): case patterns inside $( ) are parenthesised for that reason.

set -u

slug="${1:-}"
root="${2:-${TASK_OBSERVER_WORKSPACE:-}}"

if [ -z "$slug" ]; then
  echo "usage: new-observation.sh <slug> [workspace-root]" >&2; exit 1
fi
case $slug in
  (*[!a-z0-9-]*|-*|*-|*--*|"")
    echo "slug must be kebab-case (a-z, 0-9, single hyphens): got '$slug'" >&2; exit 1 ;;
esac
case $slug in
  ([0-9][0-9][0-9][0-9]-*) echo "slug must not start with an id prefix: got '$slug'" >&2; exit 1 ;;
esac
if [ -z "$root" ]; then
  echo "no workspace root: pass it as the second argument or set TASK_OBSERVER_WORKSPACE" >&2; exit 1
fi
case $root in
  (/*) ;;
  (*) echo "workspace root must be an ABSOLUTE path, never relative to the cwd: got '$root'" >&2; exit 1 ;;
esac

d="$root/skill-observations/observation-log"   # may contain a space: every expansion stays quoted
if [ ! -d "$d" ] || [ ! -d "$d/archive" ]; then
  echo "STRUCTURE MISSING — expected '$d' and '$d/archive' (Session Start step 1 creates them); HALT and re-probe, never recreate from here" >&2
  exit 1
fi

today=$(date +%F)

# --- archival sweep: stale resolved files move before the id is read --------
n_files=$(find "$d" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')
seen=$(find "$d" -maxdepth 1 -name '*.md' -print0 | { n=0
  while IFS= read -r -d '' f; do
    n=$(( n + 1 ))
    hdr=$(awk 'NR==1 && /^---[[:space:]]*$/ {fm=1; next}
               fm && /^---[[:space:]]*$/ {exit} fm' "$f")
    case $hdr in
      (*"status: actioned"*|*"status: declined"*|*"status: superseded"*) ;;
      (*) continue ;;
    esac
    r=$(printf '%s\n' "$hdr" | sed -n 's/^resolved:[[:space:]]*//p' | head -1)
    case $r in ([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;; (*) continue ;; esac
    [ "$r" != "$today" ] && \
      [ "$(printf '%s\n%s\n' "$r" "$today" | sort | head -1)" = "$r" ] && \
      mv "$f" "$d/archive/"
  done; printf %s "$n"; })
if [ "$n_files" -gt 0 ] && [ "${seen:-0}" -eq 0 ]; then
  echo "ARCHIVAL SWEEP BROKEN — $n_files files present, 0 examined" >&2; exit 1
fi

# --- id: max of active prefixes, archive prefixes and the floor, plus one ---
hi=$( { ls "$d" "$d/archive" 2>/dev/null | grep -oE '^[0-9]+'; cat "$d/archive/.id-floor" 2>/dev/null; } \
     | sed 's/^0*\([0-9]\)/\1/' | sort -n | tail -1); : "${hi:=0}"
if [ "$hi" -eq 0 ] && [ -n "$(find "$d" -maxdepth 1 -name '*.md')" ]; then
  echo "ID COMMAND BROKEN — log is non-empty but no ids extracted" >&2; exit 1
fi
next_id=$(( hi + 1 ))
prefix=$(printf '%04d' "$next_id")

# floor-staleness note: the floor lags the directory when an issuer skipped
# the floor write. Harmless here (max-of-three absorbs it) but worth a line,
# because a lagging floor is the precondition for a restart once the active
# directory is archived down.
floor=$(sed 's/^0*\([0-9]\)/\1/' "$d/archive/.id-floor" 2>/dev/null | tr -d ' \n')
top_active=$(ls "$d" 2>/dev/null | grep -oE '^[0-9]+' | sed 's/^0*\([0-9]\)/\1/' | sort -n | tail -1)
if [ -n "${floor:-}" ] && [ -n "${top_active:-}" ] && [ "$floor" -lt "$top_active" ]; then
  echo "NOTE: .id-floor ($floor) is below the highest active id ($top_active) — an issuer skipped the floor write; corrected now" >&2
fi

echo "$next_id" > "$d/archive/.id-floor" || { echo "FLOOR WRITE FAILED — $d/archive/.id-floor" >&2; exit 1; }

# --- collision guard on the id PREFIX, across active and archive ------------
# Two writers deriving the same number with different slugs produce two
# different paths; a path check passes both. The invariant is a unique
# number, so the guard is keyed on the number.
if [ -n "$(find "$d" -maxdepth 2 -name "${prefix}-*.md")" ]; then
  echo "COLLISION — id $next_id already used; re-derive (re-run this script)" >&2; exit 1
fi

# --- noclobber create: never truncate an existing file ----------------------
f="$d/${prefix}-${slug}.md"
if ! (set -C; : > "$f") 2>/dev/null; then
  echo "CREATE FAILED — $f exists or is unwritable" >&2; exit 1
fi

printf '%s\n' "$f"
