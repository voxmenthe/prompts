#!/usr/bin/env bash
# pr-export-diffs.sh
# Export key diffs for a PR as clean Markdown (and optional JSON), with summaries & context.
# Inspired by the structure/UX of pr-export-comments (resolve repo/PR, fetch base/head, clear MD).
#
# Usage:
#   ./pr-export-diffs.sh -p <PR_NUMBER|PR_URL> [-r owner/repo] [-c 5] [-o out.md] [--json out.json]
#                        [--ignore-space] [--glob 'src/**' --glob '!**/*.lock'] [--patch out.patch]
#   ./pr-export-diffs.sh --commit <COMMIT_SHA> [-r owner/repo] [-c 5] [-o out.md] [--json out.json]
#                        [--ignore-space] [--glob 'src/**' --glob '!**/*.lock'] [--patch out.patch]
#
# Examples:
#   ./pr-export-diffs.sh -p 123
#   ./pr-export-diffs.sh -p https://github.com/org/repo/pull/456 -r org/repo -c 6 -o pr-456-diffs.md --json pr-456.json
#   ./pr-export-diffs.sh -p 789 --ignore-space --glob 'src/**' --glob '!**/*.snap'
#
# Notes:
# - Run inside a clone of the repository or pass -r owner/repo.
# - We fetch the PR head and base to ensure commits exist locally.
# - Diff context uses git's -U<context> hunks; we do not download full files.

set -euo pipefail

# ---------- Defaults ----------
CONTEXT=5
REPO=""
PR_INPUT=""
PR_NUMBER=""
TARGET_MODE=""
COMMIT_INPUT=""
COMMIT_SHA=""
COMMIT_SHA_SHORT=""
TARGET_IDENTIFIER=""
TARGET_URL=""
TARGET_TITLE=""
TARGET_AUTHOR=""
TARGET_DESCRIPTION=""
DESCRIPTION_HEADING=""
DESCRIPTION_FALLBACK=""
DEFAULT_BASENAME=""
BASE_REF=""
BASE_OID=""
HEAD_OID=""
OUT_MD=""
OUT_JSON=""
OUT_PATCH=""
IGNORE_WS=0
TOTAL_FILES=0
TOTAL_ADDS=0
TOTAL_DELS=0
declare -a DEFAULT_IGNORED_FILE_GLOBS=(
  '**/package.json'
  '**/package-lock.json'
  '**/npm-shrinkwrap.json'
  '**/yarn.lock'
  '**/pnpm-lock.yaml'
  '**/bun.lockb'
)

declare -a GLOBS=("${DEFAULT_IGNORED_FILE_GLOBS[@]/#/!}")   # start with default excludes
EMPTY_TREE_HASH="4b825dc642135724b3b9f3a3b4f1b12cdad4b5d9"
FILES_JSON=""

# ---------- Helpers ----------
die() { echo "Error: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

usage() {
  sed -n '1,99p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--pr)
        [[ "$TARGET_MODE" != "commit" ]] || die "Cannot combine -p/--pr with --commit"
        TARGET_MODE="pr"
        PR_INPUT="${2:-}"
        shift 2
        ;;
      --commit)
        [[ -z "$TARGET_MODE" || "$TARGET_MODE" == "commit" ]] || die "Cannot combine --commit with -p/--pr"
        TARGET_MODE="commit"
        COMMIT_INPUT="${2:-}"
        shift 2
        ;;
      -r|--repo) REPO="${2:-}"; shift 2;;
      -c|--context) CONTEXT="${2:-}"; shift 2;;
      -o|--out) OUT_MD="${2:-}"; shift 2;;
      --json) OUT_JSON="${2:-}"; shift 2;;
      --patch) OUT_PATCH="${2:-}"; shift 2;;
      --ignore-space) IGNORE_WS=1; shift;;
      --glob) GLOBS+=("${2:-}"); shift 2;;
      -h|--help) usage;;
      *) die "Unknown arg: $1";;
    esac
  done
  if [[ "$TARGET_MODE" == "pr" ]]; then
    [[ -n "$PR_INPUT" ]] || die "Must provide -p <PR_NUMBER|PR_URL>"
  elif [[ "$TARGET_MODE" == "commit" ]]; then
    [[ -n "$COMMIT_INPUT" ]] || die "Must provide --commit <COMMIT_SHA>"
  else
    die "Must provide either -p <PR_NUMBER|PR_URL> or --commit <COMMIT_SHA>"
  fi

  [[ "$CONTEXT" =~ ^[0-9]+$ ]] || die "--context must be an integer"
}

resolve_repo() {
  if [[ -n "$REPO" ]]; then
    return
  fi
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
  [[ -n "$REPO" ]] || die "Could not infer repo; pass -r owner/repo"
}

resolve_pr_number() {
  local pr="$1"
  if [[ "$pr" =~ ^https?:// ]]; then
    PR_NUMBER="$(basename "$pr")"
    [[ "$PR_NUMBER" =~ ^[0-9]+$ ]] || die "Could not parse PR number from URL: $pr"
  else
    [[ "$pr" =~ ^[0-9]+$ ]] || die "PR must be a number or PR URL"
    PR_NUMBER="$pr"
  fi
}

load_pr_metadata() {
  local pr_meta_json
  pr_meta_json="$(gh pr view "$PR_NUMBER" -R "$REPO" --json number,title,url,author,headRefName,baseRefName,headRefOid,baseRefOid,additions,deletions,changedFiles,body)"

  TARGET_IDENTIFIER="PR #${PR_NUMBER}"
  TARGET_TITLE=$(jq -r '.title' <<<"$pr_meta_json")
  TARGET_URL=$(jq -r '.url' <<<"$pr_meta_json")
  TARGET_AUTHOR=$(jq -r '.author.login // "unknown"' <<<"$pr_meta_json")
  TARGET_DESCRIPTION="$(jq -r '.body // ""' <<<"$pr_meta_json")"
  DESCRIPTION_HEADING="PR Description"
  DESCRIPTION_FALLBACK="_No PR description set._"

  HEAD_OID=$(jq -r '.headRefOid' <<<"$pr_meta_json")
  BASE_OID=$(jq -r '.baseRefOid' <<<"$pr_meta_json")
  BASE_REF=$(jq -r '.baseRefName' <<<"$pr_meta_json")

  # Ensure we have the commits locally (best-effort, tolerate fetch failure)
  git fetch -q origin "pull/${PR_NUMBER}/head:refs/remotes/origin/pr/${PR_NUMBER}" || true
  git fetch -q origin "${BASE_REF}:${BASE_REF}" || true

  DEFAULT_BASENAME="pr-${PR_NUMBER}-diffs"
}

load_commit_metadata() {
  COMMIT_SHA="$(git rev-parse --verify "${COMMIT_INPUT}^{commit}" 2>/dev/null)" || die "Unknown commit: ${COMMIT_INPUT}"
  HEAD_OID="$COMMIT_SHA"
  COMMIT_SHA_SHORT="$(git rev-parse --short "$HEAD_OID")"

  local base_parent=""
  if base_parent="$(git rev-parse --verify "${HEAD_OID}^" 2>/dev/null)"; then
    BASE_OID="$base_parent"
  else
    BASE_OID="$EMPTY_TREE_HASH"
  fi

  TARGET_IDENTIFIER="Commit ${COMMIT_SHA_SHORT}"
  TARGET_TITLE="$(git show -s --format='%s' "$HEAD_OID")"
  TARGET_AUTHOR="$(git show -s --format='%an <%ae>' "$HEAD_OID")"
  TARGET_DESCRIPTION="$(git show -s --format='%b' "$HEAD_OID")"
  DESCRIPTION_HEADING="Commit Message"
  DESCRIPTION_FALLBACK="_Commit message has no body beyond the summary line._"

  if [[ -n "$REPO" ]]; then
    TARGET_URL="https://github.com/${REPO}/commit/${COMMIT_SHA}"
  else
    TARGET_URL=""
  fi

  DEFAULT_BASENAME="commit-${COMMIT_SHA_SHORT}-diffs"
}

build_files_json_from_git() {
  local tmp status path path_new adds dels files_json
  tmp="$(mktemp)" || die "Failed to create temp file"
  git diff --name-status -M "$BASE_OID" "$HEAD_OID" > "$tmp"
  files_json="$(jq -n '[]')"

  while IFS=$'\t' read -r status path path_new || [[ -n "$status" ]]; do
    [[ -n "$status" ]] || continue
    local final_path="$path" final_status="$status"
    case "$status" in
      R*) final_path="$path_new"; final_status="RENAMED";;
      A)  final_status="ADDED";;
      D)  final_status="REMOVED";;
      M)  final_status="MODIFIED";;
    esac

    read -r adds dels _ <<<"$(git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$final_path" | awk 'NR==1{print $1" "$2}')"
    [[ "$adds" == "-" || -z "$adds" ]] && adds=0
    [[ "$dels" == "-" || -z "$dels" ]] && dels=0

    files_json="$(jq --arg p "$final_path" --arg s "$final_status" --argjson a "$adds" --argjson d "$dels" \
      '. + [{path:$p, status:$s, additions:$a, deletions:$d}]' <<<"$files_json")"
  done < "$tmp"

  rm -f "$tmp"
  printf '%s' "$files_json"
}

build_files_json() {
  if [[ "$TARGET_MODE" == "pr" ]]; then
    FILES_JSON="$(gh pr view "$PR_NUMBER" -R "$REPO" --json files 2>/dev/null \
      | jq -c '[.files[]? | {path: .path, additions: .additions, deletions: .deletions, status: (.changeType // "MODIFIED")}]')"
  else
    FILES_JSON=""
  fi

  if [[ -z "$FILES_JSON" || "$FILES_JSON" == "null" || "$(jq 'length' <<<"$FILES_JSON" 2>/dev/null || echo 0)" -eq 0 ]]; then
    FILES_JSON="$(build_files_json_from_git)"
  fi
}

# Return 0 if PATH matches include/exclude globs; if no globs specified -> accept all.
path_matches() {
  local p="$1" inc_ok=0 exc_hit=0 had_inc=0
  if (( ${#GLOBS[@]} == 0 )); then return 0; fi
  shopt -s extglob nullglob globstar
  for g in "${GLOBS[@]}"; do
    if [[ "$g" == !* ]]; then
      local neg="${g:1}"
      [[ "$p" == $neg ]] && exc_hit=1
    else
      had_inc=1
      [[ "$p" == $g ]] && inc_ok=1
    fi
  done
  # If any exclude matches -> reject
  (( exc_hit )) && return 1
  # If we had any includes -> require one matches; else accept
  (( had_inc )) && (( inc_ok )) || { (( had_inc )) && return 1 || return 0; }
}

# Small JSON escaper for patches
json_escape() {
  jq -Rs . | sed 's/^"//; s/"$//'
}

slugify_anchor() {
  local input="$1"
  printf '%s' "$input" | sed -E 's/[^A-Za-z0-9]+/-/g' | tr '[:upper:]' '[:lower:]'
}

# ---------- Start ----------
require_cmd gh
require_cmd jq
require_cmd git
require_cmd awk
require_cmd sed

parse_args "$@"
resolve_repo

if [[ "$TARGET_MODE" == "pr" ]]; then
  resolve_pr_number "$PR_INPUT"
  load_pr_metadata
else
  load_commit_metadata
fi

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

[[ -n "$DEFAULT_BASENAME" ]] || DEFAULT_BASENAME="diff-export"
[[ -n "$OUT_MD" ]] || OUT_MD="${DEFAULT_BASENAME}.md"

build_files_json

# Filter paths per --glob
FILES_JSON="$(
  jq -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    p="$(jq -r '.path' <<<"$f")"
    if path_matches "$p"; then echo "$f"; fi
  done | jq -s '.'
)"
TOTAL_FILES=$(jq 'length' <<<"$FILES_JSON")
TOTAL_ADDS=$(jq '([.[].additions] | add) // 0' <<<"$FILES_JSON")
TOTAL_DELS=$(jq '([.[].deletions] | add) // 0' <<<"$FILES_JSON")

# Build an aggregate patch if requested
DIFF_FLAGS=(-U"$CONTEXT" -M --no-color)
(( IGNORE_WS )) && DIFF_FLAGS+=(-w)

if [[ -n "$OUT_PATCH" ]]; then
  git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" > "$OUT_PATCH" || true
fi

# ---------- Write Markdown ----------
{
  echo "# Diffs for ${TARGET_IDENTIFIER}: ${TARGET_TITLE}"
  echo ""
  echo "- Repo: \`${REPO}\`"
  if [[ "$TARGET_MODE" == "pr" ]]; then
    echo "- PR URL: ${TARGET_URL}"
    echo "- Author: @${TARGET_AUTHOR}"
  else
    if [[ -n "$TARGET_URL" ]]; then
      echo "- Commit URL: ${TARGET_URL}"
    else
      echo "- Commit URL: (local-only)"
    fi
    echo "- Author: ${TARGET_AUTHOR}"
  fi
  echo "- Exported: ${timestamp}"
  echo "- Context radius: ${CONTEXT} line(s)"
  echo "- Files changed: ${TOTAL_FILES}  (+${TOTAL_ADDS} / -${TOTAL_DELS})"
  (( IGNORE_WS )) && echo "- Whitespace: ignored (git diff -w)"
  echo ""
  echo "## ${DESCRIPTION_HEADING}"
  echo ""
  if [[ -n "$TARGET_DESCRIPTION" ]]; then
    printf "%s\n\n" "$TARGET_DESCRIPTION"
  elif [[ -n "$DESCRIPTION_FALLBACK" ]]; then
    echo "$DESCRIPTION_FALLBACK"
    echo ""
  fi

  echo "## Contents"
  jq -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    path="$(jq -r '.path' <<<"$f")"
    status="$(jq -r '.status' <<<"$f")"
    adds="$(jq -r '.additions' <<<"$f")"
    dels="$(jq -r '.deletions' <<<"$f")"
    anchor="$(slugify_anchor "$path")"
    printf -- "- [%s](#%s) — %s  (+%s / -%s)\n" "$path" "$anchor" "$status" "$adds" "$dels"
  done
  echo ""

  echo "## Changeset Overview"
  echo ""
  echo "|---|---|---:|---:|"
  jq -r '.[] | "| " + (.status|tostring) + " | " + .path + " | " + (.additions|tostring) + " | " + (.deletions|tostring) + " |"' <<<"$FILES_JSON"
  echo ""

  idx=0
  total=$(jq 'length' <<<"$FILES_JSON")
  while (( idx < total )); do
    file_json="$(jq -r --argjson i "$idx" '.[$i]' <<<"$FILES_JSON")"
    path="$(jq -r '.path' <<<"$file_json")"
    status="$(jq -r '.status' <<<"$file_json")"
    adds="$(jq -r '.additions' <<<"$file_json")"
    dels="$(jq -r '.deletions' <<<"$file_json")"

    anchor="$(slugify_anchor "$path")"
    echo "### ${path}"
    echo "<a id=\"$anchor\"></a>"
    echo ""
    echo "- **Status**: \`$status\` • **+${adds}/-${dels}** • **Context**: $CONTEXT"
    echo ""

    BIN=0
    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$path" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-' ; then
      BIN=1
    fi
    if (( BIN )); then
      echo "_Binary file change; no textual diff available._"
      echo ""
      (( idx += 1 ))
      continue
    fi

    patch="$(git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" -- "$path" || true)"

    if [[ -z "$patch" ]]; then
      echo "_No textual changes (possibly rename-only or mode change)._"
      echo ""
      (( idx += 1 ))
      continue
    fi

    echo "<details open><summary>Show diff</summary>"
    echo ""
    echo '```diff'
    printf "%s\n" "$patch"
    echo '```'
    echo ""
    echo "</details>"
    echo ""

    (( idx += 1 ))
  done
} > "$OUT_MD"

# ---------- Optional JSON emission ----------
if [[ -n "$OUT_JSON" ]]; then
  files_for_json="$(jq -r -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    p="$(jq -r '.path' <<<"$f")"
    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$p" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-'; then
      jq -n --argjson base "$f" --arg patch "" '$base + {binary:true, patch:null}'
    else
      pf="$(git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" -- "$p" || true)"
      jq -n --argjson base "$f" --arg patch "$pf" '$base + {binary:false, patch:$patch}'
    fi
  done | jq -s '.')"

  totals_json="$(jq -n --argjson adds "$TOTAL_ADDS" --argjson dels "$TOTAL_DELS" --argjson files "$TOTAL_FILES" '{additions:$adds, deletions:$dels, files:$files}')"

  if [[ "$TARGET_MODE" == "pr" ]]; then
    jq -n \
      --arg repo "$REPO" \
      --arg prNumber "$PR_NUMBER" \
      --arg prUrl "$TARGET_URL" \
      --arg prTitle "$TARGET_TITLE" \
      --arg author "$TARGET_AUTHOR" \
      --arg exported "$timestamp" \
      --argjson context "$CONTEXT" \
      --argjson totals "$totals_json" \
      --argjson files "$files_for_json" \
      --arg baseOid "$BASE_OID" \
      --arg headOid "$HEAD_OID" \
      --arg mode "$TARGET_MODE" \
      --arg identifier "$TARGET_IDENTIFIER" \
      '{
        repo: $repo,
        pr: ($prNumber|tonumber),
        url: $prUrl,
        title: $prTitle,
        author: $author,
        exported_at: $exported,
        context_radius: $context,
        totals: $totals,
        files: $files,
        oids: {base: $baseOid, head: $headOid},
        mode: $mode,
        identifier: $identifier
      }' > "$OUT_JSON"
  else
    jq -n \
      --arg repo "$REPO" \
      --arg commit "$COMMIT_SHA" \
      --arg shortCommit "$COMMIT_SHA_SHORT" \
      --arg url "$TARGET_URL" \
      --arg title "$TARGET_TITLE" \
      --arg author "$TARGET_AUTHOR" \
      --arg exported "$timestamp" \
      --argjson context "$CONTEXT" \
      --argjson totals "$totals_json" \
      --argjson files "$files_for_json" \
      --arg baseOid "$BASE_OID" \
      --arg headOid "$HEAD_OID" \
      --arg mode "$TARGET_MODE" \
      --arg identifier "$TARGET_IDENTIFIER" \
      '{
        repo: $repo,
        commit: {sha: $commit, short_sha: $shortCommit},
        url: $url,
        title: $title,
        author: $author,
        exported_at: $exported,
        context_radius: $context,
        totals: $totals,
        files: $files,
        oids: {base: $baseOid, head: $headOid},
        mode: $mode,
        identifier: $identifier
      }' > "$OUT_JSON"
  fi
fi

echo "✓ Wrote Markdown: $OUT_MD"
if [[ -n "$OUT_JSON" ]]; then
  echo "✓ Wrote JSON:     $OUT_JSON"
fi
if [[ -n "$OUT_PATCH" ]]; then
  echo "✓ Wrote Patch:     $OUT_PATCH"
fi
