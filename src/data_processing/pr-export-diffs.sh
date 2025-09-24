#!/usr/bin/env bash
# pr-export-diffs.sh
# Export key diffs for a PR as clean Markdown (and optional JSON), with summaries & context.
# Inspired by the structure/UX of pr-export-comments (resolve repo/PR, fetch base/head, clear MD).
#
# Usage:
#   ./pr-export-diffs.sh -p <PR_NUMBER|PR_URL> [-r owner/repo] [-c 5] [-o out.md] [--json out.json]
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
OUT_MD=""
OUT_JSON=""
OUT_PATCH=""
IGNORE_WS=0
declare -a GLOBS=()   # include/exclude patterns, bash-style (e.g., 'src/**' or '!**/*.lock')

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
      -p|--pr) PR_INPUT="${2:-}"; shift 2;;
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
  [[ -n "$PR_INPUT" ]] || die "Must provide -p <PR_NUMBER|PR_URL>"
  [[ "$CONTEXT" =~ ^[0-9]+$ ]] || die "--context must be an integer"
}

resolve_repo_and_pr() {
  local pr="$1"
  if [[ -z "$REPO" ]]; then
    REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
    [[ -n "$REPO" ]] || die "Could not infer repo; pass -r owner/repo"
  fi
  if [[ "$pr" =~ ^https?:// ]]; then
    PR_NUMBER="$(basename "$pr")"
    [[ "$PR_NUMBER" =~ ^[0-9]+$ ]] || die "Could not parse PR number from URL: $pr"
  else
    [[ "$pr" =~ ^[0-9]+$ ]] || die "PR must be a number or PR URL"
    PR_NUMBER="$pr"
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
resolve_repo_and_pr "$PR_INPUT"

# Pull PR metadata (including head/base OIDs and names)
PR_META_JSON="$(gh pr view "$PR_NUMBER" -R "$REPO" --json number,title,url,author,headRefName,baseRefName,headRefOid,baseRefOid,additions,deletions,changedFiles)"
PR_URL=$(jq -r '.url' <<<"$PR_META_JSON")
PR_TITLE=$(jq -r '.title' <<<"$PR_META_JSON")
HEAD_OID=$(jq -r '.headRefOid' <<<"$PR_META_JSON")
BASE_OID=$(jq -r '.baseRefOid' <<<"$PR_META_JSON")
BASE_REF=$(jq -r '.baseRefName' <<<"$PR_META_JSON")
AUTHOR=$(jq -r '.author.login // "unknown"' <<<"$PR_META_JSON")
TOTAL_ADDS=$(jq -r '.additions // 0' <<<"$PR_META_JSON")
TOTAL_DELS=$(jq -r '.deletions // 0' <<<"$PR_META_JSON")
TOTAL_FILES=$(jq -r '.changedFiles // 0' <<<"$PR_META_JSON")

# Ensure we have the commits locally
git fetch -q origin "pull/${PR_NUMBER}/head:refs/remotes/origin/pr/${PR_NUMBER}" || true
git fetch -q origin "${BASE_REF}:${BASE_REF}" || true

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
[[ -n "$OUT_MD" ]] || OUT_MD="pr-${PR_NUMBER}-diffs.md"

# Prefer GitHub's file list; fall back to git diff if necessary
FILES_JSON="$(gh pr view "$PR_NUMBER" -R "$REPO" --json files 2>/dev/null \
  | jq -c '[.files[]? | {path: .path, additions: .additions, deletions: .deletions, status: (.changeType // "MODIFIED")}]')"

if [[ -z "$FILES_JSON" || "$FILES_JSON" == "null" || "$(jq 'length' <<<"$FILES_JSON")" -eq 0 ]]; then
  # Fallback via git; note: rename appears as "Rxxx old\tnew"
  _ns_tmp="$(mktemp)" || die "Failed to create temp file"
  git diff --name-status -M "$BASE_OID" "$HEAD_OID" > "$_ns_tmp"
  if [[ ! -s "$_ns_tmp" ]]; then rm -f "$_ns_tmp"; die "No changed files found between $BASE_OID..$HEAD_OID"; fi
  FILES_JSON="$(jq -n '[]')"
  while IFS=$'\t' read -r st a b || [[ -n "$st" ]]; do
    [[ -n "$st" ]] || continue
    case "$st" in
      R*) path="$b"; status="RENAMED";;
      A)  path="$a"; status="ADDED";;
      D)  path="$a"; status="REMOVED";;
      M)  path="$a"; status="MODIFIED";;
      *)  path="$a"; status="$st";;
    esac
    # compute +/- via numstat (may be '-' for binary)
    read -r adds dels _ <<<"$(git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$path" | awk 'NR==1{print $1" "$2}')"
    adds=${adds:-0}; dels=${dels:-0}
    FILES_JSON="$(jq --arg p "$path" --arg s "$status" --argjson a "${adds//-/0}" --argjson d "${dels//-/0}" \
      '. + [{path:$p, status:$s, additions:$a, deletions:$d}]' <<<"$FILES_JSON")"
  done < "$_ns_tmp"
  rm -f "$_ns_tmp"
fi

# Filter paths per --glob
FILES_JSON="$(jq -c '.[]' <<<"$FILES_JSON" | while read -r f; do
  p="$(jq -r '.path' <<<"$f")"
  if path_matches "$p"; then echo "$f"; fi
done | jq -s '.')"

# Build an aggregate patch if requested
DIFF_FLAGS=(-U"$CONTEXT" -M --no-color)
(( IGNORE_WS )) && DIFF_FLAGS+=(-w)

if [[ -n "$OUT_PATCH" ]]; then
  git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" > "$OUT_PATCH" || true
fi

# ---------- Write Markdown ----------
{
  echo "# Diffs for PR #${PR_NUMBER}: ${PR_TITLE}"
  echo ""
  echo "- Repo: \`${REPO}\`"
  echo "- PR URL: ${PR_URL}"
  echo "- Author: @${AUTHOR}"
  echo "- Exported: ${timestamp}"
  echo "- Context radius: ${CONTEXT} line(s)"
  echo "- Files changed: ${TOTAL_FILES}  (+${TOTAL_ADDS} / -${TOTAL_DELS})"
  (( IGNORE_WS )) && echo "- Whitespace: ignored (git diff -w)"
  echo ""

  # Table of contents
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

  # Summary table
  echo "## Changeset Overview"
  echo ""
  echo "|---|---|---:|---:|"
  jq -r '.[] | "| " + (.status|tostring) + " | " + .path + " | " + (.additions|tostring) + " | " + (.deletions|tostring) + " |"' <<<"$FILES_JSON"
  echo ""

  # Per-file diffs
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

    # Detect binary by numstat '-' or if git diff yields 'Binary files differ'
    BIN=0
    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$path" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-'; then
      BIN=1
    fi
    if (( BIN )); then
      echo "_Binary file change; no textual diff available._"
      echo ""
      (( idx++ ))
      continue
    fi

    # Produce per-file patch with requested context
    patch="$(git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" -- "$path" || true)"

    if [[ -z "$patch" ]]; then
      echo "_No textual changes (possibly rename-only or mode change)._"
      echo ""
      (( idx++ ))
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

    (( idx++ ))
  done
} > "$OUT_MD"

# ---------- Optional JSON emission ----------
if [[ -n "$OUT_JSON" ]]; then
  files_for_json="$(jq -r -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    p="$(jq -r '.path' <<<"$f")"
    # Collect per-file patches (omit for binary)
    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$p" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-'; then
      # binary
      jq -n --argjson base "$f" --arg patch "" '$base + {binary:true, patch:null}'
    else
      pf="$(git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" -- "$p" || true)"
      jq -n --argjson base "$f" --arg patch "$pf" \
        '$base + {binary:false, patch:$patch}'
    fi
  done | jq -s '.')"

  jq -n \
    --arg repo "$REPO" \
    --arg prNumber "$PR_NUMBER" \
    --arg prUrl "$PR_URL" \
    --arg prTitle "$PR_TITLE" \
    --arg author "$AUTHOR" \
    --arg exported "$timestamp" \
    --argjson context "$CONTEXT" \
    --argjson totals "$(jq -n --argjson adds "$TOTAL_ADDS" --argjson dels "$TOTAL_DELS" --argjson files "$TOTAL_FILES" '{additions:$adds, deletions:$dels, files:$files}')" \
    --argjson files "$files_for_json" \
    '{
      repo: $repo,
      pr: ($prNumber|tonumber),
      url: $prUrl,
      title: $prTitle,
      author: $author,
      exported_at: $exported,
      context_radius: $context,
      totals: $totals,
      files: $files
    }' > "$OUT_JSON"
fi

echo "✓ Wrote Markdown: $OUT_MD"
[[ -n "$OUT_JSON" ]] && echo "✓ Wrote JSON:     $OUT_JSON"
[[ -n "$OUT_PATCH" ]] && echo "✓ Wrote Patch:     $OUT_PATCH"
