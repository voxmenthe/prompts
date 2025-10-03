#!/usr/bin/env bash
# commit-set-export-diffs.sh
# Export clean Markdown (and optional JSON) diffs for a set of commits stacked onto a shared base.
# The script mirrors pr-export-diffs.sh --pr-set but targets arbitrary commit SHAs.
#
# Usage:
#   ./commit-set-export-diffs.sh --commits <SHA1,SHA2,...> [-c 5]
#                                [-o out.md] [--json out.json] [--patch out.patch]
#                                [--base <BASE_SHA>] [--repo owner/repo]
#                                [--ignore-space] [--glob 'src/**' --glob '!**/*.lock']
#
# Notes:
# - Commits are applied in the order provided. Provide at least one commit.
# - If --base is omitted we use the first commit's parent, or the empty tree when unavailable.
# - A temporary worktree isolates cherry-picks so your working directory remains untouched.
# - The repo owner/name is optional and used only for URL generation in outputs.

set -euo pipefail

# ---------- Defaults ----------
CONTEXT=5
REPO=""
OUT_MD=""
OUT_JSON=""
OUT_PATCH=""
IGNORE_WS=0
BASE_OVERRIDE=""

EMPTY_TREE_HASH="4b825dc642135724b3b9f3a3b4f1b12cdad4b5d9"

declare -a DEFAULT_IGNORED_FILE_GLOBS=(
  '**/package.json'
  '**/package-lock.json'
  '**/npm-shrinkwrap.json'
  '**/yarn.lock'
  '**/pnpm-lock.yaml'
  '**/bun.lockb'
  '**/test/**'
  '**/*test*/**'
  '**/*test*'
)

declare -a GLOBS=("${DEFAULT_IGNORED_FILE_GLOBS[@]/#/!}")
declare -a COMMIT_INPUTS=()
declare -a COMMIT_OIDS=()
declare -a COMMIT_SHORTS=()
declare -a COMMIT_TITLES=()
declare -a COMMIT_AUTHORS=()
declare -a COMMIT_BODIES=()
declare -a COMMIT_URLS=()
COMMIT_SET_METADATA_JSON='[]'
COMMIT_SET_LABEL=""
TARGET_IDENTIFIER=""
TARGET_TITLE=""
TARGET_author_rollup=""
TARGET_DESCRIPTION=""
DESCRIPTION_HEADING="Commit Set Overview"
DESCRIPTION_FALLBACK="_No commit details available._"
DEFAULT_BASENAME="commit-set-diffs"
BASE_OID=""
HEAD_OID=""
TOTAL_FILES=0
TOTAL_ADDS=0
TOTAL_DELS=0
FILES_JSON=""

COMMIT_SET_WORKTREE_DIR=""
COMMIT_SET_SYNTH_HEAD=""

# ---------- Helpers ----------
die() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

usage() {
  sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

slugify_anchor() {
  local input="$1"
  printf '%s' "$input" | sed -E 's/[^A-Za-z0-9]+/-/g' | tr '[:upper:]' '[:lower:]'
}

contains_test_token() {
  local path="$1" segment lower before after has_left has_right
  IFS='/' read -r -a segments <<<"$path"
  for segment in "${segments[@]}"; do
    lower="${segment,,}"
    while [[ "$lower" == *test* ]]; do
      before=${lower%%test*}
      after=${lower#*test}
      has_left=0
      has_right=0
      if [[ -z "$before" ]]; then
        has_left=1
      else
        case "${before: -1}" in
          -|_|.) has_left=1;;
        esac
      fi
      if [[ -n "$after" ]]; then
        case "${after:0:1}" in
          -|_|.) has_right=1;;
        esac
      fi
      if (( has_left || has_right )); then
        return 0
      fi
      lower="$after"
    done
  done
  return 1
}

path_matches() {
  local p="$1" inc_ok=0 exc_hit=0 had_inc=0
  if (( ${#GLOBS[@]} == 0 )); then return 0; fi
  contains_test_token "$p" && return 1
  shopt -s extglob nullglob globstar
  for g in "${GLOBS[@]}"; do
    if [[ "$g" == !* ]]; then
      local neg="${g:1}"
      [[ $p == $neg ]] && exc_hit=1
    else
      had_inc=1
      [[ $p == $g ]] && inc_ok=1
    fi
  done
  (( exc_hit )) && return 1
  if (( had_inc )); then
    (( inc_ok )) && return 0
    return 1
  fi
  return 0
}

infer_repo_from_git() {
  local url
  url="$(git config --get remote.origin.url 2>/dev/null || true)"
  [[ -n "$url" ]] || return
  case "$url" in
    git@github.com:*)
      REPO="${url#git@github.com:}"
      ;;
    https://github.com/*)
      REPO="${url#https://github.com/}"
      ;;
    ssh://git@github.com/*)
      REPO="${url#ssh://git@github.com/}"
      ;;
    https://www.github.com/*)
      REPO="${url#https://www.github.com/}"
      ;;
    https://github.com:*/*)
      REPO="${url#https://github.com:}"
      ;;
    *)
      return
      ;;
  esac
  REPO="${REPO%.git}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --commits)
        local commit_arg="${2:-}"
        [[ -n "$commit_arg" ]] || die "--commits requires a value"
        COMMIT_INPUTS+=("$commit_arg")
        shift 2
        ;;
      --base)
        BASE_OVERRIDE="${2:-}"
        shift 2
        ;;
      -r|--repo)
        REPO="${2:-}"
        shift 2
        ;;
      -c|--context)
        CONTEXT="${2:-}"
        shift 2
        ;;
      -o|--out)
        OUT_MD="${2:-}"
        shift 2
        ;;
      --json)
        OUT_JSON="${2:-}"
        shift 2
        ;;
      --patch)
        OUT_PATCH="${2:-}"
        shift 2
        ;;
      --ignore-space)
        IGNORE_WS=1
        shift
        ;;
      --glob)
        GLOBS+=("${2:-}")
        shift 2
        ;;
      -h|--help)
        usage
        ;;
      *)
        die "Unknown arg: $1"
        ;;
    esac
  done
  (( ${#COMMIT_INPUTS[@]} > 0 )) || die "Provide at least one --commits argument"
  [[ "$CONTEXT" =~ ^[0-9]+$ ]] || die "--context must be an integer"
}

normalize_commit_inputs() {
  declare -A seen=()
  COMMIT_SET_METADATA_JSON='[]'
  COMMIT_OIDS=()
  COMMIT_SHORTS=()
  COMMIT_TITLES=()
  COMMIT_AUTHORS=()
  COMMIT_BODIES=()
  COMMIT_URLS=()

  for raw in "${COMMIT_INPUTS[@]}"; do
    IFS=',' read -r -a split <<<"$raw"
    for token in "${split[@]}"; do
      local trimmed
      trimmed="${token//[[:space:]]/}"
      [[ -n "$trimmed" ]] || continue
      if [[ -n "${seen[$trimmed]:-}" ]]; then
        echo "Warning: duplicate commit $trimmed in --commits; ignoring subsequent occurrence" >&2
        continue
      fi
      local oid
      oid="$(git rev-parse --verify "${trimmed}^{commit}" 2>/dev/null)" || die "Unknown commit: $trimmed"
      seen[$trimmed]=1
      COMMIT_OIDS+=("$oid")
      COMMIT_SHORTS+=("$(git rev-parse --short "$oid")")
      COMMIT_TITLES+=("$(git show -s --format='%s' "$oid")")
      COMMIT_AUTHORS+=("$(git show -s --format='%an <%ae>' "$oid")")
      COMMIT_BODIES+=("$(git show -s --format='%b' "$oid")")
      if [[ -n "$REPO" ]]; then
        COMMIT_URLS+=("https://github.com/${REPO}/commit/${oid}")
      else
        COMMIT_URLS+=("")
      fi
      COMMIT_SET_METADATA_JSON="$(jq \
        --arg sha "$oid" \
        --arg short "${COMMIT_SHORTS[-1]}" \
        --arg title "${COMMIT_TITLES[-1]}" \
        --arg author "${COMMIT_AUTHORS[-1]}" \
        --arg body "${COMMIT_BODIES[-1]}" \
        --arg url "${COMMIT_URLS[-1]}" \
        '. + [{sha:$sha, short:$short, title:$title, author:$author, body:$body, url:($url // null)}]' \
        <<<"$COMMIT_SET_METADATA_JSON")"
    done
  done

  (( ${#COMMIT_OIDS[@]} > 0 )) || die "No commits resolved from --commits arguments"
}

resolve_base_oid() {
  if [[ -n "$BASE_OVERRIDE" ]]; then
    BASE_OID="$(git rev-parse --verify "${BASE_OVERRIDE}^{commit}" 2>/dev/null)" || die "Unknown base commit: ${BASE_OVERRIDE}"
    return
  fi
  local parent
  if parent="$(git rev-parse --verify "${COMMIT_OIDS[0]}^" 2>/dev/null)"; then
    BASE_OID="$parent"
  else
    BASE_OID="$EMPTY_TREE_HASH"
  fi
}

build_commit_labels() {
  local first_short last_short count
  count=${#COMMIT_OIDS[@]}
  first_short="${COMMIT_SHORTS[0]}"
  last_short="${COMMIT_SHORTS[count-1]}"

  COMMIT_SET_LABEL="$first_short"
  TARGET_IDENTIFIER="Commit set ${first_short}"
  for short in "${COMMIT_SHORTS[@]:1}"; do
    COMMIT_SET_LABEL+="-$short"
    TARGET_IDENTIFIER+="→${short}"
  done

  TARGET_TITLE="Combined diff for commit set (${count} commit$( (( count > 1 )) && printf 's'))"

  local distinct_author=""
  local all_same=1
  for author in "${COMMIT_AUTHORS[@]}"; do
    if [[ -z "$distinct_author" ]]; then
      distinct_author="$author"
    elif [[ "$distinct_author" != "$author" ]]; then
      all_same=0
      break
    fi
  done
  if (( all_same )); then
    TARGET_author_rollup="$distinct_author"
  else
    TARGET_author_rollup="multiple"
  fi

  local overview
  overview=""
  local idx=0
  while (( idx < count )); do
    local short="${COMMIT_SHORTS[$idx]}"
    local title="${COMMIT_TITLES[$idx]}"
    local author="${COMMIT_AUTHORS[$idx]}"
    local url="${COMMIT_URLS[$idx]}"
    local line="- ${short} — ${title} • ${author}"
    if [[ -n "$url" ]]; then
      line+=" • ${url}"
    fi
    overview+="$line\n"
    (( idx += 1 ))
  done
  TARGET_DESCRIPTION="$overview"
  DEFAULT_BASENAME="commit-set-${COMMIT_SET_LABEL}-diffs"
}

cleanup_commit_set_worktree() {
  if [[ -n "$COMMIT_SET_WORKTREE_DIR" && -d "$COMMIT_SET_WORKTREE_DIR" ]]; then
    git worktree remove --force "$COMMIT_SET_WORKTREE_DIR" >/dev/null 2>&1 || true
    rm -rf "$COMMIT_SET_WORKTREE_DIR"
    COMMIT_SET_WORKTREE_DIR=""
  fi
}

apply_commit_to_stack() {
  local worktree_dir="$1"
  local commit_oid="$2"
  local display_short="$3"

  if git -C "$worktree_dir" cherry-pick --allow-empty "$commit_oid" >/dev/null 2>&1; then
    return
  fi

  echo "Warning: encountered conflicts cherry-picking ${display_short}; keeping conflict markers" >&2

  if ! git -C "$worktree_dir" add -A; then
    git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
    cleanup_commit_set_worktree
    die "Failed to stage conflict files for commit ${display_short}"
  fi

  if git -C "$worktree_dir" ls-files -u | grep -q .; then
    git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
    cleanup_commit_set_worktree
    die "Unresolved merge entries remain after staging conflicts for commit ${display_short}"
  fi

  if ! GIT_EDITOR=true git -C "$worktree_dir" cherry-pick --continue >/dev/null 2>&1; then
    if ! git -C "$worktree_dir" status --porcelain | grep -q .; then
      GIT_EDITOR=true git -C "$worktree_dir" cherry-pick --skip >/dev/null 2>&1 || true
      return
    fi
    git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
    cleanup_commit_set_worktree
    die "Failed to finalize conflicted cherry-pick for commit ${display_short}"
  fi
}

prepare_commit_set_worktree() {
  COMMIT_SET_WORKTREE_DIR="$(mktemp -d -t commit-set-export.XXXXXXXX)"
  [[ -n "$COMMIT_SET_WORKTREE_DIR" && -d "$COMMIT_SET_WORKTREE_DIR" ]] || die "Failed to create temporary worktree directory"
  trap cleanup_commit_set_worktree EXIT

  git worktree add --detach "$COMMIT_SET_WORKTREE_DIR" "$BASE_OID" >/dev/null 2>&1
  git -C "$COMMIT_SET_WORKTREE_DIR" config --worktree core.hooksPath /dev/null >/dev/null 2>&1 || true

  local idx=0
  local count=${#COMMIT_OIDS[@]}
  while (( idx < count )); do
    apply_commit_to_stack "$COMMIT_SET_WORKTREE_DIR" "${COMMIT_OIDS[$idx]}" "${COMMIT_SHORTS[$idx]}"
    (( idx += 1 ))
  done

  COMMIT_SET_SYNTH_HEAD="$(git -C "$COMMIT_SET_WORKTREE_DIR" rev-parse HEAD)"
  HEAD_OID="$COMMIT_SET_SYNTH_HEAD"
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
  FILES_JSON="$(build_files_json_from_git)"
}

# ---------- Start ----------
require_cmd jq
require_cmd git
require_cmd awk
require_cmd sed

parse_args "$@"
if [[ -z "$REPO" ]]; then
  infer_repo_from_git || true
fi

normalize_commit_inputs
resolve_base_oid
build_commit_labels
prepare_commit_set_worktree

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

[[ -n "$DEFAULT_BASENAME" ]] || DEFAULT_BASENAME="commit-set-diffs"
[[ -n "$OUT_MD" ]] || OUT_MD="${DEFAULT_BASENAME}.md"

build_files_json

FILES_JSON="$(
  jq -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    p="$(jq -r '.path' <<<"$f")"
    if path_matches "$p"; then echo "$f"; fi
  done | jq -s '.'
)"

TOTAL_FILES=$(jq 'length' <<<"$FILES_JSON")
TOTAL_ADDS=$(jq '([.[].additions] | add) // 0' <<<"$FILES_JSON")
TOTAL_DELS=$(jq '([.[].deletions] | add) // 0' <<<"$FILES_JSON")

DIFF_FLAGS=(-U"$CONTEXT" -M --no-color)
(( IGNORE_WS )) && DIFF_FLAGS+=(-w)

if [[ -n "$OUT_PATCH" ]]; then
  git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" > "$OUT_PATCH" || true
fi

{
  echo "# Diffs for ${TARGET_IDENTIFIER}: ${TARGET_TITLE}"
  echo ""
  if [[ -n "$REPO" ]]; then
    echo "- Repo: \`${REPO}\`"
  else
    echo "- Repo: (unknown)"
  fi
  echo "- Commits: ${#COMMIT_OIDS[@]}"
  echo "- Author(s): ${TARGET_author_rollup}"
  echo "- Exported: ${timestamp}"
  echo "- Context radius: ${CONTEXT} line(s)"
  echo "- Files changed: ${TOTAL_FILES}  (+${TOTAL_ADDS} / -${TOTAL_DELS})"
  echo "- Base commit: ${BASE_OID:0:7}"
  echo "- Head commit: ${HEAD_OID:0:7}"
  (( IGNORE_WS )) && echo "- Whitespace: ignored (git diff -w)"
  echo ""
  echo "## ${DESCRIPTION_HEADING}"
  echo ""
  if [[ -n "$TARGET_DESCRIPTION" ]]; then
    printf "%s\n" "$TARGET_DESCRIPTION"
  elif [[ -n "$DESCRIPTION_FALLBACK" ]]; then
    echo "$DESCRIPTION_FALLBACK"
  fi
  echo ""
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

    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$path" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-'; then
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

if [[ -n "$OUT_JSON" ]]; then
  files_for_json="$(jq -r -c '.[]' <<<"$FILES_JSON" | while read -r f; do
    p="$(jq -r '.path' <<<"$f")"
    if git diff --numstat "$BASE_OID" "$HEAD_OID" -- "$p" | awk 'NR==1{print $1" "$2}' | grep -q '\- \-'; then
      jq -n --argjson base "$f" --arg patch "" '$base + {binary:true, patch:null}'
    else
      pf="$(git diff "${DIFF_FLAGS[@]}" "$BASE_OID" "$HEAD_OID" -- "$p" || true)"
      tmp_patch="$(mktemp)" || die "Failed to create temp file for JSON patch"
      printf '%s' "$pf" > "$tmp_patch"
      jq -n --argjson base "$f" --rawfile patch "$tmp_patch" '$base + {binary:false, patch:$patch}'
      rm -f "$tmp_patch"
    fi
  done | jq -s '.')"

  totals_json="$(jq -n --argjson adds "$TOTAL_ADDS" --argjson dels "$TOTAL_DELS" --argjson files "$TOTAL_FILES" '{additions:$adds, deletions:$dels, files:$files}')"

  tmp_files_json="$(mktemp)" || die "Failed to create temp file for files JSON"
  tmp_commits_json="$(mktemp)" || die "Failed to create temp file for commit metadata"
  printf '%s' "$files_for_json" > "$tmp_files_json"
  printf '%s' "$COMMIT_SET_METADATA_JSON" > "$tmp_commits_json"

  jq -n \
    --arg repo "$REPO" \
    --arg exported "$timestamp" \
    --arg identifier "$TARGET_IDENTIFIER" \
    --arg title "$TARGET_TITLE" \
    --arg author "$TARGET_author_rollup" \
    --arg baseOid "$BASE_OID" \
    --arg headOid "$HEAD_OID" \
    --argjson context "$CONTEXT" \
    --argjson totals "$totals_json" \
    --rawfile files_raw "$tmp_files_json" \
    --rawfile commits_raw "$tmp_commits_json" \
    '{
      repo: ($repo // null),
      exported_at: $exported,
      identifier: $identifier,
      title: $title,
      authors: $author,
      context_radius: $context,
      totals: $totals,
      files: ($files_raw | fromjson),
      commits: ($commits_raw | fromjson),
      oids: {base: $baseOid, head: $headOid},
      mode: "commit_set"
    }' > "$OUT_JSON"

  rm -f "$tmp_files_json" "$tmp_commits_json"
fi

echo "✓ Wrote Markdown: $OUT_MD"
if [[ -n "$OUT_JSON" ]]; then
  echo "✓ Wrote JSON:     $OUT_JSON"
fi
if [[ -n "$OUT_PATCH" ]]; then
  echo "✓ Wrote Patch:     $OUT_PATCH"
fi
