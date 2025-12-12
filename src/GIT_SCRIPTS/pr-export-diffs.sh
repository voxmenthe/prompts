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
#   ./pr-export-diffs.sh --pr-set 62,63,69,86 [-r owner/repo] [-c 5] [-o out.md] [--json out.json]
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
# - --pr-set stacks PRs inside a temporary worktree; your repo history remains untouched.

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
UPSTREAM_MODE=0
TOTAL_FILES=0
TOTAL_ADDS=0
TOTAL_DELS=0
declare -a PR_SET_INPUTS=()
declare -a PR_SET_NUMBERS=()
PR_SET_LABEL=""
PR_SET_METADATA_JSON=""
PR_SET_DISPLAY_LIST=""
declare -a PR_SET_BASE_OIDS=()
declare -a PR_SET_HEAD_OIDS=()
PR_SET_WORKTREE_DIR=""
PR_SET_SYNTH_HEAD=""
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
  '**/uv.lock'
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
      --pr-set)
        [[ "$TARGET_MODE" != "commit" ]] || die "Cannot combine --pr-set with --commit"
        if [[ -n "$TARGET_MODE" && "$TARGET_MODE" != "pr_set" ]]; then
          die "Cannot combine --pr-set with -p/--pr"
        fi
        TARGET_MODE="pr_set"
        local pr_set_arg="${2:-}"
        [[ -n "$pr_set_arg" ]] || die "--pr-set requires a value"
        PR_SET_INPUTS+=("$pr_set_arg")
        shift 2
        ;;
      -r|--repo) REPO="${2:-}"; shift 2;;
      -c|--context) CONTEXT="${2:-}"; shift 2;;
      -o|--out) OUT_MD="${2:-}"; shift 2;;
      --json) OUT_JSON="${2:-}"; shift 2;;
      --patch) OUT_PATCH="${2:-}"; shift 2;;
      --ignore-space) IGNORE_WS=1; shift;;
      --upstream) UPSTREAM_MODE=1; shift;;
      --glob) GLOBS+=("${2:-}"); shift 2;;
      -h|--help) usage;;
      *) die "Unknown arg: $1";;
    esac
  done
  if [[ "$TARGET_MODE" == "pr" ]]; then
    [[ -n "$PR_INPUT" ]] || die "Must provide -p <PR_NUMBER|PR_URL>"
  elif [[ "$TARGET_MODE" == "commit" ]]; then
    [[ -n "$COMMIT_INPUT" ]] || die "Must provide --commit <COMMIT_SHA>"
  elif [[ "$TARGET_MODE" == "pr_set" ]]; then
    (( ${#PR_SET_INPUTS[@]} >= 1 )) || die "Must provide at least one --pr-set value"
  else
    die "Must provide either -p <PR_NUMBER|PR_URL>, --pr-set <PR_LIST>, or --commit <COMMIT_SHA>"
  fi

  [[ "$CONTEXT" =~ ^[0-9]+$ ]] || die "--context must be an integer"
}

normalize_pr_identifier() {
  local input="$1" pr=""
  if [[ "$input" =~ ^https?:// ]]; then
    pr="$(basename "$input")"
  else
    pr="$input"
  fi

  [[ "$pr" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$pr"
}

resolve_repo() {
  if [[ -n "$REPO" ]]; then
    : # Repo argument takes precedence, unless we construct upstream logic logic below,
      # but typically -r override means "use this repo".
      # However, if --upstream is passed, we might want to check the parent of the explicitly passed repo too?
      # Let's assume --upstream applies to whatever $REPO we resolved.
  else
    REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
    [[ -n "$REPO" ]] || die "Could not infer repo; pass -r owner/repo"
  fi

  if (( UPSTREAM_MODE )); then
     local parent
     parent="$(gh repo view "$REPO" --json parent -q '.parent | .owner.login + "/" + .name' 2>/dev/null || true)"
     if [[ -n "$parent" && "$parent" != "null" && "$parent" != "/" ]]; then
       REPO="$parent"
     else
       die "Could not determine upstream (parent) for repository $REPO"
     fi
  fi
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

resolve_pr_repo_context() {
  local pr="$1"
  # Try current repo
  if gh pr view "$pr" -R "$REPO" --json headRefOid >/dev/null 2>&1; then
     return 0
  fi

  # Fallback to parent
  local parent
  parent="$(gh repo view "$REPO" --json parent -q '.parent | .owner.login + "/" + .name' 2>/dev/null || true)"
  if [[ -n "$parent" && "$parent" != "null" && "$parent" != "/" ]]; then
     if gh pr view "$pr" -R "$parent" --json headRefOid >/dev/null 2>&1; then
       echo "PR #$pr not found in $REPO; found in $parent. Switching context." >&2
       REPO="$parent"
       return 0
     fi
  fi
  
  die "Could not find PR #$pr in $REPO or its parent"
}

load_pr_metadata() {
  resolve_pr_repo_context "$PR_NUMBER"
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
  local repo_url="https://github.com/${REPO}.git"
  git fetch -q "$repo_url" "refs/pull/${PR_NUMBER}/head:refs/remotes/origin/pr/${PR_NUMBER}" || true
  git fetch -q "$repo_url" "refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}" || true

  DEFAULT_BASENAME="pr-${PR_NUMBER}-diffs"
}

load_pr_set_metadata() {
  declare -A seen_prs=()
  PR_SET_NUMBERS=()
  PR_SET_BASE_OIDS=()
  PR_SET_HEAD_OIDS=()

  for raw in "${PR_SET_INPUTS[@]}"; do
    IFS=',' read -r -a split <<<"$raw"
    for item in "${split[@]}"; do
      local trimmed
      trimmed="${item//[[:space:]]/}"
      [[ -n "$trimmed" ]] || continue
      local pr_number
      if ! pr_number="$(normalize_pr_identifier "$trimmed")"; then
        die "Invalid PR identifier for --pr-set: $trimmed"
      fi
      if [[ -n "${seen_prs[$pr_number]:-}" ]]; then
        echo "Warning: duplicate PR #$pr_number in --pr-set; ignoring subsequent occurrence" >&2
        continue
      fi
      seen_prs[$pr_number]=1
      PR_SET_NUMBERS+=("$pr_number")
    done
  done

  if (( ${#PR_SET_NUMBERS[@]} < 2 )); then
    die "Provide at least two unique PR numbers via --pr-set"
  fi

  local first_pr="${PR_SET_NUMBERS[0]}"
  local last_index=$(( ${#PR_SET_NUMBERS[@]} - 1 ))
  local last_pr="${PR_SET_NUMBERS[$last_index]}"

  PR_SET_LABEL="$first_pr"
  PR_SET_DISPLAY_LIST="#${first_pr}"
  for pr in "${PR_SET_NUMBERS[@]:1}"; do
    PR_SET_LABEL+="-$pr"
    PR_SET_DISPLAY_LIST+=", #${pr}"
  done

  local meta_list='[]'
  local idx=0

  for pr_number in "${PR_SET_NUMBERS[@]}"; do
    resolve_pr_repo_context "$pr_number"
    local pr_meta_json
    pr_meta_json="$(gh pr view "$pr_number" -R "$REPO" --json number,title,url,author,headRefName,baseRefName,headRefOid,baseRefOid,body)"

    local base_ref_oid head_ref_oid base_ref_name
    base_ref_oid=$(jq -r '.baseRefOid' <<<"$pr_meta_json")
    head_ref_oid=$(jq -r '.headRefOid' <<<"$pr_meta_json")
    base_ref_name=$(jq -r '.baseRefName' <<<"$pr_meta_json")

    local repo_url="https://github.com/${REPO}.git"
    git fetch -q "$repo_url" "refs/pull/${pr_number}/head:refs/remotes/origin/pr/${pr_number}" || true
    git fetch -q "$repo_url" "refs/heads/${base_ref_name}:refs/remotes/origin/${base_ref_name}" || true

    if (( idx == 0 )); then
      BASE_OID="$base_ref_oid"
      BASE_REF="$base_ref_name"
    fi

    PR_SET_BASE_OIDS+=("$base_ref_oid")
    PR_SET_HEAD_OIDS+=("$head_ref_oid")

    meta_list="$(jq --argjson pr "$pr_meta_json" '. + [$pr]' <<<"$meta_list")"
    (( idx += 1 ))
  done

  HEAD_OID="${PR_SET_HEAD_OIDS[-1]}"

  TARGET_IDENTIFIER="PR set ${first_pr}->${last_pr}"
  TARGET_TITLE="Combined diff for PR set (${#PR_SET_NUMBERS[@]} PRs)"
  TARGET_AUTHOR="multiple"
  TARGET_URL=""
  DESCRIPTION_HEADING="PR Set Summary"
  DESCRIPTION_FALLBACK="_No PR details available._"

  PR_SET_METADATA_JSON="$meta_list"

  local description
  description="$(jq -r '[.[] | "- PR #" + (.number|tostring) + ": " + (.title // "(no title)") + " (base: " + (.baseRefName // "unknown") + " @ " + ((.baseRefOid // "")[:7]) + ", head: " + (.headRefName // "unknown") + " @ " + ((.headRefOid // "")[:7]) + ", author: @" + (.author.login // "unknown") + ")" ] | join("\n")' <<<"$PR_SET_METADATA_JSON")"
  if [[ -n "$description" && "$description" != "null" ]]; then
    TARGET_DESCRIPTION="$description"
  else
    TARGET_DESCRIPTION=""
  fi

  DEFAULT_BASENAME="pr-set-${PR_SET_LABEL}-diffs"
}

cleanup_pr_set_worktree() {
  if [[ -n "$PR_SET_WORKTREE_DIR" && -d "$PR_SET_WORKTREE_DIR" ]]; then
    git worktree remove --force "$PR_SET_WORKTREE_DIR" >/dev/null 2>&1 || true
    rm -rf "$PR_SET_WORKTREE_DIR"
    PR_SET_WORKTREE_DIR=""
  fi
}

apply_pr_commits_to_stack() {
  local worktree_dir="$1"
  local base_oid="$2"
  local head_oid="$3"
  local pr_number="$4"

  local commits
  commits="$(git rev-list --reverse "${base_oid}..${head_oid}" 2>/dev/null || true)"

  if [[ -z "$commits" ]]; then
    echo "Warning: PR #${pr_number} has no commits beyond base; skipping" >&2
    return
  fi

  local commit
  for commit in $commits; do
    if git -C "$worktree_dir" cherry-pick --allow-empty "$commit" >/dev/null 2>&1; then
      continue
    fi

    echo "Warning: encountered conflicts cherry-picking ${commit} from PR #${pr_number}; keeping conflict markers" >&2

    if ! git -C "$worktree_dir" add -A; then
      git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
      cleanup_pr_set_worktree
      die "Failed to stage conflict files for commit ${commit} from PR #${pr_number}"
    fi

    if git -C "$worktree_dir" ls-files -u | grep -q .; then
      git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
      cleanup_pr_set_worktree
      die "Unresolved merge entries remain after staging conflicts for commit ${commit} from PR #${pr_number}"
    fi

    if ! GIT_EDITOR=true git -C "$worktree_dir" cherry-pick --continue >/dev/null 2>&1; then
      if ! git -C "$worktree_dir" status --porcelain | grep -q .; then
        GIT_EDITOR=true git -C "$worktree_dir" cherry-pick --skip >/dev/null 2>&1 || true
        continue
      fi
      git -C "$worktree_dir" cherry-pick --abort >/dev/null 2>&1 || true
      cleanup_pr_set_worktree
      die "Failed to finalize conflicted cherry-pick for commit ${commit} from PR #${pr_number}"
    fi
  done
}

prepare_pr_set_worktree() {
  local base_commit="$BASE_OID"
  PR_SET_WORKTREE_DIR="$(mktemp -d -t pr-export-stack.XXXXXXXX)"
  [[ -n "$PR_SET_WORKTREE_DIR" && -d "$PR_SET_WORKTREE_DIR" ]] || die "Failed to create temporary worktree directory"

  trap cleanup_pr_set_worktree EXIT

  git worktree add --detach "$PR_SET_WORKTREE_DIR" "$base_commit" >/dev/null 2>&1
  git -C "$PR_SET_WORKTREE_DIR" config --worktree core.hooksPath /dev/null >/dev/null 2>&1 || true

  local idx=0
  local total=${#PR_SET_NUMBERS[@]}
  while (( idx < total )); do
    local pr_number="${PR_SET_NUMBERS[$idx]}"
    local pr_base="${PR_SET_BASE_OIDS[$idx]}"
    local pr_head="${PR_SET_HEAD_OIDS[$idx]}"
    apply_pr_commits_to_stack "$PR_SET_WORKTREE_DIR" "$pr_base" "$pr_head" "$pr_number"
    (( idx += 1 ))
  done

  PR_SET_SYNTH_HEAD="$(git -C "$PR_SET_WORKTREE_DIR" rev-parse HEAD)"
  HEAD_OID="$PR_SET_SYNTH_HEAD"
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
  contains_test_token "$p" && return 1
  shopt -s extglob nullglob globstar
  for g in "${GLOBS[@]}"; do
    if [[ "$g" == !* ]]; then
      local neg="${g:1}"
      # shellcheck disable=SC2053  # pattern match against glob
      [[ $p == $neg ]] && exc_hit=1
    else
      had_inc=1
      # shellcheck disable=SC2053
      [[ $p == $g ]] && inc_ok=1
    fi
  done
  # If any exclude matches -> reject
  (( exc_hit )) && return 1
  # If we had any includes -> require one matches; else accept
  if (( had_inc )); then
    (( inc_ok )) && return 0
    return 1
  fi
  return 0
}

# Small JSON escaper for patches
json_escape() {
  jq -Rs . | sed 's/^"//; s/"$//'
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
elif [[ "$TARGET_MODE" == "pr_set" ]]; then
  load_pr_set_metadata
  prepare_pr_set_worktree
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
  elif [[ "$TARGET_MODE" == "pr_set" ]]; then
    echo "- PR set: ${PR_SET_DISPLAY_LIST}"
    echo "- Authors: multiple (see PR Set Overview)"
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
  echo "- Base commit: ${BASE_OID:0:7}"
  echo "- Head commit: ${HEAD_OID:0:7}"
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

  if [[ "$TARGET_MODE" == "pr_set" ]]; then
    echo "## PR Set Overview"
    echo ""
    jq -r '.[] | "- PR #" + (.number|tostring) + " — " + (.title // "(no title)") + " • base " + (.baseRefName // "unknown") + " @ " + ((.baseRefOid // "")[:7]) + " • head " + (.headRefName // "unknown") + " @ " + ((.headRefOid // "")[:7]) + " • author @" + (.author.login // "unknown") + (if (.url != null) then " • " + .url else "" end)' <<<"$PR_SET_METADATA_JSON"
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
  elif [[ "$TARGET_MODE" == "pr_set" ]]; then
    jq -n \
      --arg repo "$REPO" \
      --arg exported "$timestamp" \
      --argjson context "$CONTEXT" \
      --argjson totals "$totals_json" \
      --argjson files "$files_for_json" \
      --arg baseOid "$BASE_OID" \
      --arg headOid "$HEAD_OID" \
      --arg mode "$TARGET_MODE" \
      --arg identifier "$TARGET_IDENTIFIER" \
      --arg prSetLabel "$PR_SET_LABEL" \
      --arg prSetDisplay "$PR_SET_DISPLAY_LIST" \
      --argjson prSet "$PR_SET_METADATA_JSON" \
      '{
        repo: $repo,
        pr_set_label: $prSetLabel,
        pr_set_display: $prSetDisplay,
        exported_at: $exported,
        context_radius: $context,
        totals: $totals,
        files: $files,
        oids: {base: $baseOid, head: $headOid},
        mode: $mode,
        identifier: $identifier,
        pr_set: $prSet
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
