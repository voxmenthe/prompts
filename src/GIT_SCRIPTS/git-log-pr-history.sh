#!/usr/bin/env bash
# git-log-pr-history.sh
# Emit the entire git history (default), in chronological order, annotated with
# pull-request numbers and their source branches when detectable.
#
# The script works purely from local git metadata: trailers injected by merge
# tooling (e.g. "Pull-Request: #123"), classic merge commit subjects like
# "Merge pull request #123 from owner/topic", or squash commit suffixes "(#123)".
# When a source branch or PR number cannot be inferred, we surface "-" so the
# blast radius stays contained and expectations remain explicit.
#
# Usage:
#   ./git-log-pr-history.sh                # entire history, chronological
#   ./git-log-pr-history.sh -n 50          # limit to first 50 chronological commits
#   ./git-log-pr-history.sh main..HEAD     # focus on commits not yet on main
#
# Arguments are forwarded to `git log`. The script always adds `--all` and
# `--reverse --date-order` unless you override them explicitly.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Error: git-log-pr-history.sh must be run with bash (use ./git-log-pr-history.sh or bash git-log-pr-history.sh)." >&2
  exit 1
fi

if (( BASH_VERSINFO[0] < 4 )); then
  echo "Error: git-log-pr-history.sh requires bash 4.0 or newer (detected ${BASH_VERSION})." >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage: git-log-pr-history.sh [<git log args...>]

Examples:
  git-log-pr-history.sh                # entire history in chronological order
  git-log-pr-history.sh -n 50          # limit to 50 commits
  git-log-pr-history.sh main..HEAD     # show commits on current branch only

Notes:
  * Run with bash 4+ (use ./git-log-pr-history.sh or bash git-log-pr-history.sh).
  * Defaults to "--all --reverse --date-order" unless traversal flags are
    provided explicitly.
  * PR numbers and source branches are inferred from commit trailers and
    merge-message conventions; missing data is surfaced as "-".
  * Override the primary branch detection with GIT_MAIN_BRANCH if needed.
USAGE
  exit 2
}

die() {
  echo "Error: $*" >&2
  exit 1
}

trim() {
  local s="$1"
  s="${s#${s%%[!$' \t\r\n']*}}"
  s="${s%${s##*[!$' \t\r\n']}}"
  printf '%s' "$s"
}

truncate_field() {
  local value="$1"
  local max_len="$2"
  local length=${#value}
  if (( length <= max_len )); then
    printf '%s' "$value"
  else
    printf '%s' "${value:0:max_len-3}..."
  fi
}

infer_pr_number_from_candidates() {
  local candidate pr_match
  for candidate in "$@"; do
    if [[ -n "$candidate" && "$candidate" =~ ([0-9]+) ]]; then
      pr_match="${BASH_REMATCH[1]}"
      printf '%s' "$pr_match"
      return 0
    fi
  done
  return 1
}

infer_pr_number_from_text() {
  local text="$1"
  local normalized
  normalized="${text//$'\n'/ }"
  if [[ "$normalized" =~ pull[[:space:]]request[[:space:]]\#([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$normalized" =~ \(#([0-9]+)\) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$normalized" =~ PR[[:space:]]*\#([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$normalized" =~ pull/([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

infer_branch_from_text() {
  local text="$1"
  local normalized
  normalized="${text//$'\n'/ }"
  if [[ "$normalized" =~ pull[[:space:]]request[[:space:]]#?[0-9]+[[:space:]]+from[[:space:]]+([^[:space:]]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$normalized" =~ from[[:space:]]+([^[:space:]]+)[[:space:]]+into ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$normalized" =~ Source[[:space:]]Branch:[[:space:]]*([^[:space:]]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

normalize_branch_hint() {
  local hint="$1"
  [[ -n "$hint" ]] || return 1
  hint="$(trim "$hint")"
  hint="${hint%%~*}"
  hint="${hint#refs/heads/}"
  hint="${hint#refs/remotes/}"
  hint="${hint#heads/}"
  if [[ "$hint" == remotes/* ]]; then
    hint="${hint#remotes/}"
  fi
  if [[ "$hint" == origin/* ]]; then
    :
  fi
  if [[ "$hint" == undefined ]]; then
    return 1
  fi
  printf '%s' "$hint"
}

resolve_primary_branch() {
  local candidate
  if [[ -n "${GIT_MAIN_BRANCH:-}" ]]; then
    candidate="${GIT_MAIN_BRANCH}"
    if git rev-parse --verify "${candidate}^{commit}" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
    if git rev-parse --verify "origin/${candidate}^{commit}" >/dev/null 2>&1; then
      printf '%s' "origin/${candidate}"
      return 0
    fi
  fi

  local origin_head
  origin_head="$(git symbolic-ref --quiet refs/remotes/origin/HEAD 2>/dev/null || true)"
  if [[ -n "$origin_head" ]]; then
    origin_head="${origin_head#refs/remotes/}"
    if git rev-parse --verify "${origin_head}^{commit}" >/dev/null 2>&1; then
      printf '%s' "$origin_head"
      return 0
    fi
  fi

  for candidate in main master trunk; do
    if git rev-parse --verify "${candidate}^{commit}" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
    if git rev-parse --verify "origin/${candidate}^{commit}" >/dev/null 2>&1; then
      printf '%s' "origin/${candidate}"
      return 0
    fi
  done

  printf ''
}

NAME_REV_KEYS=()
NAME_REV_VALUES=()

lookup_branch_for_commit() {
  local sha="$1"
  [[ -n "$sha" ]] || return 1

  local idx=0
  for existing_sha in "${NAME_REV_KEYS[@]}"; do
    if [[ "$existing_sha" == "$sha" ]]; then
      local cached="${NAME_REV_VALUES[$idx]}"
      if [[ -n "$cached" ]]; then
        printf '%s' "$cached"
        return 0
      fi
      return 1
    fi
    ((idx++))
  done

  local raw
  raw="$(git name-rev --name-only --refs='refs/heads/* refs/remotes/*' "$sha" 2>/dev/null || true)"
  raw="$(normalize_branch_hint "$raw" 2>/dev/null || true)"

  NAME_REV_KEYS+=("$sha")
  NAME_REV_VALUES+=("$raw")

  if [[ -n "$raw" ]]; then
    printf '%s' "$raw"
    return 0
  fi
  return 1
}

commit_in_list() {
  local sha="$1"
  local file="$2"
  [[ -n "$file" ]] || return 1
  [[ -s "$file" ]] || return 1
  if grep -F -q "$sha" "$file"; then
    return 0
  fi
  return 1
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository"

PRIMARY_BRANCH="${GIT_MAIN_BRANCH:-}"
if [[ -z "$PRIMARY_BRANCH" ]]; then
  PRIMARY_BRANCH="$(resolve_primary_branch)"
fi

PRIMARY_COMMITS_FILE=""
if [[ -n "$PRIMARY_BRANCH" ]]; then
  if git rev-parse --verify "${PRIMARY_BRANCH}^{commit}" >/dev/null 2>&1; then
    PRIMARY_COMMITS_FILE="$(mktemp 2>/dev/null || mktemp -t gitlogprimary)"
    [[ -n "$PRIMARY_COMMITS_FILE" ]] || die "Failed to create temp file for primary branch commits"
    git rev-list "$PRIMARY_BRANCH" >"$PRIMARY_COMMITS_FILE"
  fi
fi

CURRENT_COMMITS_FILE="$(mktemp 2>/dev/null || mktemp -t gitlogcurrent)"
[[ -n "$CURRENT_COMMITS_FILE" ]] || die "Failed to create temp file for current branch commits"
git rev-list HEAD >"$CURRENT_COMMITS_FILE"

cleanup_tmp_files() {
  [[ -n "$PRIMARY_COMMITS_FILE" ]] && rm -f "$PRIMARY_COMMITS_FILE"
  [[ -n "$CURRENT_COMMITS_FILE" ]] && rm -f "$CURRENT_COMMITS_FILE"
}
trap cleanup_tmp_files EXIT

declare -a base_args=("--all" "--reverse" "--date-order")
declare -a passed_args=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      ;;
    --reverse|--no-reverse|--date-order|--topo-order|--graph|--all)
      base_args=() # respect caller-provided traversal controls
      passed_args+=("$arg")
      ;;
    *)
      passed_args+=("$arg")
      ;;
  esac
done

declare -a log_args=("--no-color" "--date=short" "-z")
log_args+=("${base_args[@]}")
log_args+=("${passed_args[@]}")
log_args+=("--pretty=format:%H%x1f%h%x1f%ad%x1f%an%x1f%P%x1f%s%x1f%b%x1f%(trailers:key=Pull-Request)%x1f%(trailers:key=Pull-Requests)%x1f%(trailers:key=PR-Number)%x1f%(trailers:key=PR-URL)%x1f%(trailers:key=Merge-Request)%x1f%(trailers:key=Merge-Request-URL)%x1f%(trailers:key=Merge-Request-ID)%x1f%(trailers:key=Head-Ref)%x1f%(trailers:key=Source-Branch)%x1f%(trailers:key=Merge-Request-Source-Branch)%x1f%(trailers:key=Merge-Request-Head-Ref)")

printf '%-9s %-10s %-20s %-4s %-12s %-9s %-28s %s\n' \
  "commit" "date" "author" "type" "reach" "pr" "integration" "subject"

git log "${log_args[@]}" \
  | while IFS= read -r -d '' record; do
      IFS=$'\x1f' read -r full_sha short_sha commit_date author parents subject body \
        trailer_pull_request trailer_pull_requests trailer_pr_number trailer_pr_url \
        trailer_mr trailer_mr_url trailer_mr_id trailer_head_ref trailer_source_branch \
        trailer_mr_source trailer_mr_head <<<"$record"

      [[ -n "${full_sha}" ]] || continue

      pr_number=""
      pr_source=""
      commit_type="C"
      reach_label="other"

      parent_count=0
      parent_shas=()
      if [[ -n "$parents" ]]; then
        read -r -a parent_shas <<<"$parents"
        parent_count=${#parent_shas[@]}
      fi

      if (( parent_count > 1 )); then
        commit_type="M"
      fi

      if commit_in_list "$full_sha" "$PRIMARY_COMMITS_FILE"; then
        reach_label="main"
      elif commit_in_list "$full_sha" "$CURRENT_COMMITS_FILE"; then
        reach_label="current"
      fi

      if pr_number=$(infer_pr_number_from_candidates \
          "$trailer_pull_request" "$trailer_pull_requests" "$trailer_pr_number" \
          "$trailer_mr" "$trailer_mr_id"); then
        :
      else
        pr_number=""
      fi

      if [[ -z "$pr_number" ]]; then
        if pr_candidate=$(infer_pr_number_from_text "$subject"); then
          pr_number="$pr_candidate"
        elif pr_candidate=$(infer_pr_number_from_text "$body"); then
          pr_number="$pr_candidate"
        fi
      fi

      if [[ -z "$pr_source" ]]; then
        for ref_candidate in \
          "$trailer_head_ref" "$trailer_source_branch" "$trailer_mr_source" "$trailer_mr_head"; do
          if [[ -n "$ref_candidate" ]]; then
            pr_source="$ref_candidate"
            break
          fi
        done
      fi

      if [[ -z "$pr_source" ]]; then
        if branch_guess=$(infer_branch_from_text "$subject"); then
          pr_source="$branch_guess"
        elif branch_guess=$(infer_branch_from_text "$body"); then
          pr_source="$branch_guess"
        fi
      fi

      if [[ -z "$pr_source" && $parent_count -gt 1 ]]; then
        for (( idx=1; idx<parent_count; idx++ )); do
          if branch_guess=$(lookup_branch_for_commit "${parent_shas[$idx]}"); then
            pr_source="$branch_guess"
            break
          fi
        done
      fi

      if [[ -n "$pr_source" ]]; then
        if normalized=$(normalize_branch_hint "$pr_source" 2>/dev/null); then
          pr_source="$normalized"
        else
          pr_source="$(trim "$pr_source")"
          pr_source="${pr_source%% *}"
          pr_source="${pr_source%%,*}"
          pr_source="${pr_source%\"}"
          pr_source="${pr_source%%\"*}"
        fi
      fi

      if [[ -z "$pr_number" ]]; then
        pr_display="-"
      else
        pr_display="#${pr_number}"
      fi

      if [[ -z "$pr_source" ]]; then
        branch_display="-"
      else
        branch_display="$pr_source"
      fi

      printf '%-9s %-10s %-20s %-4s %-12s %-9s %-28s %s\n' \
        "$short_sha" \
        "$commit_date" \
        "$(truncate_field "$author" 20)" \
        "$commit_type" \
        "$reach_label" \
        "$pr_display" \
        "$(truncate_field "$branch_display" 28)" \
        "$subject"
    done || true
