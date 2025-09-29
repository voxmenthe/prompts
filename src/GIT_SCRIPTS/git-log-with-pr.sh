#!/usr/bin/env bash
# git-log-with-pr.sh
# Render recent commits annotated with the pull request number (when detectable).
#
# Usage:
#   ./git-log-with-pr.sh [<git log args...>]
#   ./git-log-with-pr.sh --all -n 50 --author "Alice"
#
# By default we show the latest 30 commits on the current branch. Pass any
# regular `git log` selectors/flags to refine the history. The PR identifier is
# inferred from trailers (e.g. "Pull-Request: #123"), commit subjects like
# "Merge pull request #123", or conventional squash merge suffixes "(#123)".
# A trailing `*` in the PR column means we surfaced a tracked PR URL.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: git-log-with-pr.sh [<git log args...>]

Examples:
  git-log-with-pr.sh            # latest 30 commits on current branch
  git-log-with-pr.sh main..HEAD # commits not yet on main
  git-log-with-pr.sh --all -n 5 # sample from entire graph

Any flags and ranges after the script name are forwarded to `git log`.
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

truncate_field() {
  local value="$1"
  local max_len="$2"
  local length
  length=${#value}
  if (( length <= max_len )); then
    printf '%s' "$value"
  else
    printf '%s' "${value:0:max_len-3}..."
  fi
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository"

declare -a log_args=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_args+=("$arg")
      ;;
  esac
done

if (( ${#log_args[@]} == 0 )); then
  log_args+=("--max-count=30")
fi

log_format='%H%x1f%h%x1f%ad%x1f%an%x1f%s%x1f%(trailers:key=Pull-Request)%x1f%(trailers:key=Pull-Requests)%x1f%(trailers:key=PR-Number)%x1f%(trailers:key=PR-URL)%x1f%(trailers:key=Merge-Request)%x1f%(trailers:key=Merge-Request-URL)%x1f%(trailers:key=Merge-Request-ID)'

printf '%-9s %-10s %-20s %-8s %s\n' "commit" "date" "author" "pr" "subject"

git log --no-color --date=short -z --pretty=format:"${log_format}" "${log_args[@]}" \
  | while IFS= read -r -d '' record; do
      IFS=$'\x1f' read -r full_sha short_sha commit_date author subject \
        trailer_pull_request trailer_pull_requests trailer_pr_number trailer_pr_url \
        trailer_mr trailer_mr_url trailer_mr_id <<<"$record"

      [[ -n "${full_sha}" ]] || continue

      pr_number=""
      pr_url=""

      for candidate in \
        "$trailer_pull_request" \
        "$trailer_pull_requests" \
        "$trailer_pr_number" \
        "$trailer_mr" \
        "$trailer_mr_id"; do
        if [[ -n "$candidate" && "$candidate" =~ ([0-9]+) ]]; then
          pr_number="${BASH_REMATCH[1]}"
          break
        fi
      done

      for candidate_url in "$trailer_pr_url" "$trailer_mr_url"; do
        if [[ -n "$candidate_url" ]]; then
          pr_url="$candidate_url"
          if [[ -z "$pr_number" && "$candidate_url" =~ ([0-9]+) ]]; then
            pr_number="${BASH_REMATCH[1]}"
          fi
          break
        fi
      done

      if [[ -z "$pr_number" ]]; then
        if [[ "$subject" =~ Merge\ pull\ request\ \#([0-9]+) ]]; then
          pr_number="${BASH_REMATCH[1]}"
        elif [[ "$subject" =~ \(#([0-9]+)\) ]]; then
          pr_number="${BASH_REMATCH[1]}"
        elif [[ "$subject" =~ PR[[:space:]]*#([0-9]+) ]]; then
          pr_number="${BASH_REMATCH[1]}"
        elif [[ "$subject" =~ pull\/([0-9]+) ]]; then
          pr_number="${BASH_REMATCH[1]}"
        fi
      fi

      pr_display="-"
      if [[ -n "$pr_number" ]]; then
        pr_display="#${pr_number}"
        if [[ -n "$pr_url" ]]; then
          pr_display+="*"
        fi
      fi

      printf '%-9s %-10s %-20s %-8s %s\n' \
        "$short_sha" \
        "$commit_date" \
        "$(truncate_field "$author" 20)" \
        "$pr_display" \
        "$subject"
    done || true
