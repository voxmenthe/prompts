#!/usr/bin/env bash
# pr-export-comments.sh
# Export all PR comments (review + issue) with code context using git + gh CLI.
# Usage:
#   ./pr-export-comments.sh -p <PR_NUMBER|PR_URL> [-r owner/repo] [-c 5] [-o out.md] [--json out.json]
# Examples:
#   ./pr-export-comments.sh -p 123
#   ./pr-export-comments.sh -p https://github.com/org/repo/pull/456 -c 8 -o pr-456.md --json pr-456.json
#
# Notes:
# - Run inside a clone of the repository or pass -r owner/repo.
# - We fetch the PR head/base to ensure the needed commits exist locally.
# - Context comes from the "side" of the diff (RIGHT=head commit, LEFT=base/original commit).
# - If the historical commit/file is unreachable (force-push, rename), we fall back to base/head or to diff_hunk.
# - If the PR number isn't found in this repo (common for forks), we auto-try the upstream parent repo.

set -euo pipefail

# ---------- Defaults ----------
CONTEXT=5
REPO=""
PR_INPUT=""
OUT_MD=""
OUT_JSON=""

# ---------- Helpers ----------
die() { echo "Error: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

usage() {
  sed -n '1,60p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--pr) PR_INPUT="$2"; shift 2;;
      -r|--repo) REPO="$2"; shift 2;;
      -c|--context) CONTEXT="$2"; shift 2;;
      -o|--out) OUT_MD="$2"; shift 2;;
      --json) OUT_JSON="$2"; shift 2;;
      -h|--help) usage;;
      *) die "Unknown arg: $1";;
    esac
  done
  [[ -n "$PR_INPUT" ]] || die "Must provide -p <PR_NUMBER|PR_URL>"
}

# Extract owner/repo and number from inputs (URL or number).
# If repo not given, infer via gh from cwd.
resolve_repo_and_pr() {
  local pr="$1"

  if [[ -z "$REPO" ]]; then
    REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
    [[ -n "$REPO" ]] || die "Could not infer repo; pass -r owner/repo"
  fi

  # If PR input is a URL, let gh parse it and confirm repo matches.
  if [[ "$pr" =~ ^https?:// ]]; then
    PR_NUMBER="$(basename "$pr")"
    [[ "$PR_NUMBER" =~ ^[0-9]+$ ]] || die "Could not parse PR number from URL: $pr"
  else
    [[ "$pr" =~ ^[0-9]+$ ]] || die "PR must be a number or PR URL"
    PR_NUMBER="$pr"
  fi
}

# If this repo is a fork, PR numbers may exist only in the upstream (parent) repo.
# Detect that case and switch REPO automatically.
resolve_pr_repo_context() {
  local pr="$1"

  # Try current repo first.
  if gh pr view "$pr" -R "$REPO" --json number >/dev/null 2>&1; then
    return 0
  fi

  # Fallback to parent/upstream repo if available.
  local parent
  parent="$(gh repo view "$REPO" --json parent -q '.parent | .owner.login + "/" + .name' 2>/dev/null || true)"
  if [[ -n "$parent" && "$parent" != "null" && "$parent" != "/" ]]; then
    if gh pr view "$pr" -R "$parent" --json number >/dev/null 2>&1; then
      echo "PR #$pr not found in $REPO; found in upstream $parent. Switching context." >&2
      REPO="$parent"
      return 0
    fi
  fi

  die "Could not find PR #$pr in $REPO or its upstream parent"
}

json_escape_md() {
  # Escapes triple-backticks within markdown bodies so our fences don't break.
  # Replaces ``` with ``\`\`\`
  sed 's/```/``\\`\\`\\`/g'
}

# Get a blob of file content for COMMIT:PATH, printing specific lines with context, and annotate the commented range.
# Args: commit path start_line end_line context
# Output: annotated text block (or empty if file@commit is missing)
print_context_block() {
  local commit="$1" path="$2" start="$3" end="$4" ctx="$5" fallback_commit="$6"
  local s e hs he use_commit

  # Bounds
  hs="$start"; he="$end"
  s=$(( start - ctx )); (( s < 1 )) && s=1
  e=$(( end + ctx ))

  # Decide which commit to use
  use_commit="$commit"
  if ! git cat-file -e "${use_commit}:${path}" 2>/dev/null; then
    # try fallback (base/head oid)
    if [[ -n "${fallback_commit:-}" ]] && git cat-file -e "${fallback_commit}:${path}" 2>/dev/null; then
      use_commit="$fallback_commit"
    else
      # give up; caller can fall back to diff_hunk
      return 1
    fi
  fi

  # Figure the max line in file to bound e
  local total_lines
  total_lines=$(git show "${use_commit}:${path}" | wc -l | awk '{print $1}')
  if [[ -n "$total_lines" && "$total_lines" =~ ^[0-9]+$ ]]; then
    (( e > total_lines )) && e=$total_lines
  fi

  # Print with markers and line numbers
  git show "${use_commit}:${path}" \
    | awk -v s="$s" -v e="$e" -v hs="$hs" -v he="$he" '
        NR>=s && NR<=e {
          pre = (NR>=hs && NR<=he) ? "▶ " : "  ";
          printf "%s%6d | %s\n", pre, NR, $0
        }'
}

# ---------- Start ----------
require_cmd gh
require_cmd jq
require_cmd git
require_cmd awk
require_cmd sed

parse_args "$@"
resolve_repo_and_pr "$PR_INPUT"
resolve_pr_repo_context "$PR_NUMBER"

# Pull PR metadata (base/head OIDs) and URL/title
PR_META_JSON="$(gh pr view "$PR_NUMBER" -R "$REPO" --json number,title,url,headRefName,baseRefName,headRefOid,baseRefOid)"
PR_URL=$(jq -r '.url' <<<"$PR_META_JSON")
PR_TITLE=$(jq -r '.title' <<<"$PR_META_JSON")
HEAD_OID=$(jq -r '.headRefOid' <<<"$PR_META_JSON")
BASE_OID=$(jq -r '.baseRefOid' <<<"$PR_META_JSON")
BASE_REF=$(jq -r '.baseRefName' <<<"$PR_META_JSON")

# Ensure we have the commits locally (best-effort; tolerate fetch failure).
# Use the resolved REPO URL so forks correctly fetch from upstream.
repo_url="https://github.com/${REPO}.git"
git fetch -q "$repo_url" "refs/pull/${PR_NUMBER}/head:refs/remotes/origin/pr/${PR_NUMBER}" || true
git fetch -q "$repo_url" "refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}" || true

# Fetch review comments (inline + replies)
RC_JSON="$(
  gh api --paginate -H "Accept: application/vnd.github+json" "/repos/${REPO}/pulls/${PR_NUMBER}/comments" \
  | jq -s 'flatten'
)"

# Fetch issue comments (top-level PR comments not attached to code)
IC_JSON="$(
  gh api --paginate -H "Accept: application/vnd.github+json" "/repos/${REPO}/issues/${PR_NUMBER}/comments" \
  | jq -s 'flatten'
)"

# Build a threads array: roots + replies
THREADS_JSON="$(
  jq '
    def roots: map(select(.in_reply_to_id == null));
    def replies: map(select(.in_reply_to_id != null)) | group_by(.in_reply_to_id) | map({root_id: (.[0].in_reply_to_id), replies: .}) | from_entries as $x | .; # placeholder

    # create map from parent id -> replies array
    def replies_map:
      (map(select(.in_reply_to_id != null)) | group_by(.in_reply_to_id) | map({(.[0].in_reply_to_id|tostring): .})) | add // {};

    . as $all
    | (roots) as $roots
    | (replies_map) as $rm
    | $roots
    | sort_by(.path, (.line // 0), (.start_line // 0), .created_at)
    | map({
        id,
        url: .html_url,
        path,
        side,
        line,
        start_line,
        original_line,
        original_start_line,
        original_commit_id,
        commit_id,
        diff_hunk,
        author: .user.login,
        created_at,
        body,
        replies: ($rm[(.id|tostring)] // []) | sort_by(.created_at) | map({
          id, url: .html_url, author: .user.login, created_at, body
        })
      })
  ' <<<"$RC_JSON"
)"

# Prepare outputs
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
[[ -n "$OUT_MD" ]] || OUT_MD="pr-${PR_NUMBER}-comments.md"

# ---------- Write Markdown ----------
{
  echo "# Comments for PR #${PR_NUMBER}: ${PR_TITLE}"
  echo ""
  echo "- Repo: \`${REPO}\`"
  echo "- PR URL: ${PR_URL}"
  echo "- Exported: ${timestamp}"
  echo "- Context radius: ${CONTEXT} line(s)"
  echo ""
  echo "## Review Threads (inline code comments)"
  echo ""

  # Iterate threads
  idx=0
  threads_len=$(jq 'length' <<<"$THREADS_JSON")
  while (( idx < threads_len )); do
    thread=$(jq -r --argjson i "$idx" '.[$i]' <<<"$THREADS_JSON")

    id=$(jq -r '.id' <<<"$thread")
    url=$(jq -r '.url' <<<"$thread")
    path=$(jq -r '.path // ""' <<<"$thread")
    side=$(jq -r '.side // ""' <<<"$thread")
    line=$(jq -r '.line // 0' <<<"$thread")
    start_line=$(jq -r '.start_line // 0' <<<"$thread")
    oline=$(jq -r '.original_line // 0' <<<"$thread")
    ostart=$(jq -r '.original_start_line // 0' <<<"$thread")
    ocmt=$(jq -r '.original_commit_id // ""' <<<"$thread")
    cmt=$(jq -r '.commit_id // ""' <<<"$thread")
    diff_hunk=$(jq -r '.diff_hunk // ""' <<<"$thread")
    author=$(jq -r '.author' <<<"$thread")
    created=$(jq -r '.created_at' <<<"$thread")
    body=$(jq -r '.body // ""' <<<"$thread" | json_escape_md)

    # Compute highlight range + commit to use by side
    hl_start=$start_line; hl_end=$line
    if (( hl_start == 0 || hl_end == 0 )); then
      # Single-line or legacy: prefer "line" else original_line
      if (( line > 0 )); then hl_start=$line; hl_end=$line;
      elif (( oline > 0 )); then hl_start=$oline; hl_end=$oline;
      else hl_start=0; hl_end=0;
      fi
    fi

    # Decide primary + fallback commit for context
    content_commit=""; fallback_commit=""
    if [[ "$side" == "RIGHT" ]]; then
      content_commit="${cmt:-$HEAD_OID}"
      fallback_commit="$HEAD_OID"
    elif [[ "$side" == "LEFT" ]]; then
      content_commit="${ocmt:-$BASE_OID}"
      fallback_commit="$BASE_OID"
    else
      # Unknown side: try head first, then base
      content_commit="${cmt:-$HEAD_OID}"
      fallback_commit="$BASE_OID"
    fi

    echo "### \`$path\`"
    echo "- **Thread**: [$id]($url) • **by** @$author • **at** $created • **side**: \`$side\` • **lines**: ${hl_start}-${hl_end}"
    echo ""
    # Context block
    if (( hl_start > 0 )); then
      if print_context_block "$content_commit" "$path" "$hl_start" "$hl_end" "$CONTEXT" "$fallback_commit" >/tmp/_ctx.$$ 2>/dev/null; then
        echo "\`\`\`text"
        cat /tmp/_ctx.$$
        echo "\`\`\`"
      else
        # Fallback to diff_hunk
        if [[ -n "$diff_hunk" ]]; then
          echo "_Exact file@commit unavailable; showing GitHub diff hunk instead:_"
          echo ""
          echo '```diff'
          # diff_hunk already contains +/- context; print as-is
          printf "%s\n" "$diff_hunk"
          echo '```'
        else
          echo "_No context available._"
        fi
      fi
      rm -f /tmp/_ctx.$$ || true
      echo ""
    fi

    echo "**Comment:**"
    echo ""
    # Preserve any ```suggestion blocks present in body
    printf "%s\n" "$body"
    echo ""

    # Replies
    rlen=$(jq '(.replies // []) | length' <<<"$thread")
    if (( rlen > 0 )); then
      echo "<details><summary>${rlen} repl$( (( rlen==1 )) && echo "y" || echo "ies")</summary>"
      echo ""
      ridx=0
      while (( ridx < rlen )); do
        reply=$(jq -r --argjson i "$ridx" '.replies[$i]' <<<"$thread")
        rauthor=$(jq -r '.author' <<<"$reply")
        rcreated=$(jq -r '.created_at' <<<"$reply")
        rbody=$(jq -r '.body // ""' <<<"$reply" | json_escape_md)
        rurl=$(jq -r '.url' <<<"$reply")
        echo "- **@$rauthor** at $rcreated — [$rurl]($rurl)"
        echo ""
        printf "%s\n\n" "$rbody"
        ((ridx++))
      done
      echo "</details>"
      echo ""
    fi

    ((idx++))
  done

  echo ""
  echo "## General Conversation (top-level PR comments)"
  ic_len=$(jq 'length' <<<"$IC_JSON")
  if (( ic_len == 0 )); then
    echo "_None_"
  else
    jq -r '
      sort_by(.created_at) |
      .[] | (
        "### " + (.user.login // "unknown") + " — " + .created_at + "  \n" +
        "[" + .html_url + "](" + .html_url + ")\n\n" +
        (.body // "")
      )
    ' <<<"$IC_JSON" | json_escape_md
  fi

} > "$OUT_MD"

# ---------- Optional JSON emission ----------
if [[ -n "$OUT_JSON" ]]; then
  jq -n \
    --arg repo "$REPO" \
    --arg prNumber "$PR_NUMBER" \
    --arg prUrl "$PR_URL" \
    --arg prTitle "$PR_TITLE" \
    --arg exported "$timestamp" \
    --argjson threads "$THREADS_JSON" \
    --argjson issueComments "$IC_JSON" \
    '{
      repo: $repo,
      pr: ($prNumber|tonumber),
      url: $prUrl,
      title: $prTitle,
      exported_at: $exported,
      context_radius: '"$CONTEXT"' ,
      review_threads: $threads,
      issue_comments: $issueComments
    }' > "$OUT_JSON"
fi

echo "✓ Wrote Markdown: $OUT_MD"
[[ -n "$OUT_JSON" ]] && echo "✓ Wrote JSON:     $OUT_JSON"
