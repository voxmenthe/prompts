#!/usr/bin/env bash
# worktree-functions.sh (consolidated & improved)
# Shell helpers for managing Git worktrees via ergonomic `wt` commands.
# - Valid bash function names (underscored)
# - Safer cleanup & base detection from remote HEAD
# - Robust worktree root anchoring
# - Usability tweaks (positional args, editor fallback, aliases)

# Prevent double-loading when sourced multiple times.
if [[ -n "${__WT_FUNCTIONS_SOURCED:-}" ]]; then
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 0
  else
    return 0
  fi
fi
__WT_FUNCTIONS_SOURCED=1

# Require Bash 4+
if [[ -z "${BASH_VERSINFO:-}" || ${BASH_VERSINFO[0]} -lt 4 ]]; then
  echo "worktree-functions.sh requires Bash 4+" >&2
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 1
  else
    return 1
  fi
fi

__wt_script_path="${BASH_SOURCE[0]}"

wt__is_execution_context() {
  [[ "${BASH_SOURCE[0]}" == "${0}" ]]
}

# In execution context, be strict
if wt__is_execution_context; then
  set -euo pipefail
fi

# ----- Colors & logging -------------------------------------------------------

wt__init_colors() {
  if [[ -n "${WT_NO_COLOR:-}" || ! -t 2 ]]; then
    WT__CLR_INFO=""
    WT__CLR_WARN=""
    WT__CLR_ERR=""
    WT__CLR_RESET=""
  else
    WT__CLR_INFO=$'\033[32m'
    WT__CLR_WARN=$'\033[33m'
    WT__CLR_ERR=$'\033[31m'
    WT__CLR_RESET=$'\033[0m'
  fi
}
wt__init_colors

wt__have_cmd() { command -v "$1" >/dev/null 2>&1; }
wt__require_cmd() { local c; for c in "$@"; do wt__have_cmd "$c" || wt_die "Missing dependency: $c"; done; }

wt_log() {
  local level="$1"; shift || true
  local prefix="[wt]" color=""
  case "$level" in
    info)  color="$WT__CLR_INFO";  prefix="[wt][info]";;
    warn)  color="$WT__CLR_WARN";  prefix="[wt][warn]";;
    error) color="$WT__CLR_ERR";   prefix="[wt][error]";;
    *)     prefix="[wt][$level]";;
  esac
  printf '%b%s %s%b\n' "$color" "$prefix" "$*" "$WT__CLR_RESET" >&2
}

wt_die() {
  wt_log error "$*"
  if wt__is_execution_context; then exit 1; else return 1; fi
}

# Read a Y/N from user (defaults to yes); reads from TTY when stdin is piped.
wt__confirm() {
  local prompt="$1" default_choice="${2:-y}"
  if [[ "${WT_ASSUME_YES:-0}" == "1" ]]; then return 0; fi
  local default_indicator="[Y/n]"
  [[ "$default_choice" =~ ^[Nn]$ ]] && default_indicator="[y/N]"
  local reply
  while true; do
    if [[ -t 0 ]]; then
      read -r -p "$prompt $default_indicator " reply || return 1
    else
      printf '%s %s ' "$prompt" "$default_indicator" > /dev/tty
      read -r reply < /dev/tty || return 1
    fi
    reply="${reply:-$default_choice}"
    case "$reply" in
      [Yy]*) return 0 ;;
      [Nn]*) return 1 ;;
    esac
  done
}

# ----- Path & repo helpers ----------------------------------------------------

wt__abs_path() {
  local path="$1"
  if wt__have_cmd realpath; then
    realpath "$path"
  else
    (
      cd "$(dirname "$path")" >/dev/null 2>&1 || return 1
      printf '%s/%s\n' "$(pwd -P)" "$(basename "$path")"
    )
  fi
}

wt__script_abs_path() { wt__abs_path "$__wt_script_path"; }
wt__repo_root() { git rev-parse --show-toplevel 2>/dev/null || wt_die "Run inside a Git repository"; }
wt__git_dir()   { git rev-parse --git-dir       2>/dev/null || wt_die "Run inside a Git repository"; }

wt__worktree_list() { git worktree list --porcelain; }

# First worktree entry, used as an anchor for stable WT_WORKTREE_ROOT
wt__primary_worktree_path() {
  local path=""
  while read -r k v; do
    if [[ "$k" == "worktree" ]]; then path="$v"; break; fi
  done < <(wt__worktree_list)
  printf '%s' "$path"
}

# Base directory for all worktrees:
# - If WT_WORKTREE_ROOT is absolute, use as-is.
# - If relative, anchor under the *primary* worktree (or repo root as fallback).
wt__worktree_root() {
  local anchor
  anchor="$(wt__primary_worktree_path || true)"
  [[ -n "$anchor" ]] || anchor="$(wt__repo_root)"
  local configured="${WT_WORKTREE_ROOT:-../wt}"
  if [[ "$configured" == /* ]]; then
    printf '%s' "$configured"
  else
    wt__abs_path "$anchor/$configured"
  fi
}

wt__ensure_worktree_root() { local r; r="$(wt__worktree_root)"; mkdir -p "$r"; printf '%s' "$r"; }

# Default base branch: WT_DEFAULT_BASE or remote HEAD (origin/main), fallback 'main'
wt__remote_default_branch() {
  local remote="${1:-origin}"
  local ref
  ref="$(git symbolic-ref --quiet --short "refs/remotes/${remote}/HEAD" 2>/dev/null || true)" # e.g., origin/main
  printf '%s' "${ref#${remote}/}"
}
wt__default_base() {
  local remote="${1:-origin}"
  if [[ -n "${WT_DEFAULT_BASE:-}" ]]; then
    printf '%s' "$WT_DEFAULT_BASE"
  else
    local b; b="$(wt__remote_default_branch "$remote")"
    printf '%s' "${b:-main}"
  fi
}

wt__rand() {
  if [[ -r /dev/urandom ]] && wt__have_cmd tr; then
    LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6
  else
    printf '%06x' "$((RANDOM * RANDOM % 1679616))"
  fi
}

# Turn a branch into a stable, URL/filename-friendly slug
wt__branch_slug() {
  local branch="$1"
  local prefix="${WT_BRANCH_PREFIX:-feature}"
  branch="${branch#refs/heads/}"
  if [[ "$branch" == "$prefix/"* ]]; then branch="${branch#"${prefix}"/}"; fi
  branch="${branch//_/ -}"    # (historical) normalize underscores temporarily
  branch="${branch,,}"        # lowercase
  branch="${branch//_/-}"     # correct: underscores -> dashes (no spaces)
  branch="${branch//[^a-z0-9-]/-}"
  while [[ "$branch" == *--* ]]; do branch="${branch//--/-}"; done
  branch="${branch##-}"; branch="${branch%%-}"
  [[ -n "$branch" ]] || branch="feature"
  printf '%s' "$branch"
}

wt__default_branch_name() {
  local prefix="${1:-${WT_BRANCH_PREFIX:-feature}}"
  local stamp; stamp="$(date +%Y%m%d)"
  printf '%s/%s-%s' "$prefix" "$stamp" "$(wt__rand)"
}

wt__dir_for_branch() { local branch="$1"; printf '%s' "${branch//\//_}"; }

wt__current_branch() { git rev-parse --abbrev-ref HEAD 2>/dev/null || true; }
wt__active_branch()  { [[ -n "${WT_ACTIVE_BRANCH:-}" ]] && printf '%s' "$WT_ACTIVE_BRANCH" || wt__current_branch; }

wt__ensure_git_ready() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || wt_die "Run inside a Git repository"
  if [[ -d "$(wt__git_dir)/rebase-merge" || -d "$(wt__git_dir)/rebase-apply" ]]; then wt_die "Rebase in progress; resolve it before continuing"; fi
  [[ ! -f "$(wt__git_dir)/MERGE_HEAD"       ]] || wt_die "Merge in progress; resolve it before continuing"
  [[ ! -f "$(wt__git_dir)/CHERRY_PICK_HEAD" ]] || wt_die "Cherry-pick in progress; resolve it before continuing"
}

wt__dirty_tree() {
  local path="${1:-}" opts=(--short --untracked-files=normal)
  if [[ -n "$path" ]]; then git -C "$path" status "${opts[@]}" | grep -q '.'; else git status "${opts[@]}" | grep -q '.'; fi
}

wt__worktree_path_for_branch() {
  local branch="$1" path="" current_path="" ref=""
  while read -r key value; do
    case "$key" in
      worktree) current_path="$value" ;;
      branch)
        ref="${value#refs/heads/}"
        if [[ "$ref" == "$branch" ]]; then path="$current_path"; fi
        ;;
    esac
  done < <(wt__worktree_list)
  printf '%s' "$path"
}

wt__fetch_ref() { local remote="$1" ref="$2"; git fetch "$remote" "$ref" >/dev/null 2>&1 || true; }

wt__ensure_branch_exists() {
  local branch="$1" base="$2" remote="$3"
  if git show-ref --verify --quiet "refs/heads/$branch"; then return 0; fi
  wt_log info "Creating branch $branch from $base"
  local start_ref=""
  if git show-ref --verify --quiet "refs/remotes/$remote/$base"; then
    start_ref="$remote/$base"
  elif git show-ref --verify --quiet "refs/heads/$base"; then
    start_ref="$base"
  else
    wt_die "Base ref $base not found locally or on $remote"
  fi
  git branch "$branch" "$start_ref"
}

wt__ensure_tracking() {
  local branch="$1" remote="$2"
  local upstream; upstream="$(git rev-parse --abbrev-ref "$branch@{upstream}" 2>/dev/null || true)"
  if [[ -z "$upstream" ]]; then
    wt_log info "Setting upstream $remote/$branch"
    git push -u "$remote" "$branch"
  fi
}

wt__remote_slug() {
  local remote="${1:-origin}" url
  url="$(git config --get remote."$remote".url 2>/dev/null || true)"
  [[ -n "$url" ]] || return 1
  case "$url" in
    git@*:* ) url="${url#*@}"; url="${url#*:}"; url="${url%.git}"; printf '%s' "$url" ;;   # owner/repo
    https://github.com/*)          printf '%s' "${url#https://github.com/%.git}" ;;
    ssh://git@github.com/*)        printf '%s' "${url#ssh://git@github.com/%.git}" ;;
    git://github.com/*)            printf '%s' "${url#git://github.com/%.git}" ;;
    *) return 1 ;;
  esac
}

# ----- Public commands --------------------------------------------------------

wt_new() {
  wt__ensure_git_ready
  local branch="" base="" dir="" enter=0 no_track=0 checkout_only=0 remote="origin" prefix_override=""
  # Parse args (also allow positional BRANCH)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      --base) base="$2"; shift 2;;
      --dir) dir="$2"; shift 2;;
      --prefix) prefix_override="$2"; shift 2;;
      --enter) enter=1; shift;;
      --no-track) no_track=1; shift;;
      --checkout-only) checkout_only=1; shift;;
      --remote) remote="$2"; shift 2;;
      -h|--help)
        cat <<'EOF'
Usage: wt new [BRANCH] [--base BASE] [--dir PATH] [--remote origin]
              [--prefix PREFIX] [--enter] [--no-track] [--checkout-only]
Create a new worktree rooted at WT_WORKTREE_ROOT (default ../wt).
EOF
        return ;;
      -*)
        wt_die "Unknown option for wt new: $1" ;;
      *)
        if [[ -z "$branch" ]]; then branch="$1"; shift; else wt_die "Unexpected argument: $1"; fi ;;
    esac
  done

  local branch_prefix="${WT_BRANCH_PREFIX:-feature}"
  [[ -n "$prefix_override" ]] && branch_prefix="$prefix_override"
  [[ -n "$branch" ]] || branch="$(wt__default_branch_name "$branch_prefix")"
  branch="${branch#refs/heads/}"
  [[ -n "$base" ]] || base="$(wt__default_base "$remote")"

  # Fast-path: existing worktree
  local existing
  existing="$(wt__worktree_path_for_branch "$branch")"
  if [[ -n "$existing" ]]; then
    wt_log warn "Branch $branch already has a worktree: $existing"
    if (( enter )); then WT_ACTIVE_BRANCH="$branch" wt_enter "$branch"; fi
    return 0
  fi

  wt_log info "Preparing worktree for branch $branch (base $base)"
  wt__fetch_ref "$remote" "$base"
  wt__fetch_ref "$remote" "$branch"

  local start_ref=""
  if git show-ref --verify --quiet "refs/remotes/$remote/$base"; then
    start_ref="$remote/$base"
  elif git show-ref --verify --quiet "refs/heads/$base"; then
    start_ref="$base"
  else
    wt_die "Base ref $base not found locally or on $remote"
  fi

  if (( checkout_only )); then
    if git show-ref --verify --quiet "refs/heads/$branch"; then
      wt_log warn "Branch $branch already exists locally"
    else
      git branch "$branch" "$start_ref"
      wt_log info "Branch created. Use: git switch $branch"
    fi
    return
  fi

  local branch_exists=0
  git show-ref --verify --quiet "refs/heads/$branch" && branch_exists=1

  local root; root="$(wt__ensure_worktree_root)"
  local target_dir
  if [[ -n "$dir" ]]; then
    if [[ "$dir" == /* ]]; then target_dir="$dir"; else target_dir="$root/$dir"; fi
  else
    target_dir="$root/$(wt__dir_for_branch "$branch")"
  fi
  [[ ! -e "$target_dir" ]] || wt_die "Worktree directory already exists: $target_dir"

  local add_args=()
  if (( branch_exists )); then
    add_args=("$target_dir" "$branch")
  else
    add_args=(-b "$branch" "$target_dir" "$start_ref")
  fi

  wt_log info "git worktree add ${add_args[*]}"
  git worktree add "${add_args[@]}"

  # Optionally try to set upstream; ignore errors (branch may not yet exist remotely)
  if (( ! no_track )); then
    ( cd "$target_dir" && git branch --set-upstream-to "$remote/$branch" >/dev/null 2>&1 || true )
  fi

  wt_log info "Worktree created at $target_dir"
  if (( enter )); then
    WT_NEW_BRANCH_CREATED=1 WT_NEW_TARGET_DIR="$target_dir" WT_ACTIVE_BRANCH="$branch" wt_enter "$branch"
  else
    wt_log info "Next: wt enter $branch"
  fi
}

wt_enter() {
  wt__ensure_git_ready
  local branch="" open_editor=0 run_cmd="" auto_create=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --command) run_cmd="$2"; shift 2;;
      --open) open_editor=1; shift;;
      --auto-create) auto_create=1; shift;;
      -h|--help)
        cat <<'EOF'
Usage: wt enter [BRANCH] [--command CMD] [--open] [--auto-create]
Enter the worktree for the given branch and start a subshell.
EOF
        return ;;
      -*)
        wt_die "Unknown option for wt enter: $1" ;;
      *)
        if [[ -z "$branch" ]]; then branch="$1"; shift; else wt_die "Unexpected argument: $1"; fi ;;
    esac
  done

  if [[ -z "$branch" ]]; then
    branch="$(wt__active_branch)"
    [[ -n "$branch" ]] || wt_die "Cannot infer branch; pass one explicitly"
  fi

  local path; path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" || ! -d "$path" ]]; then
    if (( auto_create )); then
      wt_log warn "No worktree for $branch. Creating via wt new..."
      wt_new "$branch"
      path="$(wt__worktree_path_for_branch "$branch")"
    else
      wt_die "No worktree directory found for branch $branch"
    fi
  fi
  [[ -d "$path" ]] || wt_die "Worktree path invalid: $path"

  wt_log info "Entering worktree $path"
  if [[ -n "$run_cmd" ]]; then
    ( cd "$path" || exit 1; WT_ACTIVE_BRANCH="$branch" eval "$run_cmd" )
    return
  fi
  if (( open_editor )); then wt__open_editor "$path"; fi
  ( cd "$path" || exit 1; WT_ACTIVE_BRANCH="$branch" exec "${SHELL:-bash}" )
}

wt__open_editor() {
  local path="$1"
  local ed="${WT_EDITOR:-${VISUAL:-${EDITOR:-}}}"
  if [[ -n "$ed" ]]; then ( cd "$path" && eval "$ed" . ) && return; fi
  if wt__have_cmd code; then ( cd "$path" && code . ) && return; fi
  wt_log warn "No editor configured via WT_EDITOR/VISUAL/EDITOR and VS Code unavailable"
}

wt_status() {
  wt__ensure_git_ready
  local branch=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      -h|--help)
        cat <<'EOF'
Usage: wt status [--branch BRANCH]
Show summary information for the branch worktree (default current).
EOF
        return ;;
      *) wt_die "Unknown option for wt status: $1";;
    esac
  done

  [[ -n "$branch" ]] || branch="$(wt__active_branch)"
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"

  local path; path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || path="$(wt__repo_root)"
  [[ -d "$path" ]] || wt_die "Worktree path not found for $branch"

  local git_status; git_status=$(cd "$path" && git status --short --branch)
  local head; head=$(cd "$path" && git rev-parse --short HEAD)
  local upstream; upstream=$(cd "$path" && git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo "(no upstream)")
  local base; base="$(wt__default_base origin)"

  wt_log info "Branch: $branch (@$head)"
  wt_log info "Upstream: $upstream"
  wt_log info "Default base: $base"
  wt_log info "Worktree path: $path"
  [[ -n "$git_status" ]] && printf '%s\n' "$git_status"
}

wt_sync() {
  wt__ensure_git_ready
  local branch="" remote="origin" base="" merge=0 do_stash=1 run_tests=0 test_cmd="${WT_SYNC_TEST_CMD:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      --remote) remote="$2"; shift 2;;
      --base) base="$2"; shift 2;;
      --merge) merge=1; shift;;
      --no-stash) do_stash=0; shift;;
      --run-tests) run_tests=1; shift;;
      -h|--help)
        cat <<'EOF'
Usage: wt sync [--branch BRANCH] [--remote origin] [--base BASE]
               [--merge] [--no-stash] [--run-tests]
Fetch remote and rebase (default) or merge branch on top of base.
EOF
        return ;;
      *) wt_die "Unknown option for wt sync: $1";;
    esac
  done

  [[ -n "$branch" ]] || branch="$(wt__active_branch)"
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path; path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || path="$(wt__repo_root)"
  [[ -d "$path" ]] || wt_die "Worktree path not found for $branch"

  [[ -n "$base" ]] || base="$(wt__default_base "$remote")"

  wt_log info "Syncing $branch with $remote/$base"
  ( cd "$path" || exit 1
    git fetch "$remote" "$base" >/dev/null 2>&1 || true
    git fetch "$remote" "$branch" >/dev/null 2>&1 || true
    local stashed=0
    if (( do_stash )) && wt__dirty_tree; then
      wt_log info "Stashing dirty changes"
      if git stash push -u -m "wt-sync $branch" >/dev/null 2>&1; then stashed=1; fi
    fi
    if (( merge )); then
      git merge "${remote}/${base}" || {
        (( stashed )) && git stash pop >/dev/null 2>&1 || true
        wt_die "Merge failed"
      }
    else
      git rebase "${remote}/${base}" || {
        wt_log error "Rebase failed; resolve conflicts and rerun wt sync"
        (( stashed )) && wt_log warn "Stash preserved. Use git stash list to inspect."
        exit 1
      }
    fi
    (( stashed )) && { wt_log info "Restoring stash"; git stash pop >/dev/null 2>&1 || wt_log warn "Stash conflict; resolve manually"; }
    if (( run_tests )); then
      if [[ -n "$test_cmd" ]]; then wt_log info "Running tests: $test_cmd"; eval "$test_cmd"; else wt_log warn "WT_SYNC_TEST_CMD not set; skipping tests"; fi
    fi
  )
}

wt_commit() {
  wt__ensure_git_ready
  local branch; branch="$(wt__active_branch)"
  local message="" use_all=1 signoff=0 amend=0 preview=0 skip_hooks=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --message|-m) message="$2"; shift 2;;
      --no-all) use_all=0; shift;;
      --all) use_all=1; shift;;
      --signoff) signoff=1; shift;;
      --amend) amend=1; shift;;
      --preview) preview=1; shift;;
      --skip-hooks) skip_hooks=1; shift;;
      -h|--help)
        cat <<'EOF'
Usage: wt commit [--message MSG] [--no-all] [--signoff] [--amend]
                 [--preview] [--skip-hooks]
Stage tracked changes (default) and create a commit with a helpful template.
EOF
        return ;;
      *) wt_die "Unknown option for wt commit: $1";;
    esac
  done

  [[ -n "$branch" ]] || wt_die "Cannot determine branch for commit"
  local path; path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || path="$(wt__repo_root)"

  ( cd "$path" || exit 1
    (( use_all )) && git add -A
    if git diff --cached --quiet; then wt_die "Nothing staged for commit"; fi
    wt_log info "Preparing commit on $branch"
    local slug; slug="$(wt__branch_slug "$branch")"
    local template="" ; local commit_args=("commit")
    (( signoff )) && commit_args+=("--signoff")
    (( amend ))   && commit_args+=("--amend")
    (( skip_hooks )) && commit_args+=("--no-verify")
    if [[ -n "$message" ]]; then
      commit_args+=("-m" "$message")
    else
      template="$(mktemp -t wt-commit.XXXXXX)"
      export WT_COMMIT_TEMPLATE="$template"
      printf 'feat: %s ' "$slug" >"$template"
      printf '\n\n# Add details below. Lines starting with # will be stripped.\n' >>"$template"
      commit_args+=("--template" "$template")
    fi
    (( preview )) && git diff --stat --cached
    git "${commit_args[@]}"
    [[ -n "$template" ]] && rm -f "$template"
  )
}

wt_pr() {
  wt__ensure_git_ready
  local branch; branch="$(wt__active_branch)"
  local title="" body="" draft=0 assignees=() reviewers=() labels=()
  local no_open=0 notes="" remote="origin" base="" open_browser="${WT_PR_OPEN_WEB:-1}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      --title) title="$2"; shift 2;;
      --body) body="$2"; shift 2;;
      --draft) draft=1; shift;;
      --assignee) assignees+=("$2"); shift 2;;
      --reviewer) reviewers+=("$2"); shift 2;;
      --label) labels+=("$2"); shift 2;;
      --no-open) no_open=1; open_browser=0; shift;;
      --notes) notes="$2"; shift 2;;
      --base) base="$2"; shift 2;;
      --remote) remote="$2"; shift 2;;
      -h|--help)
        cat <<'EOF'
Usage: wt pr [--title TITLE] [--body BODY] [--draft]
             [--assignee USER] [--reviewer USER] [--label LABEL]
             [--no-open] [--notes TEXT] [--base BASE] [--remote origin]
Ensure branch is pushed and create a Pull Request via GitHub CLI when available.
EOF
        return ;;
      *) wt_die "Unknown option for wt pr: $1";;
    esac
  done

  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path; path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || path="$(wt__repo_root)"
  [[ -d "$path" ]] || wt_die "Worktree path not found for $branch"

  ( cd "$path" || exit 1
    wt_log info "Pushing $branch to $remote"
    if ! git push "$remote" "$branch" --set-upstream; then
      git push "$remote" "$branch" || wt_die "Failed to push branch to $remote"
    fi

    [[ -n "$base" ]] || base="$(wt__default_base "$remote")"

    if ! wt__have_cmd gh; then
      wt_log warn "gh CLI not available. Create PR manually."
      local slug compare_url
      slug="$(wt__remote_slug "$remote")"
      if [[ -n "$slug" ]]; then
        compare_url="https://github.com/${slug}/compare/${base}...${branch}?expand=1"
        wt_log info "Open browser to: $compare_url"
      else
        wt_log info "Push succeeded. Open a PR for '$branch' against '$base' in your host UI."
      fi
      return
    fi

    [[ -n "$title" ]] || title=$(git log -1 --pretty=%s "$branch" 2>/dev/null || wt__branch_slug "$branch")

    local body_content="$body"
    if [[ -n "$notes" ]]; then
      [[ -n "$body_content" ]] && body_content+=$'\n\n'
      body_content+="$notes"
    fi

    local gh_args=("pr" "create" "--head" "$branch" "--base" "$base")
    [[ -n "$title" ]] && gh_args+=("--title" "$title")
    [[ -n "$body_content" ]] && gh_args+=("--body" "$body_content")
    (( draft )) && gh_args+=("--draft")
    (( no_open )) && gh_args+=("--fill")
    local item
    for item in "${assignees[@]}"; do gh_args+=("--assignee" "$item"); done
    for item in "${reviewers[@]}"; do gh_args+=("--reviewer" "$item"); done
    for item in "${labels[@]}"; do gh_args+=("--label" "$item"); done

    wt_log info "Creating PR via gh ${gh_args[*]}"
    gh "${gh_args[@]}" || wt_die "gh pr create failed"

    if (( ! no_open )) && [[ "$open_browser" != "0" ]]; then
      wt_log info "Opening PR in browser"
      gh pr view --web "$branch" >/dev/null 2>&1 || true
    fi
  )
}

wt_clean() {
  wt__ensure_git_ready
  local branch="" force=0 keep_branch=0 remote_delete=0 prune=0 archive=0 remote="origin" base=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      --force) force=1; shift;;
      --keep-branch) keep_branch=1; shift;;
      --remote) remote="$2"; remote_delete=1; shift 2;;
      --prune) prune=1; shift;;
      --archive) archive=1; shift;;
      --base) base="$2"; shift 2;;
      -h|--help)
        cat <<'EOF'
Usage: wt clean [--branch BRANCH] [--force] [--keep-branch]
                [--remote origin] [--base BASE] [--prune] [--archive]
Remove worktree and optionally delete branches once merged into base.
EOF
        return ;;
      *) wt_die "Unknown option for wt clean: $1";;
    esac
  done

  [[ -n "$branch" ]] || branch="$(wt__active_branch)"
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path; path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || wt_die "No worktree found for $branch"
  [[ -d "$path" ]] || wt_die "Worktree path missing: $path"

  local here; here="$(pwd -P)"
  case "$here/" in
    "$path"/*) wt_die "You are inside $path; cd elsewhere or pass --force";;
  esac

  local repo_root; repo_root="$(wt__repo_root)"
  [[ -n "$base" ]] || base="$(wt__default_base "$remote")"

  if (( ! force )); then
    if wt__dirty_tree "$path"; then wt_die "Worktree dirty. Commit, stash, or pass --force"; fi
    git -C "$repo_root" fetch "$remote" "$base" >/dev/null 2>&1 || true
    if ! git -C "$repo_root" merge-base --is-ancestor "$branch" "$remote/$base"; then
      wt_die "Branch $branch is not merged into $remote/$base. Use --force to override."
    fi
  fi

  if (( archive )); then
    local bundle; bundle="$repo_root/.git/wt-archive-${branch//\//_}-$(date +%Y%m%d%H%M%S).bundle"
    wt_log info "Archiving branch to $bundle"
    git -C "$repo_root" bundle create "$bundle" "$branch" >/dev/null 2>&1 || wt_log warn "Failed to create bundle"
  fi

  if (( force )) || wt__confirm "Remove worktree $path?" "y"; then
    wt_log info "Removing worktree $path"
    if (( force )); then
      git -C "$repo_root" worktree remove --force "$path"
    else
      git -C "$repo_root" worktree remove "$path"
    fi
  else
    wt_log warn "Aborted worktree removal"; return
  fi

  if (( ! keep_branch )); then
    wt_log info "Deleting branch $branch"
    git -C "$repo_root" branch -d "$branch" 2>/dev/null || git -C "$repo_root" branch -D "$branch"
  fi

  if (( remote_delete )); then
    local remote_slug; remote_slug="$(wt__remote_slug "$remote")"
    if [[ -n "$remote_slug" ]]; then
      wt_log info "Deleting remote branch $remote_slug/$branch"
      git -C "$repo_root" push "$remote" ":$branch" || wt_log warn "Failed to delete remote branch"
    else
      wt_log warn "Failed to parse GitHub slug from remote '$remote' (non-GitHub host?)"
    fi
  fi

  (( prune )) && { git -C "$repo_root" worktree prune; git -C "$repo_root" remote prune "$remote" || true; }
}

wt_help() {
  cat <<'EOF'
wt new        Create a worktree for a new or existing branch.
wt enter      Enter the worktree (optionally auto-create).
wt status     Show branch/worktree summary.
wt sync       Fetch & rebase/merge branch on latest base.
wt commit     Stage changes (optional) and craft a commit message.
wt pr         Push branch and open a GitHub PR (if gh available).
wt clean      Remove worktree and optionally delete branches.
wt help       Show this help.

Environment:
  WT_WORKTREE_ROOT   Base directory for worktrees (default ../wt relative to repo).
  WT_DEFAULT_BASE    Base branch (default: remote HEAD or 'main').
  WT_BRANCH_PREFIX   Prefix for generated branches (default feature).
  WT_ASSUME_YES      If 1, skip confirmations.
  WT_NO_COLOR        Disable colorized logging.
  WT_EDITOR          Preferred editor; falls back to VISUAL/EDITOR, then VS Code.
  WT_SYNC_TEST_CMD   Command run with 'wt sync --run-tests'.
  WT_PR_OPEN_WEB     If 0, do not open PR in browser after creation (default 1).
EOF
}

# Multiplexer
wt() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    new)    wt_new "$@";;
    enter)  wt_enter "$@";;
    status) wt_status "$@";;
    sync)   wt_sync "$@";;
    commit) wt_commit "$@";;
    pr)     wt_pr "$@";;
    clean)  wt_clean "$@";;
    help|-h|--help) wt_help;;
    *) wt_die "Unknown wt subcommand: $cmd";;
  esac
}

# Optional: allow users to type hyphen commands interactively (aliases)
if [[ $- == *i* ]]; then
  alias wt-new='wt new'
  alias wt-enter='wt enter'
  alias wt-status='wt status'
  alias wt-sync='wt sync'
  alias wt-commit='wt commit'
  alias wt-pr='wt pr'
  alias wt-clean='wt clean'
  alias wt-help='wt help'
fi

wt__bootstrap() {
  local path; path="$(wt__script_abs_path)"
  cat <<EOF
# Add to your shell rc file
if [[ -f "$path" ]]; then
  # shellcheck disable=SC1090
  source "$path"
fi
EOF
}

wt__dispatch_main() {
  if [[ $# -eq 0 ]]; then wt_help; return; fi
  case "$1" in
    --bootstrap) wt__bootstrap ;;
    wt) shift; wt "$@" ;;
    wt-*) # Map hyphen-style invocation to 'wt <sub>'
      local sub="${1#wt-}"; shift; wt "$sub" "$@";;
    -h|--help|help) wt_help ;;
    *) wt_die "Unrecognized argument: $1" ;;
  esac
}

if wt__is_execution_context; then
  wt__dispatch_main "$@"
fi
