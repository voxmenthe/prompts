#!/usr/bin/env bash
# worktree-functions.sh
# Shell helpers for managing Git worktrees via ergonomic `wt-*` functions.

# Prevent double-loading when sourced multiple times.
if [[ -n "${__WT_FUNCTIONS_SOURCED:-}" ]]; then
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 0
  else
    return 0
  fi
fi
__WT_FUNCTIONS_SOURCED=1

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

if wt__is_execution_context; then
  set -euo pipefail
fi

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

wt__have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

wt__require_cmd() {
  local cmd
  for cmd in "$@"; do
    wt__have_cmd "$cmd" || wt_die "Missing dependency: $cmd"
  done
}

wt_log() {
  local level="$1"; shift || true
  local prefix="[wt]"
  local color=""
  case "$level" in
    info) color="$WT__CLR_INFO"; prefix="[wt][info]";;
    warn) color="$WT__CLR_WARN"; prefix="[wt][warn]";;
    error) color="$WT__CLR_ERR"; prefix="[wt][error]";;
    *) prefix="[wt][$level]";;
  esac
  printf '%b%s %s%b\n' "$color" "$prefix" "$*" "$WT__CLR_RESET" >&2
}

wt_die() {
  wt_log error "$*"
  if wt__is_execution_context; then
    exit 1
  else
    return 1
  fi
}

wt__confirm() {
  local prompt="$1" default_choice="${2:-y}"
  if [[ "${WT_ASSUME_YES:-0}" == "1" ]]; then
    return 0
  fi
  local default_indicator="[Y/n]"
  if [[ "$default_choice" =~ ^[Nn]$ ]]; then
    default_indicator="[y/N]"
  fi
  local reply
  while true; do
    read -r -p "$prompt $default_indicator " reply || return 1
    reply="${reply:-$default_choice}"
    case "$reply" in
      [Yy]*) return 0 ;;
      [Nn]*) return 1 ;;
    esac
  done
}

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

wt__script_abs_path() {
  wt__abs_path "$__wt_script_path"
}

wt__repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || wt_die "Run inside a Git repository"
}

wt__git_dir() {
  git rev-parse --git-dir 2>/dev/null || wt_die "Run inside a Git repository"
}

wt__worktree_root() {
  local repo_root
  repo_root="$(wt__repo_root)"
  local configured="${WT_WORKTREE_ROOT:-../wt}"
  if [[ "$configured" == /* ]]; then
    printf '%s' "$configured"
  else
    wt__abs_path "$repo_root/$configured"
  fi
}

wt__ensure_worktree_root() {
  local root
  root="$(wt__worktree_root)"
  mkdir -p "$root"
  printf '%s' "$root"
}

wt__default_base() {
  printf '%s' "${WT_DEFAULT_BASE:-main}"
}

wt__rand() {
  if [[ -r /dev/urandom ]] && wt__have_cmd tr; then
    LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6
  else
    printf '%06x' "$((RANDOM * RANDOM % 1679616))"
  fi
}

wt__branch_slug() {
  local branch="$1"
  local prefix="${WT_BRANCH_PREFIX:-feature}"
  branch="${branch#refs/heads/}"
  if [[ "$branch" == "$prefix/"* ]]; then
    branch="${branch#"${prefix}"/}"
  fi
  branch="${branch//_/ -}"
  branch="${branch,,}"
  branch="${branch//_/ -}"  # ensure underscores replaced even after lowercasing
  branch="${branch//[^a-z0-9-]/-}"
  while [[ "$branch" == *--* ]]; do
    branch="${branch//--/-}"
  done
  while [[ "$branch" == -* ]]; do
    branch="${branch#-}"
  done
  while [[ "$branch" == *- ]]; do
    branch="${branch%-}"
  done
  [[ -n "$branch" ]] || branch="feature"
  printf '%s' "$branch"
}

wt__default_branch_name() {
  local prefix="${1:-${WT_BRANCH_PREFIX:-feature}}"
  local stamp
  stamp="$(date +%Y%m%d)"
  printf '%s/%s-%s' "$prefix" "$stamp" "$(wt__rand)"
}

wt__dir_for_branch() {
  local branch="$1"
  printf '%s' "${branch//\//_}"
}

wt__current_branch() {
  git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

wt__active_branch() {
  if [[ -n "${WT_ACTIVE_BRANCH:-}" ]]; then
    printf '%s' "$WT_ACTIVE_BRANCH"
  else
    wt__current_branch
  fi
}

wt__ensure_git_ready() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || wt_die "Run inside a Git repository"
  if [[ -d "$(wt__git_dir)/rebase-merge" || -d "$(wt__git_dir)/rebase-apply" ]]; then
    wt_die "Rebase in progress; resolve it before continuing"
  fi
  if [[ -f "$(wt__git_dir)/MERGE_HEAD" ]]; then
    wt_die "Merge in progress; resolve it before continuing"
  fi
  if [[ -f "$(wt__git_dir)/CHERRY_PICK_HEAD" ]]; then
    wt_die "Cherry-pick in progress; resolve it before continuing"
  fi
}

wt__dirty_tree() {
  local path="${1:-}"
  local opts=(--short --untracked-files=normal)
  if [[ -n "$path" ]]; then
    git -C "$path" status "${opts[@]}" | grep -q '.'
  else
    git status "${opts[@]}" | grep -q '.'
  fi
}

wt__worktree_list() {
  git worktree list --porcelain
}

wt__worktree_path_for_branch() {
  local branch="$1" path="" current_path="" ref=""
  while read -r key value; do
    case "$key" in
      worktree) current_path="$value" ;;
      branch)
        ref="${value#refs/heads/}"
        if [[ "$ref" == "$branch" ]]; then
          path="$current_path"
        fi
        ;;
    esac
  done < <(wt__worktree_list)
  printf '%s' "$path"
}

wt__fetch_ref() {
  local remote="$1" ref="$2"
  git fetch "$remote" "$ref" >/dev/null 2>&1 || true
}

wt__ensure_branch_exists() {
  local branch="$1" base="$2" remote="$3"
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    return 0
  fi
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
  local upstream
  upstream="$(git rev-parse --abbrev-ref "$branch@{upstream}" 2>/dev/null || true)"
  if [[ -z "$upstream" ]]; then
    wt_log info "Setting upstream $remote/$branch"
    git push -u "$remote" "$branch"
  fi
}

wt__remote_slug() {
  local remote="${1:-origin}"
  local url
  url="$(git config --get remote."$remote".url 2>/dev/null || true)"
  [[ -n "$url" ]] || return 1

  case "$url" in
    git@*:* )
      url="${url#*@}"        # github.com:owner/repo.git
      url="${url#*:}"         # owner/repo.git
      url="${url%.git}"
      printf '%s' "$url"
      ;;
    https://github.com/*)
      url="${url#https://github.com/}"
      url="${url%.git}"
      printf '%s' "$url"
      ;;
    ssh://git@github.com/*)
      url="${url#ssh://git@github.com/}"
      url="${url%.git}"
      printf '%s' "$url"
      ;;
    git://github.com/*)
      url="${url#git://github.com/}"
      url="${url%.git}"
      printf '%s' "$url"
      ;;
    *)
      return 1
      ;;
  esac
}

wt-new() {
  wt__ensure_git_ready
  local branch=""
  local base
  base="$(wt__default_base)"
  local dir=""
  local enter=0
  local no_track=0
  local checkout_only=0
  local remote="origin"
  local prefix_override=""
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
Usage: wt-new [--branch BRANCH] [--base BASE] [--dir PATH] [--remote origin]
              [--prefix PREFIX] [--enter] [--no-track] [--checkout-only]
Create a new worktree rooted at WT_WORKTREE_ROOT (default ../wt).
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-new: $1";;
    esac
  done

  local branch_prefix="${WT_BRANCH_PREFIX:-feature}"
  if [[ -n "$prefix_override" ]]; then
    branch_prefix="$prefix_override"
  fi
  if [[ -z "$branch" ]]; then
    branch="$(wt__default_branch_name "$branch_prefix")"
  fi
  branch="${branch#refs/heads/}"

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
      wt_log info "Branch created. Use git switch $branch to check it out."
    fi
    return
  fi

  local branch_exists=0
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    branch_exists=1
  fi

  local root
  root="$(wt__ensure_worktree_root)"
  local target_dir
  if [[ -n "$dir" ]]; then
    if [[ "$dir" == /* ]]; then
      target_dir="$dir"
    else
      target_dir="$root/$dir"
    fi
  else
    target_dir="$root/$(wt__dir_for_branch "$branch")"
  fi

  if [[ -e "$target_dir" ]]; then
    wt_die "Worktree directory already exists: $target_dir"
  fi

  local add_args=()
  if (( branch_exists )); then
    add_args=("$target_dir" "$branch")
  else
    add_args=("$target_dir" "$start_ref")
    add_args=("-b" "$branch" "${add_args[@]}")
    if (( ! no_track )) && git show-ref --verify --quiet "refs/remotes/$remote/$base"; then
      add_args=("--track" "${add_args[@]}")
    fi
  fi

  wt_log info "git worktree add ${add_args[*]}"
  git worktree add "${add_args[@]}"

  wt_log info "Worktree created at $target_dir"
  if (( enter )); then
    WT_NEW_BRANCH_CREATED=1 WT_NEW_TARGET_DIR="$target_dir" wt-enter "$branch"
  else
    wt_log info "Next: wt-enter $branch"
  fi
}

wt-enter() {
  wt__ensure_git_ready
  local branch=""
  local open_editor=0
  local run_cmd=""
  local auto_create=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --command) run_cmd="$2"; shift 2;;
      --open) open_editor=1; shift;;
      --auto-create) auto_create=1; shift;;
      -h|--help)
        cat <<'EOF'
Usage: wt-enter [BRANCH] [--command CMD] [--open] [--auto-create]
Enter the worktree for the given branch and start a subshell.
EOF
        return
        ;;
      *)
        if [[ -z "$branch" ]]; then
          branch="$1"; shift
        else
          wt_die "Unexpected argument: $1"
        fi
        ;;
    esac
  done

  if [[ -z "$branch" ]]; then
    branch="$(wt__active_branch)"
    [[ -n "$branch" ]] || wt_die "Cannot infer branch; pass one explicitly"
  fi

  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" || ! -d "$path" ]]; then
    if (( auto_create )); then
      wt_log warn "No worktree for $branch. Creating via wt-new..."
      wt-new --branch "$branch"
      path="$(wt__worktree_path_for_branch "$branch")"
    else
      wt_die "No worktree directory found for branch $branch"
    fi
  fi

  if [[ ! -d "$path" ]]; then
    wt_die "Worktree path invalid: $path"
  fi

  wt_log info "Entering worktree $path"
  if [[ -n "$run_cmd" ]]; then
    (
      cd "$path" || exit 1
      WT_ACTIVE_BRANCH="$branch" eval "$run_cmd"
    )
    return
  fi
  if (( open_editor )); then
    wt__open_editor "$path"
  fi
  (
    cd "$path" || exit 1
    WT_ACTIVE_BRANCH="$branch" exec "${SHELL:-bash}"
  )
}

wt__open_editor() {
  local path="$1"
  if [[ -n "${WT_EDITOR:-}" ]]; then
    ( cd "$path" && eval "$WT_EDITOR" ) && return
  fi
  if wt__have_cmd code; then
    ( cd "$path" && code . ) && return
  fi
  wt_log warn "No editor configured via WT_EDITOR and VS Code unavailable"
}

wt-status() {
  wt__ensure_git_ready
  local branch=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      -h|--help)
        cat <<'EOF'
Usage: wt-status [--branch BRANCH]
Show summary information for the branch worktree (default current).
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-status: $1";;
    esac
  done

  if [[ -z "$branch" ]]; then
    branch="$(wt__active_branch)"
  fi
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"

  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" ]]; then
    path="$(wt__repo_root)"
  fi
  if [[ ! -d "$path" ]]; then
    wt_die "Worktree path not found for $branch"
  fi

  local git_status
  git_status=$(cd "$path" && git status --short --branch)
  local head
  head=$(cd "$path" && git rev-parse --short HEAD)
  local upstream
  upstream=$(cd "$path" && git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo "(no upstream)")
  local base="${WT_DEFAULT_BASE:-main}"
  if cd "$path" && git config "branch.$branch.merge" >/dev/null 2>&1; then
    base=$(cd "$path" && git config "branch.$branch.merge")
    base="${base#refs/heads/}"
  fi

  wt_log info "Branch: $branch (@$head)"
  wt_log info "Upstream: $upstream"
  wt_log info "Default base: $base"
  if [[ -n "$path" ]]; then
    wt_log info "Worktree path: $path"
  fi
  if [[ -n "$git_status" ]]; then
    printf '%s\n' "$git_status"
  fi
}

wt-sync() {
  wt__ensure_git_ready
  local branch=""
  local remote="origin"
  local base=""
  local merge=0
  local do_stash=1
  local run_tests=0
  local test_cmd="${WT_SYNC_TEST_CMD:-}"
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
Usage: wt-sync [--branch BRANCH] [--remote origin] [--base BASE]
               [--merge] [--no-stash] [--run-tests]
Fetch remote and rebase (default) or merge branch on top of base.
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-sync: $1";;
    esac
  done

  if [[ -z "$branch" ]]; then
    branch="$(wt__active_branch)"
  fi
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" ]]; then
    path="$(wt__repo_root)"
  fi
  if [[ ! -d "$path" ]]; then
    wt_die "Worktree path not found for $branch"
  fi

  if [[ -z "$base" ]]; then
    base="${WT_DEFAULT_BASE:-main}"
    if cd "$path" && git config "branch.$branch.merge" >/dev/null 2>&1; then
      base=$(cd "$path" && git config "branch.$branch.merge")
      base="${base#refs/heads/}"
    fi
  fi

  wt_log info "Syncing $branch with $remote/$base"
  ( cd "$path" || exit 1
    git fetch "$remote" "$base" >/dev/null 2>&1 || true
    git fetch "$remote" "$branch" >/dev/null 2>&1 || true
    local stashed=0
    if (( do_stash )) && wt__dirty_tree; then
      wt_log info "Stashing dirty changes"
      if git stash push -u -m "wt-sync $branch" >/dev/null 2>&1; then
        stashed=1
      fi
    fi
    if (( merge )); then
      git merge "${remote}/${base}" || {
        if (( stashed )); then
          git stash pop >/dev/null 2>&1 || true
        fi
        wt_die "Merge failed"
      }
    else
      git rebase "${remote}/${base}" || {
        wt_log error "Rebase failed; resolve conflicts and rerun wt-sync"
        if (( stashed )); then
          wt_log warn "Stash preserved. Use git stash list to inspect."
        fi
        exit 1
      }
    fi
    if (( stashed )); then
      wt_log info "Restoring stash"
      git stash pop >/dev/null 2>&1 || wt_log warn "Stash conflict; resolve manually"
    fi
    if (( run_tests )); then
      if [[ -n "$test_cmd" ]]; then
        wt_log info "Running tests: $test_cmd"
        eval "$test_cmd"
      else
        wt_log warn "WT_SYNC_TEST_CMD not set; skipping tests"
      fi
    fi
  )
}

wt-commit() {
  wt__ensure_git_ready
  local branch
  branch="$(wt__active_branch)"
  local message=""
  local use_all=1
  local signoff=0
  local amend=0
  local preview=0
  local skip_hooks=0
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
Usage: wt-commit [--message MSG] [--no-all] [--signoff] [--amend]
                 [--preview] [--skip-hooks]
Stage tracked changes (default) and create a commit with a helpful template.
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-commit: $1";;
    esac
  done

  [[ -n "$branch" ]] || wt_die "Cannot determine branch for commit"
  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" ]]; then
    path="$(wt__repo_root)"
  fi

  ( cd "$path" || exit 1
    if (( use_all )); then
      git add -A
    fi
    if git diff --cached --quiet; then
      wt_die "Nothing staged for commit"
    fi
    wt_log info "Preparing commit on $branch"
    local slug
    slug="$(wt__branch_slug "$branch")"
    local template=""
    local commit_args=()
    commit_args+=("commit")
    if (( signoff )); then
      commit_args+=("--signoff")
    fi
    if (( amend )); then
      commit_args+=("--amend")
    fi
    if (( skip_hooks )); then
      commit_args+=("--no-verify")
    fi
    if [[ -n "$message" ]]; then
      commit_args+=("-m" "$message")
    else
      template="$(mktemp -t wt-commit.XXXXXX)"
      export WT_COMMIT_TEMPLATE="$template"
      printf 'feat: %s ' "$slug" >"$template"
      printf '\n\n# Add details below. Lines starting with # will be stripped.\n' >>"$template"
      commit_args+=("--template" "$template")
    fi
    if (( preview )); then
      git diff --stat --cached
    fi
    git "${commit_args[@]}"
    if [[ -n "$template" ]]; then
      rm -f "$template"
    fi
  )
}

wt-pr() {
  wt__ensure_git_ready
  local branch
  branch="$(wt__active_branch)"
  local title=""
  local body=""
  local draft=0
  local assignees=()
  local reviewers=()
  local labels=()
  local no_open=0
  local notes=""
  local remote="origin"
  local base=""
  local open_browser="${WT_PR_OPEN_WEB:-1}"
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
Usage: wt-pr [--title TITLE] [--body BODY] [--draft]
             [--assignee USER] [--reviewer USER] [--label LABEL]
             [--no-open] [--notes TEXT] [--base BASE] [--remote origin]
Ensure branch is pushed and create a Pull Request via GitHub CLI when available.
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-pr: $1";;
    esac
  done

  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  if [[ -z "$path" ]]; then
    path="$(wt__repo_root)"
  fi
  [[ -d "$path" ]] || wt_die "Worktree path not found for $branch"

  ( cd "$path" || exit 1
    wt_log info "Pushing $branch to $remote"
    if ! git push "$remote" "$branch" --set-upstream; then
      git push "$remote" "$branch" || wt_die "Failed to push branch to $remote"
    fi

    if [[ -z "$base" ]]; then
      base=$(git config --get "branch.$branch.merge" 2>/dev/null || true)
      base="${base#refs/heads/}"
      [[ -n "$base" ]] || base="$(wt__default_base)"
    fi

    if ! wt__have_cmd gh; then
      wt_log warn "gh CLI not available. Create PR manually."
      local slug compare_url
      slug="$(wt__remote_slug "$remote")"
      if [[ -n "$slug" ]]; then
        compare_url="https://github.com/${slug}/compare/${base}...${branch}?expand=1"
        wt_log info "Open browser to: $compare_url"
      else
        wt_log info "Run: gh pr create --head $branch"
      fi
      return
    fi

    if [[ -z "$title" ]]; then
      title=$(git log -1 --pretty=%s "$branch" 2>/dev/null || wt__branch_slug "$branch")
    fi

    local body_content="$body"
    if [[ -n "$notes" ]]; then
      if [[ -n "$body_content" ]]; then
        body_content+=$'\n\n'
      fi
      body_content+="$notes"
    fi

    local gh_args=("pr" "create" "--head" "$branch" "--base" "$base")
    [[ -n "$title" ]] && gh_args+=("--title" "$title")
    [[ -n "$body_content" ]] && gh_args+=("--body" "$body_content")
    if (( draft )); then
      gh_args+=("--draft")
    fi
    if (( no_open )); then
      gh_args+=("--fill")
    fi
    local item
    for item in "${assignees[@]}"; do
      gh_args+=("--assignee" "$item")
    done
    for item in "${reviewers[@]}"; do
      gh_args+=("--reviewer" "$item")
    done
    for item in "${labels[@]}"; do
      gh_args+=("--label" "$item")
    done

    wt_log info "Creating PR via gh ${gh_args[*]}"
    if ! gh "${gh_args[@]}"; then
      wt_die "gh pr create failed"
    fi

    if (( ! no_open )) && [[ "$open_browser" != "0" ]]; then
      wt_log info "Opening PR in browser"
      gh pr view --web "$branch" >/dev/null 2>&1 || true
    fi
  )
}

wt-clean() {
  wt__ensure_git_ready
  local branch=""
  local force=0
  local keep_branch=0
  local remote_delete=0
  local prune=0
  local archive=0
  local remote="origin"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch) branch="$2"; shift 2;;
      --force) force=1; shift;;
      --keep-branch) keep_branch=1; shift;;
      --remote) remote="$2"; remote_delete=1; shift 2;;
      --prune) prune=1; shift;;
      --archive) archive=1; shift;;
      -h|--help)
        cat <<'EOF'
Usage: wt-clean [--branch BRANCH] [--force] [--keep-branch]
                [--remote origin] [--prune] [--archive]
Remove worktree and optionally delete branches once merged.
EOF
        return
        ;;
      *) wt_die "Unknown option for wt-clean: $1";;
    esac
  done

  if [[ -z "$branch" ]]; then
    branch="$(wt__active_branch)"
  fi
  [[ -n "$branch" ]] || wt_die "Cannot determine branch"
  local path
  path="$(wt__worktree_path_for_branch "$branch")"
  [[ -n "$path" ]] || wt_die "No worktree found for $branch"
  [[ -d "$path" ]] || wt_die "Worktree path missing: $path"

  local repo_root
  repo_root="$(wt__repo_root)"

  if (( ! force )); then
    if wt__dirty_tree "$path"; then
      wt_die "Worktree dirty. Commit, stash, or pass --force"
    fi
    if ! git -C "$repo_root" branch --merged | grep -E "(^|\s)$branch$" >/dev/null 2>&1; then
      wt_die "Branch $branch is not merged. Use --force to override."
    fi
  fi

  if (( archive )); then
    local bundle
    bundle="$repo_root/.git/wt-archive-${branch//\//_}-$(date +%Y%m%d%H%M%S).bundle"
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
    wt_log warn "Aborted worktree removal"
    return
  fi

  if (( ! keep_branch )); then
    wt_log info "Deleting branch $branch"
    git -C "$repo_root" branch -d "$branch" 2>/dev/null || git -C "$repo_root" branch -D "$branch"
  fi

  if (( remote_delete )); then
    local remote_slug
    remote_slug="$(wt__remote_slug "$remote")"
    if [[ -n "$remote_slug" ]]; then
      wt_log info "Deleting remote branch $remote_slug/$branch"
      git -C "$repo_root" push "$remote" ":$branch" || wt_log warn "Failed to delete remote branch"
    else
      wt_log warn "Failed to parse remote slug from $remote"
    fi
  fi

  if (( prune )); then
    git -C "$repo_root" worktree prune
    git -C "$repo_root" remote prune "$remote" || true
  fi
}

wt-help() {
  cat <<'EOF'
wt-new        Create a worktree for a new or existing branch.
{{ ... }}
wt-status     Show branch/worktree summary.
wt-sync       Fetch & rebase/merge branch on latest base.
wt-commit     Stage changes (optional) and craft a commit message.
wt-pr         Push branch and open a GitHub PR (if gh available).
wt-clean      Remove worktree and optionally delete branches.
wt-help       Show this help.

Environment:
  WT_WORKTREE_ROOT   Base directory for worktrees (default ../wt relative to repo).
  WT_DEFAULT_BASE    Base branch (default main).
  WT_BRANCH_PREFIX   Prefix for generated branches (default feature).
  WT_ASSUME_YES      If 1, skip confirmations.
  WT_NO_COLOR        Disable colorized logging.
  WT_EDITOR          Command to open an editor within wt-enter --open.
  WT_SYNC_TEST_CMD   Command run with wt-sync --run-tests.
  WT_PR_OPEN_WEB     If set, open browser even when --no-open omitted.
EOF
}

wt() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    new) wt-new "$@";;
    enter) wt-enter "$@";;
    status) wt-status "$@";;
    sync) wt-sync "$@";;
    commit) wt-commit "$@";;
    pr) wt-pr "$@";;
    clean) wt-clean "$@";;
    help|-h|--help) wt-help;;
    *) wt_die "Unknown wt subcommand: $cmd";;
  esac
}

wt__bootstrap() {
  local path
  path="$(wt__script_abs_path)"
  cat <<EOF
# Add to your shell rc file
if [[ -f "$path" ]]; then
  # shellcheck disable=SC1090
  source "$path"
fi
EOF
}

wt__dispatch_main() {
  if [[ $# -eq 0 ]]; then
    wt-help
    return
  fi
  case "$1" in
    --bootstrap)
      wt__bootstrap
      ;;
    wt)
      shift
      wt "$@"
      ;;
    wt-*)
      local fn="$1"
      shift
      if declare -F "$fn" >/dev/null 2>&1; then
        "$fn" "$@"
      else
        wt_die "Unknown command: $fn"
      fi
      ;;
    -h|--help|help)
      wt-help
      ;;
    *)
      wt_die "Unrecognized argument: $1"
      ;;
  esac
}

if wt__is_execution_context; then
  wt__dispatch_main "$@"
fi
