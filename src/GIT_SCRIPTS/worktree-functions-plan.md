# Worktree Function Suite Plan

## Goals
- **Automate** Provide a frictionless flow for creating, using, and retiring Git worktrees without manual bookkeeping.
- **Centralize** Ship a single Bash file that defines sourced functions (`wt-new`, `wt-pr`, etc.) callable from any repo shell session.
- **Safe defaults** Assume sensible behaviors (random branch names, base `main`, sane worktree paths) while allowing overrides.
- **Composable** Allow chaining (e.g., `wt-new && wt-enter`) and integration with existing scripts like `pr-export-diffs.sh`.

## File Layout & Usage
- **Location** Store functions in `src/GIT_SCRIPTS/worktree-functions.sh`
- **Loading** Recommend `source path/to/worktree-functions.sh` in `~/.bashrc`, or run `eval "$(path/to/worktree-functions.sh --bootstrap)"` to auto-load.
- **Interface** Each function exposes a CLI-style interface (`wt-new --branch foo --base develop`), plus a `wt help` overview alias.

## Shared Behavior & Configuration
- **Worktree root** Default to `${WT_WORKTREE_ROOT:-../wt}` ensuring siblings to repo root; create on demand.
- **Base branch** Default to `${WT_DEFAULT_BASE:-main}`; fetch remote on first use.
- **Branch naming** Fallback template `feature/<YYYYmmdd>-<rand>` where `<rand>` is six lowercase alphanumerics from `/dev/urandom`; override via `--branch` or `WT_BRANCH_PREFIX`.
- **Logging** Uniform `wt_log info|warn|error` helpers with colorized output (configurable via `WT_NO_COLOR`).
- **Prompts** Respect `WT_ASSUME_YES=1` to skip confirmations; otherwise use `read -rp` with defaults displayed.
- **Safety** Every destructive action checks for dirty state, ongoing rebase/cherry-pick, and highlights next steps if blocked.
- **Dependencies** Require `git`; optionally use `gh`, `jq`, `code`, gating features gracefully if missing.

## Function Catalog
- **`wt-new`**
  - Automation: fetch default base, create random branch `feature/<stamp>-<rand>` if `--branch` omitted, add worktree at `${WT_WORKTREE_ROOT}/${branch//\//_}`.
  - Actions: `git worktree add --track -b <branch>` from base commit (default `origin/<base>` if remote exists).
  - Options: `--branch`, `--base`, `--dir`, `--no-track`, `--prefix`, `--checkout-only` (skip worktree creation when only branch desired).
  - Post steps: optional `--enter` to call `wt-enter` automatically; display next commands.

- **`wt-enter`**
  - Automation: resolves branch → worktree path; if missing and `--auto-create` set, delegates to `wt-new`.
  - Opens subshell with `cd` into worktree; exports `WT_ACTIVE_BRANCH` for downstream functions.
  - Options: `--command <cmd>` to run command then exit, `--open` to launch `$WT_EDITOR` (fallback `code .` if available).

- **`wt-status`**
  - Automation: summarize state (HEAD, base, staged changes, pending rebase) using `git status --short` and `git branch --show-current`.
  - Called implicitly by `wt-sync`, `wt-pr`, `wt-clean` prechecks.

- **`wt-sync`**
  - Automation: fetches remote base & branch, stashes uncommitted changes if needed, rebases onto updated base by default.
  - Options: `--merge` to merge instead, `--no-stash`, `--remote origin` override, `--base` override.
  - Post: optional `--run-tests` executes configured command (`WT_SYNC_TEST_CMD`).

- **`wt-commit`**
  - Automation: stages tracked modifications if `--all` (default) and generates commit message stub `feat: <branch slug> <summary>` prefilled in `$WT_COMMIT_TEMPLATE` temp file.
  - Options: `--message`, `--no-all`, `--signoff`, `--amend`, `--preview` (show diff), `--skip-hooks`.
  - Guard: refuses to commit if nothing staged; hints `wt-sync` on conflicts.

- **`wt-pr`**
  - Automation: ensures branch pushed to remote (with `--set-upstream` default), assembles PR title from latest commit or branch slug, auto-detects base via config.
  - Uses `gh pr create` when available; fallback prints `git push` + URL to create PR manually.
  - Options: `--title`, `--body`, `--draft`, `--assignee`, `--reviewer`, `--label`, `--no-open` (skip opening browser), `--notes` to append custom checklist.

- **`wt-clean`**
  - Automation: verifies branch merged (using `git branch --merged`), prompts before deletion, removes worktree directory, deletes branch locally; remote deletion with `--remote`.
  - Options: `--force` bypass safety, `--keep-branch`, `--prune` (run `git worktree prune` and `git remote prune`).
  - Handles dirty trees by offering `--archive` to `git bundle` or `git branch wt/backup` before removal.

- **`wt-help`**
  - Prints concise usage overview and environment variables.
  - Note: also each function will have their own more detailed explanatory docstrings

## Workflow Automation
- **Fresh feature**: `wt-new --enter` → `wt-sync` (if collaborating) → edits → `wt-commit` → `wt-pr --export-diff`.
- **PR wrap-up**: `wt-pr` with defaults, once merged `wt-clean --remote` returns repo to clean state.

## Implementation Notes
- **Random ID helper**: `wt__rand()` uses `LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c6`; fallback to `$RANDOM` if `/dev/urandom` unavailable.
- **Branch slugging**: derive slug from branch minus prefix, sanitized for filenames; used in worktree directory naming and commit templates.
- **State detection**: use `git rev-parse --git-dir` to anchor repo root; store metadata in `${GIT_DIR}/worktree-tools/<branch>.json` (optional) for caching base branch choices.
- **Error handling**: `set -euo pipefail` at top-level; each public function wraps body in `wt_try` trap to convert failures into readable messages.
- **Tab completion**: expose `_wt_complete()` for branch/worktree autocompletion and encourage sourcing in shell profiles.

## Follow-up Questions
- **Bootstrap**: Should we provide installer (`wt install`) that modifies shell config automatically, or keep manual instructions? Answer: Manual instructions.
- **Default editor/tests**: Decide global defaults (`code`, `npm test`, etc.) or rely on environment variables only. Answer: No tests are needed.
- **PR metadata**: Should `wt-pr` auto-generate PR body templates based on repo-specific `.github/pull_request_template.md`? Answer: Not needed at this time.
