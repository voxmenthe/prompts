**Quick plan:** you want `claude` to see a *different* set of env vars (base URL + token) but you don’t want those vars to leak into your shell or end up persisted by the CLI. The clean way is to (1) pass env vars only to the `claude` process (no `export`), (2) optionally run it in a subshell, (3) quarantine `claude`’s config/cache so nothing gets written to your real `~`, and (4) if you’re on macOS or teams, prefer an on‑demand key helper rather than a stored key. Below are pragmatic patterns—pick the one that matches your tolerance for isolation.

---

## 0) Two notes up front

* **Correct vars for Anthropic‑compatible backends.** `claude` recognizes `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` (in addition to `ANTHROPIC_API_KEY`). For Moonshot/Kimi, use the base URL `https://api.moonshot.ai/anthropic` (note `.ai`, not `.cn`). ([Reddit][1])
* **Why your function leaks:** using `export` in a shell function mutates the *parent* shell; those variables remain set after the function returns. Avoid `export` and use command‑scoped assignments.

---

## 1) Minimal, non‑leaky function (recommended starting point)

This sets vars **only for the `claude` process** and unsets any conflicting global Anthropic key for that one call:

```bash
# Put KIMI_API_KEY somewhere safe; do NOT export it globally
# e.g., in your shell RC: readonly KIMI_API_KEY=...   (no export)

kimi() {
  env -u ANTHROPIC_API_KEY \
    ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic" \
    ANTHROPIC_AUTH_TOKEN="${KIMI_API_KEY:?KIMI_API_KEY is unset}" \
    claude "$@"
}
```

# an original deepseek version for use w/ claude code
export DEEPSEEK_API_KEY="sk-YOUR-DEEPSEEK-API-KEY"

deepseek() {
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_AUTH_TOKEN=$DEEPSEEK_API_KEY
    export API_TIMEOUT_MS=600000
    export ANTHROPIC_MODEL=deepseek-chat
    export ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat
    export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
    claude $1
}


* `env VAR=... command` sets variables only for `command`.
* `-u ANTHROPIC_API_KEY` ensures your normal Anthropic key (if any) can’t be picked up by mistake.
  `claude` understands `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`, so this works against compatible backends. ([Reddit][1], [DeepSeek API Docs][2])

---

## 2) Same idea, but also quarantine `claude`’s files (no config/key leakage)

Some builds of Claude Code write to `~/.claude{,*.json}` and haven’t fully adopted XDG dirs. You can give it a throwaway config dir for each run. Parentheses `(...)` run a **subshell**, so nothing survives after exit:

```bash
kimi() (
  set -euo pipefail
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  # Quarantine Claude Code's own state:
  #  - CLAUDE_CONFIG_DIR is used by several code paths
  #  - XDG_CONFIG_HOME helps for tools that respect XDG
  env -u ANTHROPIC_API_KEY \
    CLAUDE_CONFIG_DIR="$tmp" \
    XDG_CONFIG_HOME="$tmp/.config" \
    ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic" \
    ANTHROPIC_AUTH_TOKEN="${KIMI_API_KEY:?}" \
    claude "$@"
)
```

This prevents permanent files like `~/.claude.json` or caches from being created under your real home for that session. (There are open issues about where Claude Code writes config; this keeps it out of your real `$HOME` regardless.) ([GitHub][3])

> If you still see writes into `~`, replace the `env ...` line with an **ephemeral HOME** to force *all* dotfiles into a temp area (at the cost of losing access to your real git config, ssh keys, etc. in that session):
>
> ```bash
> env -i HOME="$tmp" PATH="$PATH" SHELL="$SHELL" TERM="${TERM:-xterm-256color}" \
>   ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic" \
>   ANTHROPIC_AUTH_TOKEN="${KIMI_API_KEY:?}" \
>   claude "$@"
> ```

---

## 3) Avoid storing any key at all: use `apiKeyHelper` (on‑demand secret fetch)

Claude Code supports an **API key helper**: a shell script it runs when it needs a key. You can point it at a password manager or a command that prints the token. This avoids putting tokens in env or on disk.

1. Create a helper script:

```bash
mkdir -p ~/.claude
cat > ~/.claude/kimi_key.sh <<'EOF'
#!/usr/bin/env bash
# Example: pull from env or your secret manager
# printf %s "${KIMI_API_KEY:?}"              # from env, or:
# op read "op://Private/Kimi API Key/credential"       # 1Password
# aws secretsmanager get-secret-value --secret-id kimi_api_key --query SecretString --output text
printf %s "${KIMI_API_KEY:?}"
EOF
chmod +x ~/.claude/kimi_key.sh
```

2. In your user or project settings (e.g., `~/.claude/settings.json`), add:

```json
{
  "apiKeyHelper": "~/.claude/kimi_key.sh",
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.moonshot.ai/anthropic"
  }
}
```

3. (Optional) Tune how often Claude re‑calls your helper by setting
   `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` (e.g., `60000` for 60s) before launching:

```bash
export CLAUDE_CODE_API_KEY_HELPER_TTL_MS=60000
```

This pattern keeps the key out of persistent config and out of your interactive shell. (Anthropic documents `apiKeyHelper` and its TTL var in their IAM docs and release notes.) ([Anthropic][4], [ClaudeLog][5])

You can combine this with §2’s quarantine if you want maximal isolation.

---

## 4) Script wrapper (another simple option)

If you prefer not to define a shell function at all, a **wrapper script** naturally runs in a subprocess and can safely `export` inside itself without touching your parent shell:

```bash
# ~/bin/kimi (chmod +x ~/bin/kimi and ensure ~/bin is on PATH)
#!/usr/bin/env bash
set -euo pipefail
export ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic"
export ANTHROPIC_AUTH_TOKEN="${KIMI_API_KEY:?}"
exec claude "$@"
```

Because the script executes in its own process, those exports don’t “contaminate” your interactive environment after it exits.

---

## 5) Extra hardening and gotchas

* **Forward all arguments.** Use `"$@"`, not `$1`, so multi‑arg invocations pass through correctly.
* **Keep your default Anthropic key out of scope** to avoid accidental use: the `env -u ANTHROPIC_API_KEY` in §1/§2 handles this. ([Reddit][1])
* **If you see persistence on macOS:** Claude Code may store credentials in the macOS Keychain when you *log in*. Using an `apiKeyHelper` (and avoiding `/login`) keeps secrets ephemeral; quarantine config if you want to be extra sure nothing else is written. ([Anthropic][4])
* **Correct Moonshot base URL:** use `https://api.moonshot.ai/anthropic` (some guides show `.cn`, which won’t work for many accounts). ([Reddit][1])

---

### TL;DR

* Replace `export ...; claude` with **one‑shot env**:
  `env -u ANTHROPIC_API_KEY ANTHROPIC_BASE_URL=... ANTHROPIC_AUTH_TOKEN=... claude "$@"`.
* If you don’t want Claude Code to *persist* anything, **quarantine its config** (`CLAUDE_CONFIG_DIR`/`XDG_CONFIG_HOME`) or even run with an **ephemeral HOME** for that call.
* For teams/production, prefer **`apiKeyHelper`** so keys are supplied on demand from a secret store and never written to disk. ([Anthropic][4])

If you want, tell me your OS/shell and where you store the Kimi key (e.g., 1Password, `pass`, AWS/GCP secrets), and I’ll drop in a ready‑made `apiKeyHelper` script plus the exact `settings.json`/function variant for your setup.

[1]: https://www.reddit.com/r/ClaudeAI/comments/1lp8g4w/how_to_find_claude_code_environment_variables_and/?utm_source=chatgpt.com "How to find Claude Code environment variables and options"
[2]: https://api-docs.deepseek.com/guides/anthropic_api?utm_source=chatgpt.com "Anthropic API"
[3]: https://github.com/anthropics/claude-code/issues/1455 "Claude Code does not respect the XDG Base Directory specification · Issue #1455 · anthropics/claude-code · GitHub"
[4]: https://docs.anthropic.com/en/docs/claude-code/iam "Identity and Access Management - Anthropic"
[5]: https://www.claudelog.com/faqs/claude-code-release-notes/?utm_source=chatgpt.com "claude-code-release-notes"
