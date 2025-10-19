# uv — Expanded cheatsheet (2025)

> All commands are **uv** unless noted. Where you see `--python 3.X`, that flag works across many commands (init, run, venv, etc.). `uvx` is **exactly** `uv tool run`.

## 1) Creating projects 🧱

* `uv init` — create an **application** project in the current dir.
  Examples:

  * New app in a folder: `uv init myapp`
  * Explicit app mode: `uv init --app myapp`
  * **Packaged app** (adds build backend + `src/`): `uv init --package cli_app`
  * **Library** (implies `--package`, adds `py.typed`): `uv init --lib cool_lib`
  * **Minimal** (just `pyproject.toml`): `uv init --bare example`
  * Pick a backend (e.g., Rust via maturin): `uv init --build-backend maturin ext_proj`
  * Pin Python file alongside: `--python-pin` (writes `.python-version`)
    Structure and semantics are from the official “Creating projects” concept page.

**What uv generates**

* App: `pyproject.toml`, `main.py`, `README.md`, `.python-version`; *no* build system (not installed into env).
* Packaged app / lib: `src/<pkg>/__init__.py` plus `[build-system]` using `uv_build` by default; lib adds `py.typed`.

**Handy tips**

* Run your app immediately: `uv run main.py` or the console script it defines.
* Use `--package` for anything you intend to **install** or publish later; keep `--app` for one-off runnable projects.

---

## 2) Working with scripts 📜

* Initialize a script with inline deps (PEP 723):
  `uv init --script myscript.py --python 3.12`
* Add/update inline deps on a script:
  `uv add --script myscript.py 'requests<3' rich`
  (uv inserts a TOML “script” block at the top of the file)
* Run scripts (no project needed):

  * With deps requested on the fly: `uv run --with rich progress_demo.py`
  * Respect inline metadata automatically: `uv run myscript.py`
  * Choose interpreter ad-hoc: `uv run --python 3.10 myscript.py`
* Shebang for direct execution: first line `#!/usr/bin/env -S uv run --script`, then `chmod +x myscript`.
* **Lock** a script’s deps to `example.py.lock`: `uv lock --script example.py`
* Use alternate index for scripts: `uv add --index https://example.com/simple --script example.py httpx`

**Notes**

* When a script uses inline metadata, project deps are ignored; `--no-project` isn’t needed.

---

## 3) Project dependencies 🌱

Common:

* Add one or many: `uv add httpx 'pydantic>=2'`
* Remove: `uv remove httpx`
* Dev deps: `uv add --dev pytest` (creates/updates `[dependency-groups].dev`)
* Other groups: `uv add --group lint ruff`
* Optional extras: `uv add httpx --optional network` (=> `[project.optional-dependencies]`)
* From a file: `uv add -r requirements.txt`
* View tree: `uv tree` (try `--outdated` to see newer versions)
* Lock / upgrade:

  * `uv lock --check` (validate up-to-date)
  * `uv lock` (refresh lockfile)
  * `uv lock --upgrade` (all)
  * `uv lock --upgrade-package httpx` (one package)
* Sync explicitly (CI/editor bootstrap): `uv sync` (exact by default; remove extraneous packages unless `--inexact`)
* Export lock as `requirements.txt` for other tooling:
  `uv export --format requirements-txt > requirements.txt`

**Gotchas & power moves**

* uv resolves and **locks** automatically before `uv run`; use `--locked` (error if outdated) or `--frozen` (don’t update) for reproducibility runs.
* Default dev group is included; exclude with `--no-dev`, or install only dev with `--only-dev`. Groups have matching toggles (`--group`, `--only-group`, etc.).
* Importing from Git/URL/local path is first-class (`tool.uv.sources`) and can be switched later.

---

## 4) Running inside projects ▶️

* Run with the project env (kept in `.venv`): `uv run python -c "import mypkg; print(mypkg.__version__)"`
* Without touching the project env: `uv run --no-project some_script.py`
* Editors often need a pre-synced env: `uv sync` (or just run once with `uv run` to auto-sync+lock).

---

## 5) Python management 🐍

* List installed/available: `uv python list` (`--all-versions` shows everything)
* Install:

  * exact: `uv python install 3.12.3`
  * latest patch of a minor: `uv python install 3.12`
  * multiple: `uv python install 3.9 3.10 3.11`
  * free-threaded 3.13: request as `3.13t` or `3.13+freethreaded`
  * install CLI shims (experimental): `uv python install --default` (adds `python`, `python3`)
* Upgrade uv-managed Pythons (preview): `uv python upgrade` or `uv python upgrade 3.12`
* Pin per-project: `uv python pin 3.11` → writes `.python-version`

  * Global pin: `uv python pin --global 3.12`
* Use a specific interpreter on any command: `--python 3.10`
  (uv will auto-download if needed, unless disabled)

---

## 6) Tools ⚒️  (`uvx` = `uv tool run`)

**Run once (ephemeral env):**

* `uvx ruff@latest check`
* `uvx --from 'ruff==0.3.0' ruff check`
* `uvx pycowsay "hello from uv"`
* Try a demo: `uvx --python 3.12 textual-demo`

**Install globally (isolated venv per tool):**

* `uv tool install ruff@latest`
* List / upgrade / uninstall:

  * `uv tool list`
  * `uv tool upgrade --all` or `uv tool upgrade ruff`
  * `uv tool uninstall ruff`
* Current project as a tool (dev flow): `uv tool install --editable .`
* Ensure executables are on PATH: `uv tool update-shell` (prints/fixes shell config); show dirs: `uv tool dir` / `uv tool dir --bin`
* Include extra deps when running or installing: `--with`, `--with-requirements`
  (great for plugins, e.g., `uvx ruff --with 'ruff-lsp'`)

---

## 7) Project lifecycle 🔄

* Show version: `uv version` (`--short` for just the number)
* **Bump** version with SemVer & pre-release helpers:
  `uv version --bump patch|minor|major|alpha|beta|rc|post|dev|stable`
  (e.g., `uv version --bump minor`)
* Build dists: `uv build` → wheels and sdist in `dist/`
* Publish to index (e.g., PyPI): `uv publish` (supports token auth)

---

## 8) For old timers 🧓 (venv & pip compatibility)

* Create a venv (optionally at a path):
  `uv venv` or `uv venv .venv` or `uv venv --python 3.11 env311`
* **pip-compatible interface** (works inside the active venv or with `--system`):

  * `uv pip install -r requirements.txt`
  * `uv pip freeze` / `uv pip list` / `uv pip show` / `uv pip check`
  * Seed pip into a uv venv: `uv venv --seed` (or later `uv pip install pip`)

---

## 9) Meta / utility commands 🧭

* Help: `uv --help`, `uv init --help`, or the long form `uv help init`
* Update uv (standalone installer): `uv self update`; show version: `uv self version`
* Cache helpers: `uv cache dir | clean | prune`

---

## 10) Formatting ✨

* `uv format` — formats Python files in your project **using Ruff’s formatter**.

  * Check only: `uv format --check`
  * Show diff: `uv format --diff`
  * Pass flags through to Ruff after `--`, e.g. `uv format -- --line-length 100`

---

## Practical mini-workflows

**A. New packaged CLI in Python 3.12, with lint/format + tests**

```bash
uv init --package hello_cli --python-pin
cd hello_cli
uv add --group lint ruff
uv add --dev pytest
uv version --bump minor
uv run ruff check && uv run pytest -q
uv build
```

(Init + groups + bump + local check + build.)

**B. One-file script with pinned deps + lock for reproducibility**

```bash
uv init --script fetch.py --python 3.12
uv add --script fetch.py 'httpx<1'
uv lock --script fetch.py
uv run fetch.py
```

(Inline deps, lock beside the script.)

**C. Try and then keep a tool**

```bash
uvx ruff@latest check
uv tool install ruff@latest
uv tool list
uv tool upgrade --all
```

(Ephemeral run, then install globally.)

**D. Export for infra that needs `requirements.txt`**

```bash
uv export --format requirements-txt > requirements.txt
```

(Generate a compatible file from `uv.lock`.)

**E. Use a specific interpreter (including free-threaded 3.13)**

```bash
uv run --python 3.10 pytest -q
uv run --python 3.13t python -c "import sys; print(sys._is_gil_enabled())"
```

(Request exact interpreter per command.)

---

## Reference quick-glance (mirrors the image boxes)

**Creating projects**

* `uv init [--app|--package|--lib|--build-backend <name>|--bare] [NAME]`
* Common option: `--python 3.X` (works across uv commands)
* Project kinds: app (runnable), packaged app (installable), library (typed), extension backends (maturin/scikit-build-core).

**Working with scripts**

* `uv init --script file.py [--python 3.X]`
* `uv add --script file.py ...` ; `uv run [--with dep] file.py`
* Lock scripts: `uv lock --script file.py` ; shebang `uv run --script`.

**Project dependencies**

* `uv add ...` / `uv remove ...` / `uv tree` / `uv lock --upgrade`
* Dev/groups/optionals supported; import from requirements with `-r`.

**Project lifecycle**

* `uv version` (read) and `uv version --bump <major|minor|patch|alpha|beta|rc|post|dev|stable>`
* `uv build` / `uv publish`.

**For old timers**

* `uv venv [PATH] [--python 3.X]`
* `uv pip ...` (`install`, `freeze`, `list`, `check`, etc.).

**Python management**

* `uv python list|install|upgrade|find|pin|uninstall|dir`
* Free-threaded request: `3.13t`; default executables via `--default`.
* Use any command with `--python 3.X`.

**Tools**

* `uvx <cmd>[@<ver>]`  (= `uv tool run`)
* Install/upgrade/list/uninstall/update-shell/dir; editable installs: `uv tool install --editable .`

**Meta**

* `uv help <cmd>` ; `uv self version` ; `uv self update` ; `uv cache clean|prune|dir`.

**Misc**

* `uv format [--check|--diff] [-- <ruff-args>...]`.

---

### Footnotes / alignment with the original image

* The image shows `uvx ty` and `uvx textual-demo --from textual`; those are standard tool-run examples (any CLI on PyPI works via `uvx`, and `--from` lets you pick the package providing the command).
* The image’s “you can list 2 or more dependencies” and “remove unnecessary transitive deps” are reflected by `uv add` supporting multiple args and `uv sync` being exact by default (removes extraneous).

If you want this as a printable one-pager, say the word and I’ll output a condensed A4/Letter version.

[1]: https://docs.astral.sh/uv/concepts/python-versions/ "Python versions | uv"
[2]: https://docs.astral.sh/uv/concepts/projects/init/ "Creating projects | uv"
[3]: https://docs.astral.sh/uv/guides/scripts/ "Running scripts | uv"
[4]: https://docs.astral.sh/uv/concepts/projects/dependencies/ "Managing dependencies | uv"
[5]: https://docs.astral.sh/uv/concepts/projects/sync/ "Locking and syncing | uv"
[6]: https://docs.astral.sh/uv/concepts/projects/run/?utm_source=chatgpt.com "Running commands in projects - uv - Astral Docs"
[7]: https://docs.astral.sh/uv/guides/tools/?utm_source=chatgpt.com "Using tools | uv - Astral Docs"
[8]: https://docs.astral.sh/uv/reference/cli/ "Commands | uv"
[9]: https://docs.astral.sh/uv/guides/projects/?utm_source=chatgpt.com "Working on projects | uv - Astral Docs"
[10]: https://docs.astral.sh/uv/pip/environments/?utm_source=chatgpt.com "Using Python environments - uv - Astral Docs"
[11]: https://docs.astral.sh/uv/getting-started/help/?utm_source=chatgpt.com "Getting help | uv - Astral Docs"
