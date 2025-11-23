# UV Documentation Mirror

Use `src/DATA_SCRIPTS/create_uv_docs_mirror.py` to capture both the Markdown
sources from the `astral-sh/uv` repository and the rendered documentation at
`https://docs.astral.sh/uv/`. The script keeps the repo docs directly in the
mirror root (matching `uv/docs/`) while rendered pages live under
`rendered_site/`. By default only the simplified Markdown extracted from the
hosted site is stored; add `--keep-html` to retain raw HTML snapshots too.

## Quick start

```bash
# From repo root
uv run python src/DATA_SCRIPTS/create_uv_docs_mirror.py
```

Defaults:

- Clones (if needed) the uv repository into `~/repos/OTHER_PEOPLES_REPOS/uv`.
- Writes outputs to `SPECIALIZED_CONTEXT_AND_DOCS/UV_DOCS_MIRROR`.
- Copies `uv/docs/` into the mirror root while skipping `.overrides`, `js`,
  `stylesheets`, and `site`. Each run rebuilds from scratch to avoid stale files.
- Downloads every sitemap entry under `https://docs.astral.sh/uv`.

Key CLI options:

| Flag | Purpose |
| --- | --- |
| `--skip-repo` | Only mirror the hosted site (skip copying repo Markdown). |
| `--skip-site` | Only copy the repo Markdown (skip site scrape). |
| `--repo-skip <dir>` | Override the set of skipped directories under `uv/docs/` (repeat to list all). |
| `--force-site` | Redownload site pages even if they are already present. |
| `--keep-html` | Preserve the raw HTML for each page inside `rendered_site/html/`. |
| `--site-base` | Override the rendered site root (defaults to `https://docs.astral.sh/uv`). |
| `--sitemap` | Point to an alternate sitemap URL. |
| `--out` | Change the destination directory. |

The generated directory contains:

- Mirror root: Copy of `uv/docs/` (minus the skipped directories) so files like
  `index.md`, `concepts/`, and `guides/` sit directly inside the target folder.
- (Optional) `rendered_site/html/`: Raw HTML snapshots (requires `--keep-html`).
- `rendered_site/markdown/`: Simplified Markdown extracted from each page's `<article>` (tabs, callouts, etc. are expanded).
- `manifest.json`: Machine-readable summary (source URLs, relative file paths, counts, skip list, per-run change summary).
- `README.md`: Quick overview with run metadata.

Every invocation records which files were added/updated/removed relative to the
previous run (see the console diff summary and `manifest.json`).


options:
  -h, --help            show this help message and exit
  --repo REPO           Path to local uv repository clone.
  --out OUT             Output directory (default: /Volumes/cdrive/repos/prompts/SPECIALIZE
                        D_CONTEXT_AND_DOCS/UV_DOCS_MIRROR)
  --clean               Remove output directory before mirroring.
  --skip-repo           Skip copying repo Markdown docs.
  --skip-site           Skip sitemap download/rendered site capture.
  --force-site          Redownload site pages even if the files already exist.
  --max-workers MAX_WORKERS
                        Parallel downloads for site mirroring.
  --site-base SITE_BASE
                        Base URL for the rendered docs (default: https://docs.astral.sh/uv)
  --sitemap SITEMAP     Override sitemap URL (default: <site-base>/sitemap.xml)
