# Plan: Mirror OpenAI Platform Guides via Static JS Chunk Parsing

## Context & Lessons Learned
- Existing script now mirrors the openai-python SDK docs successfully, but the attempt to scrape platform.openai.com guides relied on Playwright-driven page rendering.
- That approach hits Cloudflare-managed hydration, adds heavy dependencies (`playwright`, Chromium downloads), and still times out: pages never expose the content div before the watchdog fires (seen with >70 timeouts during dry-run).
- We confirmed each guide’s React content already ships inside the `/static/*.js` bundles (e.g. `awftPAIqem.js` contains the full “Migrate to the Responses API” prose as `function fa`). Therefore, executing the JS bundle or parsing its React element tree offline will give us canonical text without fighting Cloudflare.

## Objectives
1. Produce full-fidelity Markdown for targeted platform guides (starting with the `guides` section, expandable later) by decoding the shipped JS bundles.
2. Remove the unused Playwright pipeline and related CLI options/dependencies.
3. Keep blast radius small: reuse existing retry/backoff primitives, keep manifest structure stable, and surface new provenance metadata (chunk names, hashes).

## Proposed Architecture
1. **Discover target routes**
   - Keep using the platform sitemap to enumerate guide URLs; continue to use `cloudscraper` (or a hardened `fetch_bytes`) because the root HTML still sits behind Cloudflare’s JavaScript challenge.
2. **Map routes → chunk modules**
   - Fetch the main Vite entry (`/static/index-*.js`) referenced in the root HTML. That file holds route definitions of the form `path:"/docs/guides/migrate-to-responses" ... children:s(Dj,{})`.
   - Parse those route records to capture the React component symbol (`Dj`) and the module path listed in the surrounding `__vite__mapDeps` array so we know which chunk exports that component.
3. **Extract React tree from chunk**
   - Download the chunk file (e.g. `static/awftPAIqem.js`). Each target component exports a wrapper (`function Dj(n={}){ ... return fa(n) }`) and a concrete render function (`function fa(n){ const t={...}; return e.jsxs(...); }`).
   - Build a small JS parser/interpreter to evaluate that render function safely:
     * Provide stubs for `e.jsx` / `e.jsxs` to emit a serializable tree.
     * Provide a stub for `r()` (the MDX component registry) returning an empty object so default HTML tags are used.
     * Execute the minimal function in `quickjs` (Python package) or `py_mini_racer` to capture a nested JSON structure representing the DOM-like tree.
4. **Convert tree to Markdown**
   - Walk the captured tree and render HTML (or directly Markdown). We can:
     * Reuse the existing `html_to_markdown` helper by first emitting HTML tags (strong, code, ul, etc.), then running markdownify.
     * Normalize code blocks by inspecting `code` nodes and applying language hints carried in the chunk (`language-python`, etc.).
   - Preserve tables, lists, headings, inline formatting, and links.
5. **Write outputs & manifest entries**
   - Store each guide at `SPECIALIZED_CONTEXT_AND_DOCS/OPENAI_API_DOCS_MIRROR/openai_platform_docs/<slug>/index.md`.
   - Extend the manifest with chunk metadata: source URL, module name, SHA-256.

## Implementation Steps
1. **Dependency & CLI cleanup**
   - Remove Playwright-specific constants, options (`--platform-render-timeout`, etc.), and execution branches.
   - Drop `playwright` from `pyproject.toml` (and `uv` lock if present); ensure Chromium caches aren’t referenced.
2. **Routing metadata extractor**
   - Add helper to fetch the entry HTML for `/docs` (or sitemap first route) and parse the `<script src="/static/index-*.js">` tag (use regex; no DOM parser needed).
   - Parse the index chunk: locate route objects for configured sections (`guides` by default) and collect `(slug, componentName, chunkPath)` tuples.
3. **Chunk evaluator**
   - Introduce a utility that:
     * Downloads a chunk, isolates the render function (`function <name>(n={})` or arrow equivalent).
     * Builds a QuickJS runtime with stubs so the function returns a JSON-like structure (define `globalThis.e = { jsx, jsxs }`, etc.).
     * Runs the render function and marshals the resulting tree back to Python.
4. **Tree → Markdown renderer**
   - Convert the collected tree into HTML/Markdown while handling:
     * Text nodes (strings)
     * Inline code vs fenced blocks (based on `code` nodes / language attr)
     * Tables (`thead`, `tbody`, `tr` etc.)
     * Lists (`ul`, `ol`, nested lists)
     * Links and emphasis (`a`, `strong`, `em`)
   - Feed the generated HTML through the existing markdown pipeline to stay consistent with prior normalization.
5. **Manifest updates & logging**
   - Record chunk URL, bytes, SHA256, and module name.
   - Emit concise logs summarizing generated/skipped docs.
6. **Validation**
   - Run `uv run python src/DATA_SCRIPTS/mirror_openai_api_docs.py --subset guides --dry-run --include-platform-guides` to ensure plan executes without network rendering.
   - Spot-check `migrate-to-responses/index.md` against `REFERENCE/openai_docpage_example_responses.md`.
7. **Cleanup tasks**
   - Remove the playwright installation artifacts if any were created (document in README or scripts).
   - Update documentation / usage text in the script header to reflect the new pipeline.

## Risks & Mitigations
- **Chunk format changes**: if OpenAI alters the bundler pattern, the parser must fail fast with clear errors; keep code modular for future adjustments.
- **New component types**: add fallback logging for unknown tags to avoid silent data loss.
- **Legal/resource strain**: respect existing rate limits by reusing download retry/backoff and batching requests.

## Next Steps
- Implement the above pipeline, then iterate on formatting fidelity (e.g., ensure copy-button boilerplate is removed, anchors preserved).
