# OpenAI Platform Static Chunk Mirroring – Progress Log

## Context
- Goal: replace Playwright scraping of `platform.openai.com/docs` guides with an offline parser that extracts React render trees from shipped static JS chunks and renders them to Markdown.
- Starting point: `mirror_platform_guides` relied on Playwright + cloudscraper. Target plan (in `PLANS/openai_platform_static_chunk_plan.md`) called for parsing Vite bundles, evaluating render functions, and reusing HTML→Markdown pipeline.

## Work Completed
### 1. Script Refactor & Dependency Changes
- Removed Playwright flow from `mirror_platform_guides`; added static chunk pipeline using QuickJS.
- Introduced helper utilities in `mirror_openai_api_docs.py`:
  - Entry module discovery, docs chunk resolution, wrapper/render extraction.
  - QuickJS evaluation scaffold (`evaluate_wrapper_to_tree`) plus HTML/Markdown conversion via existing BeautifulSoup/markdownify stack.
  - Metadata capture (chunk name, wrapper, render function) for manifest provenance.
- Swapped project dependency from `playwright` to `quickjs` (updated `pyproject.toml`).

### 2. Initial Evaluation Failures & Diagnostics
- First dry run produced `SyntaxError` / `ReferenceError` on every guide due to missing bindings.
- Instrumentation confirmed render functions reference constants (e.g., `Ms`, `ts`) defined outside the component; initial constant extractor failed to capture them (`ModuleNotFound` or `None`).

### 3. Constant Extraction Fixes
- Enhanced `extract_constant_value`:
  - Added brace-aware parsing for multi-binding declarations (`const qn={},Ln={},Yi={},…`).
  - Added literal map builder for chained declarations.
- After fix, `migrate-to-responses` rendered correctly via QuickJS (validated Markdown snippet).

### 4. Broader Dry Run & Remaining Errors
- Subsequent dry run succeeded for some guides but still failed for many with missing symbols (`Go`, `Fa`, `$l`, etc.) indicating cross-chunk dependencies and re-exported analytics/helper modules.

### 5. Dependency Discovery Improvements
- Added import path discovery and caching of fetched modules.
- Built literal/function maps per chunk and merged into evaluation environment.
- Added heuristics to collect property identifier candidates (filters uppercase/underscore tokens only) to catch CSS module maps (e.g., `Fa`).
- Guides like `conversation-state` and `deep-research` now produce Markdown when manually evaluated.

### 6. Current Results
- `uv run … --subset guides` still fails for the majority of guides. Remaining blockers:
  - Aliased imports (e.g., `import {aX as Go}`) require chasing the alias graph across multiple chunks to locate definitions.
  - Some helper names originate from global analytics utilities (`$l`, `hi`, etc.) that are not yet stubbed or resolved.
  - Additional SyntaxErrors arise when prose tokens still slip through collection heuristics (e.g., quotes leading to empty identifiers) in certain guides.

## What Didn’t Work / Lessons
- Treating every dotted token as an identifier generated invalid stubs (`const behavior={}` → syntax errors). Needed tighter filters (now only uppercase/underscore names).
- Assuming constants defined in the same chunk was insufficient; Vite splits shared data/utility modules, so we must traverse imports recursively.
- Returning empty string for unresolved constants triggered parse errors even before evaluation; must supply at least `{}` or follow actual definition.

## Recommended Next Steps
1. **Alias Graph Traversal**
   - For each import (`import {aX as Go}`), track `aX` back through dependency module literal/function maps, recursively if necessary, until the originating declaration is found.
   - Cache resolution results to avoid repeated fetches per guide.
2. **Fallback Stubs for Analytics/Telemetry Helpers**
   - Identify unresolved names after traversal; provide no-op JS definitions (e.g., `const $l = () => {};`) with clear logging to keep evaluation safe without polluting Markdown.
3. **Expand Property Identifier Filters**
   - Continue refining heuristics to avoid capturing plain prose tokens while ensuring CSS module maps (`StandaloneLi`, etc.) are included.
4. **Add Unit Tests**
   - Create targeted tests covering evaluation of representative guides (simple vs. dependency-heavy) to prevent regressions while iterating on alias resolution.
5. **Re-run Full Dry Run & Compare Output**
   - Once alias resolution is in place, re-run guides subset, verify Markdown fidelity (spot-check ChatKit docs, prompt-engineering, retrieval) and update manifest generation accordingly.

## Status Summary
- ✅ Playwright dependency removed; QuickJS pipeline operational for low-dependency guides.
- ⚠️ High: Must implement robust module graph resolution to support majority of guides.
- ⏭️ Next primary task: alias/dependency traversal (Step 1). Steps 2–5 follow naturally once alias lookups succeed.

