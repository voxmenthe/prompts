# Anthropic docs mirroring findings (2025-12-04)

## What we learned
- Claude publishes a `llms.txt` index that lists every `.md` doc URL across the site (API reference, guides, agent SDK, prompt library, etc.); it already contains 200+ English pages and is designed for LLM consumption. citeturn1view0
- The help center now points developers directly to the Claude-branded docs at `https://docs.claude.com/en/api/overview`, replacing the older console URL; this confirms the primary surface we should mirror. citeturn0search1turn0search0
- Release notes live under `/en/release-notes` and carry API-affecting changes (headers, SDK status, feature launches); the current mirror script does not pull this section. citeturn0search9turn0search4
- Site structure is broader than `/en/api`: `/en/docs` holds core how-to material (models, pricing, agent SDK), `/en/resources` hosts the prompt library, and localized variants exist for multiple languages. These paths appear in the sitemap but are skipped when we constrain to the `api` section.

## Gaps in the current script
- Defaults to `section=api`, so it fetches only ~60 pages while omitting ~100+ `/en/docs` pages, release notes, resources, and non-English locales that are in the sitemap.
- Uses the sitemap for discovery; Anthropic’s `llms.txt` is now a richer, LLM-targeted index that includes API-adjacent content the sitemap misses or organizes differently.
- No integrity checks (hash/size) or “lastmod” awareness, so we redownload blindly and can’t detect stale vs. updated pages.
- Manifest records URLs and relative paths but not content digests, language/section provenance, or a crawl source, which makes incremental updates and diffing harder.

## Recommendations
1) **Switch discovery to `llms.txt`, fallback to sitemap**: load `llms.txt` first; if unreachable, fall back to the sitemap. Deduplicate and keep source metadata (llms vs. sitemap) per entry.
2) **Expand coverage knobs**: default include `api`, `docs`, `release-notes`, `resources/prompt-library`, with optional `--all-languages`. Keep a deny/allow list to limit blast radius when needed.
3) **Add freshness + integrity**: store `sha256`, byte length, and fetch timestamp in the manifest; use sitemap `<lastmod>` (when available) or `ETag/Last-Modified` headers to skip unchanged files unless `--force`.
4) **Better observability**: per-section/page counts, failures, and a summary of skipped vs. updated files; log user-agent being sent to avoid 403 confusion.
5) **Safety rails**: fail fast on duplicate relative paths, and surface a post-run diff of added/removed pages between runs to catch doc churn.

If you want, I can implement the above as a follow-up, starting with the `llms.txt` ingestion and manifest enrichment. 

## Reusable tactics from other data scripts
- `download_gsap_docs.py`: tries multiple base domains/prefixes per path and uses a local outline fallback plus an ASCII tree/JSON run summary. Applicability: we likely don’t need multi-domain retries for Anthropic, but adopting the JSON summary + tree and a path-safety guard (refusing writes outside output root) would harden the mirror.
- `download_livekit_docs.py`: hashes normalized content to report created/modified/unchanged; strips volatile “rendered at …” footers to cut churn; optional pruning of 404s; enforces allowed hosts. Applicability: add change classification via hashes and consider pruning deleted pages; keep allowed-hosts check for safety even though Anthropic uses a single host.
- `mirror_ai_sdk_v6_docs.py`: sanitizes HTML and rewrites internal links when only HTML is available. Applicability: not needed while Anthropic serves Markdown, but useful fallback if markdown endpoints disappear.
- `create_uv_docs_mirror.py`: moves prior snapshots aside and computes added/updated/removed diffs across runs; keeps both raw HTML and extracted Markdown with a manifest. Applicability: we already emit a diff vs. manifest; could add optional HTML snapshots and a “previous snapshot” roll-forward if we ever need to capture rendered pages.
- `openai_docs_new_enhanced.py`: uses caching and a Cloudflare-safe UA with a jina.ai fallback plus retryable status codes. Applicability: we already set a UA; adding lightweight caching or a second-chance proxy endpoint could reduce flakiness if Anthropic tightens rate limits.
- `create_dspy_docs_mirror.py`: applies redirect maps and converts notebooks/API refs. Applicability: not relevant unless Anthropic publishes notebooks or redirect metadata; no action now.
