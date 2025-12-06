#!/usr/bin/env python3
"""Mirror the Claude documentation markdown locally using llms.txt + sitemap.

This script expands coverage beyond the prior version by:
- Using the public `llms.txt` index as the primary discovery source (with sitemap
  fallback) to pick up API docs, guides, release notes, and the prompt library.
- Supporting multiple sections and languages.
- Recording integrity/freshness metadata (sha256, length, etag/last-modified, lastmod).
- Skipping unchanged files when remote freshness signals indicate no updates.
- Emitting a richer manifest and a summary diff of added/removed pages.

Defaults mirror English (`en`) content for the sections: api, docs, release-notes,
and resources/prompt-library. Set `--all-languages` or multiple `--language` flags
to widen coverage. The output layout nests files under `<lang>/<section>/...`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BASE_DOMAIN = "https://docs.claude.com"
SITEMAP_URL = f"{BASE_DOMAIN}/sitemap.xml"
LLMS_URL = f"{BASE_DOMAIN}/llms.txt"
DEFAULT_HEADERS = {
    "User-Agent": "ClaudeDocsMirror/2025.12 (+https://docs.claude.com/en/api/overview)",
}
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
DEFAULT_OUTPUT_DIRECTORY = Path("SPECIALIZED_CONTEXT_AND_DOCS/ANTHROPIC_API_DOCS_MIRROR")
DEFAULT_SECTIONS = (
    "api",
    "docs",
    "release-notes",
    "resources/prompt-library",
)


@dataclass(frozen=True)
class DocPage:
    doc_url: str
    markdown_url: str
    language: str
    section: str
    relative_path: Path
    source: str  # "llms" or "sitemap"
    lastmod: str | None


@dataclass(frozen=True)
class PreviousMeta:
    etag: str | None
    last_modified: str | None
    lastmod: str | None
    sha256: str | None
    content_length: int | None
    fetched_at: str | None


def fetch_bytes(url: str, *, headers: Mapping[str, str], timeout: float = 15.0) -> bytes:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as resp:  # nosec: required for this script
        return resp.read()


def fetch_text(url: str, *, headers: Mapping[str, str], timeout: float = 15.0) -> str:
    return fetch_bytes(url, headers=headers, timeout=timeout).decode("utf-8", errors="replace")


def load_sitemap(*, headers: Mapping[str, str], timeout: float) -> ET.Element:
    raw = fetch_bytes(SITEMAP_URL, headers=headers, timeout=timeout)
    return ET.fromstring(raw)


def parse_llms_urls(text: str) -> list[str]:
    # llms.txt is Markdown-ish; collect all https links.
    return re.findall(r"https://[^\\s)]+", text)


def normalize_markdown_path(path: str) -> str:
    return path if path.endswith(".md") else f"{path}.md"


def allowed_section(path_without_lang: str, allowed_sections: Sequence[str]) -> str | None:
    for section in allowed_sections:
        if path_without_lang == section or path_without_lang.startswith(section + "/"):
            return section
    return None


def build_relative_path(language: str, path_without_lang: str) -> Path:
    return Path(language) / normalize_markdown_path(path_without_lang)


def collect_from_llms(
    *,
    headers: Mapping[str, str],
    timeout: float,
    languages: set[str],
    include_all_languages: bool,
    allowed_sections: Sequence[str],
) -> list[DocPage]:
    pages: list[DocPage] = []
    text = fetch_text(LLMS_URL, headers=headers, timeout=timeout)
    for url in parse_llms_urls(text):
        parsed = urlparse(url)
        if parsed.netloc != "docs.claude.com":
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            continue
        language = parts[0]
        if not include_all_languages and language not in languages:
            continue
        path_without_lang = "/".join(parts[1:])
        section = allowed_section(path_without_lang, allowed_sections)
        if section is None:
            continue
        rel_path = build_relative_path(language, path_without_lang)
        doc_url = url.rstrip(".md") if url.endswith(".md") else url
        pages.append(
            DocPage(
                doc_url=doc_url,
                markdown_url=normalize_markdown_path(url),
                language=language,
                section=section,
                relative_path=rel_path,
                source="llms",
                lastmod=None,
            )
        )
    return pages


def collect_from_sitemap(
    *,
    sitemap_root: ET.Element,
    languages: set[str],
    include_all_languages: bool,
    allowed_sections: Sequence[str],
    output_root: Path,
) -> list[DocPage]:
    pages: list[DocPage] = []
    for url_el in sitemap_root.findall("sm:url", SITEMAP_NS):
        loc_el = url_el.find("sm:loc", SITEMAP_NS)
        if loc_el is None or not (loc_el.text or "").strip():
            continue
        loc = loc_el.text.strip()
        parsed = urlparse(loc)
        if parsed.netloc != "docs.claude.com":
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            continue
        language = parts[0]
        if not include_all_languages and language not in languages:
            continue
        path_without_lang = "/".join(parts[1:])
        section = allowed_section(path_without_lang, allowed_sections)
        if section is None:
            continue
        lastmod_el = url_el.find("sm:lastmod", SITEMAP_NS)
        lastmod = (lastmod_el.text or "").strip() if lastmod_el is not None else None
        rel_path = build_relative_path(language, path_without_lang)
        markdown_url = normalize_markdown_path(loc)
        pages.append(
            DocPage(
                doc_url=loc,
                markdown_url=markdown_url,
                language=language,
                section=section,
                relative_path=rel_path,
                source="sitemap",
                lastmod=lastmod,
            )
        )
    return pages


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> dict[str, PreviousMeta]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = {}
    for doc in data.get("documents", []):
        entries[str(doc.get("relative_path"))] = PreviousMeta(
            etag=doc.get("etag"),
            last_modified=doc.get("last_modified"),
            lastmod=doc.get("lastmod"),
            sha256=doc.get("content_sha256"),
            content_length=doc.get("content_length"),
            fetched_at=doc.get("fetched_at"),
        )
    return entries


def conditional_fetch(
    page: DocPage,
    *,
    headers: Mapping[str, str],
    timeout: float,
    force: bool,
    dry_run: bool,
    previous: PreviousMeta | None,
) -> tuple[str, dict[str, str | int | None]]:
    """Return (status, metadata). Status includes: downloaded, not-modified, skipped-lastmod."""
    # If sitemap lastmod matches previous and not forcing, skip without a request.
    if not force and previous and page.lastmod and previous.lastmod == page.lastmod:
        return "skipped-lastmod", {
            "etag": previous.etag,
            "last_modified": previous.last_modified,
            "lastmod": previous.lastmod,
            "content_sha256": previous.sha256,
            "content_length": previous.content_length,
            "fetched_at": previous.fetched_at,
        }

    request_headers = dict(headers)
    if not force and previous:
        if previous.etag:
            request_headers["If-None-Match"] = previous.etag
        elif previous.last_modified:
            request_headers["If-Modified-Since"] = previous.last_modified

    request = Request(page.markdown_url, headers=request_headers)
    try:
        with urlopen(request, timeout=timeout) as resp:  # nosec: required
            status_code = getattr(resp, "status", resp.getcode())
            if status_code == 304:
                return "not-modified", {
                    "etag": previous.etag if previous else None,
                    "last_modified": previous.last_modified if previous else None,
                    "lastmod": page.lastmod,
                    "content_sha256": previous.sha256 if previous else None,
                    "content_length": previous.content_length if previous else None,
                    "fetched_at": previous.fetched_at if previous else None,
                }
            body = b"" if dry_run else resp.read()
            meta = {
                "etag": resp.headers.get("ETag"),
                "last_modified": resp.headers.get("Last-Modified"),
                "lastmod": page.lastmod,
                "content_sha256": hashlib.sha256(body).hexdigest() if body else previous.sha256 if previous else None,
                "content_length": len(body) if body else previous.content_length if previous else None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            return "downloaded", (body, meta)
    except HTTPError as err:
        if err.code == 304:
            return "not-modified", {
                "etag": previous.etag if previous else None,
                "last_modified": previous.last_modified if previous else None,
                "lastmod": page.lastmod,
                "content_sha256": previous.sha256 if previous else None,
                "content_length": previous.content_length if previous else None,
                "fetched_at": previous.fetched_at if previous else None,
            }
        raise RuntimeError(f"HTTP {err.code} for {page.markdown_url}") from err
    except URLError as err:  # pragma: no cover - network edge
        raise RuntimeError(f"Network error for {page.markdown_url}: {err.reason}") from err


def download_page(
    page: DocPage,
    *,
    headers: Mapping[str, str],
    timeout: float,
    force: bool,
    dry_run: bool,
    previous: PreviousMeta | None,
) -> tuple[str, dict[str, str | int | None]]:
    status, meta = conditional_fetch(
        page,
        headers=headers,
        timeout=timeout,
        force=force,
        dry_run=dry_run,
        previous=previous,
    )
    if status in {"skipped-lastmod", "not-modified"}:
        return status, meta  # type: ignore[return-value]
    if status == "downloaded":
        body, meta_dict = meta  # type: ignore[misc]
        return status, {"body": body, **meta_dict}
    return status, meta  # fallback


def write_manifest(
    *,
    output_root: Path,
    languages: Sequence[str],
    sections: Sequence[str],
    pages: Sequence[DocPage],
    metadata_by_path: Mapping[Path, dict[str, str | int | None]],
    discovery: dict[str, str],
) -> Path:
    manifest_payload = {
        "source": BASE_DOMAIN,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "languages": list(languages),
        "sections": list(sections),
        "discovery": discovery,
        "total_documents": len(pages),
        "documents": [],
    }
    for page in sorted(pages, key=lambda p: str(p.relative_path)):
        meta = metadata_by_path.get(page.relative_path, {})
        manifest_payload["documents"].append(
            {
                "doc_url": page.doc_url,
                "markdown_url": page.markdown_url,
                "relative_path": str(page.relative_path),
                "language": page.language,
                "section": page.section,
                "source": page.source,
                "lastmod": page.lastmod,
                "etag": meta.get("etag"),
                "last_modified": meta.get("last_modified"),
                "content_sha256": meta.get("content_sha256"),
                "content_length": meta.get("content_length"),
                "fetched_at": meta.get("fetched_at"),
            }
        )
    manifest_path = output_root / "manifest.json"
    ensure_directory(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def compute_diff(previous: dict[str, PreviousMeta], current_pages: Sequence[DocPage]) -> tuple[set[str], set[str]]:
    prev_paths = set(previous.keys())
    current_paths = {str(p.relative_path) for p in current_pages}
    removed = prev_paths - current_paths
    added = current_paths - prev_paths
    return added, removed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Claude docs as markdown (llms.txt + sitemap).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory to write the mirrored markdown (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        help="Language code(s) to include (default: en)",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Include every language present in the index",
    )
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        help="Section prefix(es) to include (default: api, docs, release-notes, resources/prompt-library)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they appear unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the mirror without writing any files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_HEADERS["User-Agent"],
        help="Override the User-Agent header",
    )
    parser.add_argument(
        "--skip-llms",
        action="store_true",
        help="Skip llms.txt discovery (use sitemap only)",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(message)s", level=level)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    languages = set(args.languages or ["en"])
    include_all_languages = bool(args.all_languages)
    sections: Sequence[str] = args.sections or DEFAULT_SECTIONS
    output_root = args.output_dir.resolve()
    headers = {**DEFAULT_HEADERS, "User-Agent": args.user_agent}

    previous_manifest = read_manifest(output_root / "manifest.json")

    discovery_notes = {"primary": "llms", "fallback": "sitemap"}

    pages: list[DocPage] = []
    if not args.skip_llms:
        try:
            pages.extend(
                collect_from_llms(
                    headers=headers,
                    timeout=args.timeout,
                    languages=languages,
                    include_all_languages=include_all_languages,
                    allowed_sections=sections,
                )
            )
        except Exception as err:  # pragma: no cover - network edge
            logging.warning("llms.txt discovery failed: %s; falling back to sitemap", err)
            discovery_notes["primary"] = "sitemap"

    try:
        sitemap_root = load_sitemap(headers=headers, timeout=args.timeout)
        pages.extend(
            collect_from_sitemap(
                sitemap_root=sitemap_root,
                languages=languages,
                include_all_languages=include_all_languages,
                allowed_sections=sections,
                output_root=output_root,
            )
        )
    except Exception as err:  # pragma: no cover - network edge
        if not pages:
            logging.error("Failed to fetch sitemap and no pages collected: %s", err)
            return 1
        discovery_notes["fallback"] = f"error: {err}"

    # Deduplicate, preferring llms entries when conflicts occur.
    deduped: dict[Path, DocPage] = {}
    for page in pages:
        existing = deduped.get(page.relative_path)
        if existing and existing.source == "llms":
            continue
        deduped[page.relative_path] = page
    pages = list(deduped.values())

    if not pages:
        logging.warning("No matching documentation pages found for languages=%s sections=%s", languages, sections)
        return 0

    added, removed = compute_diff(previous_manifest, pages)
    if added:
        logging.info("Detected %d added pages since last manifest", len(added))
    if removed:
        logging.info("Detected %d removed pages since last manifest", len(removed))

    download_results: Counter[str] = Counter()
    metadata_by_path: dict[Path, dict[str, str | int | None]] = {}

    def process(page: DocPage) -> tuple[DocPage, str, dict[str, str | int | None]]:
        prev = previous_manifest.get(str(page.relative_path))
        status, meta = download_page(
            page,
            headers=headers,
            timeout=args.timeout,
            force=args.force,
            dry_run=args.dry_run,
            previous=prev,
        )
        return page, status, meta

    if args.max_workers <= 1:
        for page in pages:
            try:
                page, status, meta = process(page)
            except Exception as err:  # pragma: no cover - network edge
                logging.error("Failed to download %s: %s", page.markdown_url, err)
                download_results["failed"] += 1
                continue
            download_results[status] += 1
            metadata_by_path[page.relative_path] = meta
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {pool.submit(process, page): page for page in pages}
            for future in concurrent.futures.as_completed(future_map):
                page = future_map[future]
                try:
                    _, status, meta = future.result()
                except Exception as err:  # pragma: no cover - network edge
                    logging.error("Failed to download %s: %s", page.markdown_url, err)
                    download_results["failed"] += 1
                else:
                    download_results[status] += 1
                    metadata_by_path[page.relative_path] = meta

    # Persist files for downloaded pages.
    if not args.dry_run:
        for page, meta in metadata_by_path.items():
            body = meta.pop("body", None)
            if body is None:
                continue
            destination = output_root / page
            ensure_directory(destination.parent)
            destination.write_bytes(body)

        manifest_path = write_manifest(
            output_root=output_root,
            languages=sorted(languages) if not include_all_languages else ["*"],
            sections=sections,
            pages=pages,
            metadata_by_path=metadata_by_path,
            discovery=discovery_notes,
        )
        logging.info("Wrote manifest to %s", manifest_path)

    logging.info(
        "Download summary: %s",
        ", ".join(f"{count} {key}" for key, count in sorted(download_results.items())),
    )

    return 0 if download_results.get("failed", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
