#!/usr/bin/env python3
"""Mirror the Claude API documentation markdown locally.

The script fetches the sitemap published at https://docs.claude.com/sitemap.xml,
filters pages under a chosen language/section (defaults to ``en/api``), and
writes the markdown variants (``*.md``) into a local directory that mirrors the
remote hierarchy underneath that language/section. A manifest is emitted
alongside the files to make downstream processing easier.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BASE_DOMAIN = "https://docs.claude.com"
SITEMAP_URL = f"{BASE_DOMAIN}/sitemap.xml"
DEFAULT_HEADERS = {
    "User-Agent": "ClaudeDocsMirror/2025.09 (+https://docs.claude.com/en/api/)"
}
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
DEFAULT_OUTPUT_DIRECTORY = Path("SPECIALIZED_CONTEXT_AND_DOCS/ANTHROPIC_API_DOCS_MIRROR")


@dataclass(frozen=True)
class DocPage:
    doc_url: str
    markdown_url: str
    output_path: Path


def fetch_bytes(url: str, *, headers: dict[str, str], timeout: float = 15.0) -> bytes:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as resp:  # nosec: required for this script
        return resp.read()


def load_sitemap(*, headers: dict[str, str]) -> ET.Element:
    raw = fetch_bytes(SITEMAP_URL, headers=headers)
    return ET.fromstring(raw)


def collect_doc_pages(
    sitemap_root: ET.Element,
    *,
    languages: Sequence[str],
    section: str,
    output_root: Path,
) -> list[DocPage]:
    pages: list[DocPage] = []
    seen_slugs: set[str] = set()
    assigned_relative_paths: dict[Path, str] = {}
    for loc_el in sitemap_root.findall("sm:url/sm:loc", SITEMAP_NS):
        loc_text = (loc_el.text or "").strip()
        if not loc_text:
            continue
        for lang in languages:
            prefix = f"{BASE_DOMAIN}/{lang}/{section}"
            if not loc_text.startswith(prefix + "/") and loc_text != prefix:
                continue
            slug = loc_text.removeprefix(f"{BASE_DOMAIN}/").strip("/")
            if not slug:
                continue
            if slug in seen_slugs:
                break
            seen_slugs.add(slug)
            segments = [segment for segment in slug.split("/") if segment]
            if not segments:
                continue
            prefix_segments = [lang, *[part for part in section.split("/") if part]]
            # Skip entries that correspond exactly to the language/section prefix.
            if len(segments) <= len(prefix_segments):
                break
            relative_segments = segments[len(prefix_segments) :]
            # Derive destination path that mirrors the hierarchy beneath the prefix.
            directory_segments = relative_segments[:-1]
            leaf = relative_segments[-1]
            dest_dir = output_root.joinpath(*directory_segments) if directory_segments else output_root
            relative_path = (Path(*directory_segments) / f"{leaf}.md") if directory_segments else Path(f"{leaf}.md")
            existing_slug = assigned_relative_paths.get(relative_path)
            if existing_slug and existing_slug != slug:
                raise ValueError(
                    "Conflicting documentation pages would overwrite each other: "
                    f"'{existing_slug}' and '{slug}' both map to '{relative_path}'."
                )
            assigned_relative_paths.setdefault(relative_path, slug)
            output_path = dest_dir / f"{leaf}.md"
            markdown_url = loc_text.rstrip("/") + ".md"
            pages.append(DocPage(loc_text, markdown_url, output_path))
            break
    return sorted(pages, key=lambda page: page.doc_url)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_page(
    page: DocPage,
    *,
    force: bool,
    dry_run: bool,
    headers: dict[str, str],
    timeout: float,
) -> str:
    destination = page.output_path
    if destination.exists() and not force:
        return "skipped"
    if dry_run:
        return "planned"
    ensure_directory(destination.parent)
    try:
        payload = fetch_bytes(page.markdown_url, headers=headers, timeout=timeout)
    except HTTPError as err:
        raise RuntimeError(f"HTTP {err.code} for {page.markdown_url}") from err
    except URLError as err:  # pragma: no cover - network edge
        raise RuntimeError(f"Network error for {page.markdown_url}: {err.reason}") from err
    text = payload.decode("utf-8", errors="replace")
    destination.write_text(text, encoding="utf-8")
    return "downloaded"


def write_manifest(
    pages: Sequence[DocPage],
    *,
    output_root: Path,
    languages: Sequence[str],
    section: str,
) -> Path:
    manifest_payload = {
        "source": BASE_DOMAIN,
        "section": section,
        "languages": list(languages),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [
            {
                "doc_url": page.doc_url,
                "markdown_url": page.markdown_url,
                "relative_path": str(page.output_path.relative_to(output_root)),
            }
            for page in pages
        ],
    }
    manifest_path = output_root / "manifest.json"
    ensure_directory(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Claude API docs as markdown.")
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
        "--section",
        default="api",
        help="Top-level docs section to mirror (default: %(default)s)",
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
        help="Redownload files even if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the mirror without writing any files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(message)s", level=level)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    languages = args.languages or ["en"]
    output_root = args.output_dir.resolve()
    headers = DEFAULT_HEADERS.copy()

    logging.info("Fetching sitemap from %s", SITEMAP_URL)
    try:
        sitemap_root = load_sitemap(headers=headers)
    except (HTTPError, URLError) as err:  # pragma: no cover - network edge
        logging.error("Failed to fetch sitemap: %s", err)
        return 1

    pages = collect_doc_pages(
        sitemap_root,
        languages=languages,
        section=args.section,
        output_root=output_root,
    )

    if not pages:
        logging.warning("No matching documentation pages found for %s/%s", languages, args.section)
        return 0

    logging.info("Planned %d markdown pages", len(pages))

    download_results: Counter[str] = Counter()
    if args.max_workers <= 1:
        for page in pages:
            try:
                status = download_page(
                    page,
                    force=args.force,
                    dry_run=args.dry_run,
                    headers=headers,
                    timeout=args.timeout,
                )
                download_results[status] += 1
                logging.debug("%s -> %s (%s)", page.markdown_url, page.output_path, status)
            except Exception as err:  # pragma: no cover - network edge
                logging.error("Failed to download %s: %s", page.markdown_url, err)
                download_results["failed"] += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(
                    download_page,
                    page,
                    force=args.force,
                    dry_run=args.dry_run,
                    headers=headers,
                    timeout=args.timeout,
                ): page
                for page in pages
            }
            for future in concurrent.futures.as_completed(future_map):
                page = future_map[future]
                try:
                    status = future.result()
                except Exception as err:  # pragma: no cover - network edge
                    logging.error("Failed to download %s: %s", page.markdown_url, err)
                    download_results["failed"] += 1
                else:
                    download_results[status] += 1
                    logging.debug("%s -> %s (%s)", page.markdown_url, page.output_path, status)

    if not args.dry_run:
        manifest_path = write_manifest(
            pages,
            output_root=output_root,
            languages=languages,
            section=args.section,
        )
        logging.info("Wrote manifest to %s", manifest_path)

    logging.info(
        "Download summary: %s",
        ", ".join(f"{count} {key}" for key, count in sorted(download_results.items())),
    )
    return 0 if download_results.get("failed", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
