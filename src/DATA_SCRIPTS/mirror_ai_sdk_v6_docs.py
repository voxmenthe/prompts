#!/usr/bin/env python3
"""
Mirror the AI SDK v6 documentation into a local markdown folder hierarchy.

High-level workflow:
  1. Pull `https://v6.ai-sdk.dev/sitemap.xml` to discover every `docs/` URL.
  2. Fetch each page from the v6 site, isolate the primary `<article data-docs-container>` node.
  3. Normalize the DOM (strip copy buttons, flatten code wrappers, simplify grids).
  4. Convert the sanitized HTML to Markdown (including fenced code blocks with language tags).
  5. Emit `<slug>.md` files that mirror the site hierarchy under an output directory (`/docs/foo/bar` -> `docs/foo/bar.md`).
  6. Emit a `manifest.json` describing the export for auditing.

Usage (from repo root):
    uv run python src/DATA_SCRIPTS/mirror_ai_sdk_v6_docs.py --output SPECIALIZED_CONTEXT_AND_DOCS/AI_SDK_V6_DOCS_MIRROR

The default output path is `SPECIALIZED_CONTEXT_AND_DOCS/AI_SDK_V6_DOCS_MIRROR`. Use `--clean`
to remove any prior run before mirroring.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from markdownify import MarkdownConverter  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "AI_SDK_V6_DOCS_MIRROR"
SITEMAP_URL = "https://v6.ai-sdk.dev/sitemap.xml"
SITEMAP_DOCS_PREFIX = "https://ai-sdk.dev/docs/"
DOCS_PREFIX = "https://v6.ai-sdk.dev/docs/"
USER_AGENT = "prompts-ai-sdk-v6-docs-mirror/0.1 (+https://github.com/voxmenthe/prompts)"
FETCH_RETRIES = 3
FETCH_BACKOFF_SECONDS = 2.0
MAX_WORKERS = 8

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Elements that introduce noise or navigation chrome.
ARTICLE_NOISE_SELECTORS: tuple[str, ...] = (
    "div.flex.items-center.justify-between.pb-6",
    "div.py-32",
    "div[data-docs-pagination]",
    "div[data-docs-footer]",
    "button",
    "form",
    "[data-docs-table-of-contents]",
    "[data-docs-sidebar]",
    "div[class*=tabs_tabs]",
    "div[class*=tabs_tab]",
    "div[class*=text_wrapper__]",
)


@dataclass(frozen=True)
class PageJob:
    url: str
    docs_segments: tuple[str, ...]
    is_section_intro: bool

    @property
    def rel_output(self) -> Path:
        docs_root = Path("docs")
        if not self.docs_segments:
            return docs_root / "index.md"
        *parents, leaf = self.docs_segments
        filename = f"{leaf}-folder-description.md" if self.is_section_intro else f"{leaf}.md"
        return docs_root.joinpath(*parents, filename)


@dataclass
class PageResult:
    url: str
    output_file: Path | None
    status: str
    error: str | None = None


class DocsMarkdownConverter(MarkdownConverter):
    """Customize markdown conversion for AI SDK docs content."""

    def convert_pre(
        self,
        el,
        text,
        convert_as_inline=False,
        parent=None,
        **kwargs,
    ):
        code = el.find("code")
        language = None
        for cls in el.get("class", []):
            if cls.startswith("language-"):
                language = cls.split("-", 1)[1]
                break
        if language is None and code is not None:
            for cls in code.get("class", []):
                if cls.startswith("language-"):
                    language = cls.split("-", 1)[1]
                    break
        lines = []
        for line_div in el.select("div.line"):
            lines.append(line_div.get_text("", strip=False).rstrip())
        if lines:
            snippet = "\n".join(lines)
        else:
            snippet = code.get_text() if code is not None else text
        fenced = f"\n```{language or ''}\n{snippet.strip()}\n```\n"
        return fenced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination directory for the mirrored docs (default: %(default)s)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing new files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of docs pages processed (useful for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum concurrent fetchers (default: {MAX_WORKERS}).",
    )
    return parser.parse_args()


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            with urlopen(req) as response:  # nosec: host is trusted for this task
                return response.read()
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _remap_to_v6(url: str) -> str:
    if not url.startswith(SITEMAP_DOCS_PREFIX):
        return url
    suffix = url[len(SITEMAP_DOCS_PREFIX) :]
    return f"{DOCS_PREFIX}{suffix}"


def collect_jobs(limit: int | None) -> tuple[list[PageJob], dict[str, Path]]:
    xml_bytes = fetch_bytes(SITEMAP_URL)
    root = ET.fromstring(xml_bytes)
    urls = [
        loc.text.strip()
        for loc in root.findall("sm:url/sm:loc", SITEMAP_NS)
        if loc.text and loc.text.startswith(SITEMAP_DOCS_PREFIX)
    ]
    urls = sorted(set(urls))
    if limit is not None:
        urls = urls[:limit]
    segments_list: list[tuple[str, ...]] = []
    raw_entries: list[tuple[str, tuple[str, ...]]] = []
    for url in urls:
        path = urlsplit(url).path.rstrip("/")
        segments = tuple(part for part in path.split("/") if part)
        if not segments or segments[0] != "docs":
            continue
        docs_segments = segments[1:]
        segments_list.append(docs_segments)
        raw_entries.append((_remap_to_v6(url), docs_segments))

    section_prefixes: set[tuple[str, ...]] = set()
    for segments in segments_list:
        for depth in range(1, len(segments)):
            section_prefixes.add(segments[:depth])

    jobs: list[PageJob] = []
    link_lookup: dict[str, Path] = {}
    for url, docs_segments in raw_entries:
        is_section_intro = docs_segments in section_prefixes
        job = PageJob(url=url, docs_segments=docs_segments, is_section_intro=is_section_intro)
        jobs.append(job)
        key = "docs" if not docs_segments else "docs/" + "/".join(docs_segments)
        link_lookup[key] = job.rel_output
    return jobs, link_lookup


def ensure_output_root(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        if output_dir.resolve() == Path("/"):
            raise RuntimeError("Refusing to clean filesystem root.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def sanitize_article(article: BeautifulSoup) -> None:
    factory = BeautifulSoup("", "lxml")
    for selector in ARTICLE_NOISE_SELECTORS:
        for element in article.select(selector):
            element.decompose()

    # Drop in-article SVG icons and similar decorative assets.
    for svg in article.select("svg"):
        svg.decompose()

    # Flatten Next.js code block wrappers so <pre> remains.
    for wrapper in article.select("div[class*=code-block_wrapper]"):
        pre = wrapper.find("pre")
        if pre is not None:
            wrapper.replace_with(pre)

    # Convert grid-based card layouts to unordered lists for readability.
    for grid in article.select("div.grid"):
        anchors = [a for a in grid.find_all("a", href=True, recursive=False)]
        if not anchors:
            continue
        # Skip grids that contain additional complex children (e.g., code layout).
        if any(child for child in grid.children if getattr(child, "name", None) not in {None, "a"}):
            continue
        ul = factory.new_tag("ul")
        for anchor in anchors:
            text = " ".join(anchor.get_text(" ", strip=True).split())
            href = anchor.get("href", "").strip()
            li = factory.new_tag("li")
            new_anchor = factory.new_tag("a", href=href)
            new_anchor.string = text or href or "link"
            li.append(new_anchor)
            ul.append(li)
        grid.replace_with(ul)

    # Remove heading self-links that markdownify would render as [Heading](#heading).
    for heading in article.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if len(heading.contents) == 1:
            link = heading.find("a", href=True)
            if link and link.get("href", "").startswith("#"):
                link.unwrap()


def docs_output_path_from_href(normalized_path: str, link_lookup: dict[str, Path]) -> Path | None:
    trimmed = normalized_path.strip("/")
    if not trimmed:
        return link_lookup.get("docs", Path("docs/index.md"))
    if trimmed.endswith(".md"):
        base = trimmed[:-3]
        if base.endswith("/index"):
            base = base[:-6]
        trimmed = base
    key = trimmed if trimmed.startswith("docs") else f"docs/{trimmed}"
    path = link_lookup.get(key)
    if path is not None:
        return path
    segments = tuple(part for part in trimmed.split("/") if part)
    if not segments or segments[0] != "docs":
        return None
    docs_segments = segments[1:]
    if not docs_segments:
        return link_lookup.get("docs", Path("docs/index.md"))
    if docs_segments[-1] == "index":
        base_segments = docs_segments[:-1]
        key_guess = "docs" if not base_segments else "docs/" + "/".join(base_segments)
        path = link_lookup.get(key_guess)
        if path is not None:
            return path
        docs_segments = base_segments
        if not docs_segments:
            return link_lookup.get("docs", Path("docs/index.md"))
    *parents, leaf = docs_segments
    key_guess = "docs/" + "/".join(docs_segments)
    return link_lookup.get(key_guess, Path("docs").joinpath(*parents, f"{leaf}.md"))


def relativize_internal_links(
    article: BeautifulSoup,
    current_output: Path,
    output_root: Path,
    link_lookup: dict[str, Path],
) -> None:
    try:
        current_rel = current_output.relative_to(output_root)
    except ValueError:
        current_rel = current_output
    current_dir = current_rel.parent
    for anchor in article.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            continue
        normalized_path = parsed.path or ""
        if normalized_path.startswith("/"):
            normalized_path = normalized_path.lstrip("/")
        else:
            candidate = (current_dir / normalized_path).as_posix()
            normalized_path = os.path.normpath(candidate)
        target_rel = docs_output_path_from_href(normalized_path, link_lookup)
        if target_rel is None:
            continue
        target = output_root / target_rel
        relpath = os.path.relpath(target, current_output.parent)
        if parsed.fragment:
            relpath = f"{relpath}#{parsed.fragment}"
        anchor["href"] = relpath


def convert_article_to_markdown(article: BeautifulSoup) -> str:
    converter = DocsMarkdownConverter(
        bullets="-",
        heading_style="ATX",
        newline_style="unix",
        escape_underscores=False,
        escape_asterisks=False,
    )
    markdown = converter.convert_soup(article)
    return markdown.strip() + "\n"


def fetch_article_html(url: str) -> BeautifulSoup:
    html_bytes = fetch_bytes(url)
    soup = BeautifulSoup(html_bytes, "lxml")
    article = soup.find("article", attrs={"data-docs-container": True})
    if article is None:
        raise ValueError(f"Unable to find docs article in {url}")
    return article


def process_page(job: PageJob, output_root: Path, link_lookup: dict[str, Path]) -> PageResult:
    output_file = output_root / job.rel_output
    try:
        article = fetch_article_html(job.url)
        sanitize_article(article)
        relativize_internal_links(article, output_file, output_root, link_lookup)
        markdown = convert_article_to_markdown(article)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")
        return PageResult(url=job.url, output_file=output_file, status="ok")
    except Exception as exc:  # pragma: no cover - defensive
        return PageResult(url=job.url, output_file=None, status="error", error=str(exc))


def write_manifest(output_dir: Path, results: Sequence[PageResult]) -> None:
    manifest_path = output_dir / "manifest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sitemap": SITEMAP_URL,
            "docs_prefix": DOCS_PREFIX,
            "sitemap_docs_prefix": SITEMAP_DOCS_PREFIX,
        },
        "counts": {
            "total": len(results),
            "succeeded": sum(1 for r in results if r.status == "ok"),
            "failed": sum(1 for r in results if r.status != "ok"),
        },
        "pages": [
            {
                "url": r.url,
                "output": str(r.output_file.relative_to(output_dir)) if r.output_file else None,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    ensure_output_root(output_root, clean=args.clean)

    jobs, link_lookup = collect_jobs(limit=args.limit)
    if not jobs:
        print("No documentation URLs discovered; aborting.", file=sys.stderr)
        return 1

    print(f"Discovered {len(jobs)} docs pages. Mirroring into {output_root} ...")
    results: list[PageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(process_page, job, output_root, link_lookup): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results.append(result)
            if result.status == "ok":
                rel_path = result.output_file.relative_to(output_root) if result.output_file else "?"
                print(f"[OK] {job.url} -> {rel_path}")
            else:
                print(f"[FAIL] {job.url}: {result.error}", file=sys.stderr)

    write_manifest(output_root, results)

    failures = [r for r in results if r.status != "ok"]
    summary = (
        f"Mirror complete: {len(results) - len(failures)} succeeded, {len(failures)} failed."
    )
    if failures:
        print(summary, file=sys.stderr)
        return 2
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
