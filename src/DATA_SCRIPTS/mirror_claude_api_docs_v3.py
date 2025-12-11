#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "html2text>=2025.4.15",
# ]
# ///
"""Mirror Claude documentation by crawling navigation links.

This script handles the case where sitemap.xml and llms.txt are protected by
authentication. Instead, it discovers pages by:
1. Starting from known entry point pages
2. Parsing the navigation sidebar from the HTML
3. Following internal documentation links

Usage:
    uv run python src/DATA_SCRIPTS/mirror_claude_api_docs_v3.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import html2text

BASE_DOMAIN = "https://docs.anthropic.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_OUTPUT_DIRECTORY = Path("SPECIALIZED_CONTEXT_AND_DOCS/ANTHROPIC_API_DOCS_MIRROR")

# Entry points to start crawling from
ENTRY_POINTS = [
    "/en/docs",
    "/en/api",
    "/en/release-notes",
    "/en/home",
]

# Allowed path prefixes to mirror
ALLOWED_PREFIXES = (
    "/en/docs",
    "/en/api",
    "/en/release-notes",
    "/en/home",
    "/en/resources/prompt-library",
)


class LinkExtractor(HTMLParser):
    """Extract internal documentation links from HTML."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                # Resolve relative URLs
                full_url = urljoin(self.base_url, value)
                parsed = urlparse(full_url)
                # Only keep internal docs links
                if parsed.netloc in ("docs.anthropic.com", "docs.claude.com", ""):
                    # Normalize and store path
                    path = parsed.path.rstrip("/")
                    if path and any(path.startswith(p) for p in ALLOWED_PREFIXES):
                        self.links.add(path)


@dataclass
class DocPage:
    url: str
    path: str
    content: bytes = field(default=b"", repr=False)
    sha256: str = ""
    fetched_at: str = ""


def fetch_page(url: str, *, headers: Mapping[str, str], timeout: float = 20.0) -> tuple[bytes, dict]:
    """Fetch a page and return content + response headers."""
    request = Request(url, headers=dict(headers))
    try:
        with urlopen(request, timeout=timeout) as resp:
            content = resp.read()
            return content, dict(resp.headers)
    except HTTPError as err:
        raise RuntimeError(f"HTTP {err.code} for {url}") from err
    except URLError as err:
        raise RuntimeError(f"Network error for {url}: {err.reason}") from err


def extract_links(html: bytes, base_url: str) -> set[str]:
    """Extract internal doc links from HTML content.
    
    Uses multiple strategies since links may be in:
    1. Standard anchor tags
    2. React/Next.js JSON payloads with escaped or unescaped quotes
    """
    html_str = html.decode("utf-8", errors="replace")
    links: set[str] = set()
    
    # Strategy 1: Standard HTML parser for anchor tags
    parser = LinkExtractor(base_url)
    try:
        parser.feed(html_str)
        links.update(parser.links)
    except Exception:
        pass
    
    # Strategy 2: Regex patterns for paths in React JSON
    # Pattern 1: Full paths already with language prefix like "en/api/messages"
    full_path_matches = re.findall(r'en/(?:api|docs|release-notes|home|resources)/[a-zA-Z0-9_/-]+', html_str)
    for match in full_path_matches:
        path = "/" + match.rstrip('/')
        # Skip static assets
        if any(x in path for x in ['.js', '.css', '.png', '.jpg', 'static/', 'chunk', '_next']):
            continue
        if len(path) > 8 and any(path.startswith(p) for p in ALLOWED_PREFIXES):
            links.add(path)
    
    # Pattern 2: Subpaths without language prefix like "api/messages" 
    subpath_matches = re.findall(r'(?<![a-z/])(?:api|docs|release-notes|home)/[a-zA-Z0-9_/-]+', html_str)
    for match in subpath_matches:
        path = match.rstrip('/')
        # Skip if it looks like it's part of a longer path or static asset
        if any(x in path for x in ['.js', '.css', '.png', '.jpg', 'static/', 'chunk', '_next', 'en/']):
            continue
        if len(path) < 5:
            continue
        # Add /en/ prefix
        full_path = f"/en/{path}"
        if any(full_path.startswith(p) for p in ALLOWED_PREFIXES):
            links.add(full_path)
    
    return links


def discover_pages(
    *,
    headers: Mapping[str, str],
    timeout: float,
    max_pages: int = 500,
) -> dict[str, DocPage]:
    """Discover all documentation pages by crawling from entry points."""
    discovered: dict[str, DocPage] = {}
    to_visit: set[str] = set(ENTRY_POINTS)
    visited: set[str] = set()

    logging.info("Starting page discovery from %d entry points", len(ENTRY_POINTS))

    while to_visit and len(discovered) < max_pages:
        path = to_visit.pop()
        if path in visited:
            continue
        visited.add(path)

        url = f"{BASE_DOMAIN}{path}"
        logging.debug("Discovering: %s", url)

        try:
            content, resp_headers = fetch_page(url, headers=headers, timeout=timeout)
        except Exception as err:
            logging.warning("Failed to fetch %s: %s", url, err)
            continue

        # Skip redirect pages or error pages
        if b"<title>Not Found" in content or len(content) < 1000:
            logging.debug("Skipping %s (not found or too small)", path)
            continue

        page = DocPage(
            url=url,
            path=path,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        discovered[path] = page
        logging.info("Discovered: %s (%d bytes)", path, len(content))

        # Extract links and add to queue
        new_links = extract_links(content, url)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.add(link)

    logging.info("Discovery complete: found %d pages", len(discovered))
    return discovered


def create_html2text_converter() -> html2text.HTML2Text:
    """Create a configured html2text converter for documentation pages."""
    h = html2text.HTML2Text()
    
    # Output formatting
    h.body_width = 0  # Don't wrap lines (let the reader handle it)
    h.unicode_snob = True  # Use unicode characters
    h.protect_links = True  # Don't break up links
    h.wrap_links = False  # Don't wrap links
    h.wrap_list_items = False  # Don't wrap list items
    h.wrap_tables = False  # Don't wrap tables
    
    # Content inclusion
    h.ignore_links = False  # Keep links
    h.ignore_images = False  # Keep image references
    h.ignore_emphasis = False  # Keep bold/italic
    h.ignore_tables = False  # Keep tables
    
    # Boilerplate removal
    h.skip_internal_links = False  # Keep internal anchor links
    h.inline_links = True  # Use inline link format [text](url)
    h.single_line_break = False  # Use double line breaks for paragraphs
    
    # Code handling
    h.mark_code = True  # Use backticks for code
    
    return h


def extract_markdown_content(html: bytes, url: str = "") -> str | None:
    """Extract the main article content from HTML and convert to markdown.
    
    Uses html2text for robust conversion with preprocessing to:
    1. Extract only the main content area (skip nav, header, footer)
    2. Remove script/style tags
    3. Clean up React/Next.js artifacts
    """
    html_str = html.decode("utf-8", errors="replace")
    
    # Step 1: Remove elements we definitely don't want
    # Remove script and style tags
    html_str = re.sub(r'<script[^>]*>.*?</script>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<style[^>]*>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove navigation and footer elements
    html_str = re.sub(r'<nav[^>]*>.*?</nav>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<footer[^>]*>.*?</footer>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<header[^>]*>.*?</header>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<aside[^>]*>.*?</aside>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove common sidebar patterns by class names
    sidebar_patterns = [
        r'<div[^>]*class="[^"]*(?:sidebar|sidenav|side-nav|navigation|menu|toc|table-of-contents)[^"]*"[^>]*>.*?</div>',
        r'<div[^>]*class="[^"]*(?:search|searchbar|search-box)[^"]*"[^>]*>.*?</div>',
        r'<div[^>]*class="[^"]*(?:breadcrumb)[^"]*"[^>]*>.*?</div>',
        r'<button[^>]*>.*?</button>',  # Remove buttons
        r'<input[^>]*>',  # Remove input elements
    ]
    for pattern in sidebar_patterns:
        html_str = re.sub(pattern, '', html_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove hidden elements
    html_str = re.sub(r'<[^>]+hidden[^>]*>.*?</[^>]+>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<div[^>]*aria-hidden="true"[^>]*>.*?</div>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove search placeholders and keyboard shortcuts
    html_str = re.sub(r'Search\.\.\.', '', html_str)
    html_str = re.sub(r'⌘K', '', html_str)
    
    # Remove React hydration artifacts (the __next_f arrays)
    html_str = re.sub(r'self\.__next_f\.push\([^)]+\)', '', html_str)
    
    # Step 2: Try to extract main content area
    content = None
    
    # Try to find article or main content
    patterns = [
        r'<article[^>]*>(.*?)</article>',
        r'<main[^>]*>(.*?)</main>',
        r'<div[^>]*class="[^"]*(?:content|article|docs|documentation)[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*role="main"[^>]*>(.*?)</div>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html_str, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1)
            # Verify it has substantial content (not just navigation)
            if len(content) > 500:
                break
            content = None
    
    # Fallback: use body
    if not content:
        match = re.search(r'<body[^>]*>(.*?)</body>', html_str, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1)
    
    if not content:
        return None
    
    # Step 3: Convert to markdown using html2text
    converter = create_html2text_converter()
    markdown = converter.handle(content)
    
    # Step 4: Post-process the markdown
    # Remove excessive blank lines
    markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
    
    # Remove lines that are just whitespace
    lines = markdown.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines that are just whitespace
        if not stripped:
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append('')
        else:
            cleaned_lines.append(line)
    
    markdown = '\n'.join(cleaned_lines)
    
    # Add source URL as a comment at the top
    if url:
        markdown = f"<!-- Source: {url} -->\n\n{markdown}"
    
    return markdown.strip()


def path_to_filepath(path: str) -> Path:
    """Convert URL path to local file path."""
    # Remove leading slash and add .md extension
    clean_path = path.lstrip("/")
    if not clean_path.endswith(".md"):
        clean_path = f"{clean_path}.md"
    return Path(clean_path)


def save_pages(
    pages: dict[str, DocPage],
    output_root: Path,
    *,
    save_html: bool = False,
    save_markdown: bool = True,
) -> dict[str, dict]:
    """Save discovered pages to disk."""
    metadata: dict[str, dict] = {}

    for path, page in pages.items():
        rel_path = path_to_filepath(path)
        dest = output_root / rel_path

        dest.parent.mkdir(parents=True, exist_ok=True)

        if save_markdown:
            md_content = extract_markdown_content(page.content, url=page.url)
            if md_content:
                dest.write_text(md_content, encoding="utf-8")
                logging.info("Saved: %s", dest)

        if save_html:
            html_dest = dest.with_suffix(".html")
            html_dest.write_bytes(page.content)

        metadata[str(rel_path)] = {
            "url": page.url,
            "path": page.path,
            "sha256": page.sha256,
            "fetched_at": page.fetched_at,
            "content_length": len(page.content),
        }

    return metadata


def write_manifest(output_root: Path, metadata: dict[str, dict]) -> Path:
    """Write manifest file with all page metadata."""
    manifest = {
        "source": BASE_DOMAIN,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_method": "html_crawl",
        "total_documents": len(metadata),
        "documents": metadata,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Claude docs by crawling navigation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory to write mirrored content (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Maximum number of pages to discover (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Also save raw HTML files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(message)s", level=level)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Starting Claude docs mirror (crawl method)")
    logging.info("Output directory: %s", output_root)

    # Discover pages by crawling
    pages = discover_pages(
        headers=DEFAULT_HEADERS,
        timeout=args.timeout,
        max_pages=args.max_pages,
    )

    if not pages:
        logging.error("No pages discovered!")
        return 1

    # Save pages
    metadata = save_pages(
        pages,
        output_root,
        save_html=args.save_html,
        save_markdown=True,
    )

    # Write manifest
    manifest_path = write_manifest(output_root, metadata)
    logging.info("Wrote manifest to %s", manifest_path)
    logging.info("Mirror complete: %d pages saved", len(pages))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
