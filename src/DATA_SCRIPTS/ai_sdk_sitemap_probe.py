#!/usr/bin/env python3
"""
Quick probe script to inspect the AI SDK docs sitemap.

This validates that we can rely on `https://ai-sdk.dev/sitemap.xml` to discover
documentation URLs under `/docs/`. The script prints aggregate counts alongside
examples to confirm coverage before investing in the full mirroring workflow.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from textwrap import indent
from typing import Iterable
from urllib.request import Request, urlopen

SITEMAP_URL = "https://ai-sdk.dev/sitemap.xml"
DOCS_PREFIX = "https://ai-sdk.dev/docs/"
USER_AGENT = "prompts-ai-sdk-sitemap-probe/0.1 (+https://github.com/voxmenthe/prompts)"

NAMESPACE = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass(frozen=True)
class SitemapSummary:
    total_urls: int
    docs_urls: list[str]

    @property
    def docs_count(self) -> int:
        return len(self.docs_urls)

    def sample(self, size: int = 10) -> list[str]:
        return self.docs_urls[:size]


def fetch_sitemap() -> bytes:
    req = Request(SITEMAP_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as response:  # nosec: trusted host required by task
        return response.read()


def parse_sitemap(xml_bytes: bytes) -> SitemapSummary:
    root = ET.fromstring(xml_bytes)
    urls: list[str] = [
        node.text.strip()
        for node in root.findall("sm:url/sm:loc", NAMESPACE)
        if node.text
    ]
    docs_urls = sorted(u for u in urls if u.startswith(DOCS_PREFIX))
    return SitemapSummary(total_urls=len(urls), docs_urls=docs_urls)


def render_report(summary: SitemapSummary) -> str:
    header = [
        f"Sitemap URL: {SITEMAP_URL}",
        f"Total URLs discovered: {summary.total_urls}",
        f"Documentation URLs (/{DOCS_PREFIX[len('https://ai-sdk.dev/'):]}) count: {summary.docs_count}",
    ]
    sample_block = "No documentation URLs found."
    if summary.docs_urls:
        sample_lines = [
            f"{index + 1:>2}: {url}"
            for index, url in enumerate(summary.sample())
        ]
        sample_block = "Sample docs URLs:\n" + indent("\n".join(sample_lines), "  ")
    return "\n".join(header + ["", sample_block])


def main() -> int:
    try:
        xml_bytes = fetch_sitemap()
        summary = parse_sitemap(xml_bytes)
        print(render_report(summary))
        return 0
    except Exception as exc:  # pragma: no cover - probe script
        print(f"Failed to probe sitemap: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
