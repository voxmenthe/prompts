#!/usr/bin/env python3
"""
Prototype script that fetches a single AI SDK docs page and dumps a markdown preview.

This helps validate DOM selection and cleanup heuristics before building the full
site mirroring workflow. Run it with either a full URL or a path suffix, e.g.:

    uv run python src/DATA_SCRIPTS/ai_sdk_page_probe.py docs/introduction
    uv run python src/DATA_SCRIPTS/ai_sdk_page_probe.py https://ai-sdk.dev/docs/reference
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from markdownify import MarkdownConverter  # type: ignore[import-not-found]

BASE_URL = "https://ai-sdk.dev/"
DEFAULT_PATH = "docs/introduction"
USER_AGENT = "prompts-ai-sdk-page-probe/0.1 (+https://github.com/voxmenthe/prompts)"

# Elements that add noise to the markdown export.
NOISE_SELECTORS: tuple[str, ...] = (
    "div.flex.items-center.justify-between.pb-6",  # copy markdown header row
    "div.py-32",  # prev / next nav block
    "div[data-docs-pagination]",
    "button",
    "[data-docs-table-of-contents]",
    "[data-docs-sidebar]",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_PATH,
        help="Docs URL or path suffix to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=80,
        help="Number of markdown lines to print (default: %(default)s)",
    )
    return parser.parse_args()


def build_url(target: str) -> str:
    if target.startswith("http://") or target.startswith("https://"):
        return target
    return urljoin(BASE_URL, target.lstrip("/"))


def fetch_html(url: str) -> BeautifulSoup:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as response:  # nosec: target domain required for task
        html = response.read()
    return BeautifulSoup(html, "lxml")


def strip_noise(article: BeautifulSoup) -> None:
    for selector in NOISE_SELECTORS:
        for element in article.select(selector):
            element.decompose()
    # Flatten code block wrappers to keep fenced blocks intact.
    for wrapper in article.select("div[class*=code-block_wrapper]"):
        pre = wrapper.find("pre")
        if pre:
            wrapper.replace_with(pre)


def html_to_markdown(article: BeautifulSoup) -> str:
    converter = MarkdownConverter(
        bullets="-",
        heading_style="ATX",
        newline_style="unix",
        escape_underscores=False,
    )
    return converter.convert_soup(article)


def preview_markdown(markdown: str, limit: int) -> str:
    lines = markdown.strip().splitlines()
    head = lines[:limit]
    trailer = f"... ({len(lines) - limit} more lines)" if len(lines) > limit else ""
    return "\n".join(head + ([trailer] if trailer else []))


def main() -> int:
    args = parse_args()
    url = build_url(args.target)
    soup = fetch_html(url)
    article = soup.find("article", attrs={"data-docs-container": True})
    if article is None:
        print(f"Unable to locate docs content in {url}", file=sys.stderr)
        return 1
    strip_noise(article)
    markdown = html_to_markdown(article)
    print(f"# Markdown preview for {url}\n")
    print(preview_markdown(markdown, args.lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
