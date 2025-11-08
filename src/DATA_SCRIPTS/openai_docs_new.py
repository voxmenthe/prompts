#!/usr/bin/env python3
"""
Mirror OpenAI developer docs to local Markdown, robustly.

This script is a pragmatic, resilient alternative to the previous scrapers.
It avoids fragile JS bundle introspection and headless browser automation by:

- Enumerating docs pages from the official sitemap
- Fetching each page HTML with a real-browser user agent via cloudscraper
- Falling back to r.jina.ai rendering when Cloudflare blocks direct access
- Extracting the main content area heuristically and converting to Markdown
- Optionally also downloading OpenAI's LLM-friendly .txt bundles and OpenAPI spec

Usage examples:
  python openai_docs_mirror_fixed.py --sections guides api-reference --output-dir ./OPENAI_DOCS
  python openai_docs_mirror_fixed.py --no-llm-bundles --sections guides

Notes:
- r.jina.ai is used only as a fallback to bypass Cloudflare in restricted environments.
- The Markdown output preserves all code blocks present in the server HTML (all tabs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import bs4
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


# -------------------- Configuration --------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "OPENAI_DOCS_MIRROR"

SITEMAP_URL = "https://platform.openai.com/sitemap.xml"

LLMS_BASE_URL = "https://cdn.openai.com/API/docs/txt/"
LLMS_INDEX_URL = urljoin(LLMS_BASE_URL, "llms.txt")
OPENAPI_SPEC_URL = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml"

DEFAULT_SECTIONS = (
    "guides",
    # You can also include: "api-reference", "models", "libraries", etc.
)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

TXT_HEADERS = {
    "User-Agent": "openai-docs-mirror/1.0 (+https://github.com/)",
    "Accept": "text/plain, text/markdown, application/json;q=0.9, */*;q=0.8",
}

RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504, 522, 524}


# -------------------- Data classes --------------------

@dataclass(frozen=True)
class PageResult:
    url: str
    relative_path: str
    output_path: Path
    status: str
    bytes_written: int = 0
    sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BundleResult:
    key: str
    url: str
    output_path: Path
    status: str
    bytes_written: int = 0
    sha256: str | None = None
    error: str | None = None


# -------------------- Utilities --------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)s %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str, *, force: bool) -> tuple[str, int, str]:
    ensure_dir(path.parent)
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if path.exists() and not force:
        if path.read_bytes() == data:
            return ("skipped", len(data), digest)
    path.write_bytes(data)
    return ("written", len(data), digest)


def parse_llms_index(text: str) -> dict[str, str]:
    # llms.txt is a small Markdown list of links
    mapping: dict[str, str] = {}
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        mapping[m.group(2).strip()] = m.group(1).strip()
    return mapping


def collect_urls_from_sitemap(xml_text: str, sections: Sequence[str]) -> list[str]:
    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    prefixes = [f"https://platform.openai.com/docs/{s.strip('/')}/" for s in sections]
    out: list[str] = []
    for loc in root.findall("sm:url/sm:loc", ns):
        if not loc.text:
            continue
        url = loc.text.strip()
        if any(url.startswith(p) for p in prefixes):
            out.append(url.rstrip("/"))
    # dedupe preserving order
    seen = set()
    unique: list[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def url_to_relative_path(url: str, *, root: str = "docs") -> str:
    # Converts https://platform.openai.com/docs/guides/text-generation -> guides/text-generation/index.md
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    if not path or not path.startswith(root + "/"):
        raise ValueError(f"URL does not start with /{root}: {url}")
    rel = path[len(root) + 1 :]
    if not rel:
        raise ValueError(f"URL has no segments after /{root}: {url}")
    return rel


def remove_non_content(container: bs4.BeautifulSoup) -> None:
    # Remove obvious non-content controls and widgets
    selectors = [
        "nav",
        "aside",
        "header",
        "footer",
        "[data-docs-feedback]",
        "[data-docs-pagination]",
        "[data-role='search']",
        "button",
    ]
    for sel in selectors:
        for el in container.select(sel):
            el.decompose()


def normalize_code_blocks(container: bs4.BeautifulSoup) -> None:
    # Ensure we keep clean <pre><code class="language-*"></code></pre>
    for pre in container.find_all("pre"):
        code = pre.find("code")
        if not code:
            text = pre.get_text("\n")
            new_pre = container.new_tag("pre")
            new_code = container.new_tag("code")
            new_code.string = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()
            new_pre.append(new_code)
            pre.replace_with(new_pre)
            continue
        # unwrap nested code tags
        for nested in code.find_all("code"):
            nested.unwrap()
        # strip line numbers or artifacts common in renderers
        for ln in code.select(".react-syntax-highlighter-line-number"):
            ln.decompose()
        # preserve language class if present
        lang = None
        for cls in code.get("class", []):
            if cls.startswith("language-"):
                lang = cls
                break
        text = code.get_text("\n")
        new_pre = container.new_tag("pre")
        new_code = container.new_tag("code")
        if lang:
            new_code["class"] = [lang]
        new_code.string = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()
        new_pre.append(new_code)
        pre.replace_with(new_pre)


def html_to_markdown(fragment: bs4.BeautifulSoup) -> str:
    remove_non_content(fragment)
    normalize_code_blocks(fragment)
    md = MarkdownConverter(heading_style="ATX").convert(str(fragment))
    lines = [ln.rstrip() for ln in md.splitlines()]
    return ("\n".join(lines).strip() + "\n") if lines else "\n"


def _strip_line_numbers_in_fences(md: str) -> str:
    # r.jina.ai often includes a left column of line numbers as standalone lines
    # inside fenced code blocks. Remove lines that are purely digits while inside
    # a fenced block.
    out: list[str] = []
    in_fence = False
    fence_delim = None
    for line in md.splitlines():
        if line.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_delim = line
                out.append(line)
                continue
            else:
                in_fence = False
                fence_delim = None
                out.append(line)
                continue
        if in_fence:
            if line.strip().isdigit():
                # skip standalone numeric line numbers
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def extract_main_content(soup: bs4.BeautifulSoup) -> bs4.Tag:
    # Try common containers first, fallback to <main>, then to body
    candidates = [
        "main article",
        "article",
        "main",
        "div[data-docs-content]",
        "div.docs-markdown-page.docs-markdown-content",
        "div.prose",
    ]
    for sel in candidates:
        tag = soup.select_one(sel)
        if isinstance(tag, bs4.Tag):
            return tag
    if soup.body:
        return soup.body
    return soup


def fetch_html_with_cloudscraper(url: str, *, timeout: float) -> str | None:
    try:
        import cloudscraper  # type: ignore
    except Exception:
        return None
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    try:
        resp = scraper.get(url, timeout=timeout, headers=BROWSER_HEADERS)
        if resp.status_code >= 400:
            return None
        text = resp.text
        # Detect CF challenge page
        if "Just a moment..." in text and "challenge-platform" in text:
            return None
        return text
    except Exception:
        return None


def fetch_markdown_via_jina(url: str, *, timeout: int = 120) -> str | None:
    import requests
    # r.jina.ai can render and return Markdown Content for a given URL
    # It expects the target URL to be passed with an explicit scheme after /http(s)://
    # We use http scheme to reduce TLS issues per their example, but both work
    u = urlparse(url)
    # Use http scheme to the target per r.jina.ai docs
    jina_url = f"https://r.jina.ai/http://{u.netloc}{u.path}"
    params = {"timeout": min(max(int(timeout), 1), 180)}
    try:
        r = requests.get(jina_url, params=params, timeout=timeout)
        r.raise_for_status()
        text = r.text
        # The response typically contains a header and "Markdown Content:" marker
        # Extract the part after "Markdown Content:" if possible
        marker = "Markdown Content:"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker) :].lstrip("\n")
        return text
    except Exception:
        return None


def robust_get_markdown_for_url(url: str, *, timeout: float) -> str:
    # Strategy:
    # 1) Try cloudscraper to fetch HTML and convert to Markdown
    # 2) Fallback to r.jina.ai rendered Markdown
    html_text = fetch_html_with_cloudscraper(url, timeout=timeout)
    if html_text:
        soup = BeautifulSoup(html_text, "lxml")
        main = extract_main_content(soup)
        # Work on a copy to avoid modifying soup
        frag = BeautifulSoup(str(main), "lxml")
        md_from_html = html_to_markdown(frag)
        # Heuristic: if content is too short, it's likely just a shell (client-run site).
        if len(md_from_html.strip()) >= 200:
            return _strip_line_numbers_in_fences(md_from_html)
        # Otherwise fall through to Jina fallback
    md = fetch_markdown_via_jina(url, timeout=int(timeout))
    if md:
        cleaned = _strip_line_numbers_in_fences(md if md.endswith("\n") else md + "\n")
        return cleaned
    raise RuntimeError("Failed to fetch content via both cloudscraper and r.jina.ai")


def retry(fetch_fn, *, attempts: int, backoff: float):
    last_err: Exception | None = None
    for i in range(1, max(1, attempts) + 1):
        try:
            return fetch_fn()
        except Exception as e:
            last_err = e
            if i >= attempts:
                break
            sleep_for = backoff * (2 ** (i - 1))
            time.sleep(sleep_for + random.uniform(0, sleep_for * 0.25))
    assert last_err is not None
    raise last_err


# -------------------- Main workflow --------------------

def mirror_pages(
    *,
    output_root: Path,
    sections: Sequence[str],
    force: bool,
    timeout: float,
    attempts: int,
    backoff: float,
    path_filter: str | None = None,
) -> list[PageResult]:
    import requests

    results: list[PageResult] = []
    # Fetch sitemap
    logging.info("Fetching sitemap: %s", SITEMAP_URL)
    sm = requests.get(SITEMAP_URL, timeout=timeout, headers=BROWSER_HEADERS)
    sm.raise_for_status()
    urls = collect_urls_from_sitemap(sm.text, sections)
    if not urls:
        logging.warning("No URLs found for sections: %s", ", ".join(sections))
        return results
    logging.info("Found %d URLs in sections %s", len(urls), ", ".join(sections))

    import re as _re
    regex = _re.compile(path_filter) if path_filter else None
    for url in urls:
        try:
            rel = url_to_relative_path(url, root="docs")
        except ValueError as e:
            results.append(
                PageResult(url=url, relative_path="", output_path=output_root, status="failed", error=str(e))
            )
            continue
        if regex and not regex.search(rel):
            continue
        dest = output_root / "openai_platform_docs" / Path(rel) / "index.md"

        if dest.exists() and not force:
            results.append(
                PageResult(url=url, relative_path=rel, output_path=dest, status="skipped")
            )
            continue

        logging.debug("Fetching %s", url)

        def _task():
            return robust_get_markdown_for_url(url, timeout=timeout)

        try:
            md = retry(_task, attempts=attempts, backoff=backoff)
        except Exception as e:
            results.append(
                PageResult(url=url, relative_path=rel, output_path=dest, status="failed", error=str(e))
            )
            continue

        status, nbytes, digest = write_text(dest, md, force=force)
        results.append(
            PageResult(
                url=url,
                relative_path=rel,
                output_path=dest,
                status=status,
                bytes_written=nbytes,
                sha256=digest,
            )
        )
    return results


def download_text_bundles(
    *,
    output_root: Path,
    include_bundles: bool,
    include_openapi: bool,
    timeout: float,
    attempts: int,
    backoff: float,
) -> list[BundleResult]:
    import requests

    results: list[BundleResult] = []
    if not include_bundles and not include_openapi:
        return results

    # Fetch llms index
    if include_bundles:
        logging.info("Fetching LLM bundles index: %s", LLMS_INDEX_URL)
        idx = requests.get(LLMS_INDEX_URL, timeout=timeout, headers=TXT_HEADERS)
        idx.raise_for_status()
        index_map = parse_llms_index(idx.text)
        # Choose a set of bundles that generally cover the docs
        desired = [
            "llms-api-reference.txt",
            "llms-guides.txt",
            "llms-models-pricing.txt",
            "llms-full.txt",
        ]
        for fname in desired:
            url = urljoin(LLMS_BASE_URL, fname)
            path = output_root / "txt" / fname
            try:
                def _dl():
                    r = requests.get(url, timeout=timeout, headers=TXT_HEADERS)
                    r.raise_for_status()
                    return r.text

                text = retry(_dl, attempts=attempts, backoff=backoff)
                status, nbytes, digest = write_text(path, text, force=False)
                results.append(
                    BundleResult(
                        key=fname,
                        url=url,
                        output_path=path,
                        status=status,
                        bytes_written=nbytes,
                        sha256=digest,
                    )
                )
            except Exception as e:
                results.append(
                    BundleResult(
                        key=fname,
                        url=url,
                        output_path=path,
                        status="failed",
                        error=str(e),
                    )
                )

    if include_openapi:
        url = OPENAPI_SPEC_URL
        path = output_root / "openapi" / "openapi.documented.yml"
        try:
            def _dl():
                r = requests.get(url, timeout=timeout, headers=TXT_HEADERS)
                r.raise_for_status()
                return r.text

            text = retry(_dl, attempts=attempts, backoff=backoff)
            status, nbytes, digest = write_text(path, text, force=False)
            results.append(
                BundleResult(
                    key="openapi.documented.yml",
                    url=url,
                    output_path=path,
                    status=status,
                    bytes_written=nbytes,
                    sha256=digest,
                )
            )
        except Exception as e:
            results.append(
                BundleResult(
                    key="openapi.documented.yml",
                    url=url,
                    output_path=path,
                    status="failed",
                    error=str(e),
                )
            )

    return results


def write_manifest(
    *,
    output_root: Path,
    pages: Sequence[PageResult],
    bundles: Sequence[BundleResult],
    sections: Sequence[str],
) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sitemap_url": SITEMAP_URL,
        "sections": list(sections),
        "pages": [
            {
                "url": p.url,
                "relative_path": p.relative_path,
                "relative_file": str(p.output_path.resolve().relative_to(output_root.resolve())),
                "status": p.status,
                **({"bytes": p.bytes_written} if p.bytes_written else {}),
                **({"sha256": p.sha256} if p.sha256 else {}),
                **({"error": p.error} if p.error else {}),
            }
            for p in pages
        ],
        "bundles": [
            {
                "key": b.key,
                "url": b.url,
                "relative_file": str(b.output_path.resolve().relative_to(output_root.resolve())),
                "status": b.status,
                **({"bytes": b.bytes_written} if b.bytes_written else {}),
                **({"sha256": b.sha256} if b.sha256 else {}),
                **({"error": b.error} if b.error else {}),
            }
            for b in bundles
        ],
    }
    dest = output_root / "manifest.json"
    ensure_dir(dest.parent)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mirror OpenAI docs (robust)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write files (default: %(default)s)",
    )
    p.add_argument(
        "--sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="Docs sections to mirror from sitemap (e.g., guides api-reference models)",
    )
    p.add_argument(
        "--no-llm-bundles",
        dest="include_bundles",
        action="store_false",
        help="Skip downloading cdn.openai.com LLM .txt bundles",
    )
    p.add_argument(
        "--no-openapi",
        dest="include_openapi",
        action="store_false",
        help="Skip downloading the documented OpenAPI spec",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    p.add_argument("--attempts", type=int, default=3, help="Retry attempts per request")
    p.add_argument("--backoff", type=float, default=1.5, help="Exponential backoff base seconds")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex to filter which docs relative paths to mirror (e.g., 'guides/text-generation')",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    output_root = args.output_dir.resolve()

    # Mirror pages first
    logging.info("Mirroring platform docs pages: %s", ", ".join(args.sections))
    page_results = mirror_pages(
        output_root=output_root,
        sections=args.sections,
        force=args.force,
        timeout=args.timeout,
        attempts=args.attempts,
        backoff=args.backoff,
        path_filter=args.filter,
    )
    ok_pages = sum(1 for r in page_results if r.status in {"written", "skipped"})
    logging.info("Pages: %d ok, %d failed", ok_pages, len(page_results) - ok_pages)

    # Download text bundles and spec (optional)
    bundle_results = download_text_bundles(
        output_root=output_root,
        include_bundles=args.include_bundles,
        include_openapi=args.include_openapi,
        timeout=args.timeout,
        attempts=args.attempts,
        backoff=args.backoff,
    )
    ok_bundles = sum(1 for r in bundle_results if r.status in {"written", "skipped"})
    logging.info("Bundles: %d ok, %d failed", ok_bundles, len(bundle_results) - ok_bundles)

    manifest_path = write_manifest(
        output_root=output_root,
        pages=page_results,
        bundles=bundle_results,
        sections=args.sections,
    )
    logging.info("Wrote manifest to %s", manifest_path)

    failures = any(r.status == "failed" for r in page_results + bundle_results)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
