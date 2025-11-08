#!/usr/bin/env python3
"""Mirror the OpenAI API reference documentation using LLM-friendly exports.

This script mirrors the OpenAI developer site's API reference content without
scraping the interactive frontend (which is protected by Cloudflare). It relies
on the canonical LLM-oriented `.txt` bundles published at
`https://cdn.openai.com/API/docs/txt/` as well as the official documented
OpenAPI specification from Stainless.

Key behaviors:
  - Fetch `llms.txt` to discover available text bundles.
  - Download one or more selected bundles (default: `llms-api-reference.txt`).
  - Optionally mirror the documented OpenAPI spec (`openapi.documented.yml`).
  - Emit a manifest capturing the mirrored assets, hashes, and timestamps.

Usage (from repo root):
  python src/DATA_SCRIPTS/mirror_openai_api_docs.py

To include additional bundles (e.g., guides) add `--subset guides` or specify
`--subset` multiple times. Use `--no-openapi` to skip the Stainless spec.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import random
import re
import time
import contextlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "OPENAI_API_DOCS_MIRROR"

LLMS_BASE_URL = "https://cdn.openai.com/API/docs/txt/"
LLMS_INDEX_URL = urljoin(LLMS_BASE_URL, "llms.txt")
OPENAPI_SPEC_URL = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml"

DEFAULT_HEADERS = {
    "User-Agent": "prompts-repo-openai-api-docs-downloader/1.0 (+https://github.com/)",
    "Accept": "text/plain, text/markdown, application/json;q=0.9, */*;q=0.8",
}

OPENAI_PYTHON_API_DOC_URL = (
    "https://raw.githubusercontent.com/openai/openai-python/refs/heads/main/api.md"
)
OPENAI_PYTHON_DIRNAME = "openai_python_api"
OPENAI_PYTHON_RAW_SUBDIR = "_source"
OPENAI_PYTHON_INDEX_FILENAME = "index.md"
PLATFORM_SITEMAP_URL = "https://platform.openai.com/sitemap.xml"
OPENAI_PLATFORM_DOCS_DIRNAME = "openai_platform_docs"
PLATFORM_DOC_INDEX_FILENAME = "index.md"
DEFAULT_PLATFORM_SECTIONS = ("guides",)
PLAYWRIGHT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
PLAYWRIGHT_BROWSER_ARGS = ["--disable-blink-features=AutomationControlled", "--window-size=1280,720"]

SUBSET_CHOICES = {
    "api-reference": "llms-api-reference.txt",
    "guides": "llms-guides.txt",
    "models-pricing": "llms-models-pricing.txt",
    "full": "llms-full.txt",
}

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504, 522, 524}


@dataclass(frozen=True)
class DownloadTarget:
    key: str
    description: str
    url: str
    output_path: Path


@dataclass(frozen=True)
class DownloadResult:
    target: DownloadTarget
    status: str
    bytes_written: int = 0
    sha256: str | None = None
    error: str | None = None

    def to_manifest_entry(self, output_root: Path) -> dict[str, object]:
        relative_path = str(self.target.output_path.resolve().relative_to(output_root.resolve()))
        payload: dict[str, object] = {
            "key": self.target.key,
            "description": self.target.description,
            "source_url": self.target.url,
            "relative_path": relative_path,
            "status": self.status,
        }
        if self.bytes_written:
            payload["bytes_written"] = self.bytes_written
        if self.sha256:
            payload["sha256"] = self.sha256
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class MarkdownSection:
    title: str
    level: int
    content: list[str] = field(default_factory=list)
    children: list["MarkdownSection"] = field(default_factory=list)
    slug: str | None = None


@dataclass(frozen=True)
class GeneratedDocResult:
    title: str
    level: int
    slug_path: Tuple[str, ...]
    output_path: Path
    status: str
    bytes_written: int
    sha256: str | None

    def to_manifest_entry(self, output_root: Path) -> dict[str, object]:
        relative_path = str(self.output_path.resolve().relative_to(output_root.resolve()))
        return {
            "title": self.title,
            "level": self.level,
            "slug_path": list(self.slug_path),
            "relative_path": relative_path,
            "status": self.status,
            "bytes_written": self.bytes_written,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OpenaiPythonMirrorResult:
    status: str
    raw_path: Path
    raw_bytes: int
    raw_sha256: str | None
    documents: list[GeneratedDocResult]
    error: str | None = None

    def to_manifest_block(self, output_root: Path) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_url": OPENAI_PYTHON_API_DOC_URL,
            "raw_relative_path": str(self.raw_path.resolve().relative_to(output_root.resolve())),
            "raw_status": self.status,
            "raw_bytes": self.raw_bytes,
        }
        if self.raw_sha256:
            payload["raw_sha256"] = self.raw_sha256
        if self.error:
            payload["error"] = self.error
        payload["documents"] = [
            doc.to_manifest_entry(output_root) for doc in self.documents
        ]
        return payload


@dataclass(frozen=True)
class PlatformDocResult:
    url: str
    relative_url: str
    output_path: Path
    status: str
    bytes_written: int = 0
    sha256: str | None = None
    error: str | None = None

    def to_manifest_entry(self, output_root: Path) -> dict[str, object]:
        payload: dict[str, object] = {
            "url": self.url,
            "relative_url": self.relative_url,
            "relative_path": str(self.output_path.resolve().relative_to(output_root.resolve())),
            "status": self.status,
        }
        if self.bytes_written:
            payload["bytes_written"] = self.bytes_written
        if self.sha256:
            payload["sha256"] = self.sha256
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class PlatformGuidesMirrorResult:
    status: str
    sitemap_url: str
    documents: list[PlatformDocResult]
    errors: list[str] = field(default_factory=list)

    def to_manifest_block(self, output_root: Path) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "sitemap_url": self.sitemap_url,
            "documents": [
                doc.to_manifest_entry(output_root) for doc in self.documents
            ],
        }
        if self.errors:
            payload["errors"] = list(self.errors)
        return payload


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(message)s", level=level)


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_attempts: int,
    backoff_factor: float,
    retry_status_codes: Sequence[int] | set[int] = RETRYABLE_STATUS_CODES,
) -> bytes:
    attempts = max(1, max_attempts)
    retry_codes = set(retry_status_codes)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as resp:  # nosec: required for docs mirroring
                return resp.read()
        except HTTPError as err:
            last_error = err
            should_retry = err.code in retry_codes and attempt < attempts
            if not should_retry:
                raise
            logging.warning(
                "HTTP %s for %s (attempt %d/%d); backing off %.1fs",
                err.code,
                url,
                attempt,
                attempts,
                backoff_factor * (2 ** (attempt - 1)),
            )
        except URLError as err:
            last_error = err
            if attempt >= attempts:
                raise
            logging.warning(
                "Network error for %s (attempt %d/%d): %s; backing off %.1fs",
                url,
                attempt,
                attempts,
                err.reason,
                backoff_factor * (2 ** (attempt - 1)),
            )
        sleep_for = backoff_factor * (2 ** (attempt - 1))
        jitter = sleep_for * 0.3
        time.sleep(sleep_for + random.uniform(0.0, jitter))
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts")


def parse_llms_index(text: str) -> dict[str, str]:
    discovered: dict[str, str] = {}
    for match in LINK_PATTERN.finditer(text):
        label = match.group(1).strip()
        href = match.group(2).strip()
        if not href.lower().endswith(".txt"):
            continue
        discovered[href] = label
    return discovered


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_target(
    target: DownloadTarget,
    *,
    force: bool,
    dry_run: bool,
    headers: dict[str, str],
    timeout: float,
    max_attempts: int,
    backoff_factor: float,
) -> DownloadResult:
    destination = target.output_path
    if destination.exists() and not force:
        return DownloadResult(target=target, status="skipped")
    if dry_run:
        return DownloadResult(target=target, status="planned")
    ensure_directory(destination.parent)
    try:
        payload = fetch_bytes(
            target.url,
            headers=headers,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_factor=backoff_factor,
        )
    except (HTTPError, URLError) as err:
        return DownloadResult(target=target, status="failed", error=str(err))
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return DownloadResult(
        target=target,
        status="downloaded",
        bytes_written=len(payload),
        sha256=digest,
    )


def slugify_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return slug.strip("-")


def parse_markdown_sections(markdown_text: str) -> list[MarkdownSection]:
    root = MarkdownSection(title="__root__", level=0)
    stack: list[MarkdownSection] = [root]
    for line in markdown_text.splitlines():
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            node = MarkdownSection(title=title, level=level)
            while stack and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
            continue
        stack[-1].content.append(line)
    return root.children


def assign_section_slugs(sections: Sequence[MarkdownSection]) -> None:
    seen: dict[str, int] = {}
    for section in sections:
        base_slug = slugify_title(section.title) or "section"
        count = seen.get(base_slug, 0)
        section.slug = base_slug if count == 0 else f"{base_slug}-{count + 1}"
        seen[base_slug] = count + 1
        assign_section_slugs(section.children)


def write_text_file(
    destination: Path,
    content: str,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[str, int, str]:
    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if dry_run:
        return "planned", len(encoded), digest
    ensure_directory(destination.parent)
    if destination.exists() and not force:
        existing = destination.read_bytes()
        if existing == encoded:
            return "skipped", len(encoded), digest
    destination.write_bytes(encoded)
    return "generated", len(encoded), digest


def write_section_tree(
    section: MarkdownSection,
    *,
    output_root: Path,
    parent_slugs: Tuple[str, ...],
    dry_run: bool,
    force: bool,
) -> list[GeneratedDocResult]:
    if not section.slug:
        section.slug = slugify_title(section.title) or "section"
    slug_path = (*parent_slugs, section.slug)
    section_dir = output_root.joinpath(*slug_path)
    index_path = section_dir / OPENAI_PYTHON_INDEX_FILENAME
    lines: list[str] = [f"# {section.title}", ""]
    if section.content:
        lines.extend(section.content)
        if section.content and section.content[-1].strip():
            lines.append("")
    if section.children:
        lines.append("## Subsections")
        lines.append("")
        for child in section.children:
            child_slug = child.slug or slugify_title(child.title) or "section"
            lines.append(f"- [{child.title}]({child_slug}/{OPENAI_PYTHON_INDEX_FILENAME})")
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    status, bytes_written, digest = write_text_file(
        index_path,
        content,
        dry_run=dry_run,
        force=force,
    )
    results = [
        GeneratedDocResult(
            title=section.title,
            level=section.level,
            slug_path=slug_path,
            output_path=index_path,
            status=status,
            bytes_written=bytes_written,
            sha256=digest,
        )
    ]
    for child in section.children:
        results.extend(
            write_section_tree(
                child,
                output_root=output_root,
                parent_slugs=slug_path,
                dry_run=dry_run,
                force=force,
            )
        )
    return results


def generate_openai_python_sections(
    sections: Sequence[MarkdownSection],
    *,
    output_root: Path,
    dry_run: bool,
    force: bool,
) -> list[GeneratedDocResult]:
    documents: list[GeneratedDocResult] = []
    for section in sections:
        documents.extend(
            write_section_tree(
                section,
                output_root=output_root,
                parent_slugs=tuple(),
                dry_run=dry_run,
                force=force,
            )
        )
    return documents


def mirror_openai_python_api_docs(
    *,
    output_root: Path,
    dry_run: bool,
    force: bool,
    headers: dict[str, str],
    timeout: float,
    max_attempts: int,
    backoff_factor: float,
) -> OpenaiPythonMirrorResult:
    target_root = output_root / OPENAI_PYTHON_DIRNAME
    raw_path = target_root / OPENAI_PYTHON_RAW_SUBDIR / "api.md"
    try:
        raw_bytes = fetch_bytes(
            OPENAI_PYTHON_API_DOC_URL,
            headers=headers,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_factor=backoff_factor,
        )
    except (HTTPError, URLError) as err:
        return OpenaiPythonMirrorResult(
            status="failed",
            raw_path=raw_path,
            raw_bytes=0,
            raw_sha256=None,
            documents=[],
            error=str(err),
        )
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_status = "planned" if dry_run else "downloaded"
    if not dry_run:
        ensure_directory(raw_path.parent)
        if raw_path.exists() and not force:
            existing = raw_path.read_bytes()
            if existing == raw_bytes:
                raw_status = "skipped"
            else:
                raw_path.write_bytes(raw_bytes)
                raw_status = "downloaded"
        else:
            raw_path.write_bytes(raw_bytes)
            raw_status = "downloaded"
    sections = parse_markdown_sections(raw_text)
    assign_section_slugs(sections)
    documents = generate_openai_python_sections(
        sections,
        output_root=target_root,
        dry_run=dry_run,
        force=force,
    )
    return OpenaiPythonMirrorResult(
        status=raw_status,
        raw_path=raw_path,
        raw_bytes=len(raw_bytes),
        raw_sha256=raw_sha256,
        documents=documents,
    )


def remove_elements(container: BeautifulSoup, selectors: Sequence[str]) -> None:
    for selector in selectors:
        for element in container.select(selector):
            element.decompose()


def normalize_code_blocks(container: BeautifulSoup) -> None:
    for pre in container.find_all("pre"):
        code_tag = pre.find("code")
        if not code_tag:
            text = pre.get_text()
            cleaned = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()
            new_pre = container.new_tag("pre")
            new_code = container.new_tag("code")
            new_code.string = cleaned
            new_pre.append(new_code)
            pre.replace_with(new_pre)
            continue
        for line_number in code_tag.select(".react-syntax-highlighter-line-number"):
            line_number.decompose()
        nested_codes = code_tag.find_all("code")
        for nested in nested_codes:
            nested.unwrap()
        language = None
        for cls in code_tag.get("class", []):
            if cls.startswith("language-"):
                language = cls.split("language-", 1)[1] or None
        code_text = code_tag.get_text()
        code_lines = [line.rstrip() for line in code_text.splitlines()]
        cleaned_text = "\n".join(code_lines).rstrip()
        new_pre = container.new_tag("pre")
        new_code = container.new_tag("code")
        if language:
            new_code["class"] = [f"language-{language}"]
        new_code.string = cleaned_text
        new_pre.append(new_code)
        pre.replace_with(new_pre)


def html_to_markdown(content: BeautifulSoup) -> str:
    remove_elements(
        content,
        selectors=[
            "button",
            "[data-docs-feedback]",
            "[data-docs-pagination]",
            "[data-role='search']",
            "[data-component='DocsFeedback']",
            "[data-component='DocsPagination']",
        ],
    )
    normalize_code_blocks(content)
    markdown = MarkdownConverter(heading_style="ATX").convert(str(content))
    lines = [line.rstrip() for line in markdown.splitlines()]
    trimmed = "\n".join(lines).strip()
    return trimmed + "\n"


def collect_platform_urls(
    *,
    sitemap_xml: str,
    sections: Sequence[str],
) -> list[str]:
    root = ET.fromstring(sitemap_xml)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    prefixes = [f"https://platform.openai.com/docs/{section.strip('/')}/" for section in sections]
    prefixes_set = tuple(prefixes)
    urls: list[str] = []
    for loc in root.findall("sm:url/sm:loc", namespace):
        if loc.text:
            text = loc.text.strip()
            if any(text.startswith(prefix) for prefix in prefixes_set):
                urls.append(text.rstrip("/"))
    return sorted(dict.fromkeys(urls))


def relative_segments_from_url(url: str, category_root: str = "docs") -> Tuple[str, ...]:
    if "://" in url:
        path = url.split("://", 1)[1]
        path = path.split("/", 1)[1] if "/" in path else ""
    else:
        path = url
    path = path.strip("/")
    if not path:
        raise ValueError(f"URL '{url}' has no path segments.")
    segments = tuple(part for part in path.split("/") if part)
    if not segments or segments[0] != category_root:
        raise ValueError(f"URL '{url}' does not start with '/{category_root}'.")
    relative = segments[1:]
    if not relative:
        raise ValueError(f"URL '{url}' has no segments after '/{category_root}'.")
    return relative


def mirror_platform_guides(
    *,
    output_root: Path,
    dry_run: bool,
    force: bool,
    sections: Sequence[str],
    sitemap_url: str,
    render_timeout: float,
) -> PlatformGuidesMirrorResult:
    errors: list[str] = []
    documents: list[PlatformDocResult] = []
    if not sections:
        return PlatformGuidesMirrorResult(
            status="skipped",
            sitemap_url=sitemap_url,
            documents=[],
            errors=["No platform sections specified."],
        )
    try:
        import cloudscraper
    except ImportError as err:  # pragma: no cover - dependency missing in environment
        return PlatformGuidesMirrorResult(
            status="missing-dependency",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"cloudscraper not installed: {err}"],
        )
    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
        sitemap_resp = scraper.get(sitemap_url, timeout=30)
        sitemap_resp.raise_for_status()
        sitemap_xml = sitemap_resp.text
    except Exception as err:  # pragma: no cover - network edge
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"Failed to fetch sitemap: {err}"],
        )
    target_urls = collect_platform_urls(sitemap_xml=sitemap_xml, sections=sections)
    if not target_urls:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"No matching docs found for sections: {', '.join(sections)}"],
        )
    render_timeout_ms = int(max(render_timeout, 1.0) * 1000)
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
    except ImportError as err:  # pragma: no cover
        return PlatformGuidesMirrorResult(
            status="missing-dependency",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"playwright not installed: {err}"],
        )
    with contextlib.ExitStack() as stack:
        playwright = stack.enter_context(sync_playwright())
        browser = playwright.chromium.launch(headless=True, args=PLAYWRIGHT_BROWSER_ARGS)
        stack.enter_context(contextlib.closing(browser))
        context = browser.new_context(user_agent=PLAYWRIGHT_USER_AGENT, viewport={"width": 1280, "height": 720})
        stack.enter_context(contextlib.closing(context))
        page = context.new_page()
        for url in target_urls:
            try:
                segments = relative_segments_from_url(url)
            except ValueError as err:
                errors.append(str(err))
                documents.append(
                    PlatformDocResult(
                        url=url,
                        relative_url="",
                        output_path=output_root,
                        status="failed",
                        error=str(err),
                    )
                )
                continue
            relative_url = "/".join(segments)
            destination = (
                output_root
                / OPENAI_PLATFORM_DOCS_DIRNAME
                / Path(*segments)
                / PLATFORM_DOC_INDEX_FILENAME
            )
            if destination.exists() and not force:
                documents.append(
                    PlatformDocResult(
                        url=url,
                        relative_url=relative_url,
                        output_path=destination,
                        status="skipped",
                    )
                )
                continue
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=render_timeout_ms)
                with contextlib.suppress(Exception):
                    page.wait_for_load_state("networkidle", timeout=render_timeout_ms)
                page.wait_for_selector(
                    "div.docs-markdown-page.docs-markdown-content",
                    timeout=render_timeout_ms,
                    state="attached",
                )
                html = page.content()
                soup = BeautifulSoup(html, "lxml")
                container = soup.select_one("div.docs-markdown-page.docs-markdown-content")
                if container is None:
                    raise RuntimeError("Unable to locate docs content container.")
                markdown = html_to_markdown(container)
                status_str, bytes_written, digest = write_text_file(
                    destination,
                    markdown,
                    dry_run=dry_run,
                    force=force,
                )
                documents.append(
                    PlatformDocResult(
                        url=url,
                        relative_url=relative_url,
                        output_path=destination,
                        status=status_str,
                        bytes_written=bytes_written,
                        sha256=digest if markdown else None,
                    )
                )
            except PlaywrightTimeoutError as err:  # pragma: no cover - network/render edge
                msg = f"Timeout rendering {url}: {err}"
                errors.append(msg)
                documents.append(
                    PlatformDocResult(
                        url=url,
                        relative_url=relative_url,
                        output_path=destination,
                        status="failed",
                        error=msg,
                    )
                )
            except Exception as err:  # pragma: no cover - safety net
                msg = f"Failed to mirror {url}: {err}"
                errors.append(msg)
                documents.append(
                    PlatformDocResult(
                        url=url,
                        relative_url=relative_url,
                        output_path=destination,
                        status="failed",
                        error=msg,
                    )
                )
    status = "ok"
    if errors and len(errors) == len(documents):
        status = "failed"
    elif errors:
        status = "partial"
    return PlatformGuidesMirrorResult(
        status=status,
        sitemap_url=sitemap_url,
        documents=documents,
        errors=errors,
    )


def write_manifest(
    results: Sequence[DownloadResult],
    *,
    output_root: Path,
    subsets: Sequence[str],
    include_openapi: bool,
    openai_python: OpenaiPythonMirrorResult | None,
    platform_guides: PlatformGuidesMirrorResult | None,
) -> Path:
    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llms_index_url": LLMS_INDEX_URL,
        "selected_subsets": list(subsets),
        "openapi_spec_included": include_openapi,
        "documents": [result.to_manifest_entry(output_root) for result in results],
    }
    if openai_python is not None:
        manifest_payload["openai_python_api"] = openai_python.to_manifest_block(output_root)
    if platform_guides is not None:
        manifest_payload.setdefault("platform_docs", {})
        manifest_payload["platform_docs"]["guides"] = platform_guides.to_manifest_block(output_root)
    manifest_path = output_root / "manifest.json"
    ensure_directory(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_targets(
    *,
    output_root: Path,
    subsets: Sequence[str],
    include_openapi: bool,
    index_map: dict[str, str],
) -> list[DownloadTarget]:
    targets: list[DownloadTarget] = []
    # Always mirror the index for transparency.
    targets.append(
        DownloadTarget(
            key="llms-index",
            description="Listing of available OpenAI LLM-friendly documentation bundles.",
            url=LLMS_INDEX_URL,
            output_path=output_root / "txt" / "llms.txt",
        )
    )

    for subset_key in subsets:
        filename = SUBSET_CHOICES[subset_key]
        if filename not in index_map:
            logging.warning(
                "Subset '%s' (%s) not present in index; proceeding anyway.",
                subset_key,
                filename,
            )
        targets.append(
            DownloadTarget(
                key=f"subset:{subset_key}",
                description=index_map.get(filename, f"OpenAI {subset_key} bundle"),
                url=urljoin(LLMS_BASE_URL, filename),
                output_path=output_root / "txt" / filename,
            )
        )

    if include_openapi:
        targets.append(
            DownloadTarget(
                key="openapi-documented",
                description="OpenAI documented OpenAPI specification (Stainless).",
                url=OPENAPI_SPEC_URL,
                output_path=output_root / "openapi" / "openapi.documented.yml",
            )
        )

    return targets


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror OpenAI API docs via LLM-friendly exports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write mirrored files (default: %(default)s)",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=sorted(SUBSET_CHOICES.keys()),
        help="Subset(s) to download; defaults to api-reference",
    )
    parser.add_argument(
        "--no-openapi",
        dest="include_openapi",
        action="store_false",
        help="Skip downloading the documented OpenAPI spec",
    )
    parser.add_argument(
        "--skip-openai-python",
        dest="include_openai_python",
        action="store_false",
        help="Skip mirroring the openai-python client API markdown",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
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
        default=20.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retry attempts per request (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff between retries (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-platform-guides",
        dest="include_platform_guides",
        action="store_false",
        help="Skip mirroring the platform.openai.com guides with browser rendering",
    )
    parser.add_argument(
        "--platform-section",
        dest="platform_sections",
        action="append",
        help="Platform docs section(s) to mirror (default: guides)",
    )
    parser.add_argument(
        "--platform-render-timeout",
        type=float,
        default=45.0,
        help="Per-page render timeout in seconds when using Playwright (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(
        include_openapi=True,
        include_openai_python=True,
        include_platform_guides=True,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.max_attempts < 1:
        logging.error("--max-attempts must be at least 1")
        return 1
    if args.retry_backoff <= 0:
        logging.error("--retry-backoff must be greater than 0")
        return 1

    subsets = args.subset or ["api-reference"]
    output_root = args.output_dir.resolve()
    platform_sections = args.platform_sections or list(DEFAULT_PLATFORM_SECTIONS)

    logging.info("Fetching OpenAI llms index from %s", LLMS_INDEX_URL)
    try:
        index_bytes = fetch_bytes(
            LLMS_INDEX_URL,
            headers=DEFAULT_HEADERS,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            backoff_factor=args.retry_backoff,
        )
    except (HTTPError, URLError) as err:
        logging.error("Failed to download llms index: %s", err)
        return 1
    index_text = index_bytes.decode("utf-8", errors="replace")
    index_map = parse_llms_index(index_text)

    targets = build_targets(
        output_root=output_root,
        subsets=subsets,
        include_openapi=args.include_openapi,
        index_map=index_map,
    )

    logging.info("Planned %d download targets", len(targets))

    download_results: list[DownloadResult] = []
    summary = Counter()

    if args.max_workers <= 1:
        for target in targets:
            try:
                result = download_target(
                    target,
                    force=args.force,
                    dry_run=args.dry_run,
                    headers=DEFAULT_HEADERS,
                    timeout=args.timeout,
                    max_attempts=args.max_attempts,
                    backoff_factor=args.retry_backoff,
                )
            except Exception as err:  # pragma: no cover - unexpected edge
                result = DownloadResult(target=target, status="failed", error=str(err))
            download_results.append(result)
            summary[result.status] += 1
            if result.error:
                logging.error("%s failed: %s", target.key, result.error)
            else:
                logging.debug("%s -> %s (%s)", target.url, target.output_path, result.status)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(
                    download_target,
                    target,
                    force=args.force,
                    dry_run=args.dry_run,
                    headers=DEFAULT_HEADERS,
                    timeout=args.timeout,
                    max_attempts=args.max_attempts,
                    backoff_factor=args.retry_backoff,
                ): target
                for target in targets
            }
            for future in concurrent.futures.as_completed(future_map):
                target = future_map[future]
                try:
                    result = future.result()
                except Exception as err:  # pragma: no cover - unexpected edge
                    result = DownloadResult(target=target, status="failed", error=str(err))
                download_results.append(result)
                summary[result.status] += 1
                if result.error:
                    logging.error("%s failed: %s", result.target.key, result.error)
                else:
                    logging.debug(
                        "%s -> %s (%s)",
                        result.target.url,
                        result.target.output_path,
                        result.status,
                    )

    openai_python_result: OpenaiPythonMirrorResult | None = None
    openai_python_summary = Counter()
    if args.include_openai_python:
        logging.info(
            "Mirroring openai-python client API docs from %s",
            OPENAI_PYTHON_API_DOC_URL,
        )
        try:
            openai_python_result = mirror_openai_python_api_docs(
                output_root=output_root,
                dry_run=args.dry_run,
                force=args.force,
                headers=DEFAULT_HEADERS,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                backoff_factor=args.retry_backoff,
            )
        except Exception as err:  # pragma: no cover - defensive
            logging.error("Failed to mirror openai-python docs: %s", err)
            openai_python_result = OpenaiPythonMirrorResult(
                status="failed",
                raw_path=(output_root / OPENAI_PYTHON_DIRNAME / OPENAI_PYTHON_RAW_SUBDIR / "api.md"),
                raw_bytes=0,
                raw_sha256=None,
                documents=[],
                error=str(err),
            )
        if openai_python_result.error:
            logging.error("openai-python mirror error: %s", openai_python_result.error)
        openai_python_summary[openai_python_result.status] += 1
        for doc_result in openai_python_result.documents:
            openai_python_summary[doc_result.status] += 1

    platform_guides_result: PlatformGuidesMirrorResult | None = None
    platform_guides_summary = Counter()
    if args.include_platform_guides:
        logging.info(
            "Mirroring OpenAI platform guides via Playwright (sections: %s)",
            ", ".join(platform_sections),
        )
        try:
            platform_guides_result = mirror_platform_guides(
                output_root=output_root,
                dry_run=args.dry_run,
                force=args.force,
                sections=platform_sections,
                sitemap_url=PLATFORM_SITEMAP_URL,
                render_timeout=args.platform_render_timeout,
            )
        except Exception as err:  # pragma: no cover - defensive guard
            logging.error("Failed to mirror platform guides: %s", err)
            platform_guides_result = PlatformGuidesMirrorResult(
                status="failed",
                sitemap_url=PLATFORM_SITEMAP_URL,
                documents=[],
                errors=[str(err)],
            )
        if platform_guides_result.errors:
            for error in platform_guides_result.errors:
                logging.error("platform guide error: %s", error)
        platform_guides_summary[platform_guides_result.status] += 1
        for doc_result in platform_guides_result.documents:
            platform_guides_summary[doc_result.status] += 1

    if not args.dry_run:
        # Ensure the index itself is written even if we skipped due to cache.
        ensure_directory((output_root / "txt").resolve())
        (output_root / "txt" / "llms.txt").write_text(index_text, encoding="utf-8")
        manifest_path = write_manifest(
            download_results,
            output_root=output_root,
            subsets=subsets,
            include_openapi=args.include_openapi,
            openai_python=openai_python_result,
            platform_guides=platform_guides_result,
        )
        logging.info("Wrote manifest to %s", manifest_path)
    else:
        logging.info("Dry run requested; no files written.")

    logging.info(
        "Download summary: %s",
        ", ".join(f"{count} {status}" for status, count in sorted(summary.items())),
    )
    if openai_python_result is not None and openai_python_summary:
        logging.info(
            "openai-python summary: %s",
            ", ".join(
                f"{count} {status}"
                for status, count in sorted(openai_python_summary.items())
            ),
        )
    if platform_guides_result is not None and platform_guides_summary:
        logging.info(
            "platform guides summary: %s",
            ", ".join(
                f"{count} {status}"
                for status, count in sorted(platform_guides_summary.items())
            ),
        )

    exit_code = 0 if summary.get("failed", 0) == 0 else 1
    if openai_python_result and (
        openai_python_result.status == "failed" or openai_python_result.error
    ):
        exit_code = 1
    if platform_guides_result and platform_guides_result.status == "failed":
        exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
