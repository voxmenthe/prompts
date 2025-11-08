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
import html
import json
import logging
import random
import re
import time
import posixpath
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Mapping, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
try:
    from quickjs import Context
except ImportError:  # pragma: no cover - optional dependency
    Context = None  # type: ignore[misc,assignment]


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
SUBSET_CHOICES = {
    "api-reference": "llms-api-reference.txt",
    "guides": "llms-guides.txt",
    "models-pricing": "llms-models-pricing.txt",
    "full": "llms-full.txt",
}

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504, 522, 524}
HTML_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
HTML_STANDARD_TAGS = {
    "a",
    "abbr",
    "article",
    "aside",
    "b",
    "blockquote",
    "code",
    "div",
    "em",
    "figure",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "span",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}


def split_top_level_declarations(statement: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    in_string: str | None = None
    escaped = False
    for char in statement:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char in ('"', "'", "`"):
                in_string = char
            elif char in "({[":
                depth += 1
            elif char in ")}]":
                if depth > 0:
                    depth -= 1
            elif char == "," and depth == 0:
                tokens.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    if current:
        tokens.append("".join(current).strip())
    return tokens


def build_literal_map(source: str) -> dict[str, str]:
    literal_map: dict[str, str] = {}
    declaration_pattern = re.compile(
        r"(?:^|[;{}\n\r])\s*(?:export\s+default\s+)?(?:export\s+)?(const|let|var)\s+([^;]*);",
        re.DOTALL,
    )
    for match in declaration_pattern.finditer(source):
        declarations = match.group(2)
        for segment in split_top_level_declarations(declarations):
            if "=" not in segment:
                continue
            name_part, value_part = segment.split("=", 1)
            name = name_part.strip()
            if not name:
                continue
            literal = value_part.strip()
            if literal:
                literal_map[name] = literal
    return literal_map


def build_function_map(source: str) -> dict[str, str]:
    function_map: dict[str, str] = {}
    for match in re.finditer(r"(?:^|[;{}\n\r])\s*function\s+([A-Za-z0-9_$]+)\s*\(", source):
        name = match.group(1)
        try:
            function_map[name] = extract_function_code(source, name)
        except (KeyError, ValueError):
            continue
    return function_map


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
    chunk_name: str | None = None
    wrapper_function: str | None = None
    render_function: str | None = None

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
        if self.chunk_name:
            payload["chunk_name"] = self.chunk_name
        if self.wrapper_function:
            payload["wrapper_function"] = self.wrapper_function
        if self.render_function:
            payload["render_function"] = self.render_function
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


@dataclass(frozen=True)
class ImportBinding:
    source: str
    remote_name: str
    kind: Literal["named", "default", "namespace"]


@dataclass(frozen=True)
class ReExportBinding:
    export_name: str
    remote_name: str
    source: str


@dataclass
class ModuleInfo:
    path: str
    text: str
    literals: dict[str, str]
    functions: dict[str, str]
    imports: dict[str, ImportBinding]
    export_map: dict[str, str]
    reexports: list[ReExportBinding]
    default_export: str | None = None


@dataclass(frozen=True)
class ResolvedSymbol:
    kind: Literal["literal", "function"]
    code: str
    name: str
    module_path: str


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


MODULE_SCRIPT_PATTERN = re.compile(
    r'<script[^>]+type="module"[^>]+src="(?P<src>[^"]+index-[^"]+\.js)"',
    re.IGNORECASE,
)
DOCS_CHUNK_PATTERN = re.compile(
    r'=s\.lazy\(\(\)=>E1\(\(\)=>import\("./(?P<chunk>[A-Za-z0-9_\-]+)\.js"\)\.then\(e=>e\.a\)',
)
PATH_WRAPPER_PATTERN = re.compile(
    r's\(f,\{path:"(?P<path>/docs/[^"]+)",children:s\(y,\{[^}]*?children:s\((?P<wrapper>[A-Za-z0-9_$]+),\{',
    re.DOTALL,
)
WRAPPER_RENDER_PATTERN = re.compile(
    r'function\s+(?P<wrapper>[A-Za-z0-9_$]+)\(n=\{\}\)\{const\{wrapper:t\}=\{[^}]*\};'
    r'return t\?e\.jsx\(t,\{\.{3}n,children:e\.jsx\((?P<render>[A-Za-z0-9_$]+),\{\.{3}n\}\)\}\):(?P=render)\(n\)\}',
    re.DOTALL,
)
IMPORT_DEFAULT_NAMED_PATTERN = re.compile(
    r'import\s+([A-Za-z0-9_$]+)\s*,\s*\{([^}]*)\}\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
IMPORT_NAMED_PATTERN = re.compile(
    r'import\s*\{([^}]*)\}\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
IMPORT_DEFAULT_PATTERN = re.compile(
    r'import\s+([A-Za-z0-9_$]+)\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
IMPORT_NAMESPACE_PATTERN = re.compile(
    r'import\s*\*\s*as\s*([A-Za-z0-9_$]+)\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
EXPORT_FROM_PATTERN = re.compile(
    r'export\s*\{([^}]*)\}\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
EXPORT_NAMED_PATTERN = re.compile(
    r'export\s*\{([^}]*)\}(?!\s*from)',
    re.DOTALL,
)
EXPORT_STAR_PATTERN = re.compile(
    r'export\s*\*\s*from\s*["\']([^"\']+)["\']',
    re.DOTALL,
)
EXPORT_DECL_PATTERN = re.compile(
    r'export\s+(?:const|let|var|function|class)\s+([A-Za-z0-9_$]+)',
    re.DOTALL,
)
EXPORT_DEFAULT_PATTERN = re.compile(
    r'export\s+default\s+([A-Za-z0-9_$]+)',
    re.DOTALL,
)


def normalize_static_module_path(path: str, *, base: str | None = None) -> str:
    if path.startswith(("http://", "https://")):
        return path
    base_path = ""
    if base:
        base_path = base
        if base_path.startswith(("http://", "https://")):
            base_path = base_path.split("://", 1)[1]
            base_path = base_path.split("/", 1)[1] if "/" in base_path else ""
    base_components: list[str] = []
    if base_path:
        trimmed = base_path.lstrip("/")
        base_components = [part for part in trimmed.split("/") if part]
        if base_components and base_components[-1].endswith(".js"):
            base_components = base_components[:-1]
    components: list[str] = base_components.copy()
    for raw_part in path.split("/"):
        if not raw_part or raw_part == ".":
            continue
        if raw_part == "..":
            if components:
                components.pop()
            continue
        components.append(raw_part)
    if not components or components[0] != "static":
        components.insert(0, "static")
    return "/".join(components)


def parse_named_specifiers(spec_text: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for raw in spec_text.split(","):
        part = raw.strip()
        if not part:
            continue
        match = re.match(r"(.+?)\s+as\s+(.+)", part)
        if match:
            original = match.group(1).strip()
            alias = match.group(2).strip()
        else:
            original = part
            alias = part
        results.append((original, alias))
    return results


def parse_import_map(source: str, *, current_module: str) -> dict[str, ImportBinding]:
    imports: dict[str, ImportBinding] = {}
    for match in IMPORT_DEFAULT_NAMED_PATTERN.finditer(source):
        default_local, spec_text, module_path = match.groups()
        normalized = normalize_static_module_path(module_path, base=current_module)
        default_name = default_local.strip()
        if default_name:
            imports[default_name] = ImportBinding(
                source=normalized,
                remote_name="default",
                kind="default",
            )
        for original, alias in parse_named_specifiers(spec_text):
            local_name = alias.strip() or original.strip()
            remote_name = original.strip()
            imports[local_name] = ImportBinding(
                source=normalized,
                remote_name=remote_name,
                kind="named",
            )
    for match in IMPORT_NAMED_PATTERN.finditer(source):
        spec_text, module_path = match.groups()
        normalized = normalize_static_module_path(module_path, base=current_module)
        for original, alias in parse_named_specifiers(spec_text):
            local_name = alias.strip() or original.strip()
            remote_name = original.strip()
            if local_name in imports:
                continue
            imports[local_name] = ImportBinding(
                source=normalized,
                remote_name=remote_name,
                kind="named",
            )
    for match in IMPORT_DEFAULT_PATTERN.finditer(source):
        default_local, module_path = match.groups()
        normalized = normalize_static_module_path(module_path, base=current_module)
        default_name = default_local.strip()
        if default_name and default_name not in imports:
            imports[default_name] = ImportBinding(
                source=normalized,
                remote_name="default",
                kind="default",
            )
    for match in IMPORT_NAMESPACE_PATTERN.finditer(source):
        namespace_local, module_path = match.groups()
        normalized = normalize_static_module_path(module_path, base=current_module)
        ns_name = namespace_local.strip()
        imports[ns_name] = ImportBinding(
            source=normalized,
            remote_name="*",
            kind="namespace",
        )
    return imports


def parse_export_specifiers(
    source: str,
    *,
    current_module: str,
) -> tuple[dict[str, str], list[ReExportBinding], str | None]:
    export_map: dict[str, str] = {}
    reexports: list[ReExportBinding] = []
    default_export: str | None = None

    for match in EXPORT_FROM_PATTERN.finditer(source):
        spec_text, module_path = match.groups()
        normalized = normalize_static_module_path(module_path, base=current_module)
        for original, alias in parse_named_specifiers(spec_text):
            export_name = alias.strip() or original.strip()
            remote_name = original.strip()
            reexports.append(
                ReExportBinding(
                    export_name=export_name,
                    remote_name=remote_name,
                    source=normalized,
                )
            )
    for match in EXPORT_STAR_PATTERN.finditer(source):
        module_path = match.group(1)
        normalized = normalize_static_module_path(module_path, base=current_module)
        reexports.append(
            ReExportBinding(
                export_name="*",
                remote_name="*",
                source=normalized,
            )
        )
    for match in EXPORT_NAMED_PATTERN.finditer(source):
        spec_text = match.group(1)
        for original, alias in parse_named_specifiers(spec_text):
            export_name = alias.strip() or original.strip()
            local_name = original.strip()
            export_map.setdefault(export_name, local_name)
    for match in EXPORT_DECL_PATTERN.finditer(source):
        name = match.group(1).strip()
        export_map.setdefault(name, name)
    default_match = EXPORT_DEFAULT_PATTERN.search(source)
    if default_match:
        candidate = default_match.group(1).strip()
        if candidate:
            default_export = candidate
    return export_map, reexports, default_export


def extract_entry_module_sources(html_text: str) -> list[str]:
    return [match.group("src") for match in MODULE_SCRIPT_PATTERN.finditer(html_text)]


def resolve_platform_static_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    normalized = path.lstrip("/")
    return f"https://platform.openai.com/{normalized}"


def extract_docs_chunk_name(entry_chunk_text: str) -> str:
    match = DOCS_CHUNK_PATTERN.search(entry_chunk_text)
    if not match:
        raise ValueError("Unable to locate docs chunk import in entry bundle.")
    return match.group("chunk")


def extract_function_code(source: str, function_name: str) -> str:
    pattern = f"function {function_name}"
    start = source.find(pattern)
    if start == -1:
        raise KeyError(f"Function '{function_name}' not found in source chunk.")
    index = start + len(pattern)
    paren_depth = 0
    in_string: str | None = None
    escaped = False
    while index < len(source):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char in ('"', "'", "`"):
                in_string = char
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                if paren_depth > 0:
                    paren_depth -= 1
            elif char == "{" and paren_depth == 0:
                break
        index += 1
    if index >= len(source):
        raise ValueError(f"Function '{function_name}' missing body.")
    depth = 0
    in_string: str | None = None
    escaped = False
    end = index
    while end < len(source):
        char = source[end]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char in ('"', "'", "`"):
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
        end += 1
    if depth != 0:
        raise ValueError(f"Unbalanced braces while extracting '{function_name}'.")
    return source[start:end]


def extract_path_wrapper_map(chunk_text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for match in PATH_WRAPPER_PATTERN.finditer(chunk_text):
        path = match.group("path")
        wrapper = match.group("wrapper")
        mapping[path] = wrapper
    return mapping


def extract_wrapper_render_map(chunk_text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for match in WRAPPER_RENDER_PATTERN.finditer(chunk_text):
        wrapper = match.group("wrapper")
        render = match.group("render")
        mapping[wrapper] = render
    return mapping


def collect_component_identifiers(function_source: str) -> set[str]:
    identifiers: set[str] = set()
    for match in re.finditer(r'[a-z]\.jsx[s]?\(([^,]+),', function_source):
        argument = match.group(1).strip()
        if argument.startswith(("t.", "e.", "s.", "p.")):
            continue
        if argument.startswith(('"', "'", "`")):
            continue
        if "(" in argument:  # ignore nested expressions
            continue
        identifiers.add(argument)
    return identifiers


def collect_constant_identifiers(function_source: str) -> set[str]:
    identifiers: set[str] = set()
    for match in re.finditer(r'(?:(?:code|markdown|example|value|data|table|schema)\s*:\s*)([A-Za-z0-9_$]+)', function_source):
        identifiers.add(match.group(1))
    return identifiers


def collect_property_identifier_candidates(function_source: str) -> set[str]:
    reserved = {
        "e",
        "t",
        "n",
        "s",
        "p",
        "r",
        "o",
        "a",
        "l",
        "c",
        "d",
        "i",
        "u",
        "m",
        "h",
        "x",
        "y",
        "g",
        "f",
        "b",
        "k",
        "w",
        "j",
        "q",
        "z",
        "Math",
        "JSON",
        "Number",
        "String",
        "Object",
        "Array",
    }
    candidates: set[str] = set()
    for match in re.finditer(r'\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\.', function_source):
        name = match.group(1)
        if name in reserved:
            continue
        if len(name) == 1:
            continue
        if "_" not in name and not any(ch.isupper() for ch in name):
            continue
        candidates.add(name)
    return candidates


def extract_constant_value(chunk_text: str, constant_name: str) -> str | None:
    literal_map = build_literal_map(chunk_text)
    if constant_name in literal_map:
        return literal_map[constant_name]
    stmt_pattern = re.compile(
        r"(?:^|[;{}\n\r])\s*(?:export\s+default\s+)?(?:export\s+)?(const|let|var)\s+[^;]*\b"
        + re.escape(constant_name)
        + r"\b[^;]*;",
        re.DOTALL,
    )
    statement_match = stmt_pattern.search(chunk_text)
    if not statement_match:
        return None
    statement = statement_match.group(0)
    assign_pattern = re.compile(rf"\b{re.escape(constant_name)}\b\s*=")
    assignment_match = assign_pattern.search(statement)
    if not assignment_match:
        return None
    index = assignment_match.end()
    while index < len(statement) and statement[index].isspace():
        index += 1
    depth = 0
    in_string: str | None = None
    escaped = False
    end = index
    while end < len(statement):
        char = statement[end]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char in ('"', "'", "`"):
                in_string = char
            elif char in "({[":
                depth += 1
            elif char in ")}]":
                if depth > 0:
                    depth -= 1
            elif char in (",", ";") and depth == 0:
                break
        end += 1
    literal = statement[index:end].strip()
    return literal or None


def evaluate_wrapper_to_tree(
    *,
    wrapper_name: str,
    wrapper_code: str,
    render_code: str,
    dependency_codes: Sequence[str],
    stub_names: Sequence[str],
    constant_values: Mapping[str, str],
) -> dict:
    if Context is None:  # pragma: no cover - runtime fallback
        raise RuntimeError(
            "quickjs package not available; install `quickjs` to evaluate platform docs renderers."
        )
    ctx = Context()
    script_parts: list[str] = [
        "const __HTML_FRAGMENT__ = 'fragment';",
        "function __deepClone__(value) {",
        "  if (Array.isArray(value)) { return value.map(__deepClone__); }",
        "  if (value && typeof value === 'object') {",
        "    const out = {};",
        "    for (const key of Object.keys(value)) { out[key] = __deepClone__(value[key]); }",
        "    return out;",
        "  }",
        "  return value;",
        "}",
        "function __wrap__(type, props) {",
        "  const nodeType = typeof type === 'function'",
        "    ? (type.__componentName || type.name || 'component')",
        "    : String(type);",
        "  const out = { type: nodeType, props: {} };",
        "  if (props) {",
        "    for (const key of Object.keys(props)) {",
        "      if (key === 'children') {",
        "        out.props.children = props.children;",
        "      } else {",
        "        out.props[key] = props[key];",
        "      }",
        "    }",
        "  }",
        "  return out;",
        "}",
        "function __createStub__(name) {",
        "  const fn = function(props) {",
        "    return __wrap__(name, props || {});",
        "  };",
        "  fn.__componentName = name;",
        "  return fn;",
        "}",
        "const e = {",
        "  Fragment: __HTML_FRAGMENT__,",
        "  jsx: (type, props) => __wrap__(type, props || {}),",
        "  jsxs: (type, props) => __wrap__(type, props || {}),",
        "};",
        "const s = e; const p = e;",
        "function r(){ return {}; }",
    ]
    for name, literal in constant_values.items():
        script_parts.append(f"const {name} = {literal};")
    for stub in stub_names:
        script_parts.append(f"const {stub} = __createStub__('{stub}');")
    script_parts.extend(dependency_codes)
    script_parts.append(render_code)
    script_parts.append(wrapper_code)
    script_parts.append(
        f"const __doc_tree__ = JSON.stringify(__deepClone__({wrapper_name}({{components: {{}}}})));"
    )
    ctx.eval("\n".join(script_parts))
    tree_json = ctx.eval("__doc_tree__")
    return json.loads(tree_json)


def _normalize_children(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def render_tree_to_html(node: object) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return html.escape(node)
    if isinstance(node, (int, float)):
        return html.escape(str(node))
    if isinstance(node, list):
        return "".join(render_tree_to_html(child) for child in node)
    if not isinstance(node, dict):
        return ""
    node_type = str(node.get("type") or "")
    props = node.get("props", {}) or {}
    children = props.pop("children", None)
    attributes: list[str] = []
    for key, value in props.items():
        if value in (None, False):
            continue
        attr_name = "class" if key == "className" else key
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        attributes.append(f' {attr_name}="{html.escape(str(value))}"')
    attr_str = "".join(attributes)
    if node_type.lower() in {"fragment", "react.fragment"}:
        return "".join(render_tree_to_html(child) for child in _normalize_children(children))
    if node_type in HTML_VOID_TAGS:
        return f"<{node_type}{attr_str}/>"
    child_html = "".join(render_tree_to_html(child) for child in _normalize_children(children))
    if node_type not in HTML_STANDARD_TAGS:
        return f"<div data-component=\"{html.escape(node_type)}\"{attr_str}>{child_html}</div>"
    return f"<{node_type}{attr_str}>{child_html}</{node_type}>"


def tree_to_markdown(node: object) -> str:
    html_output = render_tree_to_html(node)
    soup = BeautifulSoup(html_output, "lxml")
    return html_to_markdown(soup)




def mirror_platform_guides(
    *,
    output_root: Path,
    dry_run: bool,
    force: bool,
    sections: Sequence[str],
    sitemap_url: str,
) -> PlatformGuidesMirrorResult:
    errors: list[str] = []
    documents: list[PlatformDocResult] = []
    module_text_cache: dict[str, str] = {}
    module_info_cache: dict[str, ModuleInfo] = {}

    def get_module_text(path: str, scraper_obj) -> str:
        normalized = path
        if normalized not in module_text_cache:
            module_text_cache[normalized] = scraper_obj.get(
                resolve_platform_static_url(normalized), timeout=30
            ).text
        return module_text_cache[normalized]

    def load_module_info(
        path: str,
        *,
        scraper_obj,
        text_override: str | None = None,
        base: str | None = None,
    ) -> ModuleInfo:
        normalized = normalize_static_module_path(path, base=base)
        if normalized in module_info_cache:
            return module_info_cache[normalized]
        text = text_override if text_override is not None else get_module_text(normalized, scraper_obj)
        literals = build_literal_map(text)
        functions = build_function_map(text)
        imports = parse_import_map(text, current_module=normalized)
        export_map, reexports, default_export = parse_export_specifiers(
            text,
            current_module=normalized,
        )
        info = ModuleInfo(
            path=normalized,
            text=text,
            literals=literals,
            functions=functions,
            imports=imports,
            export_map=export_map,
            reexports=reexports,
            default_export=default_export,
        )
        module_info_cache[normalized] = info
        return info

    def resolve_identifier_value(
        module_path: str,
        identifier: str,
        visited_local: set[tuple[str, str]],
        visited_export: set[tuple[str, str]],
    ) -> ResolvedSymbol | None:
        normalized_module = normalize_static_module_path(module_path)
        key = (normalized_module, identifier)
        if key in visited_local:
            return None
        visited_local.add(key)
        try:
            info = load_module_info(normalized_module, scraper_obj=scraper)
        except Exception:
            return None
        if identifier in info.literals:
            return ResolvedSymbol(
                kind="literal",
                code=info.literals[identifier],
                name=identifier,
                module_path=normalized_module,
            )
        if identifier in info.functions:
            return ResolvedSymbol(
                kind="function",
                code=info.functions[identifier],
                name=identifier,
                module_path=normalized_module,
            )
        binding = info.imports.get(identifier)
        if binding and binding.kind != "namespace":
            return resolve_export(binding.source, binding.remote_name, visited_local, visited_export)
        return None

    def resolve_export(
        module_path: str,
        export_name: str,
        visited_local: set[tuple[str, str]],
        visited_export: set[tuple[str, str]],
    ) -> ResolvedSymbol | None:
        normalized_module = normalize_static_module_path(module_path)
        key = (normalized_module, export_name)
        if key in visited_export:
            return None
        visited_export.add(key)
        try:
            info = load_module_info(normalized_module, scraper_obj=scraper)
        except Exception:
            return None
        if export_name == "default" and info.default_export:
            result = resolve_identifier_value(
                normalized_module,
                info.default_export,
                visited_local,
                visited_export,
            )
            if result:
                return result
        if export_name in info.export_map:
            local_name = info.export_map[export_name]
            result = resolve_identifier_value(
                normalized_module,
                local_name,
                visited_local,
                visited_export,
            )
            if result:
                return result
        for binding in info.reexports:
            if binding.export_name == export_name:
                result = resolve_export(
                    binding.source,
                    binding.remote_name,
                    visited_local,
                    visited_export,
                )
                if result:
                    return result
        for binding in info.reexports:
            if binding.export_name == "*":
                remote_target = export_name if binding.remote_name == "*" else binding.remote_name
                result = resolve_export(
                    binding.source,
                    remote_target,
                    visited_local,
                    visited_export,
                )
                if result:
                    return result
        if export_name != "default":
            return resolve_identifier_value(normalized_module, export_name, visited_local, visited_export)
        return None

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
    try:
        docs_resp = scraper.get("https://platform.openai.com/docs", timeout=30)
        docs_resp.raise_for_status()
        docs_html = docs_resp.text
    except Exception as err:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"Failed to fetch docs landing page: {err}"],
        )

    module_sources = extract_entry_module_sources(docs_html)
    if not module_sources:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=["Unable to locate entry module script in docs HTML."],
        )
    entry_src = resolve_platform_static_url(module_sources[0])
    try:
        entry_chunk_text = scraper.get(entry_src, timeout=30).text
    except Exception as err:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"Failed to fetch entry module: {err}"],
        )
    try:
        docs_chunk_name = extract_docs_chunk_name(entry_chunk_text)
    except Exception as err:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"Failed to locate docs chunk name: {err}"],
        )
    if docs_chunk_name.startswith("static/"):
        chunk_asset = (
            docs_chunk_name if docs_chunk_name.endswith(".js") else f"{docs_chunk_name}.js"
        )
    else:
        chunk_asset = (
            f"static/{docs_chunk_name}.js"
            if not docs_chunk_name.endswith(".js")
            else f"static/{docs_chunk_name}"
        )
    chunk_basename = Path(chunk_asset).name
    docs_chunk_url = resolve_platform_static_url(chunk_asset)
    try:
        docs_chunk_text = scraper.get(docs_chunk_url, timeout=30).text
    except Exception as err:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=[f"Failed to fetch docs chunk {chunk_asset}: {err}"],
        )

    root_info = load_module_info(
        chunk_asset,
        scraper_obj=scraper,
        text_override=docs_chunk_text,
    )
    base_literals = dict(root_info.literals)
    base_functions = dict(root_info.functions)
    dependency_paths = {
        binding.source for binding in root_info.imports.values()
    }
    dependency_paths.add(
        normalize_static_module_path("./index-CtzF8Zkc.js", base=root_info.path)
    )
    for dep_path in sorted(dependency_paths):
        if not dep_path:
            continue
        if not (dep_path.startswith("http") or dep_path.endswith(".js")):
            continue
        try:
            dep_info = load_module_info(dep_path, scraper_obj=scraper)
        except Exception as fetch_err:  # pragma: no cover - network edge
            errors.append(f"Failed to load dependency {dep_path} for docs chunk: {fetch_err}")
            continue
        base_literals.update(dep_info.literals)
        base_functions.update(dep_info.functions)

    path_wrapper_map = extract_path_wrapper_map(docs_chunk_text)
    wrapper_render_map = extract_wrapper_render_map(docs_chunk_text)
    if not path_wrapper_map:
        return PlatformGuidesMirrorResult(
            status="failed",
            sitemap_url=sitemap_url,
            documents=[],
            errors=["Docs chunk did not expose any path → component mappings."],
        )

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
        docs_path = "/docs/" + relative_url
        wrapper_name = path_wrapper_map.get(docs_path)
        if not wrapper_name:
            msg = f"No wrapper component mapping for {docs_path}"
            errors.append(msg)
            documents.append(
                PlatformDocResult(
                    url=url,
                    relative_url=relative_url,
                    output_path=output_root,
                    status="failed",
                    error=msg,
                )
            )
            continue
        render_name = wrapper_render_map.get(wrapper_name)
        if not render_name:
            msg = f"No render function mapping for wrapper {wrapper_name}"
            errors.append(msg)
            documents.append(
                PlatformDocResult(
                    url=url,
                    relative_url=relative_url,
                    output_path=output_root,
                    status="failed",
                    error=msg,
                )
            )
            continue
        try:
            wrapper_code = extract_function_code(docs_chunk_text, wrapper_name)
            render_code = extract_function_code(docs_chunk_text, render_name)
        except (KeyError, ValueError) as err:
            msg = f"Failed to extract functions for {docs_path}: {err}"
            errors.append(msg)
            documents.append(
                PlatformDocResult(
                    url=url,
                    relative_url=relative_url,
                    output_path=output_root,
                    status="failed",
                    error=msg,
                )
            )
            continue

        combined_literals = dict(base_literals)
        combined_functions = dict(base_functions)

        stub_names = sorted(collect_component_identifiers(render_code))
        constant_candidates = collect_constant_identifiers(render_code)
        constant_candidates.update(collect_property_identifier_candidates(render_code))
        constant_values: dict[str, str] = {}
        dependency_codes: list[str] = []
        included_functions: set[str] = set()
        included_aliases: set[str] = set()

        for constant in sorted(constant_candidates):
            literal = combined_literals.get(constant)
            if literal is not None:
                constant_values[constant] = literal
                continue
            func_code = combined_functions.get(constant)
            if func_code and constant not in included_functions:
                dependency_codes.append(func_code)
                included_functions.add(constant)
                continue
            try:
                resolved = resolve_identifier_value(
                    root_info.path,
                    constant,
                    visited_local=set(),
                    visited_export=set(),
                )
            except Exception as resolution_err:  # pragma: no cover - diagnostic aid
                logging.debug("Alias resolution failed for %s: %s", constant, resolution_err)
                resolved = None
            if resolved:
                if resolved.kind == "literal":
                    constant_values[constant] = resolved.code
                    combined_literals.setdefault(constant, resolved.code)
                else:
                    if resolved.name not in included_functions:
                        dependency_codes.append(resolved.code)
                        included_functions.add(resolved.name)
                    if constant != resolved.name and constant not in included_aliases:
                        dependency_codes.append(f"const {constant} = {resolved.name};")
                        included_aliases.add(constant)
                continue
            constant_values[constant] = "{}"

        destination = (
            output_root
            / OPENAI_PLATFORM_DOCS_DIRNAME
            / Path(*segments)
            / PLATFORM_DOC_INDEX_FILENAME
        )
        if destination.exists() and not force and not dry_run:
            documents.append(
                PlatformDocResult(
                    url=url,
                    relative_url=relative_url,
                    output_path=destination,
                    status="skipped",
                    chunk_name=chunk_basename,
                    wrapper_function=wrapper_name,
                    render_function=render_name,
                )
            )
            continue

        try:
            tree = evaluate_wrapper_to_tree(
                wrapper_name=wrapper_name,
                wrapper_code=wrapper_code,
                render_code=render_code,
                dependency_codes=dependency_codes,
                stub_names=stub_names,
                constant_values=constant_values,
            )
            markdown = tree_to_markdown(tree)
        except Exception as err:  # pragma: no cover - evaluation edge
            msg = f"Failed to evaluate {docs_path}: {err}"
            errors.append(msg)
            documents.append(
                PlatformDocResult(
                    url=url,
                    relative_url=relative_url,
                    output_path=destination,
                    status="failed",
                    error=msg,
                    chunk_name=chunk_basename,
                    wrapper_function=wrapper_name,
                    render_function=render_name,
                )
            )
            continue

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
                    chunk_name=chunk_basename,
                wrapper_function=wrapper_name,
                render_function=render_name,
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
        help="Skip mirroring the platform.openai.com guides",
    )
    parser.add_argument(
        "--platform-section",
        dest="platform_sections",
        action="append",
        help="Platform docs section(s) to mirror (default: guides)",
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
            "Mirroring OpenAI platform guides via static chunks (sections: %s)",
            ", ".join(platform_sections),
        )
        try:
            platform_guides_result = mirror_platform_guides(
                output_root=output_root,
                dry_run=args.dry_run,
                force=args.force,
                sections=platform_sections,
                sitemap_url=PLATFORM_SITEMAP_URL,
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
