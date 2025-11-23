#!/usr/bin/env python3
"""
Create a comprehensive local mirror of the uv documentation.

The script pulls two complementary sources:

1. The Markdown sources hosted in the uv repository under ``docs/``.
2. The rendered documentation at https://docs.astral.sh/uv/ (via sitemap crawl).

Outputs are written under ``SPECIALIZED_CONTEXT_AND_DOCS/UV_DOCS_MIRROR`` by default
with the following layout::

    UV_DOCS_MIRROR/
      README.md                   - overview + run metadata
      manifest.json               - machine-friendly manifest
      index.md, concepts/, ...    - uv repo docs copied directly into the root
      rendered_site/html/...      - raw rendered HTML per page
      rendered_site/markdown/...  - simplified Markdown extracted from the article body

Site downloads rely exclusively on stdlib networking to keep blast radius tight,
store manifest data for auditing, and emit descriptive filenames that mirror the
remote hierarchy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UV_REPO = Path("~/repos/OTHER_PEOPLES_REPOS/uv").expanduser()
UV_REPO_URL = "https://github.com/astral-sh/uv.git"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "UV_DOCS_MIRROR"

DEFAULT_SITE_BASE_URL = "https://docs.astral.sh/uv"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
DEFAULT_REPO_SKIP_DIRS = {".overrides", "js", "stylesheets", "site"}
HOSTED_SITE_DIR = "rendered_site"

HTTP_HEADERS = {
    "User-Agent": "prompts-uv-docs-mirror/2025.11 (+https://docs.astral.sh/uv/)"
}


class TextBuffer:
    """Small helper that tracks trailing characters for efficient suffix checks."""

    __slots__ = ("parts", "tail")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.tail: str = ""

    def append(self, text: str) -> None:
        if not text:
            return
        self.parts.append(text)
        combined = (self.tail + text)[-16:]
        self.tail = combined

    def endswith(self, suffix: str) -> bool:
        if not suffix:
            return True
        if len(suffix) <= len(self.tail):
            return self.tail.endswith(suffix)
        return self.getvalue().endswith(suffix)

    def ensure(self, suffix: str) -> None:
        if not self.endswith(suffix):
            self.append(suffix)

    def getvalue(self) -> str:
        return "".join(self.parts)

    def last_char(self) -> str | None:
        if not self.tail:
            return None
        return self.tail[-1]


class ArticleSliceExtractor(HTMLParser):
    """Extract the raw HTML for the main <article> element."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.capture = False
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "article" and not self.capture:
            self.capture = True
            self.depth = 0
        if self.capture:
            self.depth += 1
            self.chunks.append(self.get_starttag_text() or f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        if not self.capture:
            return
        self.chunks.append(f"</{tag}>")
        self.depth -= 1
        if self.depth == 0:
            self.capture = False

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.chunks.append(data)

    def get_article_html(self) -> str | None:
        if not self.chunks:
            return None
        return "".join(self.chunks)


class SimpleMarkdownRenderer(HTMLParser):
    """Markdown renderer aware of MkDocs Material tabsets and admonitions."""

    def __init__(self) -> None:
        super().__init__()
        self.output = TextBuffer()
        self.anchor_stack: list[dict[str, object]] = []
        self.list_stack: list[dict[str, object]] = []
        self.blockquote_depth = 0
        self.in_pre = False
        self.inline_code_depth = 0
        self.current_heading: Optional[str] = None
        self.admonition_stack: list[dict[str, object]] = []
        self.capture_admonition_title: Optional[TextBuffer] = None
        self.tabbed_set_stack: list[dict[str, object]] = []
        self.tabbed_labels_depth = 0
        self.collecting_tab_label: Optional[TextBuffer] = None
        self.div_class_stack: list[set[str]] = []

    # Utility helpers -------------------------------------------------
    def _extract_classes(self, attrs: list[tuple[str, Optional[str]]]) -> set[str]:
        for attr, value in attrs:
            if attr == "class" and value:
                return {cls.strip() for cls in value.split() if cls.strip()}
        return set()

    def _write(self, text: str) -> None:
        if not text:
            return
        if self.anchor_stack:
            self.anchor_stack[-1]["buffer"].append(text)  # type: ignore[index]
            return
        if self.admonition_stack and not self.in_pre:
            ctx = self.admonition_stack[-1]
            if not ctx["header_emitted"]:
                self._emit_admonition_header(ctx)
            prefix = "    " * len(self.admonition_stack)
        else:
            prefix = ""
        target = self.output
        parts = text.split("\n")
        for idx, part in enumerate(parts):
            if prefix and part and self._at_line_start(target):
                target.append(prefix)
            target.append(part)
            if idx < len(parts) - 1:
                target.append("\n")

    def _at_line_start(self, buf: TextBuffer) -> bool:
        last = buf.last_char()
        return last is None or last == "\n"

    def _ensure_blank_line(self) -> None:
        if self.anchor_stack:
            return
        buf = self.output
        current = buf.getvalue()
        if current and not current.endswith("\n\n"):
            if not current.endswith("\n"):
                buf.append("\n")
            buf.append("\n")

    def _list_prefix(self) -> str:
        depth = len(self.list_stack)
        if not depth:
            return ""
        current = self.list_stack[-1]
        indent = "  " * (depth - 1)
        if current["type"] == "ol":
            current["index"] = int(current.get("index", 0)) + 1
            prefix = f"{current['index']}. "
        else:
            prefix = "- "
        return indent + prefix

    def _emit_admonition_header(self, ctx: dict[str, object]) -> None:
        ctx["header_emitted"] = True
        title = ctx.get("title")
        admon_type = ctx.get("type") or "note"
        line = f"!!! {admon_type}"
        if title:
            line += f' "{title}"'
        self.output.append(line + "\n\n")

    # HTMLParser interface --------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        classes = self._extract_classes(attrs)
        if tag == "div":
            self.div_class_stack.append(classes)
            if "tabbed-set" in classes:
                self.tabbed_set_stack.append({"labels": [], "block_index": 0})
            if "tabbed-labels" in classes and self.tabbed_set_stack:
                self.tabbed_labels_depth += 1
            if "tabbed-block" in classes and self.tabbed_set_stack:
                current = self.tabbed_set_stack[-1]
                idx = current["block_index"]
                labels: list[str] = current["labels"]
                label = labels[idx] if idx < len(labels) else f"Option {idx + 1}"
                current["block_index"] += 1
                self._ensure_blank_line()
                self._write(f'=== "{label}"\n\n')
            if "admonition" in classes:
                admon_type = next((cls for cls in classes if cls != "admonition"), "note")
                ctx = {"type": admon_type, "title": None, "header_emitted": False}
                self.admonition_stack.append(ctx)
                self._ensure_blank_line()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._ensure_blank_line()
            level = int(tag[1])
            self._write("#" * level + " ")
            self.current_heading = tag
        elif tag == "p":
            if "admonition-title" in classes and self.admonition_stack:
                self.capture_admonition_title = TextBuffer()
            else:
                self._ensure_blank_line()
        elif tag in {"ul", "ol"}:
            self.list_stack.append({"type": tag, "index": 0})
        elif tag == "li":
            self._ensure_blank_line()
            self._write(self._list_prefix())
        elif tag == "pre":
            self._ensure_blank_line()
            fence_lang = ""
            for attr, value in attrs:
                if attr == "class" and value:
                    for part in value.split():
                        if part.startswith("language-"):
                            fence_lang = part.replace("language-", "")
                            break
            self._write(f"```{fence_lang}\n")
            self.in_pre = True
        elif tag == "code":
            if self.in_pre:
                return
            self._write("`")
            self.inline_code_depth += 1
        elif tag in {"strong", "b"}:
            self._write("**")
        elif tag in {"em", "i"}:
            self._write("_")
        elif tag == "br":
            self._write("\n")
        elif tag == "blockquote":
            self.blockquote_depth += 1
            self._ensure_blank_line()
            self._write("> " * self.blockquote_depth)
        elif tag == "a":
            href = ""
            for attr, value in attrs:
                if attr == "href" and value:
                    href = value
                    break
            self.anchor_stack.append({"href": href, "buffer": TextBuffer()})
        elif tag == "img":
            alt = ""
            src = ""
            for attr, value in attrs:
                if attr == "alt" and value:
                    alt = value
                elif attr == "src" and value:
                    src = value
            if src:
                self._write(f"![{alt}]({src})")
        elif tag == "label" and self.tabbed_labels_depth > 0 and self.tabbed_set_stack:
            self.collecting_tab_label = TextBuffer()
        elif tag == "hr":
            self._ensure_blank_line()

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self.div_class_stack:
            classes = self.div_class_stack.pop()
            if "tabbed-labels" in classes and self.tabbed_labels_depth > 0:
                self.tabbed_labels_depth -= 1
            if "tabbed-set" in classes and self.tabbed_set_stack:
                self.tabbed_set_stack.pop()
            if "admonition" in classes and self.admonition_stack:
                ctx = self.admonition_stack.pop()
                if not ctx["header_emitted"]:
                    self._emit_admonition_header(ctx)
                self._ensure_blank_line()
        if tag == self.current_heading:
            self._write("\n\n")
            self.current_heading = None
        elif tag == "p":
            if self.capture_admonition_title is not None:
                title = self.capture_admonition_title.getvalue().strip()
                ctx = self.admonition_stack[-1] if self.admonition_stack else None
                if ctx is not None:
                    ctx["title"] = title or ctx.get("title") or ctx.get("type")
                self.capture_admonition_title = None
            else:
                self._write("\n\n")
        elif tag in {"ul", "ol"}:
            if self.list_stack:
                self.list_stack.pop()
            self._write("\n")
        elif tag == "li":
            self._write("\n")
        elif tag == "pre":
            if self.in_pre:
                self._write("\n```\n\n")
                self.in_pre = False
        elif tag == "code":
            if self.inline_code_depth > 0:
                self._write("`")
                self.inline_code_depth -= 1
        elif tag in {"strong", "b"}:
            self._write("**")
        elif tag in {"em", "i"}:
            self._write("_")
        elif tag == "blockquote":
            if self.blockquote_depth > 0:
                self.blockquote_depth -= 1
            self._write("\n\n")
        elif tag == "a":
            if not self.anchor_stack:
                return
            ctx = self.anchor_stack.pop()
            href = ctx.get("href", "") or ""
            text = ctx["buffer"].getvalue().strip() or href  # type: ignore[index]
            formatted = f"[{text}]({href})" if href else text
            self._write(formatted)
        elif tag == "label" and self.collecting_tab_label is not None:
            label = self.collecting_tab_label.getvalue().strip()
            if self.tabbed_set_stack:
                self.tabbed_set_stack[-1]["labels"].append(label or f"Option {len(self.tabbed_set_stack[-1]['labels']) + 1}")
            self.collecting_tab_label = None

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self.collecting_tab_label is not None:
            self.collecting_tab_label.append(data)
            return
        if self.capture_admonition_title is not None:
            self.capture_admonition_title.append(data)
            return
        if self.in_pre:
            self._write(data)
            return
        text = " ".join(data.split())
        if text:
            trailing_space = " " if not text.endswith(" ") else ""
            self._write(text + trailing_space)

    def getvalue(self) -> str:
        rendered = self.output.getvalue().strip()
        return rendered + "\n" if rendered else ""


def relocate_existing_output(output_root: Path) -> Path | None:
    """Move the existing output directory aside so we can rebuild cleanly."""
    if not output_root.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = output_root.parent / f"{output_root.name}.previous.{timestamp}"
    if backup.exists():
        shutil.rmtree(backup)
    output_root.rename(backup)
    return backup


def _list_relative_files(root: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if path.is_file():
            files[path.relative_to(root)] = path
    return files


def _files_identical(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    bufsize = 64 * 1024
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            chunk_a = fa.read(bufsize)
            chunk_b = fb.read(bufsize)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:  # EOF
                return True


def compute_change_summary(
    previous_root: Path | None,
    current_root: Path,
    include_filter: Callable[[Path], bool],
) -> dict[str, object]:
    """Compare previous and current trees and return added/updated/removed lists."""
    summary = {"added": [], "updated": [], "removed": [], "unchanged_count": 0}
    if previous_root is None or not previous_root.exists():
        current_files = _list_relative_files(current_root)
        summary["added"] = sorted(
            rel.as_posix() for rel in current_files if include_filter(rel)
        )
        return summary

    old_files = {
        rel: path for rel, path in _list_relative_files(previous_root).items() if include_filter(rel)
    }
    current_files = _list_relative_files(current_root)

    for rel, new_path in current_files.items():
        if not include_filter(rel):
            continue
        rel_str = rel.as_posix()
        old_path = old_files.pop(rel, None)
        if old_path is None:
            summary["added"].append(rel_str)
        elif _files_identical(old_path, new_path):
            summary["unchanged_count"] += 1
        else:
            summary["updated"].append(rel_str)

    summary["removed"] = sorted(rel.as_posix() for rel in old_files.keys())
    return summary


@dataclass
class SitePage:
    url: str
    relative_slug: str
    html_path: Path
    markdown_path: Path


def http_get_bytes(url: str, *, timeout: float = 20.0) -> bytes:
    request = Request(url, headers=HTTP_HEADERS, method="GET")
    with urlopen(request, timeout=timeout) as resp:  # nosec: site mirroring
        return resp.read()


def ensure_repo(repo_path: Path, allow_clone: bool) -> None:
    if (repo_path / ".git").is_dir():
        return
    if not allow_clone:
        raise SystemExit(
            f"uv repository not present at {repo_path}. Clone it manually or allow the script to clone."
        )
    repo_path.mkdir(parents=True, exist_ok=True)
    print(f"[repo] Cloning uv repo into {repo_path}")
    subprocess.run(["git", "clone", UV_REPO_URL, "."], check=True, cwd=str(repo_path))


def copy_repo_docs(repo_root: Path, dest_root: Path, skip_dirs: set[str]) -> int:
    source_docs = repo_root / "docs"
    if not source_docs.is_dir():
        raise SystemExit(f"Expected docs directory at {source_docs}")

    copied_files = 0
    skipped: list[str] = []

    # Remove previously mirrored directories that are now being skipped.
    for skip_name in skip_dirs:
        skip_target = dest_root / skip_name
        if skip_target.exists():
            shutil.rmtree(skip_target)

    for entry in sorted(source_docs.iterdir(), key=lambda p: p.name.lower()):
        if entry.name in skip_dirs:
            skipped.append(entry.name)
            continue
        target = dest_root / entry.name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target)
            copied_files += sum(1 for child in target.rglob("*") if child.is_file())
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target)
            copied_files += 1

    if skipped:
        print(f"[repo] Skipped directories: {', '.join(sorted(skipped))}")
    print(f"[repo] Copied uv/docs contents directly into {dest_root} ({copied_files} files).")
    return copied_files


def fetch_sitemap(sitemap_url: str) -> ET.Element:
    raw = http_get_bytes(sitemap_url)
    return ET.fromstring(raw)


def normalize_slug(url: str, base_url: str) -> str:
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)
    if parsed.netloc != base_parsed.netloc:
        raise ValueError(f"URL outside base domain: {url}")
    base_path = base_parsed.path.rstrip("/")
    rel = parsed.path[len(base_path) :].lstrip("/")
    rel = rel.strip("/")
    if not rel:
        return "index"
    return rel


def target_paths(slug: str, html_root: Path, markdown_root: Path) -> tuple[Path, Path]:
    slug = slug.rstrip("/")
    if not slug:
        slug = "index"
    if slug.endswith(".html"):
        rel = Path(slug)
        html_path = html_root / rel
        markdown_path = markdown_root / rel.with_suffix(".md")
    else:
        rel = Path(slug)
        html_path = html_root / rel / "index.html"
        markdown_path = markdown_root / rel / "index.md"
    return html_path, markdown_path


def collect_site_pages(
    sitemap_root: ET.Element, *, html_root: Path, markdown_root: Path, base_url: str
) -> list[SitePage]:
    pages: list[SitePage] = []
    seen: set[str] = set()
    prefix = base_url.rstrip("/")
    for loc_el in sitemap_root.findall("sm:url/sm:loc", SITEMAP_NS):
        loc_text = (loc_el.text or "").strip()
        if not loc_text or (loc_text != prefix and not loc_text.startswith(prefix + "/")):
            continue
        slug = normalize_slug(loc_text, prefix)
        if slug in seen:
            continue
        seen.add(slug)
        html_path, markdown_path = target_paths(slug, html_root, markdown_root)
        pages.append(SitePage(loc_text, slug, html_path, markdown_path))
    pages.sort(key=lambda page: page.relative_slug)
    return pages


def extract_article_markdown(html_text: str) -> str:
    extractor = ArticleSliceExtractor()
    extractor.feed(html_text)
    article_html = extractor.get_article_html()
    if not article_html:
        return ""
    renderer = SimpleMarkdownRenderer()
    renderer.feed(article_html)
    return renderer.getvalue()


def download_site_page(page: SitePage, *, force: bool, keep_html: bool) -> dict[str, object]:
    # Skip if cached
    if (keep_html is False or page.html_path.exists()) and page.markdown_path.exists() and not force:
        return {"url": page.url, "cached": True}

    html_bytes = http_get_bytes(page.url)
    if keep_html:
        page.html_path.parent.mkdir(parents=True, exist_ok=True)
        page.html_path.write_bytes(html_bytes)
    elif page.html_path.exists():
        page.html_path.unlink()

    markdown = extract_article_markdown(html_bytes.decode("utf-8", errors="replace"))
    page.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    page.markdown_path.write_text(markdown, encoding="utf-8")

    return {
        "url": page.url,
        "cached": False,
        "markdown_bytes": len(markdown.encode("utf-8")),
        "html_saved": keep_html,
    }


def mirror_site_pages(
    pages: list[SitePage],
    *,
    force: bool,
    max_workers: int,
    output_root: Path,
    keep_html: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(download_site_page, page, force=force, keep_html=keep_html): page for page in pages
        }
        for future in concurrent.futures.as_completed(future_map):
            page = future_map[future]
            try:
                res = future.result()
                results.append(res)
                status = "cached" if res.get("cached") else "downloaded"
                rel = page.markdown_path.relative_to(output_root)
                print(f"[site] {status}: {page.url} -> {rel}")
            except Exception as exc:
                error_entry = {"url": page.url, "error": str(exc)}
                results.append(error_entry)
                print(f"[site] error for {page.url}: {exc}")
    return results


def write_manifest(
    pages: list[SitePage],
    *,
    output_root: Path,
    repo_root: Path | None,
    repo_file_count: int,
    repo_skip_dirs: set[str] | None,
    site_results: list[dict[str, object]],
    site_base: str,
    keep_html: bool,
    change_summary: dict[str, dict[str, object]],
) -> Path:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root) if repo_root else None,
        "repo_skip_dirs": sorted(repo_skip_dirs or []),
        "site_base": site_base,
        "counts": {
            "site_pages": len(pages),
            "repo_docs_files": repo_file_count,
        },
        "site_pages": [
            {
                "url": page.url,
                "relative_slug": page.relative_slug,
                "html_path": str(page.html_path.relative_to(output_root)) if keep_html else None,
                "markdown_path": str(page.markdown_path.relative_to(output_root)),
            }
            for page in pages
        ],
        "site_results": site_results,
        "change_summary": change_summary,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_readme(
    output_root: Path,
    repo_source: Path | None,
    site_page_count: int,
    site_base: str,
    repo_skip_dirs: set[str] | None,
    keep_html: bool,
) -> None:
    readme_path = output_root / "README.md"
    lines = [
        "# uv Documentation Mirror",
        "",
        "This directory was generated by `create_uv_docs_mirror.py`. It captures:",
        "",
        "1. The Markdown sources from the uv repository copied directly into this directory.",
        f"2. The rendered documentation extracted from {site_base} (`{HOSTED_SITE_DIR}/`).",
        "",
        "## Layout",
        "",
        "- Root: mirror of `uv/docs/`"
        + (f" from `{repo_source}`" if repo_source else "")
        + "."
        + (
            f" Skipped directories: {', '.join(sorted(repo_skip_dirs))}."
            if repo_skip_dirs
            else ""
        ),
        (
            f"- `{HOSTED_SITE_DIR}/html/`: raw HTML per sitemap URL (mirrors the hosted structure)."
            if keep_html
            else "- (HTML snapshots skipped – rerun with `--keep-html` to include them.)"
        ),
        f"- `{HOSTED_SITE_DIR}/markdown/`: simplified Markdown derived from the `<article>` body for quick text search.",
        "- `manifest.json`: metadata about the run.",
        "",
        f"Total site pages mirrored: **{site_page_count}**.",
        "",
        "Run again with:",
        "",
        "```bash",
        "uv run python src/DATA_SCRIPTS/create_uv_docs_mirror.py",
        "```",
        "",
        "Use `--help` for advanced options (skip components, change paths, etc.).",
        "",
        "Each run rebuilds the mirror from scratch and records file deltas in `manifest.json`.",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a combined uv docs mirror (repo + site).")
    parser.add_argument("--repo", default=str(DEFAULT_UV_REPO), help="Path to local uv repository clone.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory (default: %(default)s)")
    parser.add_argument("--skip-repo", action="store_true", help="Skip copying repo Markdown docs.")
    parser.add_argument("--skip-site", action="store_true", help="Skip sitemap download/rendered site capture.")
    parser.add_argument(
        "--repo-skip",
        action="append",
        default=None,
        help=(
            "Directory names under uv/docs to skip when copying. "
            "Defaults to .overrides, js, stylesheets, site unless overridden."
        ),
    )
    parser.add_argument(
        "--force-site",
        action="store_true",
        help="Redownload site pages even if the files already exist.",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel downloads for site mirroring.")
    parser.add_argument(
        "--site-base",
        default=DEFAULT_SITE_BASE_URL,
        help="Base URL for the rendered docs (default: %(default)s)",
    )
    parser.add_argument(
        "--sitemap",
        default=None,
        help="Override sitemap URL (default: <site-base>/sitemap.xml)",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="Persist raw HTML snapshots alongside extracted Markdown (default: off).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    output_root = Path(args.out).expanduser().resolve()
    previous_snapshot = relocate_existing_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_root: Path | None = None
    repo_file_count = 0
    repo_skip_dirs: set[str] | None = None
    if not args.skip_repo:
        repo_root = Path(args.repo).expanduser().resolve()
        allow_clone = args.repo == str(DEFAULT_UV_REPO)
        ensure_repo(repo_root, allow_clone=allow_clone)
        if args.repo_skip is None:
            repo_skip_dirs = set(DEFAULT_REPO_SKIP_DIRS)
        else:
            repo_skip_dirs = {name.strip("/") for name in args.repo_skip if name}
        repo_file_count = copy_repo_docs(repo_root, output_root, skip_dirs=repo_skip_dirs)

    site_pages: list[SitePage] = []
    site_results: list[dict[str, object]] = []
    site_base = args.site_base.rstrip("/")
    sitemap_url = args.sitemap or f"{site_base}/sitemap.xml"
    keep_html = args.keep_html
    if not args.skip_site:
        print(f"[site] Fetching sitemap from {sitemap_url} ...")
        sitemap_root = fetch_sitemap(sitemap_url)
        html_root = output_root / HOSTED_SITE_DIR / "html"
        markdown_root = output_root / HOSTED_SITE_DIR / "markdown"
        if not keep_html and html_root.exists():
            shutil.rmtree(html_root)
        site_pages = collect_site_pages(
            sitemap_root,
            html_root=html_root,
            markdown_root=markdown_root,
            base_url=site_base,
        )
        print(f"[site] Planned {len(site_pages)} pages.")
        site_results = mirror_site_pages(
            site_pages,
            force=args.force_site,
            max_workers=args.max_workers,
            output_root=output_root,
            keep_html=keep_html,
        )

    BOOKKEEPING_NAMES = {"README.md", "manifest.json"}
    repo_filter = lambda rel: rel.parts and rel.parts[0] != HOSTED_SITE_DIR and rel.name not in BOOKKEEPING_NAMES
    site_md_filter = (
        lambda rel: len(rel.parts) >= 2 and rel.parts[0] == HOSTED_SITE_DIR and rel.parts[1] == "markdown"
    )
    repo_changes = compute_change_summary(previous_snapshot, output_root, repo_filter)
    site_changes = compute_change_summary(previous_snapshot, output_root, site_md_filter)

    change_summary = {"repo": repo_changes, "site_markdown": site_changes}

    print(
        "[diff] Repo docs -> added: %s, updated: %s, removed: %s, unchanged: %s"
        % (
            len(repo_changes["added"]),
            len(repo_changes["updated"]),
            len(repo_changes["removed"]),
            repo_changes["unchanged_count"],
        )
    )
    print(
        "[diff] Site markdown -> added: %s, updated: %s, removed: %s, unchanged: %s"
        % (
            len(site_changes["added"]),
            len(site_changes["updated"]),
            len(site_changes["removed"]),
            site_changes["unchanged_count"],
        )
    )

    manifest_path = write_manifest(
        site_pages,
        output_root=output_root,
        repo_root=repo_root,
        repo_file_count=repo_file_count,
        repo_skip_dirs=repo_skip_dirs,
        site_results=site_results,
        site_base=site_base,
        keep_html=keep_html,
        change_summary=change_summary,
    )
    write_readme(output_root, repo_root, len(site_pages), site_base, repo_skip_dirs, keep_html)

    if previous_snapshot and previous_snapshot.exists():
        shutil.rmtree(previous_snapshot)
    print(f"Manifest: {manifest_path}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
