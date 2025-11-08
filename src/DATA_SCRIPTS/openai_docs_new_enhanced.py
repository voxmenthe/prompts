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
from typing import Iterable, Sequence, Callable, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import bs4
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
import threading
import concurrent.futures


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

class Cache:
    def __init__(self, base: Path, enabled: bool = True, ttl: int = 0):
        self.base = base
        self.enabled = enabled
        self.ttl = max(0, int(ttl))
        self._lock = threading.Lock()

    def _paths(self, kind: str, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(f"{kind}|{url}".encode("utf-8")).hexdigest()
        dir_ = self.base / kind
        meta = dir_ / f"{key}.meta.json"
        data = dir_ / f"{key}.data"
        return meta, data

    def get_text(self, kind: str, url: str) -> Optional[str]:
        if not self.enabled:
            return None
        meta_path, data_path = self._paths(kind, url)
        if not meta_path.exists() or not data_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = meta.get("ts", 0)
            if self.ttl and (time.time() - ts > self.ttl):
                return None
            return data_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def put_text(self, kind: str, url: str, text: str) -> None:
        if not self.enabled:
            return
        meta_path, data_path = self._paths(kind, url)
        with self._lock:
            ensure_dir(meta_path.parent)
            data_path.write_text(text, encoding="utf-8")
            meta = {"ts": time.time(), "url": url}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def get_bytes(self, kind: str, url: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        meta_path, data_path = self._paths(kind, url)
        if not meta_path.exists() or not data_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = meta.get("ts", 0)
            if self.ttl and (time.time() - ts > self.ttl):
                return None
            return data_path.read_bytes()
        except Exception:
            return None

    def put_bytes(self, kind: str, url: str, data: bytes) -> None:
        if not self.enabled:
            return
        meta_path, data_path = self._paths(kind, url)
        with self._lock:
            ensure_dir(meta_path.parent)
            data_path.write_bytes(data)
            meta = {"ts": time.time(), "url": url}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

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


def _guess_language_from_code(code: str) -> Optional[str]:
    sample = code.strip().splitlines()[:20]
    joined = "\n".join(sample)
    # JSON
    if joined.startswith("{") or joined.startswith("["):
        try:
            import json as _json
            _json.loads(joined)
            return "json"
        except Exception:
            pass
    # Bash / curl
    if any(ln.strip().startswith("curl ") for ln in sample) or any(ln.strip().startswith("#!/bin/bash") for ln in sample):
        return "bash"
    # Python
    if any(kw in joined for kw in ["from ", "import ", "def ", "print(", "client = OpenAI("]):
        if ";" not in joined or "#!/usr/bin/env python" in joined:
            return "python"
    # JavaScript/TypeScript
    if any(tok in joined for tok in ["import ", "export ", "const ", "let ", "await ", "OpenAI from \"openai\""]):
        return "javascript"
    # Go
    if "package main" in joined or "func main()" in joined or "github.com/openai/openai-go" in joined:
        return "go"
    # C#
    if "using System" in joined or "class Program" in joined or "OpenAIClient" in joined:
        return "csharp"
    # Java
    if any(tok in joined for tok in ["public class ", "System.out.println", "ResponseCreateParams", "OpenAIOkHttpClient"]):
        return "java"
    return None


def _guess_language_from_context(context_lines: list[str]) -> Optional[str]:
    # Look backwards for language hints in nearby headings or lines
    hay = " ".join(line.lower() for line in context_lines[-5:])
    for lang in ["javascript", "python", "bash", "go", "java", "csharp", "json"]:
        if lang in hay or (lang == "csharp" and ("c#" in hay)):
            return lang
    return None


def _annotate_code_fences(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    # maintain a sliding window of previous non-empty lines
    recent: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            fence = line
            # if already has language, passthrough
            parts = fence.strip().split()
            has_lang = len(parts) > 1 and parts[0].startswith("```")
            if not has_lang or parts[0] == "```":
                # gather code until closing fence
                j = i + 1
                code_lines: list[str] = []
                while j < len(lines) and not lines[j].startswith("```"):
                    code_lines.append(lines[j])
                    j += 1
                code_text = "\n".join(code_lines)
                lang = _guess_language_from_context(recent) or _guess_language_from_code(code_text) or ""
                if lang:
                    out.append(f"```{lang}")
                else:
                    out.append("```")
                out.extend(code_lines)
                # append closing fence if exists
                if j < len(lines):
                    out.append(lines[j])
                    i = j + 1
                else:
                    i = j
                # update recent window with a placeholder line
                recent.append("<codeblock>")
                if len(recent) > 10:
                    recent = recent[-10:]
                continue
            else:
                # already has a language
                out.append(line)
                i += 1
                recent.append(line)
                if len(recent) > 10:
                    recent = recent[-10:]
                continue
        # non-fence
        out.append(line)
        i += 1
        if line.strip():
            recent.append(line)
            if len(recent) > 10:
                recent = recent[-10:]
    return "\n".join(out) + "\n"


def postprocess_markdown(md: str) -> str:
    md = _strip_line_numbers_in_fences(md)
    md = _annotate_code_fences(md)
    return md


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


def _download_images_in_markdown(md: str, *, page_dir: Path, cache: Optional[Cache], timeout: float) -> str:
    import requests
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    assets_dir = page_dir / "assets"
    ensure_dir(assets_dir)

    def fetch_image(url: str) -> Optional[tuple[str, bytes]]:
        # Return (ext, bytes)
        if cache:
            cached = cache.get_bytes("image", url)
            if cached is not None:
                # try to guess extension from url
                ext = Path(url).suffix or ".bin"
                return ext, cached
        try:
            r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
            r.raise_for_status()
            data = r.content
            if cache:
                cache.put_bytes("image", url, data)
            # content-type to extension
            ctype = r.headers.get("Content-Type", "").lower()
            ext = ".bin"
            if "image/png" in ctype:
                ext = ".png"
            elif "image/jpeg" in ctype or "image/jpg" in ctype:
                ext = ".jpg"
            elif "image/webp" in ctype:
                ext = ".webp"
            elif "image/svg" in ctype or url.endswith(".svg"):
                ext = ".svg"
            else:
                # fallback to url suffix
                suff = Path(url).suffix
                if suff:
                    ext = suff
            return ext, data
        except Exception:
            return None

    def replace(match: re.Match) -> str:
        alt = match.group(1)
        url = match.group(2).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return match.group(0)
        fetched = fetch_image(url)
        if not fetched:
            return match.group(0)
        ext, data = fetched
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        filename = f"img-{key}{ext}"
        (assets_dir / filename).write_bytes(data)
        rel_path = f"assets/{filename}"
        return f"![{alt}]({rel_path})"

    return image_pattern.sub(replace, md)


def fetch_html_with_cloudscraper(url: str, *, timeout: float, cache: Optional[Cache] = None) -> str | None:
    if cache:
        cached = cache.get_text("html", url)
        if cached:
            return cached
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
        if cache:
            cache.put_text("html", url, text)
        return text
    except Exception:
        return None


def fetch_markdown_via_jina(url: str, *, timeout: int = 120, cache: Optional[Cache] = None) -> str | None:
    if cache:
        cached = cache.get_text("jina", url)
        if cached:
            return cached
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
            text = text[idx + len(marker) :].lstrip("\n")
        if cache:
            cache.put_text("jina", url, text)
        return text
    except Exception:
        return None


def robust_get_markdown_for_url(url: str, *, timeout: float, cache: Optional[Cache] = None) -> str:
    if cache:
        cached_final = cache.get_text("final", url)
        if cached_final:
            return cached_final if cached_final.endswith("\n") else cached_final + "\n"
    # Strategy:
    # 1) Try cloudscraper to fetch HTML and convert to Markdown
    # 2) Fallback to r.jina.ai rendered Markdown
    html_text = fetch_html_with_cloudscraper(url, timeout=timeout, cache=cache)
    if html_text:
        soup = BeautifulSoup(html_text, "lxml")
        main = extract_main_content(soup)
        # Work on a copy to avoid modifying soup
        frag = BeautifulSoup(str(main), "lxml")
        md_from_html = html_to_markdown(frag)
        # Heuristic: if content is too short, it's likely just a shell (client-run site).
        if len(md_from_html.strip()) >= 200:
            result = postprocess_markdown(md_from_html)
            if cache:
                cache.put_text("final", url, result)
            return result
        # Otherwise fall through to Jina fallback
    md = fetch_markdown_via_jina(url, timeout=int(timeout), cache=cache)
    if md:
        cleaned = postprocess_markdown(md if md.endswith("\n") else md + "\n")
        if cache:
            cache.put_text("final", url, cleaned)
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
    max_workers: int = 8,
    cache: Optional[Cache] = None,
    download_images: bool = False,
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
    tasks: list[tuple[str, str, Path]] = []
    for url in urls:
        try:
            rel = url_to_relative_path(url, root="docs")
        except ValueError as e:
            results.append(PageResult(url=url, relative_path="", output_path=output_root, status="failed", error=str(e)))
            continue
        if regex and not regex.search(rel):
            continue
        dest = output_root / "openai_platform_docs" / Path(rel) / "index.md"
        if dest.exists() and not force:
            results.append(PageResult(url=url, relative_path=rel, output_path=dest, status="skipped"))
            continue
        tasks.append((url, rel, dest))

    if not tasks:
        return results

    def worker(url: str, rel: str, dest: Path) -> PageResult:
        logging.debug("Fetching %s", url)
        def _task():
            return robust_get_markdown_for_url(url, timeout=timeout, cache=cache)
        try:
            md = retry(_task, attempts=attempts, backoff=backoff)
            if download_images:
                md = _download_images_in_markdown(md, page_dir=dest.parent, cache=cache, timeout=timeout)
        except Exception as e:
            return PageResult(url=url, relative_path=rel, output_path=dest, status="failed", error=str(e))
        status, nbytes, digest = write_text(dest, md, force=force)
        return PageResult(url=url, relative_path=rel, output_path=dest, status=status, bytes_written=nbytes, sha256=digest)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        future_map = {pool.submit(worker, u, r, d): (u, r, d) for (u, r, d) in tasks}
        for fut in concurrent.futures.as_completed(future_map):
            res = fut.result()
            results.append(res)
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
    p.add_argument("--max-workers", type=int, default=8, help="Concurrent page workers (default: %(default)s)")
    p.add_argument("--cache-dir", type=Path, default=None, help="Cache directory (default: <output>/.cache)")
    p.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable HTTP/render cache")
    p.add_argument("--cache-ttl", type=int, default=86400, help="Cache TTL seconds (0 = no expiry; default: %(default)s)")
    p.add_argument("--download-images", action="store_true", help="Download images in markdown and rewrite links")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    output_root = args.output_dir.resolve()
    cache_dir = (args.cache_dir or (output_root / ".cache")).resolve()
    cache = Cache(cache_dir, enabled=getattr(args, "use_cache", True), ttl=int(getattr(args, "cache_ttl", 0)))

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
        max_workers=args.max_workers,
        cache=cache,
        download_images=args.download_images,
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
