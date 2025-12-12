#!/usr/bin/env python3
"""
Mirror the Hugging Face Transformers docs by scraping the rendered site.

This is a site-scraping counterpart to `create_transformers_docs_mirror.py`.
Instead of pulling the `hf-doc-build/doc-build` artifacts, this script:

1. Fetches the SvelteKit manifest for a given version/language to discover
   all docs routes (fallback: link crawl from the index page).
2. Downloads each page's rendered HTML.
3. Extracts the main docs container (`div.prose-doc`).
4. Writes raw HTML and a Markdown-converted version to a local mirror.

Default output layout:

  TRANSFORMERS_DOCS_SITE_MIRROR/
    html/<lang>/...         - raw HTML per page
    markdown/<lang>/...     - Markdown extracted from the prose container
    manifest.json           - run metadata + per-page outputs
    README.md               - overview

Usage:
  python src/DATA_SCRIPTS/create_transformers_docs_site_mirror.py --clean
  python src/DATA_SCRIPTS/create_transformers_docs_site_mirror.py --version main --languages en
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from markdownify import markdownify as md  # type: ignore[import-not-found]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "TRANSFORMERS_DOCS_SITE_MIRROR"
DEFAULT_VERSION = "main"
DEFAULT_LANGS = ("en",)

BASE_URL_TEMPLATE = "https://huggingface.co/docs/transformers/{version}/{lang}/"
MANIFEST_URL_TEMPLATE = BASE_URL_TEMPLATE + "_app/manifest.json"

USER_AGENT = "prompts-transformers-docs-site-mirror/2025.12 (+https://huggingface.co/docs/transformers)"
FETCH_RETRIES = 5
FETCH_BACKOFF_SECONDS = 2.0
MAX_WORKERS = 4
DEFAULT_MIN_DELAY_SECONDS = 1.0  # polite global throttle to avoid 429s

ROUTE_PREFIX = "src/routes/"
ROUTE_SUFFIX_RE = re.compile(r"\.(mdx|md)$", re.IGNORECASE)

_request_lock = threading.Lock()
_next_request_at = 0.0
_min_delay_seconds = DEFAULT_MIN_DELAY_SECONDS


def _throttle() -> None:
    """Global, cross-thread request pacing."""
    global _next_request_at
    sleep_for = 0.0
    with _request_lock:
        now = time.monotonic()
        if now < _next_request_at:
            sleep_for = _next_request_at - now
        _next_request_at = max(now, _next_request_at) + _min_delay_seconds
    if sleep_for > 0:
        time.sleep(sleep_for)


def _apply_global_backoff(seconds: float) -> None:
    """Push the global throttle forward after rate limiting."""
    if seconds <= 0:
        return
    global _next_request_at
    with _request_lock:
        candidate = time.monotonic() + seconds
        if candidate > _next_request_at:
            _next_request_at = candidate


@dataclass(frozen=True)
class PageJob:
    url: str
    route: str
    language: str

    @property
    def rel_stem(self) -> Path:
        if not self.route or self.route == "index":
            return Path("index")
        return Path(self.route)

    @property
    def rel_html(self) -> Path:
        stem = self.rel_stem
        return stem.parent / f"{stem.name}.html"

    @property
    def rel_markdown(self) -> Path:
        stem = self.rel_stem
        return stem.parent / f"{stem.name}.md"


@dataclass
class PageResult:
    job: PageJob
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory for the mirror (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Docs version to scrape (e.g. main, v5.0.0rc1). Default: main.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated language codes to include. Default: en.",
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
        help="Limit number of pages processed (useful for quick tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum concurrent fetchers (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=DEFAULT_MIN_DELAY_SECONDS,
        help=(
            "Minimum seconds between requests across all workers. "
            f"Default: {DEFAULT_MIN_DELAY_SECONDS}. Lower to go faster, raise to be gentler."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate and fetch pages but do not write files.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest-based enumeration and crawl links instead.",
    )
    return parser.parse_args()


def ensure_within(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:  # pragma: no cover - safety net
        raise SystemExit(f"Refusing to write outside output root: {target}") from exc


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, FETCH_RETRIES + 1):
        _throttle()
        try:
            with urlopen(req, timeout=30) as response:  # nosec: trusted host required by task
                return response.read()
        except HTTPError as exc:  # pragma: no cover - network variability
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1))
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1))
                _apply_global_backoff(wait)
                time.sleep(wait)
                continue
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def parse_manifest_routes(manifest_bytes: bytes) -> list[str] | None:
    try:
        data = json.loads(manifest_bytes)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    routes: list[str] = []
    for key in data.keys():
        if not isinstance(key, str):
            continue
        if not key.startswith(ROUTE_PREFIX):
            continue
        if key.startswith("src/routes/__"):
            continue
        if not ROUTE_SUFFIX_RE.search(key):
            continue
        rel = key[len(ROUTE_PREFIX) :]
        rel = ROUTE_SUFFIX_RE.sub("", rel)
        routes.append(rel)
    if not routes:
        return None
    return sorted(set(routes))


def load_routes_from_manifest(version: str, language: str) -> list[str] | None:
    manifest_url = MANIFEST_URL_TEMPLATE.format(version=version, lang=language)
    try:
        manifest_bytes = fetch_bytes(manifest_url)
    except Exception:
        return None
    return parse_manifest_routes(manifest_bytes)


def normalize_docs_url(url: str) -> str:
    split = urlsplit(url)
    path = split.path.rstrip("/")
    return split._replace(path=path, fragment="").geturl()


def crawl_routes(seed_url: str, base_prefix: str, limit: int | None) -> list[str]:
    seen: set[str] = set()
    queue: list[str] = [normalize_docs_url(seed_url)]
    routes: list[str] = []

    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch_bytes(url).decode("utf-8", errors="ignore")
        except Exception:
            continue

        route = urlsplit(url).path[len(urlsplit(base_prefix).path) :].strip("/")
        routes.append(route or "index")
        if limit is not None and len(routes) >= limit:
            break

        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            if not href:
                continue
            abs_url = normalize_docs_url(urljoin(url, href))
            if abs_url.startswith(base_prefix) and abs_url not in seen:
                queue.append(abs_url)

    return sorted(set(routes))


def collect_jobs(
    version: str,
    language: str,
    limit: int | None,
    no_manifest: bool,
) -> tuple[list[PageJob], str]:
    base_url = BASE_URL_TEMPLATE.format(version=version, lang=language)
    routes: list[str] | None = None
    source = "crawl"
    if not no_manifest:
        routes = load_routes_from_manifest(version, language)
        if routes:
            source = "manifest"

    if not routes:
        routes = crawl_routes(seed_url=urljoin(base_url, "index"), base_prefix=base_url, limit=limit)

    if limit is not None:
        routes = routes[:limit]

    jobs: list[PageJob] = []
    for route in routes:
        route_norm = route.strip("/")
        url = urljoin(base_url, route_norm or "index")
        jobs.append(PageJob(url=url, route=route_norm or "index", language=language))
    return jobs, source


def clean_markdown(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.strip() in {"Copied", "Copy"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def extract_prose_container(html_text: str) -> str | None:
    soup = BeautifulSoup(html_text, "html.parser")
    container = soup.select_one("div.prose-doc")
    if container is None:
        return None
    return str(container)


def html_to_markdown(container_html: str) -> str:
    cleaned = container_html.replace("<!-- HTML_TAG_START -->", "").replace("<!-- HTML_TAG_END -->", "")
    soup = BeautifulSoup(cleaned, "html.parser")
    for tag in soup(["script", "style", "link", "button", "form"]):
        tag.decompose()
    for svg in soup.find_all("svg"):
        svg.decompose()
    markdown = md(str(soup), heading_style="ATX")
    return clean_markdown(markdown)


def ensure_output_root(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        if output_dir.resolve() == Path("/"):
            raise RuntimeError("Refusing to clean filesystem root.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def process_job(job: PageJob, out_root: Path, dry_run: bool) -> PageResult:
    try:
        raw_bytes = fetch_bytes(job.url)
        raw_html = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        return PageResult(job=job, status="error", error=str(exc))

    if dry_run:
        return PageResult(job=job, status="ok")

    html_out = out_root / "html" / job.language / job.rel_html
    md_out = out_root / "markdown" / job.language / job.rel_markdown
    for target in (html_out, md_out):
        ensure_within(out_root, target)
        target.parent.mkdir(parents=True, exist_ok=True)

    html_out.write_bytes(raw_bytes)
    container_html = extract_prose_container(raw_html) or raw_html
    md_text = html_to_markdown(container_html)
    md_out.write_text(md_text, encoding="utf-8")
    return PageResult(job=job, status="ok")


def write_manifest(
    out_root: Path,
    version: str,
    language: str,
    source: str,
    jobs: Sequence[PageJob],
    results: Sequence[PageResult],
) -> None:
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "base_url": BASE_URL_TEMPLATE.format(version=version, lang=language),
            "manifest_url": MANIFEST_URL_TEMPLATE.format(version=version, lang=language),
            "enumeration": source,
        },
        "version": version,
        "language": language,
        "counts": {
            "pages_requested": len(jobs),
            "pages_ok": len(ok),
            "pages_error": len(errors),
        },
        "pages": [
            {
                "url": r.job.url,
                "route": r.job.route,
                "html_path": str(Path("html") / language / r.job.rel_html),
                "markdown_path": str(Path("markdown") / language / r.job.rel_markdown),
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
    }
    path = out_root / f"manifest-{language}.json"
    ensure_within(out_root, path)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(out_root: Path, version: str, languages: Sequence[str]) -> None:
    lines = [
        "# Transformers Docs Site Mirror",
        "",
        f"- Source site: `https://huggingface.co/docs/transformers/{version}/`",
        f"- Languages: {', '.join(languages)}",
        "",
        "Artifacts:",
        "- `html/<lang>/...` raw rendered HTML per page",
        "- `markdown/<lang>/...` Markdown extracted from the main docs container",
        "- `manifest-<lang>.json` per-language run metadata",
    ]
    path = out_root / "README.md"
    ensure_within(out_root, path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root: Path = args.output
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if not languages:
        raise SystemExit("No languages specified.")

    global _min_delay_seconds
    _min_delay_seconds = max(0.0, float(args.min_delay))

    ensure_output_root(out_root, clean=args.clean)

    for language in languages:
        jobs, source = collect_jobs(
            version=args.version,
            language=language,
            limit=args.limit,
            no_manifest=args.no_manifest,
        )
        print(f"{language}: {len(jobs)} pages ({source} enumeration)")

        results: list[PageResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_job, job, out_root, args.dry_run): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(future_map):
                job = future_map[future]
                try:
                    res = future.result()
                except Exception as exc:  # pragma: no cover - thread errors
                    res = PageResult(job=job, status="error", error=str(exc))
                results.append(res)
                if res.status != "ok":
                    print(f"ERROR fetching {job.url}: {res.error}", file=sys.stderr)

        if args.dry_run:
            ok_count = sum(1 for r in results if r.status == "ok")
            print(f"Dry run complete for {language}: ok={ok_count} errors={len(results)-ok_count}")
            continue

        write_manifest(
            out_root=out_root,
            version=args.version,
            language=language,
            source=source,
            jobs=jobs,
            results=results,
        )

    if not args.dry_run:
        write_readme(out_root, version=args.version, languages=languages)
        print(f"Done. Output at {out_root}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
