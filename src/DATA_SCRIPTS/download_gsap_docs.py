"""
Download GSAP docs listed in a markdown outline file.

Notebook-friendly: Split into cells with `# %%` to allow step-by-step execution
in editors like VS Code or Jupyter (via Python Interactive), while remaining
easy to run as a single script.

Usage (from repo root):
  uv run src/data_processing/download_gsap_docs.py
or
  python src/data_processing/download_gsap_docs.py

Configuration lives in the first cell below. Adjust `OUTLINE_FILE`, `BASE_URLS`,
`PREFIX_CANDIDATES`, and `OUTPUT_ROOT` as needed. The script will attempt each
combination of base URL and prefix until a download succeeds, with retries.
"""

# %%
# Configuration
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]

# Remote index that we fetch and parse for links
INDEX_URL = "https://gsap.com/llms.txt"

# Path to the outline file with markdown links (fallback only, used if remote fetch fails)
OUTLINE_FILE = REPO_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "gsap_greensock_llms.txt"

# Output root where the mirrored directory structure will be written
OUTPUT_ROOT = REPO_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "GSAP_DOCS_MIRROR"

# Candidate base domains. The script will try each in order per file.
# If you know the correct one, put it first for speed.
BASE_URLS: List[str] = [
    "https://gsap.com",
    "https://greensock.com",
]

# Candidate URL prefixes in front of the relative path (e.g. "/llms" + "/docs/v3/GSAP.md")
# Keep "" first to try the path as-is. Some sites host LLM markdown under a special prefix.
PREFIX_CANDIDATES: List[str] = [
    "",
    "/llms",
    "/llms-full",
]

# Concurrency and request behavior
MAX_WORKERS = 12
REQUEST_TIMEOUT_SEC = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
POLITE_DELAY_SEC = 0.0  # add a small delay between requests if needed

# If True, only preview a few items without downloading all
DRY_RUN_PREVIEW_COUNT = 0  # set > 0 to preview without downloading everything

# Skip already-downloaded files (speeds up re-runs)
SKIP_IF_EXISTS = True

# Identify ourselves politely
DEFAULT_HEADERS = {
    "User-Agent": "prompts-repo-gsap-docs-downloader/1.0 (+https://github.com/)"
}


# %%
# Link parsing helpers

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def read_text(path: Path) -> str:
    """Read a UTF-8 text file with clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Outline not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_markdown_links(markdown_text: str) -> List[Tuple[str, str]]:
    """Return list of (label, href) tuples from markdown link syntax.

    Only links ending in .md are of interest for downloads.
    """
    links: List[Tuple[str, str]] = []
    for match in LINK_PATTERN.finditer(markdown_text):
        label = match.group(1).strip()
        href = match.group(2).strip()
        if href.lower().endswith(".md"):
            links.append((label, href))
    return links


def normalize_relpath(href: str) -> str:
    """Ensure the href is treated as a site-relative path without query/fragment.

    Removes any leading and trailing whitespace, strips URL fragments and queries,
    and returns a path starting with a single leading slash.
    """
    href = href.split("#", 1)[0]
    href = href.split("?", 1)[0]
    href = href.strip()
    if not href.startswith("/"):
        href = "/" + href
    # Collapse multiple slashes
    href = re.sub(r"/+/", "/", href)
    return href


def deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# %%
# URL construction and downloading

def candidate_urls_for(relpath: str, base_urls: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """Generate candidate absolute URLs for a given site-relative path."""
    # Guarantee relpath starts with '/'
    relpath = normalize_relpath(relpath)
    urls: List[str] = []
    for base in base_urls:
        base = base.rstrip("/")
        for pref in prefixes:
            pref = pref.strip()
            if pref and not pref.startswith("/"):
                pref = "/" + pref
            urls.append(base + pref + relpath)
    return urls


def http_get_bytes(url: str, timeout: float = REQUEST_TIMEOUT_SEC, headers: Optional[Dict[str, str]] = None) -> bytes:
    """GET the URL and return response bytes or raise an error.

    Uses stdlib to avoid external dependencies.
    """
    req = Request(url, headers=headers or DEFAULT_HEADERS, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_index_text(index_url: str, output_root: Path) -> str:
    """Fetch the latest index (llms.txt) and persist a copy under output_root."""
    print(f"Fetching latest index from: {index_url}")
    output_root.mkdir(parents=True, exist_ok=True)
    content = http_get_bytes(index_url, timeout=REQUEST_TIMEOUT_SEC)
    text = content.decode("utf-8", errors="replace")
    (output_root / "llms.txt").write_text(text, encoding="utf-8")
    return text


def ensure_within_output_root(output_root: Path, relative_path: Path) -> Path:
    """Join and ensure the resolved path stays within output_root (safety guard)."""
    target = (output_root / relative_path).resolve()
    output_root_resolved = output_root.resolve()
    if not str(target).startswith(str(output_root_resolved)):
        raise ValueError(f"Refusing to write outside output root: {target}")
    return target


def local_path_for_output(output_root: Path, relpath: str) -> Path:
    """Compute local filesystem path mirroring the URL path under OUTPUT_ROOT."""
    rel = normalize_relpath(relpath)
    # Drop leading '/'
    rel = rel[1:]
    return ensure_within_output_root(output_root, Path(rel))


@dataclass
class DownloadResult:
    relative_path: str
    url_used: Optional[str]
    success: bool
    local_path: Optional[str]
    error: Optional[str]


def download_one_md(
    relpath: str,
    output_root: Path,
    base_urls: Sequence[str],
    prefixes: Sequence[str],
    max_retries: int = MAX_RETRIES,
    retry_backoff_sec: float = RETRY_BACKOFF_SEC,
    request_timeout: float = REQUEST_TIMEOUT_SEC,
    polite_delay_sec: float = POLITE_DELAY_SEC,
) -> DownloadResult:
    """Attempt to download a single .md resource trying multiple base/prefix combos."""
    local_path = local_path_for_output(output_root, relpath)
    if SKIP_IF_EXISTS and local_path.exists():
        return DownloadResult(
            relative_path=relpath,
            url_used=None,
            success=True,
            local_path=str(local_path),
            error=None,
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[str] = None
    for url in candidate_urls_for(relpath, base_urls, prefixes):
        # Simple retry loop per candidate URL
        for attempt in range(1, max_retries + 1):
            try:
                if polite_delay_sec > 0:
                    time.sleep(polite_delay_sec)
                content = http_get_bytes(url, timeout=request_timeout)
                local_path.write_bytes(content)
                return DownloadResult(
                    relative_path=relpath,
                    url_used=url,
                    success=True,
                    local_path=str(local_path),
                    error=None,
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:  # type: ignore[name-defined]
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    time.sleep(retry_backoff_sec * attempt)
                else:
                    # will try next candidate URL
                    pass
            except Exception as exc:  # unexpected error; still try next candidate URL
                last_error = f"{type(exc).__name__}: {exc}"
                break

    return DownloadResult(
        relative_path=relpath,
        url_used=None,
        success=False,
        local_path=str(local_path),
        error=last_error or "Unknown error",
    )


# %%
# Main orchestration

def main() -> int:
    print(f"Output root:  {OUTPUT_ROOT}")

    # Try fetching fresh index; fall back to local outline
    try:
        text = fetch_index_text(INDEX_URL, OUTPUT_ROOT)
        print("Index fetched successfully.")
    except Exception as exc:
        print(f"[warn] Failed to fetch index: {type(exc).__name__}: {exc}")
        print(f"Falling back to local outline file: {OUTLINE_FILE}")
        text = read_text(OUTLINE_FILE)
    link_pairs = extract_markdown_links(text)
    relpaths = [normalize_relpath(href) for _, href in link_pairs]
    relpaths = [rp for rp in relpaths if rp.lower().endswith(".md")]  # defensive filter
    relpaths = deduplicate_preserve_order(relpaths)

    print(f"Found {len(relpaths)} markdown links to download.")

    if DRY_RUN_PREVIEW_COUNT and DRY_RUN_PREVIEW_COUNT > 0:
        sample = relpaths[:DRY_RUN_PREVIEW_COUNT]
        print("Previewing candidate URLs for first items:")
        for rp in sample:
            print(f"  {rp}")
            for url in candidate_urls_for(rp, BASE_URLS, PREFIX_CANDIDATES)[:6]:
                print(f"    - {url}")
        print("Dry run complete. Set DRY_RUN_PREVIEW_COUNT=0 to download.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Download concurrently
    results: List[DownloadResult] = []
    succeeded = 0
    skipped_existing = 0
    failed = 0

    def worker(rp: str) -> DownloadResult:
        res = download_one_md(
            relpath=rp,
            output_root=OUTPUT_ROOT,
            base_urls=BASE_URLS,
            prefixes=PREFIX_CANDIDATES,
        )
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_rel = {executor.submit(worker, rp): rp for rp in relpaths}
        for fut in concurrent.futures.as_completed(future_to_rel):
            rp = future_to_rel[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.success:
                    if res.url_used is None and SKIP_IF_EXISTS and Path(res.local_path or "").exists():
                        skipped_existing += 1
                        print(f"[skip] exists: {rp}")
                    else:
                        succeeded += 1
                        print(f"[ok]  {rp} <- {res.url_used}")
                else:
                    failed += 1
                    print(f"[err] {rp} :: {res.error}")
            except Exception as exc:
                failed += 1
                print(f"[err] {rp} :: {type(exc).__name__}: {exc}")

    # Write a machine-readable summary
    summary = {
        "index_url": INDEX_URL,
        "outline_file_fallback": str(OUTLINE_FILE),
        "output_root": str(OUTPUT_ROOT),
        "base_urls": BASE_URLS,
        "prefix_candidates": PREFIX_CANDIDATES,
        "counts": {"total": len(relpaths), "succeeded": succeeded, "skipped_existing": skipped_existing, "failed": failed},
        "results": [res.__dict__ for res in results],
    }
    summary_path = OUTPUT_ROOT / "download_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate and persist an ASCII tree of downloaded files
    tree_text = generate_ascii_tree(OUTPUT_ROOT)
    tree_path = OUTPUT_ROOT / "download_tree.txt"
    tree_path.write_text(tree_text, encoding="utf-8")

    print("\nDownload complete.")
    print(f"  succeeded:        {succeeded}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  failed:           {failed}")
    print(f"  details:          {summary_path}")
    print(f"  file tree:        {tree_path}")

    return 0 if failed == 0 else 1


# %%
def generate_ascii_tree(root: Path) -> str:
    """Produce a simple ASCII tree representation for the given directory."""
    root = root.resolve()
    lines: List[str] = [root.name]

    def walk(dir_path: Path, prefix: str) -> None:
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name)
            if entry.is_dir():
                extension = "    " if idx == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    walk(root, "")
    return "\n".join(lines)


# Script entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GSAP docs from llms index.")
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--base-url", action="append", default=None, help="Add a base URL (can repeat). Defaults to gsap.com and greensock.com")
    parser.add_argument("--prefix", action="append", default=None, help="Add a URL prefix (can repeat). Defaults to '', '/llms', '/llms-full'")
    parser.add_argument("--preview", type=int, default=DRY_RUN_PREVIEW_COUNT, help="Preview first N items without downloading all")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing files")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    # Override configuration from CLI
    INDEX_URL = args.index_url
    OUTPUT_ROOT = Path(args.output_root)
    if args.base_url:
        BASE_URLS = list(args.base_url)
    if args.prefix:
        PREFIX_CANDIDATES = list(args.prefix)
    DRY_RUN_PREVIEW_COUNT = args.preview
    SKIP_IF_EXISTS = not args.no_skip
    MAX_WORKERS = args.max_workers

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
