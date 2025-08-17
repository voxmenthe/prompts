"""
Download LiveKit docs listed in the live index file.

Notebook-friendly: Split into cells with `# %%` for step-by-step execution,
but runnable end-to-end as a single script.

Default behavior:
  - Fetch and parse https://docs.livekit.io/llms.txt
  - Extract .md links for the docs site (host defaults to docs.livekit.io)
  - Download concurrently and mirror the path under OUTPUT_ROOT
  - Produce a JSON summary and an ASCII file tree of results

Usage (from repo root):
  uv run src/data_processing/download_livekit_docs.py
or
  python src/data_processing/download_livekit_docs.py
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
from urllib.parse import urlparse, urljoin
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]

# Live index URL
INDEX_URL = "https://docs.livekit.io/llms.txt"

# Output location
OUTPUT_ROOT = REPO_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "LIVEKIT_DOCS_MIRROR"

# Base URL used to resolve relative links
BASE_URL = "https://docs.livekit.io"

# Allowed hosts for downloads (to limit scope to LiveKit docs)
ALLOWED_HOSTS: List[str] = ["docs.livekit.io"]

# Networking & behavior
MAX_WORKERS = 12
REQUEST_TIMEOUT_SEC = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
POLITE_DELAY_SEC = 0.0

# Preview mode: set N > 0 to only preview first N entries
DRY_RUN_PREVIEW_COUNT = 0

# Skip already downloaded files
SKIP_IF_EXISTS = True

# Identify ourselves politely
DEFAULT_HEADERS = {
    "User-Agent": "prompts-repo-livekit-docs-downloader/1.0 (+https://github.com/)"
}


# %%
# Parsing helpers

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def http_get_bytes(url: str, timeout: float = REQUEST_TIMEOUT_SEC, headers: Optional[Dict[str, str]] = None) -> bytes:
    req = Request(url, headers=headers or DEFAULT_HEADERS, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_with_headers(
    url: str,
    timeout: float = REQUEST_TIMEOUT_SEC,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """HTTP GET that returns (status, headers, body).

    Returns 304 with empty body when server responds Not Modified.
    Raises for other HTTP errors.
    """
    req = Request(url, headers=headers or DEFAULT_HEADERS, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
            body = resp.read()
            return status, hdrs, body
    except HTTPError as e:  # type: ignore[reportGeneralTypeIssues]
        if getattr(e, "code", None) == 304:
            # 304 Not Modified – treat as success with no body
            hdrs = {k.lower(): v for k, v in (e.headers.items() if hasattr(e, "headers") else [])}
            return 304, hdrs, b""
        raise


def normalize_markdown_body(body: bytes) -> bytes:
    """Remove volatile lines (e.g., 'This document was rendered at ...') to reduce churn.

    The LiveKit docs append a timestamp line that changes on every render; stripping it
    keeps the local mirror stable when the substantive content hasn't changed.
    """
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return body
    # Remove lines that match the timestamp footer
    # Example: "This document was rendered at 2025-08-17T17:54:06.676Z."
    ts_pattern = re.compile(r"^This document was rendered at .*Z\.$", re.MULTILINE)
    text = ts_pattern.sub("", text)
    # Collapse multiple trailing blank lines that may result
    text = re.sub(r"\n{3,}$", "\n\n", text)
    return text.encode("utf-8")


def fetch_index_text(index_url: str, output_root: Path) -> str:
    print(f"Fetching latest index from: {index_url}")
    output_root.mkdir(parents=True, exist_ok=True)
    content = http_get_bytes(index_url, timeout=REQUEST_TIMEOUT_SEC)
    text = content.decode("utf-8", errors="replace")
    (output_root / "llms.txt").write_text(text, encoding="utf-8")
    return text


def extract_markdown_links(markdown_text: str) -> List[Tuple[str, str]]:
    links: List[Tuple[str, str]] = []
    for match in LINK_PATTERN.finditer(markdown_text):
        label = match.group(1).strip()
        href = match.group(2).strip()
        if href.lower().endswith(".md"):
            links.append((label, href))
    return links


def deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def strip_query_and_fragment(url: str) -> str:
    url_no_fragment = url.split("#", 1)[0]
    url_no_query = url_no_fragment.split("?", 1)[0]
    return url_no_query


def to_download_plan(
    links: List[Tuple[str, str]], base_url: str, allowed_hosts: Sequence[str], allow_external: bool
) -> List[Tuple[str, str, str]]:
    """Build a download plan of (final_url, relative_path, source_href).

    - final_url: fully-qualified URL to download
    - relative_path: path mirrored under OUTPUT_ROOT for local saving
    - source_href: original href for logging
    """
    plan: List[Tuple[str, str, str]] = []
    for _, href in links:
        href = strip_query_and_fragment(href.strip())
        if href.startswith("http://") or href.startswith("https://"):
            parsed = urlparse(href)
            if not allow_external and parsed.netloc not in allowed_hosts:
                continue
            # Only mirror path component under output; keep leading slash trimmed
            rel = parsed.path.lstrip("/")
            if not rel.lower().endswith(".md"):
                continue
            final_url = href
            relative_path = rel
        else:
            # Relative link → resolve against base_url
            final_url = urljoin(base_url.rstrip("/") + "/", href)
            parsed = urlparse(final_url)
            if parsed.netloc not in allowed_hosts and not allow_external:
                continue
            rel = parsed.path.lstrip("/")
            if not rel.lower().endswith(".md"):
                continue
            relative_path = rel

        plan.append((final_url, relative_path, href))

    # Deduplicate by relative_path to avoid redundant downloads
    unique: Dict[str, Tuple[str, str, str]] = {}
    for final_url, rel, src in plan:
        if rel not in unique:
            unique[rel] = (final_url, rel, src)
    return list(unique.values())


def ensure_within_output_root(output_root: Path, relative_path: Path) -> Path:
    target = (output_root / relative_path).resolve()
    output_root_resolved = output_root.resolve()
    if not str(target).startswith(str(output_root_resolved)):
        raise ValueError(f"Refusing to write outside output root: {target}")
    return target


def local_path_for_output(output_root: Path, relative_path: str) -> Path:
    return ensure_within_output_root(output_root, Path(relative_path))


# %%
# Download worker

@dataclass
class DownloadResult:
    relative_path: str
    url_used: Optional[str]
    success: bool
    local_path: Optional[str]
    error: Optional[str]


def meta_path_for(local_path: Path) -> Path:
    """Return sidecar JSON metadata path for a downloaded file."""
    return local_path.with_suffix(local_path.suffix + ".meta.json")


def load_sidecar_meta(local_path: Path) -> Dict[str, str]:
    """Load previously saved ETag/Last-Modified if present."""
    mp = meta_path_for(local_path)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_sidecar_meta(local_path: Path, headers: Dict[str, str]) -> None:
    mp = meta_path_for(local_path)
    meta = {
        "etag": headers.get("etag", ""),
        "last_modified": headers.get("last-modified", ""),
        "content_length": headers.get("content-length", ""),
    }
    # Do not write a sidecar file if all values are empty (not useful)
    if not (meta["etag"] or meta["last_modified"] or meta["content_length"]):
        return
    try:
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def download_one(
    final_url: str,
    relative_path: str,
    output_root: Path,
    max_retries: int = MAX_RETRIES,
    retry_backoff_sec: float = RETRY_BACKOFF_SEC,
    request_timeout: float = REQUEST_TIMEOUT_SEC,
    polite_delay_sec: float = POLITE_DELAY_SEC,
) -> DownloadResult:
    local_path = local_path_for_output(output_root, relative_path)
    # If file exists and skipping is enabled, perform a conditional GET using stored metadata.
    # If not modified (304) or content identical, we treat as a skip (url_used=None).
    if local_path.exists():
        if SKIP_IF_EXISTS:
            # Attempt conditional GET with ETag / Last-Modified
            meta = load_sidecar_meta(local_path)
            cond_headers = dict(DEFAULT_HEADERS)
            if meta.get("etag"):
                cond_headers["If-None-Match"] = meta["etag"]
            if meta.get("last_modified"):
                cond_headers["If-Modified-Since"] = meta["last_modified"]
            try:
                if polite_delay_sec > 0:
                    time.sleep(polite_delay_sec)
                status, hdrs, body = http_get_with_headers(final_url, timeout=request_timeout, headers=cond_headers)
                if status == 304:
                    return DownloadResult(relative_path=relative_path, url_used=None, success=True, local_path=str(local_path), error=None)
                # status 200 – overwrite file (we avoid an equality diff to keep it simple)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                body = normalize_markdown_body(body)
                local_path.write_bytes(body)
                save_sidecar_meta(local_path, hdrs)
                return DownloadResult(relative_path=relative_path, url_used=final_url, success=True, local_path=str(local_path), error=None)
            except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:  # type: ignore[name-defined]
                # Fall through to retry loop below (unconditional GET) on network errors
                pass
        else:
            # For --no-skip, we will force re-download below
            pass

    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            if polite_delay_sec > 0:
                time.sleep(polite_delay_sec)
            # Unconditional GET (used when file missing, --no-skip, or conditional path failed)
            status, hdrs, content = http_get_with_headers(final_url, timeout=request_timeout, headers=DEFAULT_HEADERS)
            # status should be 200 here
            body = normalize_markdown_body(content)
            local_path.write_bytes(body)
            save_sidecar_meta(local_path, hdrs)
            return DownloadResult(relative_path=relative_path, url_used=final_url, success=True, local_path=str(local_path), error=None)
        except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:  # type: ignore[name-defined]
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * attempt)
            else:
                pass
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break

    return DownloadResult(relative_path=relative_path, url_used=None, success=False, local_path=str(local_path), error=last_error or "Unknown error")


# %%
# ASCII tree helper

def generate_ascii_tree(root: Path) -> str:
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


# %%
# Main

def main() -> int:
    print(f"Output root:  {OUTPUT_ROOT}")

    try:
        text = fetch_index_text(INDEX_URL, OUTPUT_ROOT)
        print("Index fetched successfully.")
    except Exception as exc:
        print(f"[err] Failed to fetch index: {type(exc).__name__}: {exc}")
        return 1

    link_pairs = extract_markdown_links(text)
    plan = to_download_plan(
        link_pairs,
        base_url=BASE_URL,
        allowed_hosts=ALLOWED_HOSTS,
        allow_external=False,
    )
    # Keep only .md files and preserve order by relative path
    relpaths = [rel for _, rel, _ in plan]
    relpaths = deduplicate_preserve_order(relpaths)
    print(f"Found {len(relpaths)} markdown links to download.")

    if DRY_RUN_PREVIEW_COUNT and DRY_RUN_PREVIEW_COUNT > 0:
        print("Previewing first items:")
        for final_url, rel, src in plan[:DRY_RUN_PREVIEW_COUNT]:
            print(f"  {rel}\n    - {final_url}\n    - src: {src}")
        print("Dry run complete. Set DRY_RUN_PREVIEW_COUNT=0 to download.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: List[DownloadResult] = []
    succeeded = 0
    skipped_existing = 0
    failed = 0

    def worker(item: Tuple[str, str, str]) -> DownloadResult:
        final_url, rel, _ = item
        return download_one(final_url, rel, OUTPUT_ROOT)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(worker, item): item for item in plan}
        for fut in concurrent.futures.as_completed(future_to_item):
            final_url, rel, _ = future_to_item[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.success:
                    if res.url_used is None and SKIP_IF_EXISTS and Path(res.local_path or "").exists():
                        skipped_existing += 1
                        print(f"[skip] exists: {rel}")
                    else:
                        succeeded += 1
                        print(f"[ok]  {rel} <- {res.url_used}")
                else:
                    failed += 1
                    print(f"[err] {rel} :: {res.error}")
            except Exception as exc:
                failed += 1
                print(f"[err] {rel} :: {type(exc).__name__}: {exc}")

    # Summary JSON
    summary = {
        "index_url": INDEX_URL,
        "base_url": BASE_URL,
        "allowed_hosts": ALLOWED_HOSTS,
        "output_root": str(OUTPUT_ROOT),
        "counts": {"total": len(plan), "succeeded": succeeded, "skipped_existing": skipped_existing, "failed": failed},
        "results": [res.__dict__ for res in results],
    }
    summary_path = OUTPUT_ROOT / "download_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ASCII tree
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
# Entrypoint & CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LiveKit docs from llms index.")
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--allow-host", action="append", default=None, help="Allowed host (can repeat). Defaults to docs.livekit.io")
    parser.add_argument("--allow-external", action="store_true", help="Allow downloading .md from hosts outside allowed-hosts")
    parser.add_argument("--preview", type=int, default=DRY_RUN_PREVIEW_COUNT, help="Preview first N items without downloading all")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing files")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    INDEX_URL = args.index_url
    OUTPUT_ROOT = Path(args.output_root)
    BASE_URL = args.base_url
    if args.allow_host:
        ALLOWED_HOSTS = list(args.allow_host)
    DRY_RUN_PREVIEW_COUNT = args.preview
    SKIP_IF_EXISTS = not args.no_skip
    MAX_WORKERS = args.max_workers

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


