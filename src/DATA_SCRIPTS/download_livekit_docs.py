"""
Download LiveKit docs listed in the live index file.

Notebook-friendly: Split into cells with `# %%` for step-by-step execution,
but runnable end-to-end as a single script.

Default behavior:
  - Fetch and parse https://docs.livekit.io/llms.txt
  - Extract .md links for the docs site (host defaults to docs.livekit.io)
  - Download concurrently and mirror the path under OUTPUT_ROOT
  - Produce a JSON summary and an ASCII file tree of results
  - Always overwrite local files; report which docs were created or modified (ignoring timestamp-only footer)

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
import hashlib
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
# When True, delete local copies of pages that now return 404
PRUNE_MISSING = False

# Preview mode: set N > 0 to only preview first N entries
DRY_RUN_PREVIEW_COUNT = 0

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
    # One of: "created", "modified", "unchanged" (when content identical after normalization)
    change: Optional[str] = None
    http_status: Optional[int] = None
    missing: bool = False
    pruned: bool = False


def download_one(
    final_url: str,
    relative_path: str,
    output_root: Path,
    prune_missing: bool = PRUNE_MISSING,
    max_retries: int = MAX_RETRIES,
    retry_backoff_sec: float = RETRY_BACKOFF_SEC,
    request_timeout: float = REQUEST_TIMEOUT_SEC,
    polite_delay_sec: float = POLITE_DELAY_SEC,
) -> DownloadResult:
    local_path = local_path_for_output(output_root, relative_path)
    existed_before = local_path.exists()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            if polite_delay_sec > 0:
                time.sleep(polite_delay_sec)
            # Single unconditional GET
            status, hdrs, content = http_get_with_headers(final_url, timeout=request_timeout, headers=DEFAULT_HEADERS)
            # Expect 200
            body_norm = normalize_markdown_body(content)
            new_hash = hashlib.sha256(body_norm).hexdigest()
            # Determine change kind
            if existed_before:
                try:
                    old_bytes = local_path.read_bytes()
                    old_hash = hashlib.sha256(old_bytes).hexdigest()
                except Exception:
                    old_hash = ""
                change = "unchanged" if (old_hash and old_hash == new_hash) else "modified"
            else:
                change = "created"
            # Ensure dir and write file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(body_norm)
            return DownloadResult(
                relative_path=relative_path,
                url_used=final_url,
                success=True,
                local_path=str(local_path),
                error=None,
                change=change,
                http_status=status,
            )
        except HTTPError as exc:  # type: ignore[reportGeneralTypeIssues]
            status_code = getattr(exc, "code", None)
            if status_code == 404:
                # Docs occasionally move/vanish; treat 404 as a missing page, not a hard failure.
                pruned = False
                if prune_missing and local_path.exists():
                    try:
                        local_path.unlink()
                        pruned = True
                    except Exception as delete_exc:
                        last_error = f"HTTP 404 Not Found; prune failed: {delete_exc}"
                        return DownloadResult(
                            relative_path=relative_path,
                            url_used=final_url,
                            success=False,
                            local_path=str(local_path),
                            error=last_error,
                            change=None,
                            http_status=status_code,
                            missing=True,
                            pruned=False,
                        )
                return DownloadResult(
                    relative_path=relative_path,
                    url_used=final_url,
                    success=False,
                    local_path=str(local_path),
                    error=f"HTTP 404 Not Found",
                    change=None,
                    http_status=status_code,
                    missing=True,
                    pruned=pruned,
                )
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * attempt)
            else:
                pass
        except (URLError, TimeoutError, socket.timeout) as exc:  # type: ignore[name-defined]
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
    failed = 0
    missing_count = 0
    pruned_missing_count = 0
    created_count = 0
    modified_count = 0
    unchanged_count = 0

    def worker(item: Tuple[str, str, str]) -> DownloadResult:
        final_url, rel, _ = item
        return download_one(final_url, rel, OUTPUT_ROOT, prune_missing=PRUNE_MISSING)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(worker, item): item for item in plan}
        for fut in concurrent.futures.as_completed(future_to_item):
            final_url, rel, _ = future_to_item[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.success:
                    succeeded += 1
                    tag = "same"
                    if res.change == "created":
                        created_count += 1
                        tag = "new"
                    elif res.change == "modified":
                        modified_count += 1
                        tag = "mod"
                    else:
                        unchanged_count += 1
                        tag = "same"
                    print(f"[ok:{tag}] {rel}")
                else:
                    if res.missing:
                        missing_count += 1
                        if res.pruned:
                            pruned_missing_count += 1
                            print(f"[miss] {rel} :: {res.error} (pruned local copy)")
                        else:
                            print(f"[miss] {rel} :: {res.error}")
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
        "counts": {
            "total": len(plan),
            "succeeded": succeeded,
            "failed": failed,
            "missing": missing_count,
            "pruned_missing": pruned_missing_count,
            "created": created_count,
            "modified": modified_count,
            "unchanged": unchanged_count,
        },
        "results": [res.__dict__ for res in results],
    }
    summary_path = OUTPUT_ROOT / "download_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ASCII tree
    tree_text = generate_ascii_tree(OUTPUT_ROOT)
    tree_path = OUTPUT_ROOT / "download_tree.txt"
    tree_path.write_text(tree_text, encoding="utf-8")

    print("\nDownload complete.")
    print(f"  succeeded:  {succeeded}")
    print(f"  failed:     {failed}")
    print(f"  missing:    {missing_count}")
    if pruned_missing_count:
        print(f"  pruned:     {pruned_missing_count}")
    print(f"  created:    {created_count}")
    print(f"  modified:   {modified_count}")
    print(f"  unchanged:  {unchanged_count}")
    print(f"  details:          {summary_path}")
    print(f"  file tree:        {tree_path}")

    # Final message: which files had substantive changes (ignoring timestamp footer)
    created = sorted([r.relative_path for r in results if r.success and r.change == "created"])  # type: ignore[attr-defined]
    modified = sorted([r.relative_path for r in results if r.success and r.change == "modified"])  # type: ignore[attr-defined]
    missing_pruned = sorted([r.relative_path for r in results if r.missing and r.pruned])
    missing_kept = sorted([r.relative_path for r in results if r.missing and not r.pruned])

    if created or modified:
        print("\nContent changes (ignoring timestamp footer):")
        if created:
            print("  [created]")
            for p in created:
                print(f"    - {p}")
        if modified:
            print("  [modified]")
            for p in modified:
                print(f"    - {p}")
    else:
        print("\nNo content changes detected (ignoring timestamp footer).")

    if missing_pruned or missing_kept:
        if missing_pruned:
            print("\nMissing (404) pages; local copies were pruned:")
            for p in missing_pruned:
                print(f"  - {p}")
        if missing_kept:
            print("\nMissing (404) pages; local files left untouched:")
            for p in missing_kept:
                print(f"  - {p}")

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
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--prune-missing", action="store_true", help="Delete local copies of pages that now return 404")
    args = parser.parse_args()

    INDEX_URL = args.index_url
    OUTPUT_ROOT = Path(args.output_root)
    BASE_URL = args.base_url
    if args.allow_host:
        ALLOWED_HOSTS = list(args.allow_host)
    DRY_RUN_PREVIEW_COUNT = args.preview
    MAX_WORKERS = args.max_workers
    PRUNE_MISSING = args.prune_missing

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
