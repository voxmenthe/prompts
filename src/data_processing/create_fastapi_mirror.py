#!/usr/bin/env python3
"""
Create a local mirror of the English docs as Markdown.

This script mirrors the contents of `docs/en/docs` into a local output
directory (default: `SPECIALIZED_CONTEXT_AND_DOCS/FASTAPI_DOCS/` in this
repository) preserving the directory structure and assets. It also generates
a top-level README.md with a navigable table of contents by scanning the
mirrored Markdown files and extracting their first heading as the title when
available. When using the defaults, the script ensures the FastAPI repository
is cloned under `~/repos/OTHER_PEOPLES_REPOS/fastapi` before mirroring.

Goals:
- Minimize blast radius: read-only of source docs; writes are isolated to
  an explicit output directory.
- Keep it data-first: directory tree is the source of truth; no
  dependency on MkDocs or YAML parsers.
- English-only: only the `docs/en/docs` subtree is mirrored.

Usage:
  # With defaults (clones FastAPI repo locally if missing)
  python create_fastapi_mirror.py --clean

  # Override repo/output paths
  python create_fastapi_mirror.py --repo /path/to/fastapi --out /path/to/docs --clean

Notes:
- This does not run MkDocs or render macros/templates. Pages are copied as-is.
- Internal links remain relative; assets within `img`, `css`, `js` are copied.
- The generated README.md provides a browsable TOC for quick local search.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


ASSET_DIR_NAMES = {"img", "css", "js"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}

DEFAULT_FASTAPI_REPO_ROOT = Path("~/repos/OTHER_PEOPLES_REPOS/fastapi").expanduser()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "FASTAPI_DOCS_MIRROR"
FASTAPI_REPO_URL = "https://github.com/fastapi/fastapi/tree/master"


@dataclass
class TocEntry:
    path: Path
    title: str


def validate_source_dir(source_docs_dir: Path) -> None:
    if not source_docs_dir.exists():
        raise SystemExit(f"Source docs directory not found: {source_docs_dir}")
    if not source_docs_dir.is_dir():
        raise SystemExit(f"Source docs path is not a directory: {source_docs_dir}")


def is_markdown_file(path: Path) -> bool:
    return path.suffix.lower() in MARKDOWN_SUFFIXES


def discover_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Sort in-place for deterministic traversal
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            yield Path(dirpath) / name


def extract_title_from_markdown(md_path: Path) -> str:
    """Return a concise title for a Markdown file.

    Strategy:
    - First non-empty line starting with '#' is treated as a header; strip anchors.
    - Else, use the file stem.
    """
    try:
        with md_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    # Remove leading hashes and trim inline anchors like `{ #id }`
                    title = re.sub(r"\{[^}]*\}\s*$", "", line.lstrip("# ").strip())
                    # Collapse spaces and trim
                    return re.sub(r"\s+", " ", title).strip()
        return md_path.stem.replace("_", " ")
    except Exception:
        return md_path.stem.replace("_", " ")


def rel_link(from_dir: Path, to_file: Path) -> str:
    return os.path.relpath(to_file, start=from_dir).replace(os.sep, "/")


def build_toc(root: Path, ignore_dirs: set[str] | None = None) -> List[Tuple[Path, List[TocEntry]]]:
    """Build a TOC list per-directory.

    Returns a list of (directory_path, [TocEntry ...]) sorted by path depth,
    suitable for generating a nested README.
    """
    ignore_dirs = ignore_dirs or set()
    per_dir: List[Tuple[Path, List[TocEntry]]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter ignored directories from traversal but keep assets in mirror
        dirnames[:] = [d for d in sorted(dirnames) if d not in ignore_dirs]
        entries: List[TocEntry] = []
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            if is_markdown_file(fpath):
                entries.append(TocEntry(fpath, extract_title_from_markdown(fpath)))
        per_dir.append((Path(dirpath), entries))
    # Sort: shallow to deep to produce a readable nested list later
    per_dir.sort(key=lambda item: (len(item[0].relative_to(root).parts), str(item[0]).lower()))
    return per_dir


def write_root_readme(out_root: Path, toc: List[Tuple[Path, List[TocEntry]]], source_root: Path) -> None:
    lines: List[str] = []
    lines.append("# FastAPI – Local Docs Mirror (English)\n")
    lines.append(
        "This mirror was generated from `docs/en/docs` preserving structure. "
        "Links point to Markdown files within this directory.\n"
    )
    lines.append("")

    # Build nested bullet list (relative paths) with directory headings
    for dir_path, entries in toc:
        rel_dir = dir_path.relative_to(source_root)
        # Skip empty top-level bullets with no Markdown files
        if not entries and rel_dir != Path('.'):
            continue
        heading = "/".join(rel_dir.parts) if rel_dir.parts else "."
        if heading != ".":
            lines.append(f"## {heading}")
        else:
            lines.append("## . (root)")
        for entry in entries:
            rel = rel_link(out_root, out_root / entry.path.relative_to(source_root))
            lines.append(f"- [{entry.title}]({rel})")
        lines.append("")

    (out_root / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        # Copy file-by-file to allow incremental updates
        for path in discover_files(src):
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    else:
        shutil.copytree(src, dst)


def find_repo_root(explicit_repo: Optional[str]) -> Path:
    """Determine the repository root containing docs/en/docs.

    Precedence:
    1) --repo argument if provided and valid
    2) Search upwards from CWD for a directory containing docs/en/docs
    3) Search upwards from the script location for docs/en/docs
    If none found, exit with guidance.
    """
    def has_en_docs(root: Path) -> bool:
        return (root / "docs" / "en" / "docs").is_dir()

    if explicit_repo:
        root = Path(explicit_repo).expanduser().resolve()
        if has_en_docs(root):
            return root
        raise SystemExit(
            f"--repo path does not contain docs/en/docs: {root}. "
            f"Ensure you pass the FastAPI repo root."
        )

    # Try from current working directory upwards
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if has_en_docs(p):
            return p

    # Try from script location upwards
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if has_en_docs(p):
            return p

    raise SystemExit(
        "Could not locate FastAPI repo. Provide --repo /path/to/fastapi (must contain docs/en/docs)."
    )


def ensure_fastapi_repo(repo_root: Path, repo_url: str, allow_clone: bool) -> None:
    """Ensure the FastAPI repository exists at repo_root.

    Clones the upstream repository when permitted. Fails fast if cloning is
    disallowed and the repo is missing to avoid ambiguous state."""

    git_dir = repo_root / ".git"
    if git_dir.is_dir():
        return

    if not allow_clone:
        raise SystemExit(
            f"FastAPI repo not found at {repo_root}. Clone it or rerun without overriding the default path."
        )

    if repo_root.exists():
        # Prevent cloning into a directory that already has contents.
        if any(repo_root.iterdir()):
            raise SystemExit(
                f"Cannot clone FastAPI repo into non-empty directory: {repo_root}."
            )
    else:
        repo_root.mkdir(parents=True, exist_ok=True)

    print(f"Cloning FastAPI repository into {repo_root} ...")
    try:
        subprocess.run(["git", "clone", repo_url, "."], check=True, cwd=str(repo_root))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git clone failed for {repo_url} at {repo_root}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local mirror of the English docs (Markdown)")
    parser.add_argument(
        "--repo",
        default=str(DEFAULT_FASTAPI_REPO_ROOT),
        help=(
            "Path to FastAPI repo root containing docs/en/docs. "
            f"Defaults to {DEFAULT_FASTAPI_REPO_ROOT}."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output directory for the mirror. "
            f"Defaults to {DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before mirroring")
    args = parser.parse_args()

    repo_arg_default = parser.get_default("repo")
    repo_root_candidate = Path(args.repo).expanduser()
    allow_clone = args.repo == repo_arg_default or repo_root_candidate == DEFAULT_FASTAPI_REPO_ROOT
    ensure_fastapi_repo(repo_root_candidate, FASTAPI_REPO_URL, allow_clone=allow_clone)

    repo_root = find_repo_root(str(repo_root_candidate))
    source_docs_dir = (repo_root / "docs" / "en" / "docs").resolve()
    validate_source_dir(source_docs_dir)

    out_dir = Path(args.out).expanduser().resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy the entire English docs tree as-is
    copy_tree(source_docs_dir, out_dir)

    # 2) Generate a human-friendly top-level README with a TOC of Markdown pages
    toc = build_toc(out_dir, ignore_dirs=ASSET_DIR_NAMES)
    write_root_readme(out_dir, toc, source_root=out_dir)

    print(f"Mirrored English docs from {source_docs_dir} -> {out_dir}")
    print(f"Index: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
