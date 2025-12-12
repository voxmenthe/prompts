#!/usr/bin/env python3
"""
Mirror the Hugging Face Transformers docs from the public doc-build dataset.

Default behavior:
- Download `transformers/main.zip` from `hf-doc-build/doc-build`
- Extract only the English docs
- Emit raw HTML and a Markdown-converted version for each page

This keeps blast radius low by reusing the published artifacts (no heavy
`pip install -e ".[docs]"` build) while still producing Markdown for local
search/indexing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import yaml  # type: ignore[import-not-found]
from bs4 import BeautifulSoup  # type: ignore[import-not-found]
from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
from huggingface_hub.utils import HfHubHTTPError  # type: ignore[import-not-found]
from markdownify import markdownify as md  # type: ignore[import-not-found]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "SPECIALIZED_CONTEXT_AND_DOCS" / "TRANSFORMERS_DOCS_MIRROR"
DATASET_REPO = "hf-doc-build/doc-build"
DEFAULT_VERSION = "main"
DEFAULT_LANGS = ("en",)

HTML_SUFFIXES = {".html", ".htm"}
MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}
TEXT_SUFFIXES = {".txt", ".yml", ".yaml"}
ASSET_SKIP_PREFIXES = {"_app/immutable/chunks"}  # optionally skipped when --skip-assets is set


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
        help="Docs version to pull (e.g. main, v4.56.2). Default: main.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated language codes to include. Default: en.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Do not extract JS/CSS/assets (keeps output small; still writes HTML/Markdown).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of HTML files converted (for quick tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and enumerate but do not write files.",
    )
    return parser.parse_args()


def ensure_within(root: Path, target: Path) -> None:
    """Fail fast if target escapes the output root."""
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:  # pragma: no cover - safety net
        raise SystemExit(f"Refusing to write outside output root: {target}") from exc


def extract_dataset_revision(cached_path: Path) -> str | None:
    parts = cached_path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def load_versions() -> list[str]:
    versions_path = hf_hub_download(
        DATASET_REPO,
        filename="transformers/_versions.yml",
        repo_type="dataset",
    )
    with open(versions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    versions: list[str] = []
    for entry in data:
        version = entry.get("version")
        if version:
            versions.append(str(version))
    return versions


def available_languages(zip_file: ZipFile, version: str) -> set[str]:
    langs: set[str] = set()
    prefix = f"transformers/{version}/"
    for name in zip_file.namelist():
        if not name.startswith(prefix):
            continue
        parts = Path(name).parts
        if len(parts) >= 4:
            langs.add(parts[2])
    # Always assume English is available (default docs), even if not explicitly tagged
    langs.add("en")
    return langs


def clean_markdown(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        if line.strip() in {"Copied", "Copy"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def html_to_markdown(html_text: str) -> str:
    cleaned = html_text.replace("<!-- HTML_TAG_START -->", "").replace("<!-- HTML_TAG_END -->", "")
    soup = BeautifulSoup(cleaned, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    markdown = md(str(soup), heading_style="ATX")
    return clean_markdown(markdown)


def iter_language_entries(
    zip_file: ZipFile, version: str, language: str
) -> Iterable[tuple[Path, bytes]]:
    prefix = f"transformers/{version}/{language}/"
    root_prefix = f"transformers/{version}/"

    # Check if explicit language folder exists
    has_lang_folder = any(n.startswith(prefix) for n in zip_file.namelist())

    if has_lang_folder:
        for name in zip_file.namelist():
            if not name.startswith(prefix):
                continue
            if name.endswith("/"):
                continue  # directory entry
            rel = Path(name).relative_to(prefix)
            yield rel, zip_file.read(name)
    elif language == "en":
        # Fallback to root, excluding other likely language dirs (2-letter codes)
        for name in zip_file.namelist():
            if not name.startswith(root_prefix):
                continue
            if name.endswith("/"):
                continue

            rel = Path(name).relative_to(root_prefix)
            # Heuristic: exclude top-level 2-letter directories (e.g. fr/, es/)
            if len(rel.parts) > 1 and len(rel.parts[0]) == 2:
                continue

            yield rel, zip_file.read(name)


def mirror_language(
    zip_file: ZipFile,
    version: str,
    language: str,
    out_root: Path,
    skip_assets: bool,
    limit: int | None,
    dry_run: bool,
) -> dict[str, int]:
    counts = {"html": 0, "markdown": 0, "assets": 0, "raw_md": 0}
    html_written = 0
    for rel_path, blob in iter_language_entries(zip_file, version, language):
        suffix = rel_path.suffix.lower()
        rel_str = rel_path.as_posix()

        if suffix in HTML_SUFFIXES:
            if limit is not None and html_written >= limit:
                continue
            html_written += 1
            counts["html"] += 1
            html_out = out_root / "html" / language / rel_path
            md_out = out_root / "markdown" / language / rel_path.with_suffix(".md")
            if not dry_run:
                for target in (html_out, md_out):
                    ensure_within(out_root, target)
                html_out.parent.mkdir(parents=True, exist_ok=True)
                md_out.parent.mkdir(parents=True, exist_ok=True)
                html_out.write_bytes(blob)
                markdown = html_to_markdown(blob.decode("utf-8", errors="ignore"))
                md_out.write_text(markdown, encoding="utf-8")
            counts["markdown"] += 1
            continue

        if suffix in MARKDOWN_SUFFIXES:
            counts["raw_md"] += 1
            md_out = out_root / "markdown" / language / rel_path
            if not dry_run:
                ensure_within(out_root, md_out)
                md_out.parent.mkdir(parents=True, exist_ok=True)
                md_out.write_bytes(blob)
            continue

        if suffix in TEXT_SUFFIXES:
            counts["raw_md"] += 1
            txt_out = out_root / "markdown" / language / rel_path
            if not dry_run:
                ensure_within(out_root, txt_out)
                txt_out.parent.mkdir(parents=True, exist_ok=True)
                txt_out.write_bytes(blob)
            continue

        # Assets (JS/CSS/images/etc.)
        is_chunk = any(rel_str.startswith(prefix) for prefix in ASSET_SKIP_PREFIXES)
        if skip_assets and is_chunk:
            continue
        counts["assets"] += 1
        if dry_run:
            continue
        asset_out = out_root / "assets" / language / rel_path
        ensure_within(out_root, asset_out)
        asset_out.parent.mkdir(parents=True, exist_ok=True)
        asset_out.write_bytes(blob)
    return counts


def write_manifest(
    out_root: Path,
    version: str,
    requested_version: str,
    dataset_revision: str | None,
    languages: Sequence[str],
    per_lang_counts: dict[str, dict[str, int]],
    source_zip: Path,
    skip_assets: bool,
    limit: int | None,
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": DATASET_REPO,
            "version": version,
            "requested_version": requested_version,
            "dataset_revision": dataset_revision,
            "zip_path": str(source_zip),
        },
        "languages": languages,
        "skip_assets": skip_assets,
        "limit": limit,
        "counts": per_lang_counts,
        "output_root": str(out_root),
    }
    path = out_root / "manifest.json"
    ensure_within(out_root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(
    out_root: Path,
    version: str,
    requested_version: str,
    languages: Sequence[str],
    skip_assets: bool,
) -> None:
    lines = [
        "# Transformers Docs Mirror",
        "",
        f"- Source dataset: `{DATASET_REPO}` (`transformers/{version}.zip`)",
        f"- Requested version: {requested_version}",
        f"- Languages: {', '.join(languages)}",
        f"- Assets included: {'no (chunks skipped)' if skip_assets else 'yes'}",
        "",
        "Artifacts:",
        "- `html/<lang>/...` raw pages from the dataset",
        "- `markdown/<lang>/...` converted from HTML (or copied when already markdown/txt)",
        "- `assets/<lang>/...` site assets (JS/CSS/images) when included",
        "- `manifest.json` run metadata",
    ]
    path = out_root / "README.md"
    ensure_within(out_root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.output
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    requested_version = args.version
    versions_ordered = load_versions()

    def try_version(ver: str) -> tuple[Path, set[str]] | None:
        print(f"Checking version {ver} for languages {languages}...")
        try:
            candidate_zip = hf_hub_download(
                DATASET_REPO,
                filename=f"transformers/{ver}.zip",
                repo_type="dataset",
            )
        except HfHubHTTPError:
            return None
        with ZipFile(candidate_zip, "r") as zf:
            available = available_languages(zf, ver)
        return Path(candidate_zip), available

    chosen_version: str | None = None
    zip_path: Path | None = None

    # Try requested version first, then walk the versions list until all languages are present.
    sequence: list[str] = [requested_version] + [v for v in versions_ordered if v != requested_version]
    for ver in sequence:
        candidate = try_version(ver)
        if not candidate:
            continue
        zip_file_path, langs_available = candidate
        if set(languages).issubset(langs_available):
            chosen_version = ver
            zip_path = zip_file_path
            if ver != requested_version:
                print(
                    f"Requested version '{requested_version}' lacks {languages}; "
                    f"using latest available version with those languages: '{ver}'."
                )
            break

    if chosen_version is None or zip_path is None:
        raise SystemExit(
            f"Unable to find a transformers docs archive containing languages {languages}. "
            f"Checked versions: {sequence[:10]}..."
        )

    dataset_revision = extract_dataset_revision(zip_path)
    per_lang_counts: dict[str, dict[str, int]] = {}

    with ZipFile(zip_path, "r") as zf:
        for lang in languages:
            counts = mirror_language(
                zip_file=zf,
                version=chosen_version,
                language=lang,
                out_root=out_root,
                skip_assets=args.skip_assets,
                limit=args.limit,
                dry_run=args.dry_run,
            )
            per_lang_counts[lang] = counts

    if args.dry_run:
        print("Dry run complete. Counts:", per_lang_counts)
        sys.exit(0)

    write_manifest(
        out_root=out_root,
        version=chosen_version,
        requested_version=requested_version,
        dataset_revision=dataset_revision,
        languages=languages,
        per_lang_counts=per_lang_counts,
        source_zip=Path(zip_path),
        skip_assets=args.skip_assets,
        limit=args.limit,
    )
    write_readme(
        out_root=out_root,
        version=chosen_version,
        requested_version=requested_version,
        languages=languages,
        skip_assets=args.skip_assets,
    )
    print(f"Done. Output at {out_root}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
