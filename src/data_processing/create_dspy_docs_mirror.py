#!/usr/bin/env python3
"""
Create a local Markdown mirror of the DSPy docs site.

This script mirrors the content under `docs/docs/` from a DSPy repo into a new
directory `DSPY_DOCS/`, preserving the folder structure. It also:

- Converts `.ipynb` notebooks to Markdown (basic conversion: Markdown and code cells).
- Copies common static assets (images, JS, CSS, SVG, etc.).
- Applies simple redirect mappings from `docs/mkdocs.yml` (if present),
  creating additional `.md` files that mirror those mappings.

Dependencies: Python 3 standard library only.

Configuration:
- Edit the config section near the top of this file to set:
  - `DSPY_REPO_ROOT`: absolute path to the DSPy repo (defaults to
    `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy`).
  - `OUTPUT_DIR`: absolute path to the output mirror directory. If `None`,
    uses `./DSPY_DOCS` relative to the current working directory.
  - `CLEAN_OUTPUT`: if `True`, removes any existing output directory first.

Usage:
    python mirror_docs.py

Notes:
- The notebook to Markdown conversion does not include rich outputs. It includes
  Markdown and code cells only, which suffices for a readable text mirror.
- Asset links are preserved relative to their mirrored locations. The script copies
  the entire directory structure under `docs/docs`, so most relative links should work.
- Output defaults to ./DSPY_DOCS where you run the script.
- API expansion uses DSPY_REPO_ROOT to find source for signatures and docstrings.
- If you want logs in your terminal, stderr is used for status; redirect with `python
mirror_docs.py 2>&1 | tee mirror.log` if helpful.
"""

from __future__ import annotations
import json
import subprocess
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import ast


# --------------------
# Config (edit below)
# --------------------

# Absolute path to the DSPy repo root when running this script outside the repo.
# Default matches your current setup. Change if needed.
DSPY_REPO_ROOT = Path("/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy")

# Output directory to create the DSPY_DOCS mirror in. If None, uses
# a folder named 'DSPY_DOCS' in the current working directory.
OUTPUT_DIR: Optional[Path] = Path("/Volumes/cdrive/repos/prompts/SPECIALIZED_CONTEXT_AND_DOCS/DSPY_DOCS_MIRROR")

# Whether to remove existing OUTPUT_DIR before mirroring.
CLEAN_OUTPUT: bool = True

# Whether to run `git pull` in the DSPy repo before mirroring.
GIT_PULL: bool = True


ASSET_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".bmp",
    ".tiff",
    ".css",
    ".js",
    ".json",
    ".pdf",
}


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def load_redirect_maps(mkdocs_yml: Path) -> Dict[str, str]:
    """
    Extract a minimal redirects mapping from `mkdocs.yml` without requiring PyYAML.
    It looks for a `redirects` -> `redirect_maps:` block and collects lines of the form:

        "from/path.md": "to/path.ipynb"

    Returns a dict mapping the (normalized) from-path (relative to docs root) to the to-path.
    """
    if not mkdocs_yml.exists():
        return {}

    text = mkdocs_yml.read_text(encoding="utf-8")

    # Find the redirects block first.
    redirects_idx = text.find("\n    - redirects:")
    if redirects_idx == -1:
        return {}

    sub = text[redirects_idx:]
    maps_idx = sub.find("redirect_maps:")
    if maps_idx == -1:
        return {}

    after_maps = sub[maps_idx + len("redirect_maps:") :]

    # Collect quoted pairs until the block obviously ends (naive but effective
    # for this repo). Stop when dedent to a top-level or new plugin section.
    mappings: Dict[str, str] = {}
    for line in after_maps.splitlines():
        # Stop when we seem to leave the indented block
        if re.match(r"^\S", line):  # non-indented line
            break
        m = re.search(r'"([^"]+)"\s*:\s*"([^"]+)"', line)
        if m:
            from_path, to_path = m.group(1), m.group(2)
            # Normalize: the mkdocs config sometimes includes a leading "docs/"
            # because docs_dir = "docs/". We want paths relative to docs_dir.
            if from_path.startswith("docs/"):
                from_path = from_path[len("docs/") :]
            mappings[from_path] = to_path

    return mappings


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        # Safety: only remove if it's a directory inside the repo, not root.
        if path.is_dir() and path.name == "DSPY_DOCS":
            shutil.rmtree(path)
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_notebook_language(nb: dict) -> str:
    try:
        return nb.get("metadata", {}).get("language_info", {}).get("name", "python") or "python"
    except Exception:
        return "python"


def convert_ipynb_to_markdown(ipynb_path: Path) -> str:
    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    language = read_notebook_language(nb)

    out_lines: List[str] = []
    cells: Iterable[dict] = nb.get("cells", [])
    for cell in cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        # Some notebooks store source as string, normalize to list
        if isinstance(source, str):
            source_lines = source.splitlines()
        else:
            source_lines = [s.rstrip("\n") for s in source]

        if cell_type == "markdown":
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.extend(source_lines)
            out_lines.append("")
        elif cell_type == "code":
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(f"```{language}")
            out_lines.extend(source_lines)
            out_lines.append("```")
            out_lines.append("")
            # Optionally include simple stream outputs (stdout). Skipping rich outputs
            outputs = cell.get("outputs", [])
            for out in outputs:
                if out.get("output_type") == "stream" and out.get("text"):
                    txt = out["text"]
                    if isinstance(txt, list):
                        txt_lines = [t.rstrip("\n") for t in txt]
                    else:
                        txt_lines = str(txt).splitlines()
                    out_lines.append("```text")
                    out_lines.extend(txt_lines)
                    out_lines.append("```")
                    out_lines.append("")
        else:
            # Unknown cell type; skip but keep spacing
            if out_lines and out_lines[-1] != "":
                out_lines.append("")

    # Trim trailing whitespace lines
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    header = [
        f"<!-- Auto-generated from {ipynb_path.as_posix()} on {ts} -->",
        "",
    ]
    return "\n".join(header + out_lines) + "\n"


def copy_or_convert_file(src_file: Path, dst_file: Path) -> Tuple[str, Optional[str]]:
    """
    Copy or convert a source file to the destination path.
    Returns a tuple of (action, extra) for logging, e.g., ("copy", "image"), ("convert", "ipynb").
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    ext = src_file.suffix.lower()

    if ext == ".md":
        shutil.copy2(src_file, dst_file)
        return ("copy", "md")
    elif ext == ".ipynb":
        md_text = convert_ipynb_to_markdown(src_file)
        dst_md = dst_file.with_suffix(".md")
        dst_md.write_text(md_text, encoding="utf-8")
        return ("convert", "ipynb")
    elif ext in ASSET_EXTS:
        shutil.copy2(src_file, dst_file)
        return ("copy", ext.lstrip("."))
    else:
        # Skip other file types silently, but allow copying small helpful assets
        # like YAML files placed alongside docs if any.
        if ext in {".yml", ".yaml"}:
            shutil.copy2(src_file, dst_file)
            return ("copy", ext.lstrip("."))
        return ("skip", ext.lstrip("."))


def mirror_tree(src_root: Path, dst_root: Path) -> Dict[str, int]:
    stats = {
        "copied": 0,
        "converted": 0,
        "skipped": 0,
    }

    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        rel = root_path.relative_to(src_root)
        out_dir = dst_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_file = root_path / name
            dst_file = out_dir / name
            action, kind = copy_or_convert_file(src_file, dst_file)
            if action == "copy":
                stats["copied"] += 1
            elif action == "convert":
                stats["converted"] += 1
            else:
                stats["skipped"] += 1
    return stats


#############################
# API DOCS POST-PROCESSING  #
#############################


@dataclass
class ApiOptions:
    members: Optional[List[str]] = None
    heading_level: int = 2
    show_root_heading: bool = True
    show_source: bool = True


@dataclass
class SymbolEntry:
    kind: str  # 'class' or 'function'
    module: str  # dotted module path
    file: Path
    name: str
    node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass
class PythonIndex:
    package_root: Path
    modules: Dict[str, Path]
    symbols_by_fqn: Dict[str, SymbolEntry]
    symbols_by_name: Dict[str, List[SymbolEntry]]
    reexports: Dict[str, str]


def _iter_python_files(root: Path) -> Iterable[Tuple[str, Path]]:
    base = root.resolve()
    for path in base.rglob("*.py"):
        # Skip __pycache__ and hidden
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        rel = path.relative_to(base)
        mod = ".".join([root.name] + list(rel.with_suffix("").parts))
        yield mod, path


def _format_default(expr: Optional[ast.AST]) -> Optional[str]:
    if expr is None:
        return None
    try:
        # Try Python 3.9+ ast.unparse if available
        unparse = getattr(ast, "unparse", None)
        if callable(unparse):
            return unparse(expr)
    except Exception:
        pass
    # Fallback: best-effort for simple constants/names
    if isinstance(expr, ast.Constant):
        return repr(expr.value)
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        parts = []
        cur: Optional[ast.AST] = expr
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return "..."


def _format_parameters(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    a = fn.args
    parts: List[str] = []
    pos = list(a.posonlyargs) + list(a.args)
    defaults = list(a.defaults)
    # Align defaults to the last N positional args
    pos_default_start = len(pos) - len(defaults)
    for i, arg in enumerate(pos):
        name = arg.arg
        default = None
        if i >= pos_default_start:
            default = _format_default(defaults[i - pos_default_start])
        if default is not None:
            parts.append(f"{name}={default}")
        else:
            parts.append(name)
    if a.vararg is not None:
        parts.append(f"*{a.vararg.arg}")
    elif a.kwonlyargs:
        # Add bare * to indicate start of kw-only section
        parts.append("*")
    for kwarg, kwdef in zip(a.kwonlyargs, a.kw_defaults):
        name = kwarg.arg
        default = _format_default(kwdef)
        if default is not None:
            parts.append(f"{name}={default}")
        else:
            parts.append(name)
    if a.kwarg is not None:
        parts.append(f"**{a.kwarg.arg}")
    return ", ".join(parts)


def _format_function_signature(name: str, fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    params = _format_parameters(fn)
    prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {name}({params})"


def _format_class_signature(cls: ast.ClassDef) -> str:
    # Find __init__ if present for signature; else empty parens
    init_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None
    for n in cls.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__":
            init_node = n
            break
    if init_node:
        params = _format_parameters(init_node)
        # drop leading self
        params_list = [p.strip() for p in params.split(", ") if p.strip()]
        if params_list and params_list[0] == "self":
            params_list = params_list[1:]
        return f"class {cls.name}({', '.join(params_list)})"
    return f"class {cls.name}"


def build_python_index(package_root: Path) -> PythonIndex:
    modules: Dict[str, Path] = {}
    symbols_by_fqn: Dict[str, SymbolEntry] = {}
    symbols_by_name: Dict[str, List[SymbolEntry]] = {}
    reexports: Dict[str, str] = {}

    # Map module name to path
    for mod, path in _iter_python_files(package_root):
        modules[mod] = path

    # Parse modules to collect symbols and re-exports
    for mod, path in modules.items():
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue

        # Re-exports from __init__.py files
        if path.name == "__init__.py":
            pkg = mod  # package module name
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    # Resolve absolute module name (handle relative imports)
                    if node.level and pkg:
                        # Determine base package for relative import
                        pkg_parts = pkg.split(".")
                        base_parts = pkg_parts[: -node.level] if node.level <= len(pkg_parts) else []
                        if node.module:
                            base_parts += node.module.split(".")
                        abs_mod = ".".join(base_parts) if base_parts else (node.module or "")
                    else:
                        abs_mod = node.module or ""

                    for alias in node.names:
                        exported = alias.asname or alias.name
                        # e.g., dspy.Adapter -> dspy.adapters.Adapter
                        from_fqn = f"{pkg}.{exported}"
                        to_fqn = f"{abs_mod}.{alias.name}" if abs_mod else alias.name
                        reexports[from_fqn] = to_fqn

        # Top-level class and function defs
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                fqn = f"{mod}.{node.name}"
                entry = SymbolEntry("class", mod, path, node.name, node)
                symbols_by_fqn[fqn] = entry
                symbols_by_name.setdefault(node.name, []).append(entry)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fqn = f"{mod}.{node.name}"
                entry = SymbolEntry("function", mod, path, node.name, node)
                symbols_by_fqn[fqn] = entry
                symbols_by_name.setdefault(node.name, []).append(entry)

    return PythonIndex(package_root, modules, symbols_by_fqn, symbols_by_name, reexports)


def resolve_reexport(fqn: str, index: PythonIndex, max_depth: int = 8) -> str:
    seen = set()
    cur = fqn
    while cur in index.reexports and cur not in seen and max_depth > 0:
        seen.add(cur)
        cur = index.reexports[cur]
        max_depth -= 1
    return cur


def find_symbol(object_path: str, index: PythonIndex) -> Optional[SymbolEntry]:
    # Exact match first
    resolved = resolve_reexport(object_path, index)
    if resolved in index.symbols_by_fqn:
        return index.symbols_by_fqn[resolved]

    # Try as module.symbol if object_path points to module (not reexported)
    parts = object_path.split(".")
    if len(parts) >= 2:
        mod, name = ".".join(parts[:-1]), parts[-1]
        resolved_mod = resolve_reexport(mod, index)
        fqn = f"{resolved_mod}.{name}"
        if fqn in index.symbols_by_fqn:
            return index.symbols_by_fqn[fqn]

    # Fallback: match by trailing name only
    name = parts[-1]
    candidates = index.symbols_by_name.get(name, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Heuristic: prefer candidates whose module path contains any intermediate segments
    segs = set(parts[1:-1])
    for c in candidates:
        if any(seg in c.module.split(".") for seg in segs):
            return c
    # Otherwise return the first one
    return candidates[0]


def _extract_methods(cls: ast.ClassDef) -> Dict[str, Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    methods: Dict[str, Union[ast.FunctionDef, ast.AsyncFunctionDef]] = {}
    for n in cls.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[n.name] = n
    return methods


def generate_api_markdown(object_path: str, opts: ApiOptions, index: PythonIndex) -> str:
    entry = find_symbol(object_path, index)
    if entry is None:
        return f"> API reference unavailable for `{object_path}`."

    lines: List[str] = []
    heading_prefix = "#" * max(1, opts.heading_level)

    # Root heading
    if opts.show_root_heading:
        lines.append(f"{heading_prefix} {object_path}")
        lines.append("")

    # Signature and docstring
    if entry.kind == "class":
        cls = entry.node  # type: ignore[assignment]
        sig = _format_class_signature(cls)
        lines.append("```python")
        lines.append(sig)
        lines.append("```")
        doc = ast.get_docstring(cls) or ""
        if doc:
            lines.append("")
            lines.append(doc)
            lines.append("")

        # Members
        methods = _extract_methods(cls)
        members = opts.members or []
        for name in members:
            m = methods.get(name)
            if not m:
                continue
            lines.append("")
            lines.append(f"{heading_prefix}# {name}")
            lines.append("")
            lines.append("```python")
            lines.append(_format_function_signature(name, m))
            lines.append("```")
            mdoc = ast.get_docstring(m) or ""
            if mdoc:
                lines.append("")
                lines.append(mdoc)
                lines.append("")
    else:
        fn = entry.node  # type: ignore[assignment]
        lines.append("```python")
        lines.append(_format_function_signature(entry.name, fn))
        lines.append("```")
        doc = ast.get_docstring(fn) or ""
        if doc:
            lines.append("")
            lines.append(doc)
            lines.append("")

    # Source info
    if opts.show_source:
        try:
            lineno = entry.node.lineno  # type: ignore[attr-defined]
            end = getattr(entry.node, "end_lineno", None)
            if end:
                lines.append(f"Source: `{entry.file.as_posix()}` (lines {lineno}–{end})")
            else:
                lines.append(f"Source: `{entry.file.as_posix()}` (line {lineno})")
        except Exception:
            lines.append(f"Source: `{entry.file.as_posix()}`")

    return "\n".join(lines).rstrip() + "\n"


def parse_api_options(block_text: str) -> Tuple[str, ApiOptions]:
    """Extract the object path after ':::' and a subset of options from the block."""
    # Extract object path after ':::'
    m = re.search(r"^:::\s*([^\s]+)", block_text, flags=re.MULTILINE)
    obj = m.group(1).strip() if m else ""

    opts = ApiOptions()
    # members
    mem_block = re.search(r"^\s*options:\s*(?:\n|\r\n)([\s\S]*?)^(?=\S|:::)",
                          block_text + "\n:::", flags=re.MULTILINE)
    if mem_block:
        ob = mem_block.group(1)
        # heading_level
        m_h = re.search(r"^\s*heading_level:\s*(\d+)", ob, flags=re.MULTILINE)
        if m_h:
            try:
                opts.heading_level = int(m_h.group(1))
            except Exception:
                pass
        # booleans
        for key in ["show_root_heading", "show_source"]:
            m_b = re.search(rf"^\s*{key}:\s*(true|false)", ob, flags=re.MULTILINE | re.IGNORECASE)
            if m_b:
                val = m_b.group(1).lower() == "true"
                setattr(opts, key, val)
        # members list
        m_m = re.search(r"^\s*members:\s*(?:\n|\r\n)([\s\S]*?)(?=^\s*\w|\Z)", ob, flags=re.MULTILINE)
        if m_m:
            items = re.findall(r"^\s*-\s*([^\s#]+)\s*$", m_m.group(1), flags=re.MULTILINE)
            if items:
                opts.members = items

    return obj, opts


def transform_api_blocks_in_text(text: str, index: PythonIndex) -> Tuple[str, int]:
    """Replace START/END_API_REF blocks with generated Markdown from source code."""
    pattern = re.compile(r"<!--\s*START_API_REF\s*-->[\s\S]*?<!--\s*END_API_REF\s*-->", re.MULTILINE)
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        block = match.group(0)
        obj, opts = parse_api_options(block)
        if not obj:
            return block  # leave unchanged
        rendered = generate_api_markdown(obj, opts, index)
        count += 1
        return rendered

    new_text = pattern.sub(repl, text)
    return new_text, count


def postprocess_api_docs(dst_root: Path, repo_root: Path) -> Dict[str, int]:
    api_dir = dst_root / "api"
    stats = {"processed_files": 0, "blocks_replaced": 0}
    if not api_dir.exists():
        return stats

    pkg_root = repo_root / "dspy"
    if not pkg_root.exists():
        return stats

    index = build_python_index(pkg_root)

    for md in api_dir.rglob("*.md"):
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        new_text, replaced = transform_api_blocks_in_text(text, index)
        if replaced:
            md.write_text(new_text, encoding="utf-8")
            stats["processed_files"] += 1
            stats["blocks_replaced"] += replaced
    return stats


def apply_redirects(src_root: Path, dst_root: Path, redirects: Dict[str, str]) -> Dict[str, int]:
    """
    For each redirect mapping, create the destination `.md` file at the `from` path
    using the content resolved from the `to` path. Supports `.md` sources (copy) and
    `.ipynb` sources (convert to `.md`).
    """
    stats = {"created": 0, "missing": 0}

    for from_rel, to_rel in redirects.items():
        from_rel_path = Path(from_rel)
        to_rel_path = Path(to_rel)

        src_path = (src_root / to_rel_path).resolve()
        if not src_path.exists():
            eprint(f"[redirect] Missing source for {from_rel} -> {to_rel} ({src_path})")
            stats["missing"] += 1
            continue

        # Ensure .md extension for the redirect output
        from_out = (dst_root / from_rel_path).with_suffix(".md")
        from_out.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix.lower() == ".md":
            content = src_path.read_text(encoding="utf-8")
            from_out.write_text(content, encoding="utf-8")
        elif src_path.suffix.lower() == ".ipynb":
            content = convert_ipynb_to_markdown(src_path)
            from_out.write_text(content, encoding="utf-8")
        else:
            # For non-doc sources, write a small pointer page.
            from_out.write_text(
                f"This page redirects to `{to_rel}` which is not a Markdown or Notebook file.",
                encoding="utf-8",
            )

        stats["created"] += 1

    return stats


def main(argv: Optional[List[str]] = None) -> int:
    # Resolve paths based on config
    repo_root = DSPY_REPO_ROOT.resolve()
    src_root = (repo_root / "docs/docs").resolve()
    mkdocs_yml = (repo_root / "docs/mkdocs.yml").resolve()

    if OUTPUT_DIR is None:
        dst_root = (Path.cwd() / "DSPY_DOCS").resolve()
    else:
        dst_root = OUTPUT_DIR.resolve()

    if not repo_root.exists():
        eprint(f"Repo root not found: {repo_root}")
        return 2

    # Update repo if configured and if it looks like a git repo
    created_stash = False
    if GIT_PULL and (repo_root / ".git").exists():
        try:
            # If there are local changes, stash them first (including untracked)
            status = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if status.returncode == 0 and status.stdout.strip():
                eprint("Stashing local changes before pull ...")
                ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                stash = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo_root),
                        "stash",
                        "push",
                        "-u",
                        "-m",
                        f"mirror_docs auto-stash {ts}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if stash.returncode == 0:
                    created_stash = True
                else:
                    eprint("git stash failed (continuing):")
                    eprint(stash.stdout)

            eprint(f"Updating repo via git pull in {repo_root} ...")
            # Use --ff-only to avoid creating merge commits unintentionally
            res = subprocess.run(
                ["git", "-C", str(repo_root), "pull", "--ff-only"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                eprint("git pull failed (continuing):")
                eprint(res.stdout)
            else:
                eprint("git pull succeeded.")
        except Exception as ex:
            eprint(f"git commands raised exception (continuing): {ex}")

    if not src_root.exists():
        eprint(f"Source directory not found: {src_root}")
        return 2

    # Always remove the existing mirror directory before creating a new one
    if dst_root.exists():
        eprint(f"Removing existing destination: {dst_root}")
        ensure_clean_dir(dst_root)
    else:
        dst_root.mkdir(parents=True, exist_ok=True)

    eprint(f"Mirroring docs from {src_root} -> {dst_root}")
    stats = mirror_tree(src_root, dst_root)

    redirects = load_redirect_maps(mkdocs_yml)
    if redirects:
        eprint(f"Applying redirects from {mkdocs_yml} ({len(redirects)} mappings)")
        rstats = apply_redirects(src_root, dst_root, redirects)
    else:
        rstats = {"created": 0, "missing": 0}

    # Post-process API docs: expand mkdocstrings blocks into readable Markdown
    apistats = postprocess_api_docs(dst_root, repo_root)
    if apistats["processed_files"]:
        eprint(
            f"Expanded API refs in {apistats['processed_files']} files, blocks replaced: {apistats['blocks_replaced']}"
        )

    # Write a small README in the mirror root
    readme = dst_root / "README.md"
    readme.write_text(
        (
            f"# DSPY_DOCS (Local Mirror)\n\n"
            f"- Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
            f"- Source: {src_root.as_posix()}\n\n"
            f"This directory mirrors the DSPy documentation as Markdown.\n"
            f"Notebooks were converted to Markdown (code + markdown cells, minimal outputs).\n"
        ),
        encoding="utf-8",
    )

    eprint(
        (
            f"Done. Copied: {stats['copied']}, Converted notebooks: {stats['converted']}, "
            f"Skipped: {stats['skipped']}. Redirect pages created: {rstats['created']}, Missing sources: {rstats['missing']}"
        )
    )

    # Pop the stash after the mirror has been created, if we created one
    if GIT_PULL and created_stash:
        eprint("Restoring stashed changes ...")
        try:
            pop = subprocess.run(
                ["git", "-C", str(repo_root), "stash", "pop"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if pop.returncode != 0:
                eprint("git stash pop reported issues:")
                eprint(pop.stdout)
        except Exception as ex:
            eprint(f"git stash pop raised exception: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
