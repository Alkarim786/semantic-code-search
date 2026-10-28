"""Fetch GitHub repositories and walk their source files."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .chunker import LANGUAGES, Chunk, chunk_source

SKIP_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "third_party",
    "site-packages",
}
MAX_FILE_BYTES = 300_000
_GH = re.compile(r"^(?:https://github\.com/)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def parse_repo(spec: str) -> tuple[str, str]:
    m = _GH.match(spec.strip())
    if not m:
        raise ValueError(f"not a GitHub repo: {spec!r} (use owner/name or https://github.com/owner/name)")
    return m.group(1), m.group(2)


def clone(spec: str, dest_root: Path) -> Path:
    """Shallow clone (depth 1); reuse an existing checkout."""
    owner, name = parse_repo(spec)
    dest = dest_root / f"{owner}__{name}"
    if not (dest / ".git").exists():
        dest_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{owner}/{name}.git", str(dest)],
            check=True,
            timeout=600,
        )
    return dest


def source_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_file() and p.suffix.lower() in LANGUAGES and p.stat().st_size <= MAX_FILE_BYTES:
            yield p


def chunk_repo(root: Path, repo_name: str, strip_docstrings: bool = False) -> list[Chunk]:
    out: list[Chunk] = []
    for f in source_files(root):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        out.extend(chunk_source(text, f.relative_to(root).as_posix(), repo_name, strip_docstrings=strip_docstrings))
    return out
