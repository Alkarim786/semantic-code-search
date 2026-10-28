"""AST-aware code chunking with tree-sitter.

A "chunk" is a definition (function, method, class) with its qualified name
(e.g. ``Session.request``), signature, docstring and exact line span, which is
what a developer actually searches for. Definitions longer than ``max_lines``
are split into overlapping windows; code outside definitions (module-level
statements) and files in unsupported languages fall back to line windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

# Node types that define something searchable, per language.
DEFINITIONS: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "class_declaration", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "class_declaration", "interface_declaration"},
    "tsx": {"function_declaration", "method_definition", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"method_declaration", "class_declaration", "interface_declaration", "constructor_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item", "enum_item"},
}
CONTAINERS = {"class_definition", "class_declaration", "impl_item", "interface_declaration", "trait_item"}
BODY_TYPES = {"block", "class_body", "declaration_list", "statement_block", "interface_body", "field_declaration_list"}


@dataclass
class Chunk:
    repo: str
    path: str
    language: str
    kind: str  # function | method | class | window
    name: str  # qualified, e.g. "Session.request"
    start_line: int  # 1-based, inclusive
    end_line: int
    code: str
    docstring: str = ""
    signature: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.repo}:{self.path}:{self.start_line}-{self.end_line}"


@cache
def _parser(language: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(language)


def language_for(path: str) -> str | None:
    return LANGUAGES.get(Path(path).suffix.lower())


def _text(src: bytes, node) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name(src: bytes, node) -> str:
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(src, n)
    if node.type == "impl_item":  # Rust: impl Foo / impl Trait for Foo
        t = node.child_by_field_name("type")
        return _text(src, t) if t is not None else "impl"
    if node.type == "type_declaration":  # Go: type Foo struct {...}
        for c in node.named_children:
            n = c.child_by_field_name("name")
            if n is not None:
                return _text(src, n)
    return "<anonymous>"


def _python_docstring(src: bytes, node) -> tuple[str, tuple[int, int] | None]:
    body = node.child_by_field_name("body")
    if body is None or not body.named_children:
        return "", None
    first = body.named_children[0]
    # Grammar versions differ: the docstring is either a bare `string` node or a
    # `string` wrapped in an `expression_statement`.
    node = first if first.type == "string" else None
    if first.type == "expression_statement" and first.named_children and first.named_children[0].type == "string":
        node = first.named_children[0]
    if node is not None:
        raw = _text(src, node)
        doc = re.sub(r'^[rRbBuU]*("""|\'\'\'|"|\')|("""|\'\'\'|"|\')$', "", raw).strip()
        return doc, (first.start_byte, first.end_byte)
    return "", None


def _leading_comment(src: bytes, node) -> str:
    """Comment block immediately above a definition (JSDoc, Go doc, Javadoc, ///)."""
    lines = []
    prev = node.prev_named_sibling
    while (
        prev is not None
        and prev.type in ("comment", "line_comment", "block_comment")
        and node.start_point[0] - prev.end_point[0] <= 1 + len(lines)
    ):
        lines.insert(0, _text(src, prev))
        prev = prev.prev_named_sibling
    text = "\n".join(lines)
    text = re.sub(r"^\s*(/\*\*?|\*/|\*|//+|#)", "", text, flags=re.MULTILINE)
    return text.strip()


def _signature(code: str) -> str:
    first = code.split("\n", 1)[0]
    return first.rstrip(" {:").strip()[:200]


def chunk_source(
    source: str,
    path: str,
    repo: str = "local",
    max_lines: int = 80,
    window: int = 40,
    strip_docstrings: bool = False,
) -> list[Chunk]:
    """Chunk one file. ``strip_docstrings`` removes docstrings/doc comments from
    the indexed code (used by the evaluation so queries can't leak)."""
    lang = language_for(path)
    if lang is None:
        return _windows(source, path, repo, "text", window, 1)
    src = source.encode("utf-8")
    tree = _parser(lang).parse(src)
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []
    defs = DEFINITIONS[lang]

    def visit(node, scope: list[str]) -> None:
        for child in node.named_children:
            if child.type in defs:
                name = _name(src, child)
                qual = ".".join([*scope, name])
                if lang == "python":
                    doc, doc_span = _python_docstring(src, child)
                else:
                    doc, doc_span = _leading_comment(src, child), None
                code = _text(src, child)
                held_out = ""
                if strip_docstrings:
                    # Evaluation mode: the docstring becomes the query, so it must not
                    # appear anywhere in what gets indexed (code text or doc field).
                    held_out, doc = doc, ""
                    if doc_span:
                        a, b = doc_span[0] - child.start_byte, doc_span[1] - child.start_byte
                        code = (code.encode()[:a] + b'""' + code.encode()[b:]).decode("utf-8", errors="replace")
                kind = (
                    "class"
                    if child.type in CONTAINERS or child.type in ("struct_item", "enum_item", "type_declaration")
                    else ("method" if scope else "function")
                )
                start, end = child.start_point[0] + 1, child.end_point[0] + 1
                is_container = child.type in CONTAINERS
                if is_container:
                    # Index the container's header/summary, then its members separately.
                    header = "\n".join(code.split("\n")[: min(15, max_lines)])
                    chunks.append(
                        Chunk(
                            repo,
                            path,
                            lang,
                            kind,
                            qual,
                            start,
                            min(end, start + 14),
                            header,
                            doc,
                            _signature(code),
                            {"held_out_doc": held_out},
                        )
                    )
                    body = child.child_by_field_name("body") or next(
                        (c for c in child.named_children if c.type in BODY_TYPES), None
                    )
                    if body is not None:
                        visit(body, [*scope, name])
                elif end - start + 1 > max_lines:
                    for w in _windows(code, path, repo, lang, max_lines, start, overlap=10):
                        w.kind, w.name, w.docstring, w.signature = kind, qual, doc, _signature(code)
                        w.extra = {"held_out_doc": held_out, "part": True}
                        chunks.append(w)
                else:
                    chunks.append(
                        Chunk(
                            repo,
                            path,
                            lang,
                            kind,
                            qual,
                            start,
                            end,
                            code,
                            doc,
                            _signature(code),
                            {"held_out_doc": held_out},
                        )
                    )
                covered.append((start, end))
            elif child.type in ("decorated_definition", "export_statement", "lexical_declaration", "impl_item"):
                visit(child, scope)

    visit(tree.root_node, [])
    if not chunks:
        return _windows(source, path, repo, lang, window, 1)
    return chunks


def _windows(text: str, path: str, repo: str, lang: str, size: int, first_line: int, overlap: int = 5) -> list[Chunk]:
    lines = text.split("\n")
    out: list[Chunk] = []
    step = max(1, size - overlap)
    for i in range(0, len(lines), step):
        part = lines[i : i + size]
        if not "".join(part).strip():
            continue
        out.append(
            Chunk(
                repo,
                path,
                lang,
                "window",
                Path(path).name,
                first_line + i,
                first_line + i + len(part) - 1,
                "\n".join(part),
            )
        )
        if i + size >= len(lines):
            break
    return out
