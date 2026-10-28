"""Hybrid search: BM25 over identifiers + dense ANN, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from .chunker import Chunk
from .index import CodeIndex

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Result:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    s: dict[int, float] = defaultdict(float)
    for r in rankings:
        for pos, d in enumerate(r, 1):
            s[d] += 1 / (k + pos)
    return sorted(s.items(), key=lambda x: (-x[1], x[0]))


def search(
    index: CodeIndex,
    query: str,
    k: int = 10,
    mode: str = "hybrid",
    language: str | None = None,
    repo: str | None = None,
    path_contains: str | None = None,
    candidates: int = 100,
) -> tuple[list[Result], float]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()

    def ok(i: int) -> bool:
        c = index.chunks[i]
        return (
            (language is None or c.language == language)
            and (repo is None or c.repo == repo)
            and (path_contains is None or path_contains in c.path)
        )

    filtered = any(v is not None for v in (language, repo, path_contains))
    allowed = {i for i in range(len(index.chunks)) if ok(i)} if filtered else None
    n = candidates * (4 if filtered else 1)
    bm = index.bm25.search(query, n, allowed) if mode != "dense" else []
    de = [(i, s) for i, s in index.dense(query, n) if ok(i)] if mode != "bm25" else []
    bm_rank = {d: r for r, (d, _) in enumerate(bm, 1)}
    de_rank = {d: r for r, (d, _) in enumerate(de, 1)}
    if mode == "bm25":
        ranked = bm
    elif mode == "dense":
        ranked = de
    else:
        ranked = rrf([[d for d, _ in bm], [d for d, _ in de]])
    out = [Result(index.chunks[d], s, bm_rank.get(d), de_rank.get(d)) for d, s in ranked[:k]]
    return out, (time.perf_counter() - t0) * 1000
