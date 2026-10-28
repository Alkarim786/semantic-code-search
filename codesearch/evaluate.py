"""CodeSearchNet-style evaluation on real repositories.

For every function/method with a docstring we use the docstring's first
sentence as a natural-language query. The index is built with docstrings
*removed* from the code (``strip_docstrings=True``), so a hit can't come from
matching the query text against the docstring itself. The gold answer is the
chunk(s) of that same definition.

    python -m codesearch.evaluate --repos psf/requests pallets/flask --max-queries 400
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np

from .index import CodeIndex, HashingEmbedder, STEmbedder
from .repos import chunk_repo, clone, parse_repo
from .search import MODES, search


def first_sentence(doc: str) -> str:
    doc = doc.strip().split("\n\n")[0]
    s = re.split(r"(?<=[.!?])\s", " ".join(doc.split()))[0]
    return s.strip()


def reciprocal_rank(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], 1):
        if r in gold:
            return 1 / i
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=["psf/requests", "pallets/flask", "pallets/click", "encode/httpx"])
    ap.add_argument("--work", type=Path, default=Path("data/repos"))
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument("--embedder", default="st", choices=["st", "hashing"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = []
    for spec in args.repos:
        owner, name = parse_repo(spec)
        root = clone(spec, args.work)
        chunks.extend(chunk_repo(root, f"{owner}/{name}", strip_docstrings=True))
    # Queries: definitions whose held-out docstring gives a usable summary (>= 4 words).
    by_def: dict[tuple[str, str, str], list[str]] = {}
    queries: dict[tuple[str, str, str], str] = {}
    for c in chunks:
        key = (c.repo, c.path, c.name)
        by_def.setdefault(key, []).append(c.id)
        q = first_sentence(c.extra.get("held_out_doc", ""))
        if c.kind in ("function", "method") and len(q.split()) >= 4 and key not in queries:
            queries[key] = q
    keys = sorted(queries)
    random.Random(args.seed).shuffle(keys)
    keys = keys[: args.max_queries]

    embedder = STEmbedder() if args.embedder == "st" else HashingEmbedder()
    t0 = time.perf_counter()
    exact = CodeIndex(embedder, ann="flat")
    exact.add(chunks)
    build_s = time.perf_counter() - t0
    hnsw = CodeIndex(embedder, ann="hnsw")
    hnsw.embedder, hnsw.chunks, hnsw.bm25, hnsw.vectors = exact.embedder, exact.chunks, exact.bm25, exact.vectors
    t1 = time.perf_counter()
    hnsw._build_ann()
    hnsw_build_s = time.perf_counter() - t1

    results: dict[str, dict] = {}
    for mode in MODES:
        rr, r1, r10, lat = [], [], [], []
        for key in keys:
            gold = set(by_def[key])
            hits, ms = search(hnsw, queries[key], k=10, mode=mode)
            ids = [h.chunk.id for h in hits]
            rr.append(reciprocal_rank(ids, gold))
            r1.append(float(bool(set(ids[:1]) & gold)))
            r10.append(float(bool(set(ids[:10]) & gold)))
            lat.append(ms)
        results[mode] = {
            "MRR@10": round(float(np.mean(rr)), 4),
            "Recall@1": round(float(np.mean(r1)), 4),
            "Recall@10": round(float(np.mean(r10)), 4),
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
        }

    # ANN quality: overlap of HNSW top-10 with exact top-10, and pure vector-search latency.
    overlap, t_exact, t_ann = [], [], []
    qv = embedder.encode([queries[k] for k in keys])
    for v in qv:
        a = time.perf_counter()
        ex = set(np.argsort(-(exact.vectors @ v))[:10].tolist())
        b = time.perf_counter()
        _, ids = hnsw.faiss.search(v[None, :], 10)
        c = time.perf_counter()
        overlap.append(len(ex & set(ids[0].tolist())) / 10)
        t_exact.append((b - a) * 1000)
        t_ann.append((c - b) * 1000)

    summary = {
        "repos": args.repos,
        "chunks": len(chunks),
        "definitions": len(by_def),
        "queries": len(keys),
        "embedder": embedder.name,
        "index_build_s": round(build_s, 1),
        "hnsw_build_s": round(hnsw_build_s, 2),
        "retrieval": results,
        "ann": {
            "hnsw_recall@10_vs_exact": round(float(np.mean(overlap)), 4),
            "exact_p50_ms": round(float(np.percentile(t_exact, 50)), 3),
            "hnsw_p50_ms": round(float(np.percentile(t_ann, 50)), 3),
        },
        "cpu": platform.processor(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
