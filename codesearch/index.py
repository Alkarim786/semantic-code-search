"""The search index: chunks + BM25 + embeddings in FAISS (HNSW or exact)."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunker import Chunk
from .lexical import BM25


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class STEmbedder:
    """sentence-transformers model; default is trained on CodeSearchNet (NL to code)."""

    def __init__(self, model: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = 256
        getdim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(getdim())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )


class HashingEmbedder:
    """Deterministic, download-free stand-in for tests and CI."""

    def __init__(self, dim: int = 256) -> None:
        from .lexical import code_tokens

        self._tok = code_tokens
        self.name, self.dim = f"hashing-{dim}", dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in self._tok(t):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0 if h >> 63 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


def embed_text(c: Chunk) -> str:
    """What the embedding model sees: name, signature, doc and code, truncated."""
    head = f"{c.language} {c.kind} {c.name}\n{c.signature}\n{c.docstring}\n"
    return head + c.code[:1500]


def lexical_text(c: Chunk) -> str:
    return f"{c.name} {c.name} {c.signature} {c.docstring} {c.path} {c.code}"


class CodeIndex:
    def __init__(self, embedder: Embedder, ann: str = "hnsw", hnsw_m: int = 32, ef_search: int = 64) -> None:
        self.embedder = embedder
        self.ann = ann
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.vectors = np.zeros((0, embedder.dim), np.float32)
        self.faiss = None
        self.build_seconds: dict[str, float] = {}

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        t0 = time.perf_counter()
        vecs = self.embedder.encode([embed_text(c) for c in chunks])
        t1 = time.perf_counter()
        self.chunks.extend(chunks)
        self.bm25 = BM25()
        self.bm25.add(lexical_text(c) for c in self.chunks)
        self.vectors = np.vstack([self.vectors, vecs])
        self._build_ann()
        self.build_seconds = {"embed": round(t1 - t0, 2), "index": round(time.perf_counter() - t1, 2)}

    def _build_ann(self) -> None:
        import faiss

        if self.ann == "hnsw":
            idx = faiss.IndexHNSWFlat(self.embedder.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 80
            idx.hnsw.efSearch = self.ef_search
        else:
            idx = faiss.IndexFlatIP(self.embedder.dim)
        if len(self.vectors):
            idx.add(self.vectors)
        self.faiss = idx

    def dense(self, query: str, k: int, exact: bool = False) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query])
        if exact:
            sims = self.vectors @ q[0]
            top = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in top]
        scores, ids = self.faiss.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    # Persistence ---------------------------------------------------------
    def save(self, d: Path) -> None:
        import faiss

        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in self.chunks), encoding="utf-8")
        np.save(d / "vectors.npy", self.vectors)
        (d / "bm25.json").write_text(json.dumps(self.bm25.state()), encoding="utf-8")
        faiss.write_index(self.faiss, str(d / "ann.faiss"))
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "ann": self.ann}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path, embedder: Embedder) -> CodeIndex:
        import faiss

        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index built with {meta['embedder']}, not {embedder.name}")
        idx = cls(embedder, meta["ann"])
        lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        idx.chunks = [Chunk(**json.loads(x)) for x in lines if x]
        idx.vectors = np.load(d / "vectors.npy")
        idx.bm25 = BM25.from_state(json.loads((d / "bm25.json").read_text(encoding="utf-8")))
        idx.faiss = faiss.read_index(str(d / "ann.faiss"))
        if hasattr(idx.faiss, "hnsw"):
            idx.faiss.hnsw.efSearch = idx.ef_search
        return idx
