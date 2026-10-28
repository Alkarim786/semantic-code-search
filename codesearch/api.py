"""FastAPI service + web UI.

uvicorn codesearch.api:app --port 8000      # then open http://localhost:8000
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .index import CodeIndex, HashingEmbedder, STEmbedder
from .repos import chunk_repo, clone, parse_repo
from .search import MODES, search

DATA = Path(os.environ.get("CODESEARCH_DATA", "data"))
STATIC = Path(__file__).parent / "static"


class IndexRequest(BaseModel):
    repo: str = Field(examples=["psf/requests"], max_length=200)


def make_embedder():
    return (
        HashingEmbedder()
        if os.environ.get("CODESEARCH_EMBEDDER") == "hashing"
        else STEmbedder(os.environ.get("CODESEARCH_MODEL", "flax-sentence-embeddings/st-codesearch-distilroberta-base"))
    )


def create_app(index: CodeIndex | None = None) -> FastAPI:
    lock = threading.Lock()
    jobs: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.index is None:
            emb = make_embedder()
            try:
                app.state.index = CodeIndex.load(DATA / "index", emb)
            except (FileNotFoundError, ValueError):
                app.state.index = CodeIndex(emb)
        yield

    app = FastAPI(title="Semantic Code Search", version="0.1.0", lifespan=lifespan)
    app.state.index = index

    @app.get("/")
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/search")
    def api_search(
        q: str = Query(min_length=2, max_length=300),
        k: int = Query(10, ge=1, le=50),
        mode: str = Query("hybrid"),
        language: str | None = None,
        repo: str | None = None,
        path: str | None = None,
    ):
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        hits, ms = search(app.state.index, q, k, mode, language, repo, path)
        return {
            "query": q,
            "took_ms": round(ms, 2),
            "results": [
                {
                    "repo": h.chunk.repo,
                    "path": h.chunk.path,
                    "language": h.chunk.language,
                    "kind": h.chunk.kind,
                    "name": h.chunk.name,
                    "start_line": h.chunk.start_line,
                    "end_line": h.chunk.end_line,
                    "url": f"https://github.com/{h.chunk.repo}/blob/HEAD/{h.chunk.path}#L{h.chunk.start_line}-L{h.chunk.end_line}"
                    if "/" in h.chunk.repo
                    else None,
                    "score": round(h.score, 5),
                    "bm25_rank": h.bm25_rank,
                    "dense_rank": h.dense_rank,
                    "code": h.chunk.code[:4000],
                }
                for h in hits
            ],
        }

    @app.get("/api/stats")
    def stats():
        idx: CodeIndex = app.state.index
        repos: dict[str, int] = {}
        langs: dict[str, int] = {}
        for c in idx.chunks:
            repos[c.repo] = repos.get(c.repo, 0) + 1
            langs[c.language] = langs.get(c.language, 0) + 1
        return {
            "chunks": len(idx.chunks),
            "repos": repos,
            "languages": langs,
            "embedder": idx.embedder.name,
            "jobs": jobs,
        }

    def _index_job(spec: str) -> None:
        owner, name = parse_repo(spec)
        key = f"{owner}/{name}"
        try:
            jobs[key] = "cloning"
            root = clone(spec, DATA / "repos")
            jobs[key] = "indexing"
            chunks = chunk_repo(root, key)
            with lock:
                idx: CodeIndex = app.state.index
                if any(c.repo == key for c in idx.chunks):
                    jobs[key] = "already indexed"
                    return
                idx.add(chunks)
                idx.save(DATA / "index")
            jobs[key] = f"done: {len(chunks)} chunks"
        except Exception as e:  # report, don't crash the server
            jobs[key] = f"failed: {str(e)[:200]}"

    @app.post("/api/index", status_code=202)
    def index_repo(req: IndexRequest, bg: BackgroundTasks):
        try:
            owner, name = parse_repo(req.repo)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        bg.add_task(_index_job, req.repo)
        return {"repo": f"{owner}/{name}", "status": "queued"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
