import numpy as np
import pytest
from fastapi.testclient import TestClient

from codesearch.api import create_app
from codesearch.chunker import chunk_source
from codesearch.evaluate import first_sentence, reciprocal_rank
from codesearch.index import CodeIndex, HashingEmbedder
from codesearch.lexical import BM25, code_tokens, split_identifier
from codesearch.repos import parse_repo
from codesearch.search import rrf, search

PY = '''
import os

class Session:
    """A persistent HTTP session."""

    def request(self, method, url, retries=3):
        """Send a request, retrying with exponential backoff."""
        for attempt in range(retries):
            pass

    @property
    def closed(self):
        return False

def parse_url(url):
    """Split a URL into scheme, host and path."""
    return url.split("/")
'''

JS = """
/** Debounce a function call. */
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

class Cache {
  get(key) { return this.m.get(key); }
}
"""

GO = """package store

// Get returns the value stored under key.
func (s *Store) Get(key string) (string, bool) {
	v, ok := s.m[key]
	return v, ok
}

type Store struct { m map[string]string }
"""


def test_python_chunks_have_qualified_names_docs_and_lines():
    chunks = chunk_source(PY, "lib/session.py", "acme/http")
    by_name = {c.name: c for c in chunks}
    assert {"Session", "Session.request", "Session.closed", "parse_url"} <= set(by_name)
    req = by_name["Session.request"]
    assert req.kind == "method" and req.docstring.startswith("Send a request")
    assert req.start_line == 7 and req.code.lstrip().startswith("def request")
    assert by_name["parse_url"].kind == "function"
    assert by_name["Session"].kind == "class"


def test_strip_docstrings_removes_query_text_everywhere():
    chunks = chunk_source(PY, "s.py", strip_docstrings=True)
    req = next(c for c in chunks if c.name == "Session.request")
    assert "exponential" not in req.code and req.docstring == ""
    assert req.extra["held_out_doc"].startswith("Send a request")


def test_other_languages():
    js = {c.name for c in chunk_source(JS, "a.js")}
    assert {"debounce", "Cache", "Cache.get"} <= js
    go = {c.name: c for c in chunk_source(GO, "s.go")}
    assert "Get" in go and go["Get"].docstring.startswith("Get returns")
    assert "Store" in go


def test_long_functions_split_and_unknown_files_window():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    chunks = chunk_source(f"def big():\n{body}\n", "big.py", max_lines=80)
    assert len(chunks) >= 3 and all(c.name == "big" for c in chunks)
    assert all(c.end_line - c.start_line + 1 <= 80 for c in chunks)
    txt = chunk_source("line\n" * 100, "README.txt")
    assert txt and txt[0].kind == "window"


def test_identifier_splitting_and_tokens():
    assert split_identifier("getHTTPResponseCode") == ["get", "http", "response", "code"]
    assert split_identifier("max_retry_count") == ["max", "retry", "count"]
    toks = code_tokens("def parseURL(self): return self.base_url")
    assert {"parseurl", "parse", "url", "base"} <= set(toks)
    assert "self" not in toks and "def" not in toks


def test_bm25_and_rrf():
    bm = BM25()
    bm.add(["def connect_database(host)", "def parse_json(text)", "def close_database_connection()"])
    assert bm.search("database connection", 3)[0][0] == 2
    again = BM25.from_state(bm.state())
    assert again.search("json", 1) == bm.search("json", 1)
    fused = dict(rrf([[1, 2, 3], [3, 1]]))
    assert max(fused, key=fused.get) == 1


@pytest.fixture
def index():
    idx = CodeIndex(HashingEmbedder())
    idx.add(
        chunk_source(PY, "lib/session.py", "acme/http")
        + chunk_source(JS, "web/util.js", "acme/web")
        + chunk_source(GO, "store/store.go", "acme/store")
    )
    return idx


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes_and_filters(index, mode):
    hits, ms = search(index, "split url into host and path", k=3, mode=mode)
    assert hits[0].chunk.name == "parse_url" and ms >= 0
    only_js, _ = search(index, "debounce function", k=5, mode=mode, language="javascript")
    assert only_js and all(h.chunk.language == "javascript" for h in only_js)


def test_hnsw_matches_exact_on_small_index(index):
    q = "retry request backoff"
    exact = [i for i, _ in index.dense(q, 5, exact=True)]
    ann = [i for i, _ in index.dense(q, 5)]
    assert exact[0] == ann[0]


def test_persistence(index, tmp_path):
    index.save(tmp_path / "ix")
    loaded = CodeIndex.load(tmp_path / "ix", HashingEmbedder())
    a, _ = search(index, "cache get key", k=3)
    b, _ = search(loaded, "cache get key", k=3)
    assert [h.chunk.id for h in a] == [h.chunk.id for h in b]
    assert np.allclose(loaded.vectors, index.vectors)


def test_api(index):
    c = TestClient(create_app(index))
    r = c.get("/api/search", params={"q": "store value under key", "language": "go"})
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["name"] == "Get" and top["url"].startswith("https://github.com/acme/store/blob/HEAD/store/store.go#L")
    assert c.get("/api/search", params={"q": "x"}).status_code == 422
    assert c.get("/api/search", params={"q": "abc", "mode": "magic"}).status_code == 422
    assert c.get("/api/stats").json()["chunks"] == len(index.chunks)
    assert c.post("/api/index", json={"repo": "not a repo!"}).status_code == 422
    assert "Semantic Code Search" in c.get("/").text


def test_repo_parsing_and_eval_helpers():
    assert parse_repo("https://github.com/psf/requests") == ("psf", "requests")
    assert parse_repo("pallets/flask.git") == ("pallets", "flask")
    with pytest.raises(ValueError):
        parse_repo("../etc/passwd")
    assert first_sentence("Send a request.  It retries.\n\nMore text") == "Send a request."
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5 and reciprocal_rank(["a"], {"z"}) == 0.0
