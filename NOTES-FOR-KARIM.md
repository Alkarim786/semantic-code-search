# Notes for Karim: Semantic Code Search

## Resume bullets (real numbers)
- Built a **semantic code search engine** (tree-sitter AST chunking for 6 languages, code embeddings, FAISS HNSW, identifier-aware BM25, FastAPI + web UI) that indexes any GitHub repo.
- On a leak-free evaluation over 4 real repos (4,253 functions, 400 queries), semantic retrieval reached **MRR@10 0.539 / Recall@10 0.805 vs 0.400 / 0.645 for keyword BM25**.
- HNSW matched exact search at **99.7% recall@10 while being 2.9× faster** (0.59 vs 1.70 ms).

## 5 likely interview questions
1. **"Why chunk by AST instead of fixed-size windows?"** Search results should be the unit a developer reads: a function or method. Windows cut functions in half and mix unrelated code into one embedding. Tree-sitter gives exact spans and qualified names (`Class.method`) for 6 languages with one API.
2. **"How did you avoid evaluation leakage?"** The query is the docstring, so the docstring is stripped from both the indexed code and the stored doc field. Otherwise BM25 trivially matches the query to itself. A unit test enforces it.
3. **"Why did hybrid lose to pure semantic?"** RRF weights both lists equally. When one retriever is much weaker, fusion pulls the good ranking down. Fixes: weighted RRF, score-based fusion with normalisation, or a cross-encoder reranker. Hybrid still helps exact-identifier queries, which this eval doesn't measure.
4. **"HNSW vs exact search?"**
   - HNSW is a multi-layer proximity graph searched greedily.
   - Tunables: M (degree), efConstruction (build quality), efSearch (query-time breadth).
   - Measured 99.7% recall at about 3× the speed here; the gap grows with N, since exact is O(N·d) and HNSW is roughly O(log N).
5. **"How would you scale to all of GitHub?"**
   - Shard the index by repo or org.
   - Embed on GPUs in batch, with incremental indexing driven by push events.
   - Quantised vectors (IVF-PQ) for memory.
   - A cache for popular queries; a reranker on the top-k.

## Understand this before interviews
- `chunker.visit`: containers (classes, impls) emit a header chunk, then recurse into their body with the class name pushed onto the scope. That's how `Session.request` gets its qualified name.
- The docstring grammar gotcha: the bare `string` node vs `expression_statement`. It was found by a failing test.
- `split_identifier`: the camelCase regex `[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+` handles `HTTPResponse` → http, response.
- Query latency breakdown: about 70 ms is the CPU query encoding, under 1 ms the HNSW search. Optimise the model (ONNX or a smaller model), not the index.
