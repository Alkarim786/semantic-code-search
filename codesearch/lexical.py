"""Code-aware tokenisation and BM25.

Identifiers are split the way developers read them: ``getHTTPResponse`` →
get, http, response; ``max_retry_count`` → max, retry, count. The original
identifier is kept as a token too, so exact-name queries still win.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOP = frozenset(
    "the a an of to in is it and or for on with as by be this that from at are was if else return def self "
    "function func fn let var const new public private static void class import export package true false none "
    "null nil".split()
)


def split_identifier(ident: str) -> list[str]:
    parts: list[str] = []
    for piece in ident.split("_"):
        parts.extend(m.lower() for m in _CAMEL.findall(piece))
    return [p for p in parts if p]


def code_tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text):
        sub = split_identifier(w)
        if len(sub) > 1:
            out.append(w.lower())
        out.extend(sub)
    return [t for t in out if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        self.idf: dict[str, float] = {}

    def add(self, docs: Iterable[str]) -> None:
        for text in docs:
            i = len(self.lengths)
            c = Counter(code_tokens(text))
            self.lengths.append(sum(c.values()))
            for t, tf in c.items():
                self.postings[t].append((i, tf))
        n = len(self.lengths)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        for t in set(code_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for d, tf in self.postings[t]:
                if allowed is not None and d not in allowed:
                    continue
                scores[d] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[d] / avg))
        return heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))

    def state(self) -> dict:
        return {"k1": self.k1, "b": self.b, "postings": self.postings, "lengths": self.lengths}

    @classmethod
    def from_state(cls, s: dict) -> BM25:
        m = cls(s["k1"], s["b"])
        m.postings = defaultdict(list, {t: [tuple(x) for x in v] for t, v in s["postings"].items()})
        m.lengths = s["lengths"]
        n = len(m.lengths)
        m.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in m.postings.items()}
        return m
