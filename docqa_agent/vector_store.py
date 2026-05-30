from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .models import Chunk, Evidence

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")


class TfidfStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_token_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.idf = self._build_idf(self.doc_tokens)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        scored: list[tuple[float, Chunk]] = []
        for chunk, counts, doc_len in zip(self.chunks, self.doc_token_counts, self.doc_lens):
            score = self._bm25_score(query_counter, counts, doc_len)
            score += phrase_boost(query, chunk.text)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        max_score = scored[0][0] if scored else 1.0
        return [
            Evidence(
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                text=chunk.text,
                score=round(score / max_score, 4),
                clause_id=chunk.clause_id,
                kind=chunk.kind,
            )
            for score, chunk in scored[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        payload = {"chunks": [chunk.__dict__ for chunk in self.chunks]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TfidfStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**item) for item in payload["chunks"]])

    def _build_idf(self, docs: list[list[str]]) -> dict[str, float]:
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in docs:
            for token in set(tokens):
                df[token] += 1
        n = max(len(docs), 1)
        return {token: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

    def _bm25_score(self, query: Counter[str], doc: Counter[str], doc_len: int) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token, query_freq in query.items():
            freq = doc.get(token, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
            score += self.idf.get(token, 0.0) * freq * (k1 + 1) / denom * min(query_freq, 3)
        return score


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.extend(chinese_ngrams(value))
        else:
            tokens.append(value)
    return tokens


def chinese_ngrams(text: str) -> list[str]:
    chars = list(text)
    grams = chars[:]
    for size in (2, 3, 4):
        grams.extend("".join(chars[i : i + size]) for i in range(0, max(len(chars) - size + 1, 0)))
    return grams


def phrase_boost(query: str, text: str) -> float:
    compact_query = compact(query)
    compact_text = compact(text)
    if not compact_query or not compact_text:
        return 0.0
    boost = 0.0
    if compact_query in compact_text:
        boost += 3.0
    for term in important_terms(compact_query):
        if term in compact_text:
            boost += 0.4
    return boost


def important_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", text.lower())
    expanded: list[str] = []
    for term in terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            expanded.extend("".join(list(term)[i : i + 2]) for i in range(max(len(term) - 1, 0)))
        expanded.append(term)
    return expanded


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())
