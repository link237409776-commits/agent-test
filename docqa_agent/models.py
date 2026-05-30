from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageContent:
    page: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass(frozen=True)
class PdfProfile:
    path: str
    page_count: int
    kind: str
    strategy: str
    text_chars: int
    image_blocks: int
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str
    clause_id: str | None = None
    kind: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    page: int
    text: str
    score: float
    clause_id: str | None = None
    kind: str = "text"


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    evidences: list[Evidence]
    self_check: dict[str, Any]
