from __future__ import annotations

import re

from .models import Chunk, PageContent

CLAUSE_RE = re.compile(
    r"(?m)^\s*(第[一二三四五六七八九十百千万\d]+[章节条款项]|"
    r"\d+(?:\.\d+){0,4}|[（(]?[一二三四五六七八九十]+[)）])\s*[、.．:：]?"
)


def build_chunks(pages: list[PageContent], target_chars: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(_text_chunks(page, target_chars, overlap))
        chunks.extend(_table_chunks(page))
    return chunks


def _text_chunks(page: PageContent, target_chars: int, overlap: int) -> list[Chunk]:
    text = normalize_text(page.text)
    if not text:
        return []

    spans = list(CLAUSE_RE.finditer(text))
    if spans:
        pieces: list[tuple[str | None, str]] = []
        for index, match in enumerate(spans):
            start = match.start()
            end = spans[index + 1].start() if index + 1 < len(spans) else len(text)
            piece = text[start:end].strip()
            pieces.append((match.group(1).strip(), piece))
    else:
        pieces = [(None, text)]

    chunks: list[Chunk] = []
    seq = 1
    for clause_id, piece in pieces:
        for part in split_long_text(piece, target_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"p{page.page}-t{seq}",
                    page=page.page,
                    text=part,
                    clause_id=clause_id,
                    kind="text",
                )
            )
            seq += 1
    return chunks


def _table_chunks(page: PageContent) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, table in enumerate(page.tables, start=1):
        rows = [" | ".join(cell for cell in row) for row in table if any(cell.strip() for cell in row)]
        if not rows:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"p{page.page}-table{i}",
                page=page.page,
                text="表格：\n" + "\n".join(rows),
                kind="table",
                metadata={"rows": len(rows)},
            )
        )
    return chunks


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_long_text(text: str, target_chars: int, overlap: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [part for part in parts if part]
