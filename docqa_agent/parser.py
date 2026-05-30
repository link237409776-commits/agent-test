from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from .models import PageContent, PdfProfile

MIN_TEXT_CHARS = 30
OCR_DPI = 200


def parse_pdf(path: str | Path, profile: PdfProfile | None = None) -> list[PageContent]:
    pages = _parse_text_and_tables(path)
    if not _should_run_ocr(pages, profile):
        return pages
    return _fill_sparse_pages_with_ocr(path, pages)


def _parse_text_and_tables(path: str | Path) -> list[PageContent]:
    doc = fitz.open(str(path))
    pages = [PageContent(page=i + 1, text=page.get_text("text").strip()) for i, page in enumerate(doc)]

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables() or []
            if i < len(pages):
                pages[i].tables.extend(_clean_table(table) for table in tables if table)

    return pages


def _should_run_ocr(pages: list[PageContent], profile: PdfProfile | None) -> bool:
    if os.getenv("DOCQA_ENABLE_OCR", "1").lower() in {"0", "false", "no"}:
        return False
    if profile and profile.strategy == "ocr_required":
        return True
    return any(len(page.text.strip()) < MIN_TEXT_CHARS for page in pages)


def _fill_sparse_pages_with_ocr(path: str | Path, pages: list[PageContent]) -> list[PageContent]:
    ocr = _get_ocr_engine()
    doc = fitz.open(str(path))
    cache_path = _ocr_cache_path(path)
    cache = _load_ocr_cache(cache_path)
    patched: list[PageContent] = []

    for index, page_content in enumerate(pages):
        if len(page_content.text.strip()) >= MIN_TEXT_CHARS:
            patched.append(page_content)
            continue

        page_key = str(page_content.page)
        text = str(cache.get(page_key, "")).strip()
        if not text:
            text = _ocr_page(doc[index], ocr).strip()
            cache[page_key] = text
        if not text:
            text = "[本页未提取到可复制文字，OCR 也未识别到可靠文本。]"
        patched.append(PageContent(page=page_content.page, text=text, tables=page_content.tables))

    _save_ocr_cache(cache_path, cache)
    return patched


def _get_ocr_engine() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("OCR 需要安装 PaddleOCR：python -m pip install paddleocr") from exc

    lang = os.getenv("DOCQA_OCR_LANG", "ch")
    ocr_version = os.getenv("DOCQA_OCR_VERSION", "PP-OCRv4")
    return PaddleOCR(
        lang=lang,
        ocr_version=ocr_version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _ocr_page(page: fitz.Page, ocr: Any) -> str:
    scale = OCR_DPI / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    fd, image_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    pixmap.save(image_path)

    try:
        result = _run_ocr(ocr, image_path)
    finally:
        Path(image_path).unlink(missing_ok=True)

    return "\n".join(_extract_text_lines(result))


def _run_ocr(ocr: Any, image_path: str) -> Any:
    if hasattr(ocr, "ocr"):
        return ocr.ocr(image_path)
    return ocr.predict(image_path)


def _extract_text_lines(result: Any) -> list[str]:
    lines: list[str] = []
    if result is None:
        return lines

    if isinstance(result, dict):
        for key in ("rec_texts", "text", "texts"):
            value = result.get(key)
            if isinstance(value, list):
                lines.extend(str(item) for item in value if item)
            elif isinstance(value, str):
                lines.append(value)
        return lines

    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                lines.extend(_extract_text_lines(item))
            elif isinstance(item, (list, tuple)):
                if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
                    lines.append(str(item[1][0]))
                else:
                    lines.extend(_extract_text_lines(item))
    return [line.strip() for line in lines if line and line.strip()]


def _clean_table(table: list[list[str | None]]) -> list[list[str]]:
    return [["" if cell is None else " ".join(str(cell).split()) for cell in row] for row in table]


def _ocr_cache_path(path: str | Path) -> Path:
    pdf_path = Path(path)
    stat = pdf_path.stat()
    digest = hashlib.sha256(f"{pdf_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    cache_dir = Path(os.getenv("DOCQA_OCR_CACHE_DIR", ".docqa_cache/ocr"))
    return cache_dir / f"{digest}.json"


def _load_ocr_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _save_ocr_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
