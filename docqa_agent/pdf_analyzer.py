from __future__ import annotations

from pathlib import Path

import fitz

from .models import PdfProfile


def analyze_pdf(path: str | Path) -> PdfProfile:
    pdf_path = str(path)
    doc = fitz.open(pdf_path)
    text_chars = 0
    image_blocks = 0
    notes: list[str] = []

    for page in doc:
        text_chars += len(page.get_text("text").strip())
        blocks = page.get_text("dict").get("blocks", [])
        image_blocks += sum(1 for block in blocks if block.get("type") == 1)

    page_count = doc.page_count
    avg_chars = text_chars / max(page_count, 1)
    avg_images = image_blocks / max(page_count, 1)

    if avg_chars >= 250:
        kind = "digital_text_pdf"
        strategy = "pymupdf_text_plus_pdfplumber_tables"
    elif avg_chars >= 30 and avg_images > 0:
        kind = "mixed_pdf"
        strategy = "pymupdf_text_plus_optional_ocr_for_sparse_pages"
        notes.append("部分页面文字较少且含图片，建议在生产环境启用 OCR 补充。")
    else:
        kind = "scanned_or_image_pdf"
        strategy = "ocr_required"
        notes.append("当前原型不会自动安装 OCR；可接入 PaddleOCR、Tesseract 或云 OCR。")

    if page_count == 0:
        notes.append("PDF 没有页面。")

    return PdfProfile(
        path=pdf_path,
        page_count=page_count,
        kind=kind,
        strategy=strategy,
        text_chars=text_chars,
        image_blocks=image_blocks,
        notes=notes,
    )
