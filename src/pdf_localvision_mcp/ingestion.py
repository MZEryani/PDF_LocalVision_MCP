from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PATCH_MULTIPLE = 28
DEFAULT_MAX_IMAGE_SIDE = 2044


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    data_uri: str
    width: int
    height: int


@dataclass(frozen=True)
class RasterizedDocument:
    path: Path
    total_pages: int
    pages: list[RenderedPage]


def _snap_dimensions_to_28(width: int, height: int, max_side: int) -> tuple[int, int]:
    capped_max = max(PATCH_MULTIPLE, (max_side // PATCH_MULTIPLE) * PATCH_MULTIPLE)
    longest = max(width, height)
    scale = min(1.0, capped_max / longest) if longest > 0 else 1.0
    scaled_w = width * scale
    scaled_h = height * scale
    snapped_w = min(capped_max, max(PATCH_MULTIPLE, int(round(scaled_w / PATCH_MULTIPLE)) * PATCH_MULTIPLE))
    snapped_h = min(capped_max, max(PATCH_MULTIPLE, int(round(scaled_h / PATCH_MULTIPLE)) * PATCH_MULTIPLE))
    return snapped_w, snapped_h


def validate_single_pdf_target(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Target Path does not exist: '{path}'")
    if path.is_file() and path.suffix.lower() != ".pdf":
        raise ValueError(f"Target Path must have a .pdf extension, got '{path.name}'")
    return path


def rasterize_pdf_file(path: Path, dpi: int = 150) -> RasterizedDocument:
    validate_single_pdf_target(path)
    max_side = int(os.environ.get("OMLX_MAX_IMAGE_SIDE", str(DEFAULT_MAX_IMAGE_SIDE)))
    doc = pymupdf.open(str(path))
    try:
        total_pages = len(doc)
        rendered_pages: list[RenderedPage] = []
        for idx in range(total_pages):
            page = doc[idx]
            raw_pix = page.get_pixmap(dpi=dpi, alpha=False)
            target_w, target_h = _snap_dimensions_to_28(raw_pix.width, raw_pix.height, max_side)
            if (raw_pix.width, raw_pix.height) != (target_w, target_h):
                pix = pymupdf.Pixmap(raw_pix, target_w, target_h, None)
            else:
                pix = raw_pix
            png_bytes = pix.tobytes("png")
            b64_str = base64.b64encode(png_bytes).decode("ascii")
            rendered_pages.append(
                RenderedPage(
                    page_number=idx + 1,
                    data_uri=f"data:image/png;base64,{b64_str}",
                    width=pix.width,
                    height=pix.height,
                )
            )
        return RasterizedDocument(path=path, total_pages=total_pages, pages=rendered_pages)
    finally:
        doc.close()
