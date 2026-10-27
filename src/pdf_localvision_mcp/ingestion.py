from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PATCH_MULTIPLE = 28
DEFAULT_MAX_IMAGE_SIDE = 1792


class PdfIngestionError(Exception):
    """Base exception for PDF ingestion failures."""


class EncryptedPdfError(PdfIngestionError):
    """Raised when a PDF is password-protected or encrypted."""


class CorruptedPdfError(PdfIngestionError):
    """Raised when a PDF is corrupted or unreadable."""


def parse_page_selection(expression: str) -> list[int]:
    """Parse a 1-based page selection expression (e.g. '1-3, 5') into a deduplicated,
    sorted list of 1-based page numbers.

    Raises ValueError on malformed or empty expressions.
    """
    if not expression or not expression.strip():
        raise ValueError("Invalid page selection: expression cannot be empty.")

    pages: set[int] = set()
    parts = expression.split(",")
    for part in parts:
        segment = part.strip()
        if not segment:
            raise ValueError(f"Invalid page selection: empty segment in '{expression}'.")
        if "-" in segment:
            range_parts = segment.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid page selection range: '{segment}'.")
            start_str, end_str = range_parts[0].strip(), range_parts[1].strip()
            if not start_str.isdigit() or not end_str.isdigit():
                raise ValueError(f"Invalid page selection range: '{segment}'.")
            start, end = int(start_str), int(end_str)
            if start <= 0 or end <= 0:
                raise ValueError(f"Invalid page selection: page numbers must be >= 1, got '{segment}'.")
            if start > end:
                raise ValueError(f"Invalid page selection: start page ({start}) cannot exceed end page ({end}).")
            pages.update(range(start, end + 1))
        else:
            if not segment.isdigit():
                raise ValueError(f"Invalid page selection: expected positive integer, got '{segment}'.")
            val = int(segment)
            if val <= 0:
                raise ValueError(f"Invalid page selection: page numbers must be >= 1, got '{val}'.")
            pages.add(val)

    return sorted(pages)


def resolve_pages(pages_expression: str | None, total_pages: int) -> list[int]:
    """Resolve an optional page selection expression against a document's total page count.

    Returns the clamped list of 1-based page numbers. If completely out-of-bounds,
    returns an empty list.
    """
    if pages_expression is None:
        return list(range(1, total_pages + 1))
    requested_pages = parse_page_selection(pages_expression)
    return [p for p in requested_pages if 1 <= p <= total_pages]


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
    pages_processed: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pages_processed and self.pages:
            object.__setattr__(self, "pages_processed", [p.page_number for p in self.pages])


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
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Target Path must be a .pdf file, got '{path}'")
    return path


def discover_pdf_files(target_dir: Path) -> list[Path]:
    resolved_dir = target_dir.expanduser()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Target Path does not exist: '{target_dir}'")
    if not resolved_dir.is_dir():
        raise ValueError(f"Target Path is not a directory: '{target_dir}'")

    visited_dirs: set[Path] = set()
    canonical_files: dict[Path, Path] = {}
    stack: list[Path] = [resolved_dir]

    while stack:
        current_dir = stack.pop()
        try:
            canonical_dir = current_dir.resolve()
        except (OSError, RuntimeError):
            continue

        if canonical_dir in visited_dirs:
            continue
        visited_dirs.add(canonical_dir)

        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            continue

        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file() and entry.suffix.lower() == ".pdf":
                    try:
                        canonical_path = entry.resolve()
                    except (OSError, RuntimeError):
                        canonical_path = entry
                    rel_posix = entry.relative_to(resolved_dir).as_posix()
                    if (
                        canonical_path not in canonical_files
                        or rel_posix < canonical_files[canonical_path].relative_to(resolved_dir).as_posix()
                    ):
                        canonical_files[canonical_path] = entry
            except OSError:
                continue

    discovered = list(canonical_files.values())
    discovered.sort(key=lambda p: p.relative_to(resolved_dir).as_posix())
    return discovered


def rasterize_pdf_file(
    path: Path,
    dpi: int = 200,
    max_image_side: int | None = None,
    pages: str | None = None,
) -> RasterizedDocument:
    if pages is not None:
        parse_page_selection(pages)

    validate_single_pdf_target(path)
    resolved_max_side = (
        max_image_side
        if max_image_side is not None
        else int(os.environ.get("OMLX_MAX_IMAGE_SIDE", str(DEFAULT_MAX_IMAGE_SIDE)))
    )
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise CorruptedPdfError(f"Target file is corrupted / unreadable PDF: '{path.name}'") from exc

    try:
        if getattr(doc, "is_encrypted", False) and getattr(doc, "needs_pass", 0) != 0:
            raise EncryptedPdfError(f"Target file is encrypted / password-protected: '{path.name}'")
        total_pages = len(doc)
        target_pages = resolve_pages(pages, total_pages)

        if not target_pages:
            return RasterizedDocument(
                path=path,
                total_pages=total_pages,
                pages=[],
                pages_processed=[],
            )

        rendered_pages: list[RenderedPage] = []
        for page_num in target_pages:
            idx = page_num - 1
            try:
                page = doc[idx]
            except Exception as exc:
                if getattr(doc, "is_encrypted", False) and getattr(doc, "needs_pass", 0) != 0:
                    raise EncryptedPdfError(f"Target file is encrypted / password-protected: '{path.name}'") from exc
                raise CorruptedPdfError(f"Target file is corrupted / unreadable PDF: '{path.name}'") from exc

            try:
                raw_pix = page.get_pixmap(dpi=dpi, alpha=False)
            except Exception as exc:
                raise CorruptedPdfError(f"Target file is corrupted / unreadable PDF: '{path.name}'") from exc

            target_w, target_h = _snap_dimensions_to_28(raw_pix.width, raw_pix.height, resolved_max_side)
            if (raw_pix.width, raw_pix.height) != (target_w, target_h):
                pix = pymupdf.Pixmap(raw_pix, target_w, target_h, None)
            else:
                pix = raw_pix
            png_bytes = pix.tobytes("png")
            b64_str = base64.b64encode(png_bytes).decode("ascii")
            rendered_pages.append(
                RenderedPage(
                    page_number=page_num,
                    data_uri=f"data:image/png;base64,{b64_str}",
                    width=pix.width,
                    height=pix.height,
                )
            )
        return RasterizedDocument(
            path=path,
            total_pages=total_pages,
            pages=rendered_pages,
            pages_processed=target_pages,
        )
    finally:
        doc.close()
