from __future__ import annotations

from pathlib import Path

from typing import Any

import pymupdf
import pytest

from pdf_localvision_mcp.ingestion import parse_page_selection, rasterize_pdf_file, resolve_pages
from pdf_localvision_mcp.server import read_pdf_vision


class RecordingVlmAdapter:
    def __init__(self, response_text: str | list[str] | None = None) -> None:
        self.response_text = response_text
        self.calls: list[list[dict[str, Any]]] = []

    async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        if isinstance(self.response_text, list):
            idx = len(self.calls) - 1
            if idx < len(self.response_text):
                return self.response_text[idx]
            return self.response_text[-1]
        return self.response_text or ""


def _create_sample_pdf(path: Path, page_texts: list[str], width: float = 612, height: float = 792) -> Path:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


def test_parse_page_selection_valid_ranges_and_deduplication() -> None:
    # Single pages, ranges, whitespace tolerance
    assert parse_page_selection("1") == [1]
    assert parse_page_selection("1, 3, 5") == [1, 3, 5]
    assert parse_page_selection("1-3, 5") == [1, 2, 3, 5]
    assert parse_page_selection("1-5, 8") == [1, 2, 3, 4, 5, 8]
    assert parse_page_selection(" 1 - 3 ,  5 ") == [1, 2, 3, 5]

    # Deduplication and sorting
    assert parse_page_selection("5, 2, 1") == [1, 2, 5]
    assert parse_page_selection("1-3, 2-4") == [1, 2, 3, 4]
    assert parse_page_selection("4, 4, 4") == [4]


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "abc",
        "0",
        "-3",
        "1--3",
        "5-2",  # descending range
        "1, , 3",
        "1-3-5",
        "1,",
        ",1",
        "1-",
        "-1",
        "one, two",
        "1.5",
    ],
)
def test_parse_page_selection_rejects_malformed_syntax(invalid_input: str) -> None:
    with pytest.raises(ValueError, match="Invalid page selection"):
        parse_page_selection(invalid_input)


def test_resolve_pages_defaults_and_clamping() -> None:
    # None defaults to all pages
    assert resolve_pages(None, total_pages=4) == [1, 2, 3, 4]

    # In-bounds subset
    assert resolve_pages("2-3", total_pages=5) == [2, 3]

    # Partially out-of-bounds clamps to available pages
    assert resolve_pages("1-10", total_pages=4) == [1, 2, 3, 4]
    assert resolve_pages("3-6", total_pages=4) == [3, 4]

    # Completely out-of-bounds returns empty list
    assert resolve_pages("10-12", total_pages=4) == []
    assert resolve_pages("5", total_pages=4) == []


def test_rasterize_pdf_file_with_page_selection(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(
        tmp_path / "multi_page.pdf",
        ["Page 1 text", "Page 2 text", "Page 3 text", "Page 4 text"],
    )

    # Selective rasterization preserves true page numbers
    doc = rasterize_pdf_file(pdf_path, pages="2, 4")
    assert doc.total_pages == 4
    assert doc.pages_processed == [2, 4]
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 2
    assert doc.pages[1].page_number == 4

    # Clamping partially out-of-bounds
    doc_clamped = rasterize_pdf_file(pdf_path, pages="3-10")
    assert doc_clamped.total_pages == 4
    assert doc_clamped.pages_processed == [3, 4]
    assert [p.page_number for p in doc_clamped.pages] == [3, 4]

    # Totally out-of-bounds produces empty pages list
    doc_oob = rasterize_pdf_file(pdf_path, pages="10-12")
    assert doc_oob.total_pages == 4
    assert doc_oob.pages_processed == []
    assert len(doc_oob.pages) == 0

    # Malformed syntax rejected before rasterization
    with pytest.raises(ValueError, match="Invalid page selection"):
        rasterize_pdf_file(pdf_path, pages="abc")


async def test_read_pdf_vision_page_selection_preserves_true_page_numbers(tmp_path: Path) -> None:
    page_texts = [f"Page {i} content" for i in range(1, 7)]
    pdf_path = _create_sample_pdf(tmp_path / "report.pdf", page_texts)

    model_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 4]\n"
        'Strategic metric for page 4 (> "page 4 quote" [Page 4]).\n\n'
        "## [Page 5]\n"
        'Detailed metric for page 5 (> "page 5 quote" [Page 5]).\n'
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Executive Summary: Insights from page 4 [Page 4] and page 5 [Page 5]."
    )
    adapter = RecordingVlmAdapter(model_output)

    result = await read_pdf_vision(str(pdf_path), pages="4-5", vlm_adapter=adapter)

    # Document Metadata assertions
    assert "- **Pages**: 6" in result
    assert "- **Pages Processed**: 4-5" in result
    assert "- **Status**: `completed`" in result

    # Grounded Page Breakdown & citations assertions
    assert "## [Page 4]" in result
    assert "## [Page 5]" in result
    assert "## [Page 1]" not in result
    assert "Insights from page 4 [Page 4] and page 5 [Page 5]" in result

    # Modal prompt payload inspection across seam
    assert len(adapter.calls) == 1
    user_items = adapter.calls[0][-1]["content"]
    text_blocks = [it["text"] for it in user_items if it["type"] == "text"]
    assert "=== PAGE 4 ===" in text_blocks
    assert "=== PAGE 5 ===" in text_blocks
    assert "=== PAGE 1 ===" not in text_blocks
    image_blocks = [it for it in user_items if it["type"] == "image_url"]
    assert len(image_blocks) == 2


async def test_read_pdf_vision_malformed_pages_rejected_immediately_before_io(tmp_path: Path) -> None:
    adapter = RecordingVlmAdapter("unused")
    nonexistent_path = tmp_path / "does_not_exist.pdf"

    # Malformed pages syntax is rejected before opening or validating path
    with pytest.raises(ValueError, match="Invalid page selection"):
        await read_pdf_vision(str(nonexistent_path), pages="abc", vlm_adapter=adapter)

    assert len(adapter.calls) == 0


async def test_read_pdf_vision_partially_out_of_bounds_clamps(tmp_path: Path) -> None:
    page_texts = [f"Page {i}" for i in range(1, 4)]
    pdf_path = _create_sample_pdf(tmp_path / "three_pages.pdf", page_texts)

    model_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\nPage 1 breakdown\n\n"
        "## [Page 2]\nPage 2 breakdown\n\n"
        "## [Page 3]\nPage 3 breakdown\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Three page executive synthesis [Page 1][Page 3]."
    )
    adapter = RecordingVlmAdapter(model_output)

    result = await read_pdf_vision(str(pdf_path), pages="1-10", vlm_adapter=adapter)

    assert "- **Pages**: 3" in result
    assert "- **Pages Processed**: 1-3" in result
    assert "- **Status**: `completed`" in result
    assert len(adapter.calls) == 1
    user_items = adapter.calls[0][-1]["content"]
    image_blocks = [it for it in user_items if it["type"] == "image_url"]
    assert len(image_blocks) == 3


async def test_read_pdf_vision_totally_out_of_bounds_skips_inference(tmp_path: Path) -> None:
    page_texts = [f"Page {i}" for i in range(1, 5)]
    pdf_path = _create_sample_pdf(tmp_path / "four_pages.pdf", page_texts)

    adapter = RecordingVlmAdapter("unused")

    result = await read_pdf_vision(str(pdf_path), pages="10-12", vlm_adapter=adapter)

    # Zero inference calls made
    assert len(adapter.calls) == 0

    # Clean skipped status message
    assert "- **Pages**: 4" in result
    assert "- **Status**: `skipped`" in result
    assert "exceed" in result.lower()
    assert "10-12" in result



