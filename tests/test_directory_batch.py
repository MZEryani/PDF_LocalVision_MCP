from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from pdf_localvision_mcp.server import read_pdf_vision


def _create_sample_pdf(path: Path, page_texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


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


async def test_empty_directory_returns_clear_diagnostic_message(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()
    adapter = RecordingVlmAdapter("unused")

    result = await read_pdf_vision(str(empty_dir), vlm_adapter=adapter)

    assert "No .pdf files found in Target Path" in result
    assert "empty_folder" in result
    assert len(adapter.calls) == 0


async def test_directory_batching_recursive_case_insensitive_and_deterministic_order(
    tmp_path: Path,
) -> None:
    # Setup directory structure:
    # batch_root/
    #   sub_b/doc_b.PDF
    #   sub_a/doc_a.pdf
    #   .hidden/ignored.pdf
    #   not_a_pdf.txt
    _create_sample_pdf(tmp_path / "sub_b" / "doc_b.PDF", ["Page 1 of doc B"])
    _create_sample_pdf(tmp_path / "sub_a" / "doc_a.pdf", ["Page 1 of doc A"])
    _create_sample_pdf(tmp_path / ".hidden" / "ignored.pdf", ["Hidden page"])
    (tmp_path / "not_a_pdf.txt").write_text("Hello", encoding="utf-8")

    doc_a_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Content of doc A [Page 1]\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Executive synthesis for doc A [Page 1]."
    )
    doc_b_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Content of doc B [Page 1]\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Executive synthesis for doc B [Page 1]."
    )
    adapter = RecordingVlmAdapter([doc_a_output, doc_b_output])

    result = await read_pdf_vision(str(tmp_path), vlm_adapter=adapter)

    # 1. Batch Summary header check
    assert "## Batch Summary" in result
    assert "- **Total Found**: 2" in result
    assert "- **Succeeded**: 2" in result
    assert "- **Failed**: 0" in result
    assert "AGENT NOTICE" not in result

    # 2. Deterministic relative-path order: sub_a/doc_a.pdf before sub_b/doc_b.PDF
    pos_a = result.index("doc_a.pdf")
    pos_b = result.index("doc_b.PDF")
    assert pos_a < pos_b
    assert "Executive synthesis for doc A" in result
    assert "Executive synthesis for doc B" in result
    assert "ignored.pdf" not in result

    # 3. Two calls to VLM adapter
    assert len(adapter.calls) == 2


async def test_directory_batching_deduplicates_symlinks_in_report(tmp_path: Path) -> None:
    doc_real = _create_sample_pdf(tmp_path / "docs" / "real.pdf", ["Page 1"])
    sym_dir = tmp_path / "symlinks"
    sym_dir.mkdir(parents=True, exist_ok=True)
    sym_file = sym_dir / "alias.pdf"
    sym_file.symlink_to(doc_real)

    doc_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Page 1 [Page 1]\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Synthesis [Page 1]."
    )
    adapter = RecordingVlmAdapter([doc_output])

    result = await read_pdf_vision(str(tmp_path), vlm_adapter=adapter)

    assert "- **Total Found**: 1" in result
    assert "- **Succeeded**: 1" in result
    assert "- **Failed**: 0" in result
    assert len(adapter.calls) == 1



async def test_directory_batch_resilience_isolates_corrupted_and_encrypted_files(
    tmp_path: Path,
) -> None:
    # Setup mixed files:
    # 1. 01_valid.pdf -> valid 1-page PDF
    # 2. 02_encrypted.pdf -> password-protected PDF
    # 3. 03_corrupted.pdf -> malformed garbage bytes
    # 4. 04_valid.pdf -> valid 1-page PDF
    _create_sample_pdf(tmp_path / "01_valid.pdf", ["Page 1 of doc 1"])

    enc_doc = pymupdf.open()
    enc_doc.new_page()
    enc_path = tmp_path / "02_encrypted.pdf"
    enc_method = getattr(pymupdf, "PDF_ENCRYPT_AES_256", 4)
    enc_doc.save(str(enc_path), encryption=enc_method, user_pw="secret")
    enc_doc.close()

    (tmp_path / "03_corrupted.pdf").write_bytes(b"%PDF-1.4\ncorrupted garbage content\n%%EOF")

    _create_sample_pdf(tmp_path / "04_valid.pdf", ["Page 1 of doc 4"])

    doc1_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Valid doc 1 text [Page 1]\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Executive synthesis 1 [Page 1]."
    )
    doc4_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Valid doc 4 text [Page 1]\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "Executive synthesis 4 [Page 1]."
    )
    adapter = RecordingVlmAdapter([doc1_output, doc4_output])

    result = await read_pdf_vision(str(tmp_path), vlm_adapter=adapter)

    # 1. Batch Summary header check
    assert "## Batch Summary" in result
    assert "- **Total Found**: 4" in result
    assert "- **Succeeded**: 2" in result
    assert "- **Failed**: 2" in result

    # 2. Agent Warning Notice Banner is prominent
    assert "> [!WARNING] AGENT NOTICE" in result
    assert "inform the user which files or pages failed and why" in result
    assert "02_encrypted.pdf" in result
    assert "encrypted / password-protected" in result
    assert "03_corrupted.pdf" in result
    assert "corrupted / unreadable PDF" in result

    # 3. Valid files were successfully processed
    assert "Executive synthesis 1" in result
    assert "Executive synthesis 4" in result
    assert len(adapter.calls) == 2


async def test_directory_batching_partial_document_failure_emits_agent_warning_with_pages(
    tmp_path: Path,
) -> None:
    # 2 documents: doc1 has 6 pages and partially fails on pages 4-6; doc2 has 1 page and succeeds
    page_texts_1 = [f"Page {i}" for i in range(1, 7)]
    _create_sample_pdf(tmp_path / "doc1_partial.pdf", page_texts_1)
    _create_sample_pdf(tmp_path / "doc2_valid.pdf", ["Single page"])

    class PartialBatchAdapter:
        async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
            user_content = messages[-1]["content"]
            user_text = user_content if isinstance(user_content, str) else " ".join(c.get("text", "") for c in user_content)
            if "=== PAGE 4 ===" in user_text:
                raise RuntimeError("Page 4 error")
            if "DOCUMENT GROUNDED PAGE BREAKDOWNS" in user_text:
                return "### Executive Summary\nPartial synthesis [Page 1]."
            return "## [Page 1]\nValid content"

    adapter = PartialBatchAdapter()
    result = await read_pdf_vision(str(tmp_path), vlm_adapter=adapter)

    assert "## Batch Summary" in result
    assert "> [!WARNING] AGENT NOTICE" in result
    assert "doc1_partial.pdf" in result
    assert "pages 4-6" in result




