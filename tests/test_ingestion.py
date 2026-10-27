from __future__ import annotations

from pathlib import Path

import pymupdf

from pdf_localvision_mcp.ingestion import (
    CorruptedPdfError,
    EncryptedPdfError,
    discover_pdf_files,
    rasterize_pdf_file,
)


def _create_minimal_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def test_discover_pdf_files_case_insensitive_sorted_and_skips_hidden_dirs(tmp_path: Path) -> None:
    # Setup files:
    # sub_b/doc2.PDF (uppercase)
    # sub_a/doc1.pdf (lowercase)
    # .hidden/secret.pdf (should be ignored)
    # notes.txt (should be ignored)
    # root.pdf
    _create_minimal_pdf(tmp_path / "sub_b" / "doc2.PDF")
    _create_minimal_pdf(tmp_path / "sub_a" / "doc1.pdf")
    _create_minimal_pdf(tmp_path / ".hidden" / "secret.pdf")
    (tmp_path / "notes.txt").write_text("not a pdf", encoding="utf-8")
    _create_minimal_pdf(tmp_path / "root.pdf")

    results = discover_pdf_files(tmp_path)
    rel_paths = [p.relative_to(tmp_path).as_posix() for p in results]

    assert rel_paths == ["root.pdf", "sub_a/doc1.pdf", "sub_b/doc2.PDF"]


def test_discover_pdf_files_follows_symlinks_deduplicates_and_guards_cycles(tmp_path: Path) -> None:
    # 1. Real PDF file
    doc_real = _create_minimal_pdf(tmp_path / "docs" / "real.pdf")

    # 2. Symlink to real.pdf under another folder
    symlink_folder = tmp_path / "symlinks"
    symlink_folder.mkdir(parents=True, exist_ok=True)
    symlink_file = symlink_folder / "alias.pdf"
    symlink_file.symlink_to(doc_real)

    # 3. Directory symlink creating a recursive cycle
    loop_symlink = tmp_path / "docs" / "cycle_loop"
    loop_symlink.symlink_to(tmp_path / "docs", target_is_directory=True)

    results = discover_pdf_files(tmp_path)
    # real.pdf and alias.pdf resolve to the exact same canonical path, so only one should be returned
    assert len(results) == 1
    # cycle_loop must not cause an infinite recursion or crash
    assert results[0].resolve() == doc_real.resolve()


def test_rasterize_pdf_file_raises_encrypted_pdf_error_for_password_protected_file(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page()
    encrypted_path = tmp_path / "protected.pdf"
    enc_method = getattr(pymupdf, "PDF_ENCRYPT_AES_256", 4)
    doc.save(str(encrypted_path), encryption=enc_method, user_pw="secret")
    doc.close()

    import pytest

    with pytest.raises(EncryptedPdfError, match="encrypted / password-protected"):
        rasterize_pdf_file(encrypted_path)


def test_rasterize_pdf_file_raises_corrupted_pdf_error_for_malformed_file(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupted.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4\ncorrupted garbage content\n%%EOF")

    import pytest

    with pytest.raises(CorruptedPdfError, match="corrupted / unreadable PDF"):
        rasterize_pdf_file(corrupt_path)


