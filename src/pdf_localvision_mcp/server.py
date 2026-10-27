from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from pdf_localvision_mcp.ingestion import (
    CorruptedPdfError,
    EncryptedPdfError,
    RasterizedDocument,
    RenderedPage,
    discover_pdf_files,
    parse_page_selection,
    rasterize_pdf_file,
)
from pdf_localvision_mcp.omlx_client import OmlxClient, OmlxConnectionError
from pdf_localvision_mcp.prompts import (
    HIERARCHICAL_SYNTHESIS_THRESHOLD,
    build_hierarchical_intermediate_messages,
    build_single_window_messages,
    build_stage2_synthesis_messages,
    build_window_stage1_messages,
    format_page_ranges,
    format_visual_reading_markdown,
    parse_single_window_response,
    strip_context_page_section,
)

mcp = MCPServer("pdf-localvision-mcp")

PAGE_WINDOW_SIZE = 5
DEFAULT_MAX_CONCURRENCY = 1


@dataclass(frozen=True)
class WindowReading:
    page_numbers: list[int]
    content: str | None
    error: Exception | None = None


class VlmAdapter(Protocol):
    async def complete_messages(self, messages: list[dict[str, Any]]) -> str: ...


async def _process_window_stage1(
    adapter: VlmAdapter,
    target_pages: list[RenderedPage],
    context_page: RenderedPage | None,
    semaphore: asyncio.Semaphore,
) -> WindowReading:
    async with semaphore:
        messages = build_window_stage1_messages(target_pages, context_page=context_page)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                res = await adapter.complete_messages(messages)
                cleaned_res = strip_context_page_section(
                    res,
                    context_page.page_number if context_page is not None else None,
                    target_pages[0].page_number,
                )
                return WindowReading(
                    page_numbers=[p.page_number for p in target_pages],
                    content=cleaned_res,
                )
            except OmlxConnectionError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
        return WindowReading(
            page_numbers=[p.page_number for p in target_pages],
            content=None,
            error=last_error,
        )


async def _run_stage2_synthesis(
    adapter: VlmAdapter,
    window_readings: list[str],
    prompt: str | None = None,
    hierarchical_threshold: int = HIERARCHICAL_SYNTHESIS_THRESHOLD,
) -> str:
    combined = "\n\n".join(window_readings).strip()
    if len(combined) <= hierarchical_threshold:
        messages = build_stage2_synthesis_messages(combined, prompt=prompt)
        return await adapter.complete_messages(messages)

    # Hierarchical folding: reduce intermediate clusters to stay safely within context limit
    cluster_size = max(1, len(window_readings) // 2)
    intermediate_summaries: list[str] = []
    for i in range(0, len(window_readings), cluster_size):
        cluster_text = "\n\n".join(window_readings[i : i + cluster_size])
        inter_messages = build_hierarchical_intermediate_messages(cluster_text)
        summary = await adapter.complete_messages(inter_messages)
        intermediate_summaries.append(summary.strip())

    folded_combined = "\n\n".join(intermediate_summaries)
    messages = build_stage2_synthesis_messages(folded_combined, prompt=prompt)
    return await adapter.complete_messages(messages)


async def _read_single_pdf_document(
    target_path: Path,
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
    *,
    adapter: VlmAdapter,
) -> str:
    doc = rasterize_pdf_file(target_path, dpi=dpi, pages=pages)

    if not doc.pages:
        return format_visual_reading_markdown(
            doc=doc,
            executive_synthesis=f"Inference skipped: requested pages ({pages}) exceed document page count ({doc.total_pages} pages).",
            grounded_pages="No pages were processed.",
            status="skipped",
        )

    if len(doc.pages) <= PAGE_WINDOW_SIZE:
        messages = build_single_window_messages(doc.pages, prompt=prompt)
        raw_response = await adapter.complete_messages(messages)
        parsed = parse_single_window_response(raw_response)
        return format_visual_reading_markdown(
            doc=doc,
            executive_synthesis=parsed.executive_synthesis,
            grounded_pages=parsed.grounded_pages,
        )

    # Multi-window Map-Reduce pipeline
    concurrency = int(os.environ.get("OMLX_MAX_CONCURRENCY", str(DEFAULT_MAX_CONCURRENCY)))
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for idx in range(0, len(doc.pages), PAGE_WINDOW_SIZE):
        window_pages = doc.pages[idx : idx + PAGE_WINDOW_SIZE]
        context_page = (
            doc.pages[idx - 1]
            if idx > 0 and doc.pages[idx - 1].page_number == window_pages[0].page_number - 1
            else None
        )
        tasks.append(_process_window_stage1(adapter, window_pages, context_page, semaphore))

    window_readings = await asyncio.gather(*tasks)

    # Order results by initial window sequence
    succeeded_readings: list[tuple[int, str]] = []
    failed_page_numbers: list[int] = []
    for idx, reading in enumerate(window_readings):
        if reading.content is not None:
            succeeded_readings.append((idx, reading.content.strip()))
        else:
            failed_page_numbers.extend(reading.page_numbers)

    if not succeeded_readings:
        failed_ranges = format_page_ranges(failed_page_numbers)
        raise RuntimeError(f"All Page Windows failed during visual reading (pages {failed_ranges})")

    ordered_breakdowns = [text for _, text in sorted(succeeded_readings, key=lambda x: x[0])]
    combined_grounded_breakdown = "\n\n".join(ordered_breakdowns)

    executive_synthesis = await _run_stage2_synthesis(
        adapter=adapter,
        window_readings=ordered_breakdowns,
        prompt=prompt,
    )

    if failed_page_numbers:
        failed_ranges = format_page_ranges(failed_page_numbers)
        warning_notice = (
            f"> [!WARNING] AGENT NOTICE\n"
            f"> Visual reading partially failed for pages {failed_ranges} after retry. "
            f"You MUST explicitly inform the user that pages {failed_ranges} could not be processed and are excluded from this analysis."
        )
        return format_visual_reading_markdown(
            doc=doc,
            executive_synthesis=executive_synthesis,
            grounded_pages=combined_grounded_breakdown,
            status="partial",
            failed_pages=failed_page_numbers,
            warning_notice=warning_notice,
        )

    return format_visual_reading_markdown(
        doc=doc,
        executive_synthesis=executive_synthesis,
        grounded_pages=combined_grounded_breakdown,
        status="completed",
    )


async def _process_directory_batch(
    target_dir: Path,
    pdf_files: list[Path],
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
    *,
    adapter: VlmAdapter,
) -> str:
    total_found = len(pdf_files)
    succeeded = 0
    failed = 0
    failure_details: list[tuple[str, str]] = []
    doc_sections: list[str] = []

    for file_path in pdf_files:
        rel_path = file_path.relative_to(target_dir).as_posix()
        try:
            doc_output = await _read_single_pdf_document(
                target_path=file_path,
                prompt=prompt,
                dpi=dpi,
                pages=pages,
                adapter=adapter,
            )
            succeeded += 1
            doc_sections.append(doc_output)
            if "- **Status**: `partial`" in doc_output:
                failed_match = re.search(r"- \*\*Failed Pages\*\*: ([^\n]+)", doc_output)
                failed_page_range = failed_match.group(1) if failed_match else "unknown"
                failure_details.append((rel_path, f"partial page reading failure (pages {failed_page_range})"))
        except (EncryptedPdfError, CorruptedPdfError, Exception) as exc:
            failed += 1
            if isinstance(exc, EncryptedPdfError):
                reason = "encrypted / password-protected"
            elif isinstance(exc, CorruptedPdfError):
                reason = "corrupted / unreadable PDF"
            else:
                reason = str(exc)
            failure_details.append((rel_path, reason))
            failed_section = (
                f"# Visual Reading: {file_path.name}\n\n"
                f"## Document Metadata\n"
                f"- **Filename**: `{file_path.name}`\n"
                f"- **Relative Path**: `{rel_path}`\n"
                f"- **Status**: `failed`\n"
                f"- **Error**: {reason}\n"
            )
            doc_sections.append(failed_section)

    summary_lines = [
        f"# Batch Visual Reading: {target_dir.name}",
        "",
        "## Batch Summary",
        f"- **Total Found**: {total_found}",
        f"- **Succeeded**: {succeeded}",
        f"- **Failed**: {failed}",
    ]

    if failed > 0 or any("partial page reading failure" in d[1] for d in failure_details):
        warning_lines = [
            "",
            "> [!WARNING] AGENT NOTICE",
            "> One or more documents or pages failed during directory batch processing. "
            "You MUST explicitly inform the user which files or pages failed and why:",
        ]
        for rel_path, reason in failure_details:
            warning_lines.append(f"> - `{rel_path}`: {reason}")
        summary_lines.extend(warning_lines)

    all_blocks = ["\n".join(summary_lines)]
    all_blocks.extend(doc_sections)
    return "\n\n---\n\n".join(all_blocks).strip() + "\n"


async def read_pdf_vision(
    path: str,
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
    *,
    vlm_adapter: VlmAdapter | None = None,
) -> str:
    if pages is not None:
        parse_page_selection(pages)

    target_path = Path(path).expanduser()
    adapter: VlmAdapter = vlm_adapter if vlm_adapter is not None else OmlxClient()

    if target_path.exists() and target_path.is_dir():
        pdf_files = discover_pdf_files(target_path)
        if not pdf_files:
            return f"No .pdf files found in Target Path: '{target_path}'"
        return await _process_directory_batch(
            target_dir=target_path,
            pdf_files=pdf_files,
            prompt=prompt,
            dpi=dpi,
            pages=pages,
            adapter=adapter,
        )

    return await _read_single_pdf_document(
        target_path=target_path,
        prompt=prompt,
        dpi=dpi,
        pages=pages,
        adapter=adapter,
    )


@mcp.tool(name="read_pdf_vision")
async def _read_pdf_vision_tool(
    path: str,
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
) -> str:
    """Perform a native multimodal Visual Reading of a PDF file or directory of PDFs using local oMLX Q-Vision."""
    return await read_pdf_vision(path=path, prompt=prompt, dpi=dpi, pages=pages)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
