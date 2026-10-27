from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pdf_localvision_mcp.ingestion import RasterizedDocument, RenderedPage

GROUNDED_PAGES_MARKER = "<!-- GROUNDED_PAGES -->"
EXECUTIVE_SYNTHESIS_MARKER = "<!-- EXECUTIVE_SYNTHESIS -->"
HIERARCHICAL_SYNTHESIS_THRESHOLD = 140_000


@dataclass(frozen=True)
class ParsedSingleWindowReading:
    executive_synthesis: str
    grounded_pages: str


SINGLE_WINDOW_SYSTEM_PROMPT = """You are a high-precision multimodal document reading system modeled after Google Gemini and NotebookLM.
You analyze rendered PDF page images directly without OCR pre-parsing.

CRITICAL RULES:
1. Every page image is preceded by an explicit `=== PAGE N ===` marker. Attribute all content strictly to its true page number `N`.
2. Generate output in TWO explicitly delimited sections IN THIS EXACT ORDER:
   - First: `<!-- GROUNDED_PAGES -->` containing a `## [Page N]` section for every page in ascending order:
     * Natural reading-order prose in Markdown (headers, lists, footnotes).
     * Tables as complete HTML `<table>` blocks preserving `<thead>`, `<tbody>`, `rowspan`, `colspan`, and exact cell values.
     * Mathematical equations in LaTeX (`\\(...\\)` inline or `\\[...\\]` display).
     * Figures, charts, and diagrams as structured Visual Data Blocks (chart type, axes/labels, linearized HTML data table of plotted values, or `mermaid` flowcharts for process/architecture diagrams).
     * Key verbatim anchor quotes formatted as `> "exact quote" [Page N]`.
   - Second: `<!-- EXECUTIVE_SYNTHESIS -->` grounded strictly in the page breakdowns you just produced, citing every claim with `[Page N]`.
"""

STAGE1_WINDOW_SYSTEM_PROMPT = """You are a high-precision multimodal document reading system modeled after Google Gemini and NotebookLM.
You analyze rendered PDF page images directly without OCR pre-parsing.

CRITICAL RULES:
1. Target page images are preceded by explicit `=== PAGE N ===` markers. Attribute all content strictly to its true page number `N`.
2. If a preceding page image is provided marked with `=== READ-ONLY VISUAL CONTEXT: PAGE N ===`, treat it strictly as context to resolve continuing sentences, table headers, and diagrams across page boundaries. Do NOT transcribe the context page; transcribe only the target pages.
3. For each target page, generate a `## [Page N]` section containing:
   * Natural reading-order prose in Markdown (headers, lists, footnotes).
   * Tables as complete HTML `<table>` blocks preserving `<thead>`, `<tbody>`, `rowspan`, `colspan`, and exact cell values.
   * Mathematical equations in LaTeX (`\\(...\\)` inline or `\\[...\\]` display).
   * Figures, charts, and diagrams as structured Visual Data Blocks (chart type, axes/labels, linearized HTML data table of plotted values, or `mermaid` flowcharts for process/architecture diagrams).
   * Key verbatim anchor quotes formatted as `> "exact quote" [Page N]`.
4. Output ONLY the `## [Page N]` sections for the target pages. Do not produce an executive synthesis.
"""

STAGE2_SYNTHESIS_SYSTEM_PROMPT = """You are a high-precision document executive synthesis system modeled after Google NotebookLM.
You receive structured Grounded Page Breakdowns of a document and synthesize them into an authoritative, citation-anchored Executive Synthesis.

CRITICAL RULES:
1. Every claim, finding, table figure, and quote MUST carry an inline `[Page N]` citation grounded strictly in the provided page breakdowns.
2. Ground all numbers and tables verbatim.
"""

STAGE2_HIERARCHICAL_INTERMEDIATE_PROMPT = """You are a document synthesis cluster reducer.
Condense the following page breakdowns into an intermediate grounded synthesis preserving all key metrics, findings, table figures, and inline `[Page N]` citations.
"""


DEFAULT_EXECUTIVE_SYNTHESIS_STRUCTURE = (
    "Provide an exhaustive Executive Synthesis structured into:\n"
    "### Executive Summary\n"
    "### Key Narratives & Findings\n"
    "### Extracted Tabular Data & Metrics\n"
    "### Diagrams & Visual Analyses\n"
    "### Critical Citations Index\n"
    "Every claim, figure, and quote MUST carry an inline `[Page N]` citation."
)


def format_page_ranges(pages: list[int]) -> str:
    if not pages:
        return ""
    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = sorted_pages[0]
    prev = start
    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


def build_window_stage1_messages(
    pages: list[RenderedPage],
    context_page: RenderedPage | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if context_page is not None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"=== READ-ONLY VISUAL CONTEXT: PAGE {context_page.page_number} ===\n"
                    f"Use this page only as read-only visual context to resolve split sentences and continuing table headers. "
                    f"Do NOT transcribe Page {context_page.page_number}."
                ),
            }
        )
        content.append({"type": "image_url", "image_url": {"url": context_page.data_uri}})

    for page in pages:
        content.append({"type": "text", "text": f"=== PAGE {page.page_number} ==="})
        content.append({"type": "image_url", "image_url": {"url": page.data_uri}})

    first_page = pages[0].page_number
    last_page = pages[-1].page_number
    content.append(
        {
            "type": "text",
            "text": (
                f"Transcribe ONLY target pages {first_page} through {last_page} into explicit `## [Page N]` sections.\n"
                f"Do not produce any executive synthesis."
            ),
        }
    )
    return [
        {"role": "system", "content": STAGE1_WINDOW_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_stage2_synthesis_messages(
    combined_breakdown: str,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    if prompt and prompt.strip():
        instruction = (
            f"Directly and thoroughly answer the following user query using mandatory `[Page N]` citations "
            f"and verbatim quotes (`> \"...\" [Page N]`) grounded strictly in the page breakdowns below:\n"
            f"USER QUERY: {prompt.strip()}"
        )
    else:
        instruction = DEFAULT_EXECUTIVE_SYNTHESIS_STRUCTURE

    content = f"{instruction}\n\nDOCUMENT GROUNDED PAGE BREAKDOWNS:\n\n{combined_breakdown}"
    return [
        {"role": "system", "content": STAGE2_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_hierarchical_intermediate_messages(
    cluster_breakdown: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": STAGE2_HIERARCHICAL_INTERMEDIATE_PROMPT},
        {
            "role": "user",
            "content": f"Summarize this section into an intermediate grounded synthesis retaining all `[Page N]` citations:\n\n{cluster_breakdown}",
        },
    ]


def build_single_window_messages(
    pages: list[RenderedPage],
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for page in pages:
        content.append({"type": "text", "text": f"=== PAGE {page.page_number} ==="})
        content.append({"type": "image_url", "image_url": {"url": page.data_uri}})

    if prompt and prompt.strip():
        synthesis_instruction = (
            f"Under `{EXECUTIVE_SYNTHESIS_MARKER}`, directly and thoroughly answer the following user query "
            f"using mandatory `[Page N]` citations and verbatim quotes (`> \"...\" [Page N]`) grounded in the pages above:\n"
            f"USER QUERY: {prompt.strip()}"
        )
    else:
        synthesis_instruction = (
            f"Under `{EXECUTIVE_SYNTHESIS_MARKER}`, {DEFAULT_EXECUTIVE_SYNTHESIS_STRUCTURE}"
        )

    content.append(
        {
            "type": "text",
            "text": (
                f"First output `{GROUNDED_PAGES_MARKER}` followed by `## [Page N]` sections for all provided pages.\n"
                f"Then output `{EXECUTIVE_SYNTHESIS_MARKER}`.\n{synthesis_instruction}"
            ),
        }
    )

    return [
        {"role": "system", "content": SINGLE_WINDOW_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_single_window_response(raw_response: str) -> ParsedSingleWindowReading:
    cleaned = raw_response.replace(GROUNDED_PAGES_MARKER, "").strip()
    if EXECUTIVE_SYNTHESIS_MARKER in raw_response:
        before_syn, after_syn = raw_response.split(EXECUTIVE_SYNTHESIS_MARKER, 1)
        return ParsedSingleWindowReading(
            executive_synthesis=after_syn.strip(),
            grounded_pages=before_syn.replace(GROUNDED_PAGES_MARKER, "").strip(),
        )
    return ParsedSingleWindowReading(
        executive_synthesis=cleaned,
        grounded_pages=cleaned,
    )


def format_visual_reading_markdown(
    doc: RasterizedDocument,
    executive_synthesis: str,
    grounded_pages: str,
    status: str = "completed",
    failed_pages: list[int] | None = None,
    warning_notice: str | None = None,
) -> str:
    lines: list[str] = []
    if warning_notice and warning_notice.strip():
        lines.append(warning_notice.strip())
        lines.append("")

    processed_str = format_page_ranges(doc.pages_processed) if doc.pages_processed else "none"
    lines.extend(
        [
            f"# Visual Reading: {doc.path.name}",
            "",
            "## Document Metadata",
            f"- **Filename**: `{doc.path.name}`",
            f"- **Pages**: {doc.total_pages}",
            f"- **Pages Processed**: {processed_str}",
            f"- **Status**: `{status}`",
        ]
    )
    if failed_pages:
        ranges = format_page_ranges(failed_pages)
        lines.append(f"- **Failed Pages**: {ranges}")

    lines.extend(
        [
            "",
            "## Executive Synthesis",
            executive_synthesis.strip(),
            "",
            "---",
            "",
            "## Grounded Page Breakdown",
            grounded_pages.strip(),
        ]
    )
    return "\n".join(lines).strip() + "\n"
