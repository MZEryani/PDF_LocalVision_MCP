from __future__ import annotations

import re
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
     * Natural reading-order prose in Markdown (headers, lists, footnotes, speaker/author tags).
     * Tables as complete HTML `<table>` blocks preserving `<thead>`, `<tbody>`, `rowspan`, `colspan`, and exact cell values.
     * Mathematical equations in LaTeX (`\\(...\\)` inline or `\\[...\\]` display).
     * Embedded Photographs, Diagrams & Flowcharts: Inspect inside every embedded photograph, screenshot, diagram, and flowchart. Transcribe ALL overlaid scene-text, numbered phases (e.g., "Phase #1: ...", "Phase #2: ..."), directional arrows, bounding-box callouts, and road-sign or badge values (e.g., "50", "50km/h") verbatim.
     * Charts, Timelines & Gantt Charts: First list the full Legend mapping every category name to its visual bar/series color. For Gantt charts, timelines, and horizontal/vertical bar charts, trace each bar's exact left/start edge and right/end edge straight down to the vertical X-axis gridlines (using odd-numbered midpoints when a bar starts or ends halfway between even-numbered gridlines) and emit a 4-column table: `| Task / Activity | Category (from Legend Color) | Start (X-Axis Gridline) | End (X-Axis Gridline) |`.
     * Key verbatim anchor quotes formatted as `> "exact quote" [Page N]`.
   - Second: `<!-- EXECUTIVE_SYNTHESIS -->` grounded strictly in the page breakdowns you just produced, citing every claim with `[Page N]`.
3. NEVER emit placeholder image links (`![](...)` or `<img ...>`). Transcribe EVERY visible word, number, percentage, callout card, badge, and caption on the page, and describe all photographs, visual callouts, charts, and diagrams in detail.
"""

STAGE1_WINDOW_SYSTEM_PROMPT = """You are a high-precision multimodal document reading system modeled after Google Gemini and NotebookLM.
You analyze rendered PDF page images directly without OCR pre-parsing.

CRITICAL RULES:
1. Every page image is preceded by an explicit page marker (`=== READ-ONLY VISUAL CONTEXT: PAGE K ===` or `=== PAGE N ===`). Attribute all content strictly to the exact page number in its marker. Never copy text, tables, or diagrams from one page image onto another page's section.
2. For each provided page image, generate a `## [Page N]` section in exact 1-to-1 order containing:
   * Natural reading-order prose in Markdown (headers, lists, footnotes, speaker/author tags).
   * Tables as complete HTML `<table>` blocks preserving `<thead>`, `<tbody>`, `rowspan`, `colspan`, and exact cell values.
   * Mathematical equations in LaTeX (`\\(...\\)` inline or `\\[...\\]` display).
   * Embedded Photographs, Diagrams & Flowcharts: After transcribing all main text and tables on a page, include a `### Visual & Diagram Callouts` sub-section at the bottom of that page's section transcribing any overlaid scene-text, numbered phases (e.g., "Phase #1: ...", "Phase #2: ..."), arrows, and road-sign or badge values (e.g., "50", "50km/h") inside photographs or diagrams on that specific page (or write "None." if that page has no embedded photograph/diagram).
   * Charts, Timelines & Gantt Charts:
     a) First list every Legend item and its exact color.
     b) List all X-axis tick marks/gridline values.
     c) Output a table with columns: `| Task / Activity | Category (from Legend Color) | Start (X-Axis Gridline) | End (X-Axis Gridline) |` by carefully tracing the left and right edges of each horizontal bar down to the vertical gridlines on the X-axis (using odd-numbered week midpoints like 7, 11, 13, 15, 17 when a bar edge lies halfway between even-numbered gridlines).
   * Key verbatim anchor quotes formatted as `> "exact quote" [Page N]`.
3. NEVER emit placeholder image links (`![](...)` or `<img ...>`). Transcribe EVERY visible word, number, percentage, callout card, badge, and caption on each page verbatim.
4. Output ONLY the `## [Page ...]` sections. Do not produce an executive synthesis.
"""

STAGE2_SYNTHESIS_SYSTEM_PROMPT = """You are a high-precision document executive synthesis system modeled after Google NotebookLM.
You receive structured Grounded Page Breakdowns of a document and synthesize them into an authoritative, citation-anchored Executive Synthesis.

CRITICAL RULES:
1. Every claim, finding, table figure, and quote MUST carry an inline `[Page N]` citation grounded strictly in the provided page breakdowns.
2. Ground all numbers and tables verbatim.
3. Under `### Diagrams & Visual Analyses`, preserve all extracted flowchart phases, embedded diagram callouts and sign values, chart legend categories, and Gantt timeline task start/end ranges verbatim with `[Page N]` citations.
"""

STAGE2_HIERARCHICAL_INTERMEDIATE_PROMPT = """You are a document synthesis cluster reducer.
Condense the following page breakdowns into an intermediate grounded synthesis preserving all key metrics, findings, table figures, diagram phases, chart legends, Gantt timeline spans, and inline `[Page N]` citations.
"""


DEFAULT_EXECUTIVE_SYNTHESIS_STRUCTURE = (
    "Provide an exhaustive Executive Synthesis structured into:\n"
    "### Executive Summary\n"
    "### Key Narratives & Findings\n"
    "### Extracted Tabular Data & Metrics\n"
    "### Diagrams & Visual Analyses (include all flowchart phases, embedded sign/badge values, chart legend categories, and Gantt task start/end ranges)\n"
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
                    f"This image is Page {context_page.page_number} (`## [Page {context_page.page_number}]`), provided as preceding visual context to resolve split sentences and continuing table headers."
                ),
            }
        )
        content.append({"type": "image_url", "image_url": {"url": context_page.data_uri}})

    for page in pages:
        content.append({"type": "text", "text": f"=== PAGE {page.page_number} ==="})
        content.append({"type": "image_url", "image_url": {"url": page.data_uri}})

    all_page_numbers = (
        [context_page.page_number, *[p.page_number for p in pages]]
        if context_page is not None
        else [p.page_number for p in pages]
    )
    all_page_ranges = format_page_ranges(all_page_numbers)
    all_headings = ", ".join(f"`## [Page {p}]`" for p in all_page_numbers)
    closing_text = (
        f"Extract pages ({all_page_ranges}) into explicit, full {all_headings} sections in exact order.\n"
        f"Under each `## [Page N]` heading, first write that page's main slide title, then transcribe all text cards, lists, and tables in full, and finally include a `### Visual & Diagram Callouts` sub-section transcribing any overlaid scene-text, phase labels (e.g., 'Phase #1', 'Phase #2'), and sign values (e.g., '50', '50km/h') inside photographs/diagrams on that page (or 'None' if none).\n"
        f"On any page with a Gantt or timeline chart, list every Legend category and its color, then output a 4-column table (`| Task / Activity | Category (from Legend Color) | Start (X-Axis Gridline) | End (X-Axis Gridline) |`).\n"
        f"NEVER use `![](...)` placeholder image tags; transcribe all visible text, numbers, percentages, tables, and timelines in full.\n"
        f"Do not produce any executive synthesis."
    )
    content.append({"type": "text", "text": closing_text})
    return [
        {"role": "system", "content": STAGE1_WINDOW_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def strip_context_page_section(
    raw_window_output: str,
    context_page_number: int | None,
    first_target_page_number: int,
    target_page_numbers: list[int] | None = None,
) -> str:
    cleaned = _strip_outer_code_fence(raw_window_output)
    cleaned = re.sub(r"(?m)^(?:#+\s+)+## (\[Page \d+\])", r"## \1", cleaned)
    cleaned = re.sub(
        r"(?mi)\n*### Visual & Diagram Callouts\s*\n+(?:[-*]\s*)?(?:none\.?|no [^\n]+)\s*",
        "\n\n",
        cleaned,
    ).strip()
    if context_page_number is None:
        return cleaned
    target_heading = f"## [Page {first_target_page_number}]"
    context_heading = f"## [Page {context_page_number}]"
    if context_heading in cleaned and target_heading in cleaned:
        idx = cleaned.index(target_heading)
        return cleaned[idx:].strip()

    # Fallback: if the model started numbering at first_target_page_number on the context image
    # and emitted len(target_page_numbers) + 1 sections, drop the first section and re-index.
    if target_page_numbers:
        matches = list(re.finditer(r"(?m)^## \[Page (\d+)\]", cleaned))
        if len(matches) == len(target_page_numbers) + 1:
            reindexed_parts: list[str] = []
            for idx_m, page_num in enumerate(target_page_numbers):
                sec_start = matches[idx_m + 1].start()
                sec_end = matches[idx_m + 2].start() if idx_m + 2 < len(matches) else len(cleaned)
                sec_text = cleaned[sec_start:sec_end]
                sec_text = re.sub(
                    r"^## \[Page \d+\]",
                    f"## [Page {page_num}]",
                    sec_text,
                    count=1,
                )
                reindexed_parts.append(sec_text.strip())
            return "\n\n".join(reindexed_parts).strip()

    return cleaned


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


def _strip_outer_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def parse_single_window_response(raw_response: str) -> ParsedSingleWindowReading:
    unwrapped = _strip_outer_code_fence(raw_response)
    cleaned = unwrapped.replace(GROUNDED_PAGES_MARKER, "").strip()
    if EXECUTIVE_SYNTHESIS_MARKER in unwrapped:
        before_syn, after_syn = unwrapped.split(EXECUTIVE_SYNTHESIS_MARKER, 1)
        return ParsedSingleWindowReading(
            executive_synthesis=_strip_outer_code_fence(after_syn),
            grounded_pages=_strip_outer_code_fence(before_syn.replace(GROUNDED_PAGES_MARKER, "")),
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
