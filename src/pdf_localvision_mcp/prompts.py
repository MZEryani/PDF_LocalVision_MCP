from __future__ import annotations

from typing import Any

from pdf_localvision_mcp.ingestion import RasterizedDocument, RenderedPage

GROUNDED_PAGES_MARKER = "<!-- GROUNDED_PAGES -->"
EXECUTIVE_SYNTHESIS_MARKER = "<!-- EXECUTIVE_SYNTHESIS -->"

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
            f"Under `{EXECUTIVE_SYNTHESIS_MARKER}`, provide an exhaustive synthesis structured into:\n"
            "### Executive Summary\n"
            "### Key Narratives & Findings\n"
            "### Extracted Tabular Data & Metrics\n"
            "### Diagrams & Visual Analyses\n"
            "### Critical Citations Index\n"
            "Every claim, figure, and quote MUST carry an inline `[Page N]` citation."
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


def parse_single_window_response(raw_response: str) -> tuple[str, str]:
    if EXECUTIVE_SYNTHESIS_MARKER in raw_response:
        before_syn, after_syn = raw_response.split(EXECUTIVE_SYNTHESIS_MARKER, 1)
        grounded_pages = before_syn.replace(GROUNDED_PAGES_MARKER, "").strip()
        executive_synthesis = after_syn.strip()
        return executive_synthesis, grounded_pages
    return raw_response.replace(GROUNDED_PAGES_MARKER, "").strip(), ""


def format_visual_reading_markdown(
    doc: RasterizedDocument,
    executive_synthesis: str,
    grounded_pages: str,
    status: str = "completed",
) -> str:
    lines = [
        f"# Visual Reading: {doc.path.name}",
        "",
        "## Document Metadata",
        f"- **Filename**: `{doc.path.name}`",
        f"- **Pages**: {doc.total_pages}",
        f"- **Status**: `{status}`",
        "",
        "## Executive Synthesis",
        executive_synthesis.strip(),
    ]
    if grounded_pages.strip():
        lines.extend(
            [
                "",
                "---",
                "",
                "## Grounded Page Breakdown",
                grounded_pages.strip(),
            ]
        )
    return "\n".join(lines).strip() + "\n"
