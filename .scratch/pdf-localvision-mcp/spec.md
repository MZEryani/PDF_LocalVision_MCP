# PDF LocalVision MCP Specification

## Overview

An MCP server that performs native multimodal **Visual Readings** of PDF documents using a local oMLX vision language model (`Q-Vision`). Instead of extracting embedded PDF text layers or running separate OCR pre-parsers, the server rasterizes PDF pages directly into PNG images in-memory and processes them through a two-tier grounded visual reading pipeline.

## MCP Tool Contract

- **Tool Name**: `read_pdf_vision`
- **Arguments**:
  - `path` (string, required): The **Target Path** — either a single `.pdf` file or a directory containing `.pdf` files.
  - `prompt` (string | null, optional, default `null`): Specific extraction goal or query. Defaults to an exhaustive structural and semantic **Visual Reading**.
  - `dpi` (integer, optional, default `150`): Rasterization resolution in dots per inch.
  - `pages` (string | null, optional, default `null`): 1-based page selection expression (e.g., `"1-5, 8"`). Defaults to all pages in the document.

## Visual Rasterization & Sizing

- Pages are rendered in-memory to PNG using PyMuPDF without intermediate text extraction.
- Rendered dimensions are snapped to multiples of `28` with the longest edge capped at `2044` pixels (the largest multiple of `28` within oMLX's `2048` pixel limit, configurable via `OMLX_MAX_IMAGE_SIDE`).

## Two-Tier Grounded Visual Reading Pipeline

- **Environment Configuration**:
  - `OMLX_BASE_URL` (default `http://localhost:8000/v1`)
  - `OMLX_MODEL` (default `Q-Vision`)
  - `OMLX_API_KEY` (default `"omlx"`)
  - `OMLX_MAX_CONCURRENCY` (default `2`)
  - `OMLX_MAX_IMAGE_SIDE` (default `2044`)
- **Single-Window PDFs (`<= 5` pages)**:
  - Processed in a single multimodal call where each page image is preceded by an explicit `=== PAGE N ===` marker.
  - Autoregressively generates the **Grounded Page Breakdown** (`## [Page N]` sections containing reading-order Markdown, HTML `<table>` tags with `rowspan`/`colspan`, LaTeX equations, linearized chart data tables/Mermaid diagrams, and verbatim anchor quotes) first, followed by the **Executive Synthesis**, then reorders the **Executive Synthesis** above the **Grounded Page Breakdown** in the returned Markdown.
- **Multi-Window PDFs (`> 5` pages)**:
  - Sliced into 5-page **Page Windows** processed with bounded concurrency (`2` by default) and 1 retry on transient window errors.
  - Windows `2..K` prepend page `N-1` as a read-only visual context image so continuing sentences and table `<thead>` headers are preserved across window boundaries.
  - Stage 2 synthesizes all **Window Readings** into a unified **Executive Synthesis** (with automatic two-level hierarchical synthesis if combined transcripts exceed 140,000 characters) placed above the full **Grounded Page Breakdown**.
- **Failure Isolation & Agent Notice Banner**:
  - Unreachable oMLX endpoints fail fast with host details and `omlx start` / `omlx serve` remediation instructions.
  - Partial window failures or per-file failures in a directory batch emit a prominent `> [!WARNING] AGENT NOTICE` banner instructing the calling agent to inform the user of the specific pages or files that failed.
