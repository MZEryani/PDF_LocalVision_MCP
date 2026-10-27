# 01: Single-window PDF Visual Reading tracer bullet

**What to build:** Expose the `read_pdf_vision(path, prompt=None, dpi=150, pages=None)` tool over an installable stdio MCP server that takes a single-PDF **Target Path** (up to 5 pages), verifies local oMLX availability (`OMLX_BASE_URL`, `OMLX_MODEL`, `OMLX_API_KEY`), rasterizes pages in-memory to PNG with multiple-of-28 smart resizing (longest edge capped at `2044px` via `OMLX_MAX_IMAGE_SIDE`), sends a single **Page Window** with interleaved `=== PAGE N ===` markers to `Q-Vision`, generates the **Grounded Page Breakdown** before the **Executive Synthesis** in a single multimodal call, and returns structured Markdown with Document Metadata, **Executive Synthesis**, and **Grounded Page Breakdown** in that order.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Running the server via `mcp run`, `uv run`, or `stdio` exposes `read_pdf_vision(path: str, prompt: str | None = None, dpi: int = 150, pages: str | None = None)`.
- [x] Validating a single-file **Target Path** rejects missing paths and non-PDF files with a descriptive error.
- [x] Each PDF page is rasterized in-memory to PNG at the requested `dpi` with width and height snapped to multiples of `28` and longest edge clamped to `<= 2044` pixels (`OMLX_MAX_IMAGE_SIDE`), encoded as `data:image/png;base64,...` without intermediate text-layer extraction.
- [x] The multimodal payload interleaves an explicit `=== PAGE N ===` text block before each page image.
- [x] For a single-window PDF (`<= 5` pages), the model generates the per-page **Grounded Page Breakdown** (`## [Page N]` with reading-order prose, HTML `<table>` tags, LaTeX math, chart data tables/Mermaid diagrams, and verbatim quotes) before the **Executive Synthesis**, and the returned Markdown places Document Metadata and **Executive Synthesis** above the **Grounded Page Breakdown**.
- [x] When a custom `prompt` is supplied, the **Executive Synthesis** directly answers the user's query with mandatory `[Page N]` citations and verbatim quotes grounded in the page breakdown.
- [x] When the oMLX endpoint is unreachable or the configured model is missing, the tool raises an actionable error containing the target URL, model name, and `omlx start` / `omlx serve` remediation steps.
- [x] Automated tests verify single-PDF **Visual Reading**, multiple-of-28 PNG base64 payload generation, custom `prompt` handling, and unreachable oMLX error formatting.
