# 03: Page selection (`pages`) filtering and out-of-bounds clamping

**What to build:** Support the optional `pages` parameter (`"1-5, 8"`, etc.) end-to-end in `read_pdf_vision`, validating syntax upfront, rasterizing only the selected 1-based pages while preserving their true document page numbers in `=== PAGE N ===` markers and `[Page N]` citations, clamping partially out-of-bounds ranges to the document's actual page count, and reporting a clean skipped status when the requested range lies completely outside a document's page count.

**Blocked by:** 01: Single-window PDF Visual Reading tracer bullet

**Status:** ready-for-agent

- [ ] Passing a 1-based page selection expression (e.g., `"1-3, 5"`) rasterizes and analyzes only those pages, deduplicating overlapping ranges and sorting page numbers in ascending order.
- [ ] True document page numbers are preserved in `=== PAGE N ===` multimodal labels, `## [Page N]` headings, and `[Page N]` synthesis citations (e.g., `pages="4-5"` labels the images `=== PAGE 4 ===` and `=== PAGE 5 ===`, not `1` and `2`).
- [ ] Malformed `pages` syntax (e.g., `"abc"`, `"0"`, `"-3"`, `"5-2"`) is rejected immediately with a clear validation error before any PDF rasterization or inference calls.
- [ ] Partially out-of-bounds ranges (e.g., `pages="1-10"` on a 4-page PDF) clamp to the available pages (`[1, 2, 3, 4]`) and record both total page count and `pages_processed` in Document Metadata.
- [ ] Totally out-of-bounds ranges (e.g., `pages="10-12"` on a 4-page PDF) skip inference for that document and return a clear status message indicating that the requested pages exceed the document's page count.
- [ ] Automated tests verify valid range parsing, true page-number preservation, malformed syntax rejection, partial clamping, and completely out-of-bounds skipping.
