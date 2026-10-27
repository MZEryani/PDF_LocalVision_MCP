# 02: Multi-window Map-Reduce Visual Reading and boundary continuity

**What to build:** Extend `read_pdf_vision` so documents larger than 5 pages are processed across 5-page **Page Windows** with bounded concurrency (`OMLX_MAX_CONCURRENCY`, default `2`), a 1-page read-only visual context prefix (`page N-1`) on subsequent windows to preserve split sentences and table headers across boundaries, transient per-window retry (`1` retry with backoff), Stage 2 document-level **Executive Synthesis** (with automatic hierarchical folding above 140,000 characters), and graceful degradation to a `"partial"` **Visual Reading** with a prominent agent warning banner if a window fails after retry.

**Blocked by:** 01: Single-window PDF Visual Reading tracer bullet

**Status:** ready-for-human

- [x] Documents with `> 5` selected pages are partitioned into 5-page **Page Windows** and processed concurrently up to `OMLX_MAX_CONCURRENCY` (default `2`).
- [x] Every **Page Window** after the first prepends the rendered image of the immediately preceding page (`N-1`) labeled as read-only visual context so the model can identify continuing sentences and repeat `<thead>` column headers on split tables without re-transcribing page `N-1`.
- [x] Stage 2 synthesizes all **Window Readings** into a unified, `[Page N]`-cited **Executive Synthesis** placed above the combined **Grounded Page Breakdown** (`## [Page 1]` through `## [Page N]`).
- [x] When combined Stage 1 transcripts exceed 140,000 characters, Stage 2 automatically performs two-level hierarchical synthesis so the request stays within the model's 64k context window.
- [x] Transient errors on an individual **Page Window** are retried once; if the window still fails while oMLX remains reachable, the tool returns a `"partial"` **Visual Reading** covering the surviving windows, records the failed page range in Document Metadata, and prepends a `> [!WARNING] AGENT NOTICE` banner instructing the calling agent to inform the user about the failed pages.
- [x] Automated tests verify 5-page window slicing, `N-1` boundary context image injection, concurrency limiting, hierarchical synthesis fallback, and partial-failure agent warning banners.
