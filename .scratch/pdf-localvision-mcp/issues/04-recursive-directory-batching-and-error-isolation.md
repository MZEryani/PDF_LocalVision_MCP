# 04: Recursive directory Target Path batching and per-file error isolation

**What to build:** Extend `read_pdf_vision` so passing a directory as the **Target Path** recursively discovers all `.pdf` files (case-insensitively), skips hidden directories, follows symlinked `.pdf` files while deduplicating by canonical path and guarding against directory cycles, processes each file in deterministic relative-path order, isolates corrupted or password-protected PDFs per-file without crashing the batch, emits a prominent agent warning banner when any file fails, and returns an aggregated batch report.

**Blocked by:** 01: Single-window PDF Visual Reading tracer bullet

**Status:** resolved

- [x] Passing a directory **Target Path** recursively discovers `.pdf` files case-insensitively (`*.pdf`, `*.PDF`) and sorts them deterministically by relative path.
- [x] Directory traversal skips hidden directories (names starting with `.`) and guards against directory symlink cycles, while following symlinked `.pdf` files and deduplicating by canonical resolved path so the same file is never processed twice.
- [x] Passing an empty directory (containing zero `.pdf` files) returns a clear diagnostic message stating no `.pdf` files were found in the **Target Path**.
- [x] Corrupted PDFs and password-protected PDFs are caught per-file and reported with their specific failure reason (`encrypted / password-protected` vs. `corrupted / unreadable PDF`) without aborting the remaining files in the directory batch.
- [x] The directory output begins with a Batch Summary header (`total_found`, `succeeded`, `failed`) and, whenever `failed > 0` (or any document has a partial failure), includes a `> [!WARNING] AGENT NOTICE` banner instructing the calling agent to inform the user which files or pages failed and why.
- [x] Automated tests verify recursive directory ingestion, case-insensitive matching, hidden-directory skipping, symlinked PDF deduplication, empty-directory reporting, and batch resilience when encrypted or corrupted PDFs are mixed with valid PDFs.
