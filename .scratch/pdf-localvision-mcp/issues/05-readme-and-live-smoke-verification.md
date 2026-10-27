# 05: Client registration docs (`README.md`) and live `Q-Vision` smoke verification

**What to build:** Provide a concise `README.md` documenting oMLX + `Q-Vision` prerequisites, environment variables, and exact registration commands for MCP clients (Claude Desktop, Antigravity, Cursor), along with a live smoke verification test against the local oMLX `Q-Vision` server that runs when `localhost:8000` is reachable and skips cleanly when offline.

**Blocked by:** 02: Multi-window Map-Reduce Visual Reading and boundary continuity, 03: Page selection (`pages`) filtering and out-of-bounds clamping, 04: Recursive directory Target Path batching and per-file error isolation

**Status:** resolved

- [x] `README.md` documents how to install and start oMLX (`omlx start` and `omlx serve`) with the `Q-Vision` model alias (`Qwen2.5-VL-7B-Instruct-8bit`).
- [x] `README.md` provides copy-pasteable MCP server registration commands and JSON configuration snippets for Claude Desktop, Antigravity, and Cursor (`uvx`, `uv run`, and `mcp run`).
- [x] `README.md` documents all supported environment variables (`OMLX_BASE_URL`, `OMLX_MODEL`, `OMLX_API_KEY`, `OMLX_MAX_CONCURRENCY`, `OMLX_MAX_IMAGE_SIDE`) and tool parameters (`path`, `prompt`, `dpi`, `pages`).
- [x] The test suite includes a live smoke test that exercises `read_pdf_vision` end-to-end against the local oMLX `Q-Vision` server when reachable (and skips cleanly when oMLX is not running).
