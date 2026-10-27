# PDF LocalVision MCP

An MCP server that performs native multimodal **Visual Readings** of PDF documents using a local [oMLX](https://github.com/jundot/omlx) vision-language model (`Q-Vision`).

Instead of extracting embedded PDF text layers or running separate OCR pre-parsers, `pdf-localvision-mcp` rasterizes PDF pages in-memory to PNG (`pymupdf`) with dimensions snapped to multiples of `28` (`<= 2044px`) and processes them through a two-tier grounded reading pipeline:

1. **Grounded Page Breakdown (`## [Page N]`)**: Reading-order Markdown prose, HTML `<table>` tags (preserving `rowspan`/`colspan`), LaTeX equations (`\(...\)` / `\[...\]`), linearized chart data tables & Mermaid diagrams, and verbatim anchor quotes (`> "..." [Page N]`).
2. **Executive Synthesis**: A source-grounded document synthesis where every claim carries an inline `[Page N]` citation.

---

## Prerequisites: oMLX & `Q-Vision`

1. **Install oMLX** (Apple Silicon LLM/VLM server) and ensure the `Q-Vision` alias is mapped to `Qwen2.5-VL-7B-Instruct-8bit` in `~/.omlx/model_settings.json`.
2. **Start the oMLX server** on `http://localhost:8000/v1`:

```bash
# Option A: Start the managed background oMLX daemon (loads configured aliases including Q-Vision)
omlx start

# Option B: Serve Qwen2.5-VL directly on port 8000
omlx serve mlx-community/Qwen2.5-VL-7B-Instruct-8bit --port 8000
```

3. **Verify `Q-Vision` is available**:

```bash
curl http://localhost:8000/v1/models
```

---

## Running & Registering the MCP Server

### CLI Execution (`uv run` / `uvx` / `mcp run`)

```bash
# Run via uv from the repository directory
uv --directory /Users/majd/Documents/Coding/PDF_LocalVision_MCP run pdf-localvision-mcp

# Run via uvx directly from the local repository path
uvx --from /Users/majd/Documents/Coding/PDF_LocalVision_MCP pdf-localvision-mcp

# Run via the MCP CLI
uv --directory /Users/majd/Documents/Coding/PDF_LocalVision_MCP run mcp run src/pdf_localvision_mcp/server.py
```

### 1. Antigravity (`~/.gemini/config/mcp_config.json`)

Add `pdf-localvision` to `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "/Users/majd/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/majd/Documents/Coding/PDF_LocalVision_MCP",
        "run",
        "pdf-localvision-mcp"
      ],
      "env": {
        "OMLX_BASE_URL": "http://localhost:8000/v1",
        "OMLX_MODEL": "Q-Vision"
      },
      "timeout": 300000
    }
  }
}
```

### 2. Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "/Users/majd/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/majd/Documents/Coding/PDF_LocalVision_MCP",
        "run",
        "pdf-localvision-mcp"
      ],
      "env": {
        "OMLX_BASE_URL": "http://localhost:8000/v1",
        "OMLX_MODEL": "Q-Vision"
      }
    }
  }
}
```

Or with Claude Code CLI:

```bash
claude mcp add pdf-localvision -- uv --directory /Users/majd/Documents/Coding/PDF_LocalVision_MCP run pdf-localvision-mcp
```

### 3. Cursor (`~/.cursor/mcp.json` or `.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "/Users/majd/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/majd/Documents/Coding/PDF_LocalVision_MCP",
        "run",
        "pdf-localvision-mcp"
      ],
      "env": {
        "OMLX_BASE_URL": "http://localhost:8000/v1",
        "OMLX_MODEL": "Q-Vision"
      }
    }
  }
}
```

---

## MCP Tool Reference: `read_pdf_vision`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | *(required)* | **Target Path**: path to a single `.pdf` file or a directory containing `.pdf` files (recursively scanned). |
| `prompt` | `str \| None` | `None` | Optional query or extraction goal. When omitted, performs an exhaustive 5-part **Executive Synthesis** + **Grounded Page Breakdown**. |
| `dpi` | `int` | `150` | Rasterization resolution in dots per inch before multiple-of-`28` smart resizing. |
| `pages` | `str \| None` | `None` | Optional 1-based page selection expression (e.g., `"1-5, 8"`). Clamps partial out-of-bounds ranges to document length. |

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OMLX_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL of the local oMLX server. |
| `OMLX_MODEL` | `Q-Vision` | Vision model alias or ID served by oMLX. |
| `OMLX_API_KEY` | `omlx` | Bearer token sent in the `Authorization` header to oMLX. |
| `OMLX_MAX_CONCURRENCY` | `2` | Maximum concurrent 5-page **Page Windows** processed in multi-window PDFs. |
| `OMLX_MAX_IMAGE_SIDE` | `2044` | Maximum pixel length of the longest rendered page edge (snapped down to a multiple of `28`). |

---

## Running Tests

```bash
uv run pytest
uv run pyright
```
