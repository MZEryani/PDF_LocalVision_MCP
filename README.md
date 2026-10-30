# PDF LocalVision MCP

An MCP server that performs native multimodal **Visual Readings** of PDF documents using local vision-language models (via [oMLX](https://github.com/jundot/omlx) on Apple Silicon or any OpenAI-compatible vision endpoint such as Ollama, vLLM, or LM Studio).

Instead of extracting embedded PDF text layers or running separate OCR pre-parsers, `pdf-localvision-mcp` rasterizes PDF pages in-memory to PNG (`pymupdf`) with dimensions snapped to multiples of `28` (`<= 1792px` by default) and processes them through a two-tier grounded reading pipeline:

1. **Grounded Page Breakdown (`## [Page N]`)**: Reading-order Markdown prose, HTML `<table>` tags (preserving `rowspan`/`colspan`), LaTeX equations (`\(...\)` / `\[...\]`), linearized chart data tables & diagrams, and verbatim anchor quotes (`> "..." [Page N]`).
2. **Executive Synthesis**: A source-grounded document synthesis where every claim carries an inline `[Page N]` citation.

---

## Prerequisites: Vision Model Backend

### Option A: Local oMLX (Recommended for Apple Silicon)

1. **Install oMLX**:

```bash
brew tap jundot/omlx && brew install omlx
# or via pip / uv:
uv tool install omlx
```

2. **Configure the `Q-Vision` alias** in `~/.omlx/model_settings.json` mapped to `Qwen2.5-VL-7B-Instruct-8bit`:

```json
{
  "version": 1,
  "models": {
    "Qwen2.5-VL-7B-Instruct-8bit": {
      "model_alias": "Q-Vision"
    }
  }
}
```

3. **Start the oMLX server** on `http://localhost:8000/v1`:

```bash
# Option 1: Start the managed background daemon
omlx start

# Option 2: Serve directly
omlx serve mlx-community/Qwen2.5-VL-7B-Instruct-8bit --port 8000
```

### Option B: Ollama, vLLM, or any OpenAI-Compatible Vision Server

Run any vision model supporting image inputs (e.g., `llama3.2-vision`, `qwen2.5-vl`, etc.):

```bash
ollama run llama3.2-vision
```

Then point `pdf-localvision-mcp` to it via `VISION_BASE_URL` (or `OPENAI_BASE_URL`) and `VISION_MODEL` (or `OPENAI_MODEL`).

---

## Running & Registering the MCP Server

### Quick Start with `uvx` (No clone needed)

```bash
# Run directly from GitHub
uvx --from git+https://github.com/majdaleryani/pdf-localvision-mcp pdf-localvision-mcp
```

### From Local Clone

```bash
git clone https://github.com/majdaleryani/pdf-localvision-mcp.git
cd pdf-localvision-mcp

# Run via uv
uv run pdf-localvision-mcp
```

---

### Client Configuration Examples

#### 1. Antigravity (`~/.gemini/config/mcp_config.json`)

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/majdaleryani/pdf-localvision-mcp",
        "pdf-localvision-mcp"
      ],
      "env": {
        "VISION_BASE_URL": "http://localhost:8000/v1",
        "VISION_MODEL": "Q-Vision"
      },
      "timeout": 300000
    }
  }
}
```

#### 2. Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/majdaleryani/pdf-localvision-mcp",
        "pdf-localvision-mcp"
      ],
      "env": {
        "VISION_BASE_URL": "http://localhost:8000/v1",
        "VISION_MODEL": "Q-Vision"
      }
    }
  }
}
```

Or via Claude Code CLI:

```bash
claude mcp add pdf-localvision -- uvx --from git+https://github.com/majdaleryani/pdf-localvision-mcp pdf-localvision-mcp
```

#### 3. Cursor (`~/.cursor/mcp.json` or `.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "pdf-localvision": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/majdaleryani/pdf-localvision-mcp",
        "pdf-localvision-mcp"
      ],
      "env": {
        "VISION_BASE_URL": "http://localhost:8000/v1",
        "VISION_MODEL": "Q-Vision"
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

| Variable | Fallback Keys | Default | Description |
| :--- | :--- | :--- | :--- |
| `VISION_BASE_URL` | `OPENAI_BASE_URL`, `OMLX_BASE_URL` | `http://localhost:8000/v1` | Base URL of the OpenAI-compatible vision server. |
| `VISION_MODEL` | `OPENAI_MODEL`, `OMLX_MODEL` | `Q-Vision` | Vision model ID or alias. If omitted and endpoint hosts a single model, auto-selects that model. |
| `VISION_API_KEY` | `OPENAI_API_KEY`, `OMLX_API_KEY` | `omlx` | Bearer token for authorization. |
| `OMLX_MAX_CONCURRENCY` | — | `1` | Maximum concurrent 3-page **Page Windows** processed in multi-window PDFs. |
| `OMLX_MAX_IMAGE_SIDE` | — | `1792` | Maximum pixel length of the longest rendered page edge (snapped down to a multiple of `28`). |

---

## Running Tests

```bash
uv run pytest
uv run pyright
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
