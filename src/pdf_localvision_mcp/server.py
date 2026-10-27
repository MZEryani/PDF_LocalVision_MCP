from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from pdf_localvision_mcp.ingestion import rasterize_pdf_file
from pdf_localvision_mcp.omlx_client import OmlxClient
from pdf_localvision_mcp.prompts import (
    build_single_window_messages,
    format_visual_reading_markdown,
    parse_single_window_response,
)

mcp = MCPServer("pdf-localvision-mcp")


class VlmAdapter(Protocol):
    async def complete_messages(self, messages: list[dict[str, Any]]) -> str: ...


async def read_pdf_vision(
    path: str,
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
    *,
    vlm_adapter: VlmAdapter | None = None,
) -> str:
    target_path = Path(path).expanduser()
    doc = rasterize_pdf_file(target_path, dpi=dpi)
    adapter: VlmAdapter = vlm_adapter if vlm_adapter is not None else OmlxClient()

    messages = build_single_window_messages(doc.pages, prompt=prompt)
    raw_response = await adapter.complete_messages(messages)
    parsed = parse_single_window_response(raw_response)
    return format_visual_reading_markdown(
        doc=doc,
        executive_synthesis=parsed.executive_synthesis,
        grounded_pages=parsed.grounded_pages,
    )


@mcp.tool(name="read_pdf_vision")
async def _read_pdf_vision_tool(
    path: str,
    prompt: str | None = None,
    dpi: int = 150,
    pages: str | None = None,
) -> str:
    """Perform a native multimodal Visual Reading of a PDF file or directory of PDFs using local oMLX Q-Vision."""
    return await read_pdf_vision(path=path, prompt=prompt, dpi=dpi, pages=pages)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
