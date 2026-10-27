from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest

from pdf_localvision_mcp.omlx_client import OmlxClient, OmlxConnectionError
from pdf_localvision_mcp.server import mcp, read_pdf_vision


def _create_sample_pdf(path: Path, page_texts: list[str], width: float = 612, height: float = 792) -> Path:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


class RecordingVlmAdapter:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[list[dict[str, Any]]] = []

    async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        return self.response_text


async def test_single_window_pdf_visual_reading_places_executive_synthesis_above_grounded_page_breakdown(
    tmp_path: Path,
) -> None:
    pdf_path = _create_sample_pdf(
        tmp_path / "quarterly_report.pdf",
        ["Page 1: Revenue Table", "Page 2: Architecture Diagram"],
    )
    model_output = (
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "<table><thead><tr><th>Quarter</th><th>Revenue</th></tr></thead>"
        "<tbody><tr><td>Q3</td><td>$42M</td></tr></tbody></table>\n"
        '> "Q3 revenue reached $42M." [Page 1]\n\n'
        "## [Page 2]\n"
        "System architecture flow diagram.\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "### Executive Summary\n"
        "Q3 revenue reached $42M [Page 1] with a modernized architecture [Page 2]."
    )
    adapter = RecordingVlmAdapter(model_output)

    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    # Document Metadata -> Executive Synthesis -> Grounded Page Breakdown ordering
    assert "quarterly_report.pdf" in result
    assert "- **Pages**: 2" in result
    synthesis_idx = result.index("### Executive Summary")
    breakdown_idx = result.index("## [Page 1]")
    assert synthesis_idx < breakdown_idx

    # Verify the multimodal payload sent across the oMLX seam
    assert len(adapter.calls) == 1
    user_content = adapter.calls[0][-1]["content"]
    text_blocks = [item["text"] for item in user_content if item["type"] == "text"]
    image_blocks = [item["image_url"]["url"] for item in user_content if item["type"] == "image_url"]

    assert "=== PAGE 1 ===" in text_blocks
    assert "=== PAGE 2 ===" in text_blocks
    assert len(image_blocks) == 2

    for data_uri in image_blocks:
        assert data_uri.startswith("data:image/png;base64,")
        raw_png = base64.b64decode(data_uri.removeprefix("data:image/png;base64,"))
        pix = pymupdf.Pixmap(raw_png)
        assert pix.width % 28 == 0
        assert pix.height % 28 == 0
        assert max(pix.width, pix.height) <= 2044


async def test_target_path_validation_rejects_missing_or_non_pdf_files(
    tmp_path: Path,
) -> None:
    adapter = RecordingVlmAdapter("unused")
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError, match="does_not_exist.pdf"):
        await read_pdf_vision(str(missing_path), vlm_adapter=adapter)

    non_pdf = tmp_path / "notes.txt"
    non_pdf.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.pdf"):
        await read_pdf_vision(str(non_pdf), vlm_adapter=adapter)


async def test_unreachable_omlx_endpoint_raises_actionable_remediation_error(
    tmp_path: Path,
) -> None:
    pdf_path = _create_sample_pdf(tmp_path / "doc.pdf", ["Page 1"])

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    client = OmlxClient(
        base_url="http://localhost:8000/v1",
        model="Q-Vision",
        api_key="omlx",
        http_client=http_client,
    )

    with pytest.raises(OmlxConnectionError) as exc_info:
        await read_pdf_vision(str(pdf_path), vlm_adapter=client)

    message = str(exc_info.value)
    assert "http://localhost:8000/v1" in message
    assert "Q-Vision" in message
    assert "omlx start" in message
    assert "omlx serve" in message


async def test_custom_prompt_and_max_image_side_clamping_through_omlx_http_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OMLX_MODEL", "Q-Vision")
    monkeypatch.setenv("OMLX_API_KEY", "sk-test-omlx")
    monkeypatch.setenv("OMLX_MAX_IMAGE_SIDE", "1120")

    pdf_path = _create_sample_pdf(tmp_path / "metrics.pdf", ["Page 1: Net Retention 118%"])
    captured_requests: list[httpx.Request] = []

    def mock_omlx_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "Q-Vision", "object": "model"}]},
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<!-- GROUNDED_PAGES -->\n"
                                    "## [Page 1]\n"
                                    '> "Net Retention 118%" [Page 1]\n'
                                    "<!-- EXECUTIVE_SYNTHESIS -->\n"
                                    'Net retention is **118%** [Page 1] (> "Net Retention 118%" [Page 1]).'
                                )
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_omlx_handler))
    client = OmlxClient(http_client=http_client)

    result = await read_pdf_vision(
        str(pdf_path),
        prompt="What is the net retention rate?",
        dpi=300,
        vlm_adapter=client,
    )

    assert 'Net retention is **118%** [Page 1] (> "Net Retention 118%" [Page 1]).' in result
    assert len(captured_requests) == 2
    assert captured_requests[1].headers["Authorization"] == "Bearer sk-test-omlx"

    import json

    body = json.loads(captured_requests[1].content.decode("utf-8"))
    assert body["model"] == "Q-Vision"
    user_items = body["messages"][-1]["content"]
    prompt_texts = [item["text"] for item in user_items if item["type"] == "text"]
    assert any("What is the net retention rate?" in t for t in prompt_texts)

    image_uri = next(item["image_url"]["url"] for item in user_items if item["type"] == "image_url")
    raw_png = base64.b64decode(image_uri.removeprefix("data:image/png;base64,"))
    pix = pymupdf.Pixmap(raw_png)
    assert max(pix.width, pix.height) == 1120
    assert pix.width % 28 == 0
    assert pix.height % 28 == 0


async def test_missing_target_model_on_omlx_raises_remediation_error(
    tmp_path: Path,
) -> None:
    pdf_path = _create_sample_pdf(tmp_path / "doc.pdf", ["Page 1"])

    def missing_model_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "Q-Code", "object": "model"}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(missing_model_handler))
    client = OmlxClient(model="Q-Vision", http_client=http_client)

    with pytest.raises(OmlxConnectionError, match="not found in available oMLX models"):
        await read_pdf_vision(str(pdf_path), vlm_adapter=client)


async def test_mcp_server_registers_read_pdf_vision_tool_schema() -> None:
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "read_pdf_vision" in tool_names
    tool = next(t for t in tools if t.name == "read_pdf_vision")
    props = tool.input_schema.get("properties", {})
    assert set(props.keys()) == {"path", "prompt", "dpi", "pages"}



