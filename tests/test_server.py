from __future__ import annotations

import asyncio
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
    def __init__(self, response_text: str | list[str] | None = None) -> None:
        self.response_text = response_text
        self.calls: list[list[dict[str, Any]]] = []

    async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        if isinstance(self.response_text, list):
            idx = len(self.calls) - 1
            if idx < len(self.response_text):
                return self.response_text[idx]
            return self.response_text[-1]
        return self.response_text or ""


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

    with pytest.raises(OmlxConnectionError, match=r"not found in available (?:oMLX )?models"):
        await read_pdf_vision(str(pdf_path), vlm_adapter=client)


async def test_mcp_server_registers_read_pdf_vision_tool_schema() -> None:
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "read_pdf_vision" in tool_names
    tool = next(t for t in tools if t.name == "read_pdf_vision")
    props = tool.input_schema.get("properties", {})
    assert set(props.keys()) == {"path", "prompt", "dpi", "pages"}


async def test_multi_window_map_reduce_visual_reading_with_boundary_context(
    tmp_path: Path,
) -> None:
    page_texts = [f"Page {i}: Section {i} content" for i in range(1, 6)]
    pdf_path = _create_sample_pdf(tmp_path / "long_document.pdf", page_texts)

    window_1_output = (
        "## [Page 1]\nPage 1 breakdown\n\n"
        "## [Page 2]\nPage 2 breakdown\n\n"
        "## [Page 3]\nPage 3 breakdown table continuing onto page 4\n"
    )
    window_2_output = (
        "### ## [Page 3]\nContext topic note for Page 3 that must be stripped\n\n"
        "### ## [Page 4]\nPage 4 continuing table row 42\n\n"
        "### ## [Page 5]\nPage 5 concluding notes\n"
    )
    stage_2_output = (
        "### Executive Summary\n"
        "Multi-window analysis indicates continuous progress across 5 pages [Page 1][Page 5].\n"
        "### Key Findings\n"
        "Table spans pages 3 and 4 [Page 3][Page 4]."
    )

    adapter = RecordingVlmAdapter([window_1_output, window_2_output, stage_2_output])

    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    # 1. Output structure checks
    assert "long_document.pdf" in result
    assert "- **Pages**: 5" in result
    assert "- **Status**: `completed`" in result
    assert "### Executive Summary" in result
    assert "Multi-window analysis indicates continuous progress" in result
    assert "## [Page 1]" in result
    assert "## [Page 3]" in result
    assert "## [Page 4]" in result
    assert "## [Page 5]" in result
    assert "Context topic note for Page 3 that must be stripped" not in result

    # Ordering: Executive Synthesis must precede Grounded Page Breakdown
    synthesis_pos = result.index("### Executive Summary")
    breakdown_pos = result.index("## [Page 1]")
    assert synthesis_pos < breakdown_pos

    # 2. Seam verification: 2 window calls (pages 1-3, pages 4-5) + 1 stage-2 synthesis call
    assert len(adapter.calls) == 3

    # Window 1: Pages 1-3, no boundary context prefix
    w1_user_items = adapter.calls[0][-1]["content"]
    w1_texts = [it["text"] for it in w1_user_items if it["type"] == "text"]
    assert "=== PAGE 1 ===" in w1_texts
    assert "=== PAGE 3 ===" in w1_texts
    assert "=== PAGE 4 ===" not in w1_texts
    assert not any("READ-ONLY" in t for t in w1_texts)

    # Window 2: Preceding page 3 as read-only visual context, then target pages 4-5
    w2_user_items = adapter.calls[1][-1]["content"]
    w2_texts = [it["text"] for it in w2_user_items if it["type"] == "text"]
    context_text = next(t for t in w2_texts if "READ-ONLY" in t and "PAGE 3" in t)
    assert "Do NOT generate a breakdown" not in context_text
    assert "## [Page 3]" in context_text
    assert "=== PAGE 4 ===" in w2_texts
    assert "=== PAGE 5 ===" in w2_texts

    # Window 2 visual context is page 3 image
    context_idx = next(i for i, t in enumerate(w2_user_items) if t.get("type") == "text" and "READ-ONLY" in t.get("text", ""))
    assert w2_user_items[context_idx + 1]["type"] == "image_url"

    # Stage 2: Synthesis receives combined transcripts and produces Executive Synthesis
    s2_content = adapter.calls[2][-1]["content"]
    assert isinstance(s2_content, str) or isinstance(s2_content, list)
    s2_text = s2_content if isinstance(s2_content, str) else " ".join(c.get("text", "") for c in s2_content)
    assert "## [Page 1]" in s2_text
    assert "## [Page 5]" in s2_text
    assert "Context topic note for Page 3 that must be stripped" not in s2_text


async def test_multi_window_concurrency_is_bounded_by_omlx_max_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMLX_MAX_CONCURRENCY", "2")
    page_texts = [f"Page {i}" for i in range(1, 16)]  # 3 full windows of 5 pages
    pdf_path = _create_sample_pdf(tmp_path / "15_pages.pdf", page_texts)

    class ConcurrencyTrackingAdapter:
        def __init__(self) -> None:
            self.current_in_flight = 0
            self.max_in_flight = 0

        async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
            self.current_in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.current_in_flight)
            await asyncio.sleep(0.04)
            self.current_in_flight -= 1
            return "## [Page N]\nSample breakdown content"

    adapter = ConcurrencyTrackingAdapter()
    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    assert "- **Pages**: 15" in result
    assert adapter.max_in_flight == 2


async def test_hierarchical_synthesis_fallback_when_transcripts_exceed_140k_chars(
    tmp_path: Path,
) -> None:
    page_texts = [f"Page {i}" for i in range(1, 7)]  # 2 windows of 3 pages (1-3, 4-6)
    pdf_path = _create_sample_pdf(tmp_path / "huge_doc.pdf", page_texts)

    # Generate huge breakdowns > 140,000 characters total
    w1_huge = "## [Page 1]\n" + ("Detailed quarterly data metric analysis. " * 2000)  # ~80k chars
    w2_huge = "## [Page 4]\n" + ("Continuing enterprise architecture breakdown. " * 2000)  # ~80k chars
    assert len(w1_huge) + len(w2_huge) > 140_000

    intermediate_1 = "### Intermediate Cluster 1\nSummarized cluster 1 metrics [Page 1]."
    intermediate_2 = "### Intermediate Cluster 2\nSummarized cluster 2 architecture [Page 4]."
    final_synthesis = "### Executive Summary\nUnified synthesis from hierarchical folding [Page 1][Page 4]."

    adapter = RecordingVlmAdapter([w1_huge, w2_huge, intermediate_1, intermediate_2, final_synthesis])
    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    assert "Unified synthesis from hierarchical folding [Page 1][Page 4]." in result
    assert "## [Page 1]" in result
    assert "## [Page 4]" in result

    # Check that hierarchical intermediate synthesis occurred across the seam
    # Window 1 (idx 0), Window 2 (idx 1), Intermediate 1 (idx 2), Intermediate 2 (idx 3), Final (idx 4)
    assert len(adapter.calls) >= 4
    inter_call = adapter.calls[2]
    inter_role = inter_call[0]["role"]
    assert inter_role == "system"
    assert "cluster reducer" in inter_call[0]["content"]


async def test_transient_window_error_retries_once_and_succeeds(
    tmp_path: Path,
) -> None:
    page_texts = [f"Page {i}" for i in range(1, 11)]  # 2 windows of 5 pages
    pdf_path = _create_sample_pdf(tmp_path / "retry_doc.pdf", page_texts)

    class TransientFailureAdapter:
        def __init__(self) -> None:
            self.call_count = 0

        async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
            self.call_count += 1
            # First window call fails once, then succeeds on retry
            if self.call_count == 1:
                raise RuntimeError("Temporary 503 Server Busy")
            if "## [Page" in str(messages):
                return "### Executive Summary\nAll 10 pages synthesized successfully [Page 1][Page 10]."
            return "## [Page N]\nSuccessful window breakdown"

    adapter = TransientFailureAdapter()
    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    assert "- **Status**: `completed`" in result
    assert "All 10 pages synthesized successfully" in result
    assert "AGENT NOTICE" not in result


async def test_persistent_window_failure_emits_partial_status_and_agent_warning_banner(
    tmp_path: Path,
) -> None:
    page_texts = [f"Page {i}" for i in range(1, 7)]  # Window 1 (1-3), Window 2 (4-6)
    pdf_path = _create_sample_pdf(tmp_path / "partial_doc.pdf", page_texts)

    class FailingSecondWindowAdapter:
        async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
            user_content = messages[-1]["content"]
            user_text = user_content if isinstance(user_content, str) else " ".join(c.get("text", "") for c in user_content)
            # Window 2 contains === PAGE 4 ===
            if "=== PAGE 4 ===" in user_text:
                raise RuntimeError("Model timeout on complex graphic on page 4")
            if "DOCUMENT GROUNDED PAGE BREAKDOWNS" in user_text:
                return "### Executive Summary\nSynthesis based on available surviving pages 1-3 [Page 1]."
            return "## [Page 1]\nPage 1 content\n\n## [Page 2]\nPage 2 content\n\n## [Page 3]\nPage 3 content"

    adapter = FailingSecondWindowAdapter()
    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    # Partial status and metadata
    assert "- **Status**: `partial`" in result
    assert "- **Failed Pages**: 4-6" in result

    # Agent warning banner at top
    assert result.startswith("> [!WARNING] AGENT NOTICE")
    assert "pages 4-6" in result
    assert "inform the user" in result

    # Surviving content is present
    assert "Synthesis based on available surviving pages 1-3" in result
    assert "## [Page 1]" in result


async def _require_live_omlx() -> None:
    client = OmlxClient()
    try:
        await client.ensure_ready()
    except OmlxConnectionError as exc:
        pytest.skip(f"Local oMLX Q-Vision server is not reachable: {exc}")


async def test_single_window_pdf_visual_reading_unwraps_outer_markdown_code_fences(
    tmp_path: Path,
) -> None:
    pdf_path = _create_sample_pdf(tmp_path / "fenced.pdf", ["Page 1: Code 9482"])
    fenced_output = (
        "```html\n"
        "<!-- GROUNDED_PAGES -->\n"
        "## [Page 1]\n"
        "Verification code 9482.\n"
        "<!-- EXECUTIVE_SYNTHESIS -->\n"
        "The verification code is 9482 [Page 1].\n"
        "```"
    )
    adapter = RecordingVlmAdapter(fenced_output)

    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    assert "The verification code is 9482 [Page 1]." in result
    assert "## [Page 1]\nVerification code 9482." in result
    assert "```html" not in result


async def test_default_rasterization_uses_200_dpi_and_1792_max_image_side(
    tmp_path: Path,
) -> None:
    # 16:9 widescreen slide (960x540 pt -> 2667x1500 px at 200 DPI -> capped at 1792x1008)
    wide_pdf = _create_sample_pdf(
        tmp_path / "widescreen_slide.pdf",
        ["Slide 11: Gantt Chart"],
        width=960,
        height=540,
    )
    adapter = RecordingVlmAdapter("<!-- GROUNDED_PAGES -->\n## [Page 1]\nSlide\n<!-- EXECUTIVE_SYNTHESIS -->\nSyn")
    await read_pdf_vision(str(wide_pdf), vlm_adapter=adapter)

    user_items = adapter.calls[0][-1]["content"]
    image_uri = next(it["image_url"]["url"] for it in user_items if it["type"] == "image_url")
    raw_png = base64.b64decode(image_uri.removeprefix("data:image/png;base64,"))
    pix = pymupdf.Pixmap(raw_png)
    assert pix.width == 1792
    assert pix.height == 1008
    assert pix.width % 28 == 0
    assert pix.height % 28 == 0

    # 7x7 inch square page (504x504 pt -> 1400x1400 px at 200 DPI, vs 1050x1050 at 150 DPI)
    square_pdf = _create_sample_pdf(
        tmp_path / "square_page.pdf",
        ["Square page"],
        width=504,
        height=504,
    )
    adapter_sq = RecordingVlmAdapter("<!-- GROUNDED_PAGES -->\n## [Page 1]\nSq\n<!-- EXECUTIVE_SYNTHESIS -->\nSyn")
    await read_pdf_vision(str(square_pdf), vlm_adapter=adapter_sq)

    sq_items = adapter_sq.calls[0][-1]["content"]
    sq_uri = next(it["image_url"]["url"] for it in sq_items if it["type"] == "image_url")
    sq_png = base64.b64decode(sq_uri.removeprefix("data:image/png;base64,"))
    sq_pix = pymupdf.Pixmap(sq_png)
    assert sq_pix.width == 1400
    assert sq_pix.height == 1400


async def test_prompts_enforce_embedded_diagram_scene_text_and_4_column_chromatic_gantt_schema(
    tmp_path: Path,
) -> None:
    # 1. Single-window PDF (2 pages)
    single_pdf = _create_sample_pdf(tmp_path / "single.pdf", ["Slide 8", "Slide 11"])
    single_adapter = RecordingVlmAdapter(
        "<!-- GROUNDED_PAGES -->\n## [Page 1]\nP1\n## [Page 2]\nP2\n<!-- EXECUTIVE_SYNTHESIS -->\nSyn"
    )
    await read_pdf_vision(str(single_pdf), vlm_adapter=single_adapter)

    single_sys = single_adapter.calls[0][0]["content"]
    assert "Phase #1" in single_sys
    assert "50km/h" in single_sys
    assert "Category (from Legend Color)" in single_sys
    assert "Start (X-Axis Gridline)" in single_sys
    assert "End (X-Axis Gridline)" in single_sys

    # 2. Multi-window PDF (5 pages -> 2 Stage 1 calls + 1 Stage 2 synthesis call)
    multi_pdf = _create_sample_pdf(tmp_path / "multi.pdf", [f"Slide {i}" for i in range(1, 6)])
    multi_adapter = RecordingVlmAdapter(
        [
            "## [Page 1]\nP1\n## [Page 2]\nP2\n## [Page 3]\nP3",
            "## [Page 3]\nCtx\n## [Page 4]\nP4\n## [Page 5]\nP5",
            "### Executive Summary\nSyn [Page 1].",
        ]
    )
    await read_pdf_vision(str(multi_pdf), vlm_adapter=multi_adapter)

    stage1_sys = multi_adapter.calls[0][0]["content"]
    stage1_user_texts = [
        it["text"] for it in multi_adapter.calls[0][-1]["content"] if it["type"] == "text"
    ]
    stage1_combined = stage1_sys + "\n" + "\n".join(stage1_user_texts)
    assert "Phase #1" in stage1_combined
    assert "50km/h" in stage1_combined
    assert "Category (from Legend Color)" in stage1_combined
    assert "Start (X-Axis Gridline)" in stage1_combined
    assert "End (X-Axis Gridline)" in stage1_combined

    stage2_sys = multi_adapter.calls[2][0]["content"]
    stage2_user = multi_adapter.calls[2][-1]["content"]
    stage2_combined = f"{stage2_sys}\n{stage2_user}"
    assert "Diagrams & Visual Analyses" in stage2_combined
    assert "Gantt" in stage2_combined or "legend" in stage2_combined.lower()


async def test_multi_window_reindexes_shifted_context_sections_and_strips_empty_callouts(
    tmp_path: Path,
) -> None:
    page_texts = [f"Page {i}" for i in range(1, 7)]  # Window 1: 1-3, Window 2: ctx 3 + target 4-6
    pdf_path = _create_sample_pdf(tmp_path / "shifted_context.pdf", page_texts)

    w1_out = (
        "## [Page 1]\nSlide 1\n### Visual & Diagram Callouts\n- None.\n\n"
        "## [Page 2]\nSlide 2\n### Visual & Diagram Callouts\n- Phase #1: Detect traffic sign\n\n"
        "## [Page 3]\nSlide 3 SMART Goals\n"
    )
    # Simulate model labeling the 4 images in Window 2 (ctx 3 + target 4, 5, 6) as 4, 5, 6, 7
    w2_out = (
        "## [Page 4]\nSlide 3 SMART Goals repeated on context image\n\n"
        "## [Page 5]\nSlide 4 Project Scope Table\n\n"
        "## [Page 6]\nSlide 5 Schedule Timeline\n\n"
        "## [Page 7]\nSlide 6 Conclusion\n"
    )
    s2_out = "### Executive Summary\nAll 6 pages aligned [Page 2][Page 6]."

    adapter = RecordingVlmAdapter([w1_out, w2_out, s2_out])
    result = await read_pdf_vision(str(pdf_path), vlm_adapter=adapter)

    assert "Slide 3 SMART Goals repeated on context image" not in result
    assert "## [Page 4]\nSlide 4 Project Scope Table" in result
    assert "## [Page 5]\nSlide 5 Schedule Timeline" in result
    assert "## [Page 6]\nSlide 6 Conclusion" in result
    assert "## [Page 7]" not in result
    assert "- Phase #1: Detect traffic sign" in result
    assert "- None." not in result





async def test_live_omlx_q_vision_smoke_reading(tmp_path: Path) -> None:
    await _require_live_omlx()

    pdf_path = _create_sample_pdf(
        tmp_path / "live_smoke.pdf",
        ["ACME Corp Q3 2026 Report\nTotal Revenue: $128M\nOperating Margin: 24%"],
    )

    result = await read_pdf_vision(
        str(pdf_path),
        prompt="What is ACME Corp's Q3 2026 Total Revenue and Operating Margin?",
        dpi=150,
    )

    assert "# Visual Reading: live_smoke.pdf" in result
    assert "## Document Metadata" in result
    assert "- **Status**: `completed`" in result
    assert "## Executive Synthesis" in result
    assert "## Grounded Page Breakdown" in result
    assert "128" in result
    assert "24%" in result


async def test_live_smoke_check_skips_cleanly_when_omlx_is_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(pytest.skip.Exception, match="Local oMLX Q-Vision server is not reachable"):
        await _require_live_omlx()



