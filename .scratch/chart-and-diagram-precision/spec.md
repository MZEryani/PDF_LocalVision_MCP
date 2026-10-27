# Chart, Diagram, and Multi-Window Visual Reading Precision Specification

Status: resolved

## Problem Statement

When performing a **Visual Reading** on visually complex PDF documents (such as engineering presentation decks containing embedded road photographs with overlaid flowchart callouts, color-coded Gantt timeline charts with faint vertical gridlines, and dense evaluation matrices), the current pipeline drops critical graphical and spatial information in two specific ways:

1. **Embedded Diagram & Scene-Text Compression + Context-Page Heading Drift**:
   - When a **Page Window** contains 5 target pages plus 1 read-only context page (6 total page images in a single `Q-Vision` inference request), the 7B vision-language model compresses its per-page token budget on slides that follow dense matrices. As a result, it transcribes large typographic cards on a slide while skipping smaller scene-text, phase labels, arrows, and sign values embedded inside photographs or flowchart diagrams (e.g., `"Phase #1: Detect traffic sign"` and `"Phase #2: Recognize traffic sign (50km/h)"`).
   - Additionally, the Stage 1 multi-window prompt contains a self-contradiction regarding the read-only context page: the text block immediately above the context image instructs the model *not* to generate a breakdown for `Page K`, while the closing prompt block instructs the model to emit a 1-line `## [Page K]` note. When the model follows the first instruction and skips `## [Page K]`, its first emitted heading (`## [Page K+1]`) binds to the context image (`Page K`), shifting all **Grounded Page Breakdown** headings in the **Window Reading** by +1 page.

2. **Loss of Chromatic Legend Metadata & Horizontal Bar Extents on Charts**:
   - The current **Grounded Page Breakdown** prompt requests only a generic "linearized table of plotted values/milestones" for figures, charts, and timelines. When reading a color-coded Gantt chart or bar chart, `Q-Vision` emits a two-column `| Week | Activity |` table that completely omits the color-coded legend categories and guesses sequential integers instead of tracing each horizontal bar's left and right edges to the X-axis gridlines.
   - Furthermore, rasterizing widescreen presentation slides at `150` DPI with a default maximum image side of `1344` pixels (`1344 × 756`) downsamples 1-pixel `#eeeeee` vertical chart gridlines until anti-aliasing washes them out against the white background, preventing the vision encoder from projecting horizontal bar start/end boundaries down to the timeline axis.

## Solution

Upgrade the **Visual Reading** rasterization, **Page Window** sizing, context-page alignment, and multimodal structural prompts so that embedded diagram scene-text, flowchart callouts, color-coded chart legends, and continuous horizontal bar spans are captured with high fidelity:

1. **Higher-Fidelity Default Rasterization for Chart Gridlines**:
   - Increase the default rasterization DPI from `150` to `200` and raise the default maximum image side from `1344` pixels to `1792` pixels (`64 × 28`, safely inside oMLX's `2048`-pixel limit and aligned to `Qwen2.5-VL`'s `28 × 28` patch grid). At `1792 × 1008` on 16:9 slides, faint 1-pixel vertical gridlines and small embedded diagram labels remain crisp without exceeding per-window visual token budgets.
2. **Focused 3-Page Windows (`PAGE_WINDOW_SIZE = 3`)**:
   - Reduce the **Page Window** size from `5` pages to `3` pages (with at most `1` read-only context page, capping any single Stage 1 call at 4 images / `~8k` visual tokens). This prevents multi-image attention compression on `Q-Vision` so embedded photograph callouts, flowchart phases, and fine annotations are transcribed completely alongside main slide cards.
3. **Unambiguous 1-to-1 Read-Only Context Page Alignment & Stripping**:
   - Unify the `=== READ-ONLY VISUAL CONTEXT: PAGE K ===` prompt block and closing instruction so both consistently require a single 1-line `## [Page K]` alignment heading for the context image before emitting full `## [Page N]` sections for target pages.
   - Harden context-page section stripping so any heading prefix variant (`## [Page K]` or `### ## [Page K]`) is cleanly removed before assembling the **Window Reading**, guaranteeing zero off-by-one page shifts across window boundaries.
4. **Explicit Scene-Text Callout & Chromatic Gantt/Chart Extraction Prompts**:
   - Update both the single-window and Stage 1 multi-window prompts to require:
     - Verbatim extraction of all text nested inside embedded photographs, diagrams, flowcharts, callout boxes, badges, arrows, and traffic/UI signs (including numbered phases such as `Phase #1`, `Phase #2`, and unit labels like `50km/h`).
     - Structured extraction for Gantt charts, timelines, and multi-series charts that first lists every **Legend** category with its visual color/pattern, and then emits a 4-column linearized Markdown or HTML table (`| Task / Activity | Category (from Legend Color) | Start (X-Axis Gridline) | End (X-Axis Gridline) |`) by tracing each bar's left start edge and right end edge directly to the vertical axis gridlines.
   - Update the Stage 2 **Executive Synthesis** prompt so that under `### Diagrams & Visual Analyses`, all extracted flowchart phases, embedded diagram callouts, chart legend categories, and Gantt task start/end ranges are preserved verbatim with `[Page N]` citations rather than summarized away.

## User Stories

1. As an AI agent calling `read_pdf_vision`, I want every embedded diagram callout, phase label, arrow annotation, and road-sign value inside a slide photograph to be transcribed verbatim in the **Grounded Page Breakdown**, so that I can answer detailed questions about flowcharts and visual examples without missing text locked inside images.
2. As an AI agent calling `read_pdf_vision`, I want **Page Windows** to be sized at 3 target pages per multimodal request, so that `Q-Vision` allocates sufficient visual attention and output tokens to transcribe both primary layout cards and secondary embedded diagrams on every page.
3. As an AI agent analyzing a multi-window PDF document, I want the read-only context page (`Page K`) to maintain strict 1-to-1 image-to-heading alignment during Stage 1 inference and be stripped cleanly afterward, so that no **Grounded Page Breakdown** section `## [Page N]` ever shifts onto an adjacent page's content.
4. As an AI agent reading a PDF containing a Gantt chart or project timeline, I want the **Grounded Page Breakdown** to explicitly list all color-coded legend categories alongside their visual colors, so that I can identify which category each task belongs to.
5. As an AI agent reading a PDF containing a Gantt chart or horizontal bar chart, I want each bar's start and end positions traced against the vertical X-axis gridlines into a 4-column table (`Task / Activity`, `Category (from Legend Color)`, `Start (X-Axis Gridline)`, `End (X-Axis Gridline)`), so that I can accurately determine which tasks are active during a specific week and the exact week range of any activity.
6. As an AI agent calling `read_pdf_vision` with default parameters, I want PDF pages rasterized at `200` DPI with a `1792`-pixel maximum side length snapped to multiples of `28`, so that faint 1-pixel chart gridlines and small annotations remain sharp enough for the vision encoder to resolve.
7. As a user configuring the MCP server via environment variables, I want `OMLX_MAX_IMAGE_SIDE` to continue overriding the `1792`-pixel default (clamped to `2044`), so that I can tune image resolution for custom hardware constraints.
8. As an AI agent reading the **Executive Synthesis** of a multi-window PDF, I want the `### Diagrams & Visual Analyses` section to retain specific flowchart phase labels, embedded sign values, legend categories, and Gantt chart timeline ranges with `[Page N]` citations, so that both top-level synthesis and per-page breakdowns contain complete visual data.
9. As a developer running the test suite, I want deterministic tests across the `read_pdf_vision` seam verifying 3-page window slicing, `1792`-pixel / `200`-DPI default rasterization, consistent context-page prompt instructions, robust `### ## [Page K]` context stripping, and diagram/Gantt prompt contracts, so that regressions in visual precision are caught automatically.

## Implementation Decisions

- **Default Rasterization Resolution & Side Cap (`ingestion` module)**:
  - Change `DEFAULT_MAX_IMAGE_SIDE` from `1344` to `1792` (`64 × 28`), which remains safely below oMLX's `2048`-pixel limit (`HARD_MAX_IMAGE_SIDE = 2044`).
  - Change the default `dpi` parameter on `rasterize_pdf_file`, `_read_single_pdf_document`, `_process_directory_batch`, `read_pdf_vision`, and the registered MCP tool `_read_pdf_vision_tool` from `150` to `200`.
- **Page Window Sizing (`server` module)**:
  - Reduce `PAGE_WINDOW_SIZE` from `5` to `3`. Documents with `<= 3` selected pages use the single-window pass; documents with `> 3` selected pages use the two-stage Map-Reduce pipeline with 3-page **Page Windows** plus 1 preceding read-only context page (`<= 4` images per Stage 1 call).
- **Consistent Read-Only Context Page Alignment & Stripping (`prompts` module)**:
  - Remove the contradictory `"Do NOT generate a breakdown for Page {context_page.page_number}."` sentence from the per-image text marker in `build_window_stage1_messages`. Instead, instruct the model right above the context image:
    `"Emit ONLY a single 1-line '## [Page K]' topic note for this context image to maintain 1-to-1 image alignment, and do NOT copy its tables or body onto subsequent pages."`
  - Update `strip_context_page_section` using a regex match (`r"(?:^|\n)(?:#+\s*)*## \[Page " + str(first_target_page_number) + r"\]"`) so that even if the model prefixes headings with `### ## [Page N]`, everything prior to the first target page's `[Page N]` heading (i.e., the context page's `## [Page K]` note) is cleanly stripped and normalized to start with `## [Page N]`.
- **Embedded Scene-Text & Chromatic Chart/Gantt Prompt Rules (`prompts` module)**:
  - Based on the `[DEBUG-d811]` diagnostic prototype on `Mid-sem-presentation.pdf`, embed explicit visual extraction rules into both `SINGLE_WINDOW_SYSTEM_PROMPT` and `STAGE1_WINDOW_SYSTEM_PROMPT` (and reinforce them in the user closing prompt):
    ```text
    - Embedded Photographs, Diagrams & Flowcharts: Inspect inside every embedded photograph, screenshot, diagram, and flowchart. Transcribe ALL overlaid scene-text, numbered phases (e.g., "Phase #1: ...", "Phase #2: ..."), directional arrows, bounding-box callouts, and road-sign or badge values (e.g., "50", "50km/h") verbatim.
    - Charts, Timelines & Gantt Charts:
      1. First extract the full Legend mapping every category name to its visual bar/series color.
      2. For Gantt charts, timelines, and horizontal/vertical bar charts, trace each bar's exact left/start edge and right/end edge down to the axis gridlines (including midpoints between even-numbered ticks) and emit a 4-column table:
         `Task / Activity | Category (from Legend Color) | Start (X-Axis Gridline) | End (X-Axis Gridline)`
    ```
  - Update `STAGE2_SYNTHESIS_SYSTEM_PROMPT` and `DEFAULT_EXECUTIVE_SYNTHESIS_STRUCTURE` so `### Diagrams & Visual Analyses` explicitly preserves numbered diagram phases, sign/badge values, chart legend categories, and Gantt task start/end spans with `[Page N]` citations.

## Testing Decisions

- **What makes a good test**:
  - Tests must exercise observable behavior exclusively through the top-level `read_pdf_vision(path, prompt, dpi, pages, *, vlm_adapter)` entrypoint using the existing `VlmAdapter` (`RecordingVlmAdapter`) boundary. Tests must not call private helpers directly or assert internal implementation details.
- **Modules under test**:
  - `server`, `prompts`, and `ingestion` — all exercised together through `read_pdf_vision`.
- **Prior art**:
  - Existing end-to-end behavioral tests in `tests/test_server.py`, `tests/test_page_selection.py`, and `tests/test_directory_batch.py`, which generate synthetic multi-page PDFs with `pymupdf` in `tmp_path`, pass a `RecordingVlmAdapter`, and assert on both the multimodal payloads sent to the VLM (`=== PAGE N ===` markers, PNG dimensions, system/user prompt instructions) and the formatted **Visual Reading** Markdown returned.

## Out of Scope

- Adding external OCR engines, third-party object detectors, or two-pass bounding-box image-cropping pipelines (`Q-Vision` resolves both Slide 8 and Slide 11 in a single pass once window density, rasterization resolution, and prompt schemas are corrected).
- Changing the MCP tool name (`read_pdf_vision`) or adding new parameters to its public JSON schema.
- Modifying oMLX server daemon configuration outside this repository.

## Further Notes

- Updating `PAGE_WINDOW_SIZE` from `5` to `3` and `DEFAULT_MAX_IMAGE_SIDE` to `1792` at `200` DPI keeps per-window visual token count at `~7,500–8,500` tokens (4 widescreen images at `1792 × 1008` = `64 × 36 = 2,304` patches = `~576` visual tokens per image after `Qwen2.5-VL`'s `2×2` spatial merger), which is actually *lower* total prefill per window than 6 images at `1344px` while delivering `1.78×` higher pixel density per page for faint gridlines and small diagram callouts.

## Comments

- Implemented across `src/pdf_localvision_mcp/ingestion.py`, `src/pdf_localvision_mcp/prompts.py`, `src/pdf_localvision_mcp/omlx_client.py`, and `src/pdf_localvision_mcp/server.py`.
- Verified with 48 automated tests (`uv run pytest`), strict typechecking (`uv run pyright`), and live multi-window verification on `Mid-sem-presentation.pdf` Slides 7–12 (`=== FEEDBACK LOOP VERDICT: GREEN ===`).

