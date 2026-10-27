# VLM-Based PDF Reading & Structural Extraction (`Qwen2.5-VL-7B-Instruct`)

Research into primary sources on how Vision-Language Models (specifically `Qwen2.5-VL-7B-Instruct-8bit` served via local oMLX as `Q-Vision` with `max_image_side_length: 2048` and `max_model_len: 64000`) can achieve Gemini / NotebookLM-caliber PDF reading, structural fidelity, and source-grounded synthesis.

---

## 1. Primary Sources

1. **Qwen2.5-VL Technical Report** — Bai et al., *Qwen2.5-VL Technical Report*, arXiv:2502.13923 (Feb 2025).
   - Primary URL: <https://arxiv.org/abs/2502.13923> (HTML: <https://arxiv.org/html/2502.13923v1>)
2. **Qwen2.5-VL Official Repository & Cookbooks** — Qwen Team (`QwenLM/Qwen2.5-VL`).
   - Repository: <https://github.com/QwenLM/Qwen2.5-VL>
   - Document Parsing Cookbook (`QwenVL HTML` & Markdown): <https://github.com/QwenLM/Qwen2.5-VL/blob/a065f96c32c1dd022118dc0db488e7f66158b70d/cookbooks/document_parsing.ipynb>
   - OCR & Visual Information Extraction Cookbook: <https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb>
   - Long Document Understanding Cookbook: <https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/long_document_understanding.ipynb>
3. **Google Gemini API Document Understanding Documentation** — Google AI for Developers.
   - Primary URL: <https://ai.google.dev/gemini-api/docs/document-processing>
4. **Gemini 1.5 Technical Report (Long-Context Multimodal Grounding)** — Gemini Team, Google, *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context*, arXiv:2403.05530.
   - Primary URL: <https://arxiv.org/abs/2403.05530>
5. **Google NotebookLM Source-Grounding Documentation** — Google Labs / NotebookLM Help Center.
   - Primary URL: <https://support.google.com/notebooklm/answer/16164461>
   - Google Blog architectural overview: <https://blog.google/technology/ai/notebooklm-google-ai/>
6. **AllenAI `olmOCR` Technical Report & Prompt Implementation** — Poznanski et al., *olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models*, arXiv:2502.18443 (Feb 2025).
   - Primary URL: <https://arxiv.org/abs/2502.18443> (HTML: <https://arxiv.org/html/2502.18443v1>)
   - Prompt source code (`prompts.py`): <https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py>
7. **Docling VLM Pipeline & Prompt Specifications** — IBM Research /LF AI & Data (`docling-project/docling`).
   - VLM Model Specs (`vlm_model_specs.py`): <https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_model_specs.py>
   - VLM Prompts (`vlm_prompts.py`): <https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_prompts.py>
8. **Marker LLM/VLM Processors** — Vik Paruchuri / Datalab (`VikParuchuri/marker`).
   - Table Processor (`llm_table.py`): <https://github.com/VikParuchuri/marker/blob/master/marker/processors/llm/llm_table.py>
   - Image/Chart Description Processor (`llm_image_description.py`): <https://github.com/VikParuchuri/marker/blob/master/marker/processors/llm/llm_image_description.py>
   - Page Correction Processor (`llm_page_correction.py`): <https://github.com/VikParuchuri/marker/blob/master/marker/processors/llm/llm_page_correction.py>

---

## 2. How Qwen2.5-VL Is Trained for Omni-Document Parsing

### 2.1 Native Dynamic Resolution & the Multiple-of-28 Rule
Per the Qwen2.5-VL Technical Report (Section 2.1 & Table 1, <https://arxiv.org/html/2502.13923v1>):
- **Vision Encoder (ViT):** Uses a patch size and stride of **14**, with 2D-RoPE and windowed attention (112×112 window in 28 of 32 layers; full self-attention at layers `{7, 15, 23, 31}`) so computational cost scales linearly with patch count.
- **MLP Vision-Language Merger:** Groups spatially adjacent sets of $2 \times 2 = 4$ patch features (`14 × 2 = 28` pixels per side) and projects them through a two-layer MLP into a single visual token aligned with the LLM hidden dimension (`3584` for `Qwen2.5-VL-7B`).
- **Resolution & Token Math:** Both height $H$ and width $W$ are rounded to multiples of **28** (`smart_resize(height, width, factor=28)`). The number of visual tokens for a single page image of resized dimensions $H' \times W'$ is:
  $$\text{Visual Tokens} = \frac{H'}{28} \times \frac{W'}{28}$$
- **Implications for `Q-Vision` (`max_image_side_length: 2048`, `max_model_len: 64000`):**
  - Standard US Letter page ($8.5 \times 11$ in) at **150 DPI** is $1275 \times 1650$ px $\rightarrow$ aligned to multiples of 28 as $1260 \times 1652$ px = $45 \times 59 = \mathbf{2,655\text{ tokens/page}}$.
  - At **200 DPI** (or longest side capped near `2044` px, the largest multiple of 28 $\le 2048$), a Letter page is $1568 \times 2044$ px = $56 \times 73 = \mathbf{4,088\text{ tokens/page}}$.
  - In Qwen's official `document_parsing.ipynb` (<https://github.com/QwenLM/Qwen2.5-VL/blob/a065f96c32c1dd022118dc0db488e7f66158b70d/cookbooks/document_parsing.ipynb>), single-page parsing uses `min_pixels = 512 * 28 * 28` and `max_pixels = 2048 * 28 * 28` (up to 2,048 visual tokens per page, or ~1176×1540 px for portrait pages).
  - For a 5-page **Page Window**, 5 pages $\times$ 2,048–4,088 tokens/page = **10,240–20,440 input visual tokens**, leaving **43,000+ tokens** in the 64k context window for full verbatim structural output without context pressure.

### 2.2 `QwenVL HTML` & Omni-Document Parsing Ground Truth
In Section 2.2.2 (*Omni-Document Parsing*, <https://arxiv.org/html/2502.13923v1>), Qwen2.5-VL was specifically trained on a unified representation that converts heterogeneous page elements into structured tags in natural reading order:
- **Paragraphs & Headings:** `<p data-bbox="x1 y1 x2 y2">content</p>`, `<h1>`–`<h6>`
- **Tables:** `<table data-bbox="x1 y1 x2 y2" class="table{id}">...</table>` (standard HTML `<tr>`, `<th>`, `<td>` with `colspan`/`rowspan`)
- **Charts (Data Linearization):** `<div class="chart" data-bbox="x1 y1 x2 y2"><img data-bbox="..." /><table>chart content</table></div>` — **crucially, Qwen2.5-VL was trained to convert visual charts directly into structured HTML data tables** inside the chart block.
- **Formulas:** `<div class="formula" data-bbox="..."><img .../><div>formula content (LaTeX)</div></div>`
- **Figures / Illustrations:** `<div class="image caption">...</div>` and `<div class="image ocr">...</div>`
- **Chemical Formulas:** `<div class="chemical formula" format="smile">...</div>`

Because Qwen2.5-VL's post-training corpus includes both `QwenVL HTML` (with `data-bbox`) and clean Markdown/HTML linearization (without bounding boxes), prompting `Qwen2.5-VL-7B-Instruct` to emit **Markdown prose + HTML `<table>` blocks + LaTeX math + structured chart/figure data tables/descriptions** directly activates its native Omni-Document training distribution without spending output tokens on 4-integer bounding-box coordinates for every line.

---

## 3. What Makes Gemini & NotebookLM Effective at PDF Reading

### 3.1 Gemini Multimodal Document Processing Principles
From the Google Gemini API Document Understanding documentation (<https://ai.google.dev/gemini-api/docs/document-processing>):
1. **Holistic Page Vision:** Pages are interpreted visually (scaling small pages to at least $768 \times 768$ and large pages up to $3072 \times 3072$ preserving aspect ratio) so layout hierarchy, tables, diagrams, and charts are read in context rather than stripped by a text-only lexer.
2. **Prompt Ordering:** Google explicitly recommends placing the **image/page content before the text prompt** in the multimodal message array (`[...page_images, {"type": "text", "text": prompt}]`), which also matches Qwen2.5-VL's `long_document_understanding.ipynb` convention.
3. **Orientation & Visual Clarity:** Pages must be right-side up; rotated pages degrade extraction unless detected and corrected.

### 3.2 NotebookLM Source-Grounding Architecture
From Google's NotebookLM documentation (<https://support.google.com/notebooklm/answer/16164461>) and Gemini 1.5 long-context grounding (<https://arxiv.org/abs/2403.05530>):
1. **Closed-Corpus Faithfulness Contract:** The model is strictly forbidden from using parametric knowledge to fill gaps; if a claim is not present in the source pages, it must state that the source does not contain it.
2. **Claim-Level Inline Citations (`[Page N]`):** Every factual assertion, metric, or synthesized point carries an inline citation anchored to the specific page(s) where the evidence appears.
3. **Verbatim Evidence Pairing:** Synthesis is never detached from the raw text. High-value claims are backed by **exact verbatim quotes** or **exact table/chart values** from the cited page so the user can verify the claim without re-reading the PDF.
4. **Structural Preservation of Non-Prose Elements:** Tables, quantitative charts, and architectural diagrams are translated into lossless structured representations (tables + axis/series values + component relationships) rather than vague one-line summaries ("Page 4 shows a bar chart of revenue").

---

## 4. Lessons from `olmOCR`, `Docling`, and `Marker` for 7B VLMs

### 4.1 AllenAI `olmOCR` (Built on `Qwen2-VL-7B` / `Qwen2.5-VL-7B`)
From Poznanski et al. (<https://arxiv.org/html/2502.18443v1>) and `olmocr/prompts/prompts.py` (<https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py>):
- **HTML for Tables, Markdown for Prose:** In `build_openai_silver_data_prompt_v3_simple` and `build_no_anchoring_v4_yaml_prompt`, AllenAI moved from Markdown tables to **HTML tables**:
  > *"Convert tables into HTML format. Keep the syntax simple, but use `<th>` for header rows, and use `rowspan` and `colspans` appropriately. Don't use `<br>` inside of table cells, just split that into new rows as needed. Do NOT use LaTeX or Markdown table syntax."*
  - **Why:** Markdown tables cannot represent `rowspan`/`colspan` or multi-level headers, causing 7B models to misalign columns on financial, scientific, and specification PDFs.
- **Strict LaTeX Math Delimiters:**
  > *"Turn equations and math symbols into a LaTeX representation, make sure to use `\(` and `\)` as a delimiter for inline math, and `\[` and `\]` for block math. Do NOT use ascii or unicode math symbols... Do not hallucinate."*
- **Cross-Page Sentence Continuity:**
  > *"This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are."*
- **Repetition Loop Mitigation on 7B VLMs (Section 4.2, *Decoding*):**
  - The primary failure mode of 7B VLMs on dense or noisy pages is **token/line repetition collapse** at greedy/near-zero temperature.
  - `olmOCR` detects repetition or length overflow and retries with a slightly higher temperature (or presence/repetition penalty, as shown in Qwen's `long_document_understanding.ipynb`: `temperature=0.01`, `presence_penalty=1.5`, `top_p=0.8`).

### 4.2 `Docling` & `Marker` VLM Prompting Patterns
From Docling's `vlm_prompts.py` (<https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_prompts.py>) and Marker's `llm_image_description.py` (<https://github.com/VikParuchuri/marker/blob/master/marker/processors/llm/llm_image_description.py>):
- **Quantitative Chart & Diagram Extraction:**
  - Marker instructs: *"Output a faithful description of the image. Make sure there is enough specific detail to accurately reconstruct the image. If the image is a figure or contains numeric data, include the numeric data in the output."*
  - Docling (`CHANDRA_PROMPT_ENDING`) instructs: *"Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid."*
- **Running Headers/Footers vs. Footnotes:**
  - Both `olmOCR` and `Docling` instruct the VLM to strip repetitive running headers/footers/watermarks while **preserving footnotes, references, and captions**.

---

## 5. Architectural Comparison: Why Naive Multi-Page Summarization Fails vs. Grounded Extraction + Synthesis

When reading a multi-page PDF in **Page Windows** (e.g., 5 pages per window) with a 7B VLM (`Qwen2.5-VL-7B-Instruct-8bit`):

### Failure Mode of Naive "Summarize These Pages" in the Window Reading (Map) Step
If the **Window Reading** prompt asks the model to *"Summarize the key points of these pages"*:
1. **Attention Dilution Across Multiple Images:** When 5 page images are fed simultaneously with an open-ended summarization prompt, a 7B model attends disproportionately to big headings and the first/last pages in the window, skipping dense middle-page paragraphs and footnotes.
2. **Irreversible Information Loss (The "Summary of Summaries" Problem):** Exact numbers, table rows, mathematical formulas, and verbatim quotes are abstracted away during the Window Reading step. Once lost in the Map step, the document-level Reduce step **cannot recover them**—resulting in vague, ungrounded prose that fails specific user queries.
3. **Page Attribution Drift:** Without forcing an explicit per-page boundary (`## [Page N]`) in the Window Reading output, a 5-page window summary blends facts from pages $k\dots k+4$ and attributes them to the wrong page number.

### The NotebookLM-Caliber Two-Stage Architecture

| Stage | Role | Input | Output Structure |
| :--- | :--- | :--- | :--- |
| **Stage 1: Window Reading (Grounded Structural Extraction)** | Lossless visual transcription & structural parsing per page within each Page Window | $1\text{–}5$ rendered page images (resized to multiples of 28, long edge $\le 2044\text{ px}$) + Page-Indexed Extraction Prompt | Strictly partitioned `## Page {N}` blocks containing:<br>1. **Page Metadata** (page type, orientation/legibility, visual elements present)<br>2. **Structured Content (Markdown + HTML `<table>` + LaTeX math)** in natural reading order<br>3. **Visual & Chart Data** (full numeric series / data table + Mermaid or topological breakdown for diagrams)<br>4. **Key Verbatim Quotes** (exact high-signal quotes with `[Page N]` anchor) |
| **Stage 2: Visual Reading (Source-Grounded Synthesis)** | Cross-document reasoning, executive synthesis, or targeted query answering | Concatenated **Window Readings** across all Page Windows (pure structured text, easily fitting 50–100+ pages in the 64k token context window) | NotebookLM-style grounded response:<br>1. **Executive Synthesis / Direct Answer** with inline `[Page N]` citations on every claim<br>2. **Section-by-Section / Thematic Breakdown** with embedded **Verbatim Evidence Quotes** (`> "..." [Page N]`)<br>3. **Consolidated Tables, Equations & Visual Index** preserving exact figures and data tables from the Window Readings |

### Why Single-Page or Small-Window Concurrency Excels in Stage 1
- In `olmOCR` (<https://arxiv.org/html/2502.18443v1>) and `Docling` (<https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_model_specs.py>), Stage 1 is executed **1 page per VLM request** (or small bounded windows of 1–3 pages) at `temperature=0.0` (with retry fallback at `temperature=0.2–0.4` + `presence_penalty=1.2` if repetition occurs).
- Even when multiple pages (e.g., a 3–5 page **Page Window**) are sent in a single request to preserve cross-page sentence/table continuity, **explicitly labeling each image part** (`"Page {N}:"` text part immediately before each `image_url` part) and requiring an explicit `## [Page N]` section per page prevents cross-page attribution drift and ensures no page is skipped.

---

## 6. Concrete Prompt Engineering & Output Structure for `Qwen2.5-VL-7B-Instruct-8bit`

### 6.1 Interleaved Multimodal Message Construction
Rather than dumping 5 anonymous images followed by a prompt, interleave explicit page markers before each image so `Qwen2.5-VL-7B` has a deterministic 1-to-1 binding between image and page number:

```json
[
  {"type": "text", "text": "=== PAGE 1 ==="},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "=== PAGE 2 ==="},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "<Stage 1 Extraction Instructions>"}
]
```

### 6.2 Stage 1 Prompt Template (Window Reading)
Combines the proven constraints from `Qwen2.5-VL` Omni-Document Parsing, `olmOCR` v3/v4, `Docling`, and `Marker`:

```markdown
You are a precision document vision parser. Process each attached PDF page image in order and produce a complete, source-grounded Window Reading.

For EVERY page provided, output a top-level section `## [Page N]` (matching the exact page number) containing:

1. **Structured Text & Layout**:
   - Reproduce all document text in natural human reading order using clean Markdown headings (`###`, `####`), paragraphs, and lists.
   - Strip repetitive running page headers, footers, and watermarks, but preserve all footnotes, citations, captions, and margin notes.
   - Preserve sentences that continue from the previous page or onto the next page verbatim.
   - Do not summarize, paraphrase, or omit body text. Do not hallucinate.

2. **Tables (HTML Format)**:
   - Convert every table into clean HTML (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`) using `colspan` and `rowspan` where cells span multiple columns or rows.
   - Reproduce every row, column header, and cell value faithfully.

3. **Equations & Formulas (LaTeX)**:
   - Convert all inline math to `\(...\)` and block equations to `\[...\]`. Use standard LaTeX commands instead of raw Unicode math symbols.

4. **Charts, Figures & Diagrams (High-Fidelity Data)**:
   - For every chart or plot: output a `<figure data-page="N">` block with its caption, a prose description of axes/trends, AND a Markdown/HTML table of all readable numeric data points/series.
   - For every architecture diagram or flowchart: output a `<figure data-page="N">` block with its caption, a Mermaid diagram (` ```mermaid `) or structured node-edge list representing the flow, and all labels inside the diagram.

5. **Verbatim Anchor Quotes**:
   - At the end of each page section, list 1–3 high-signal verbatim sentences from that page under `**Anchor Quotes [Page N]:**` for downstream citation grounding.
```

### 6.3 Stage 2 Prompt Template (Document-Level Visual Reading / Synthesis)
Operates over the assembled **Window Readings** (plus an optional user query/focus if provided):

```markdown
You are a source-grounded document research engine operating with strict NotebookLM-style attribution rules.
Below are the page-indexed Window Readings extracted from the PDF document.

STRICT GROUNDING RULES:
1. Rely EXCLUSIVELY on the provided Window Readings. Never invent facts, numbers, or extrapolations not supported by the pages.
2. Every factual claim, metric, or argument MUST carry an inline page citation in the format `[Page N]` or `[Pages N–M]`.
3. Pair key synthesized findings with exact verbatim quotes (`> "..." [Page N]`) from the source pages.
4. Preserve quantitative tables, formulas, and chart data in full structural detail—never flatten a table or chart into a vague sentence.
```
