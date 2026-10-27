# PDF LocalVision MCP

An MCP server that performs native multimodal visual readings of PDF documents using a local oMLX vision language model (`Q-Vision`).

## Language

**Target Path**:
The filesystem path supplied to the tool, resolving to either a single PDF document or a recursive collection of PDF documents.
_Avoid_: Ingestion target, input blob, target payload

**Page Window**:
A bounded, contiguous slice of rendered PDF page images submitted together in a single multimodal inference request.
_Avoid_: Chunk, shard, split

**Window Reading**:
The intermediate structured visual analysis produced by the vision model for a single Page Window before document-level synthesis.
_Avoid_: Chunk summary, partial completion, batch output

**Grounded Page Breakdown**:
The per-page structural extraction (`## [Page N]`) containing reading-order prose, HTML `<table>` blocks, LaTeX equations, chart data tables, and verbatim anchor quotes.
_Avoid_: Raw OCR, page dump

**Executive Synthesis**:
The document-level, citation-anchored report (executive summary, key narratives, consolidated tables, visual analyses, and critical citations) synthesized from the Grounded Page Breakdowns.
_Avoid_: Abstract, blurb

**Visual Reading**:
The complete, page-cited semantic and structural analysis of a PDF—comprising both the Executive Synthesis and the Grounded Page Breakdown—derived directly from its rendered page images without intermediate text-layer extraction or separate OCR pre-parsing.
_Avoid_: OCR output, text dump, transcription
