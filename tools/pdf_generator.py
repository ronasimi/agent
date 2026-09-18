# ==========================================
# FILE: tools/pdf_generator.py
# ==========================================
"""Generate PDF reports beneath the workspace without allowing path escape."""
from __future__ import annotations

import os
from pathlib import Path

from weasyprint import CSS, HTML

WORKSPACE = Path("/app/workspace").resolve()

PDF_CSS = """
@page {
    size: A4;
    margin: 1.8cm 1.9cm 2cm 1.9cm;
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        font-family: 'DejaVu Sans', sans-serif;
        color: #718096;
    }
}
body {
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #2d3748;
}
.report-content h1 {
    font-size: 20pt;
    color: #1a202c;
    border-bottom: 2px solid #3182ce;
    padding-bottom: 6px;
    margin-top: 0;
    break-after: avoid;
}
.report-content h2 {
    font-size: 14pt;
    color: #2b6cb0;
    margin-top: 1.45em;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    break-after: avoid;
}
.report-content h3 {
    font-size: 11pt;
    color: #2c5282;
    margin-top: 1.2em;
    break-after: avoid;
}
.report-content p { margin-bottom: 0.9em; text-align: justify; }
.report-content ul, .report-content ol { margin-bottom: 1em; }
.report-content li { margin-bottom: 0.25em; }
.report-content code { font-family: monospace; background-color: #edf2f7; padding: 2px 4px; border-radius: 3px; font-size: 9pt; }
.report-content pre {
    background-color: #1a202c;
    color: #f7fafc;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 8.5pt;
    break-inside: avoid;
}
.report-content blockquote {
    border-left: 4px solid #3182ce;
    padding: 0.8em 1em;
    color: #4a5568;
    margin: 1em 0;
    background-color: #ebf8ff;
    break-inside: avoid;
}
.report-content img {
    max-width: 100%;
    max-height: 10.5cm;
    width: auto;
    height: auto;
    display: block;
    margin: 1.2em auto 0.35em auto;
    object-fit: contain;
    break-inside: avoid;
}
.report-content .figure-caption {
    margin: 0 auto 1.2em auto;
    max-width: 92%;
    font-size: 8.5pt;
    line-height: 1.35;
    color: #718096;
    text-align: center;
    break-before: avoid;
    break-inside: avoid;
}
.report-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0;
    break-inside: avoid;
}
.report-content th, .report-content td { border: 1px solid #cbd5e0; padding: 7px 10px; text-align: left; font-size: 9pt; }
.report-content th { background-color: #ebf8ff; color: #2b6cb0; font-weight: bold; }
.report-content tr:nth-child(even) { background-color: #f7fafc; }
.report-content a { color: #2b6cb0; text-decoration: none; }
"""


def _safe_output_path(output_filename: str) -> Path:
    raw = str(output_filename).strip() or "research_report.pdf"
    if not raw.lower().endswith(".pdf"):
        raw += ".pdf"

    requested = Path(raw)
    candidate = requested if requested.is_absolute() else WORKSPACE / requested
    resolved = candidate.resolve(strict=False)
    try:
        common = os.path.commonpath([str(WORKSPACE), str(resolved)])
    except ValueError as exc:
        raise ValueError("Output path must remain inside /app/workspace.") from exc
    if common != str(WORKSPACE):
        raise ValueError("Output path must remain inside /app/workspace.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def generate_pdf_report(markdown_content: str, output_filename: str = "research_report.pdf") -> str:
    """Generate a PDF from Markdown under /app/workspace; local relative media is supported."""
    try:
        output_path = _safe_output_path(output_filename)
        import markdown

        html_body = markdown.markdown(
            str(markdown_content),
            extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
        )
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Research Report</title></head>
        <body><div class="report-content">{html_body}</div></body>
        </html>
        """
        # Resolve Markdown image paths relative to the generated report rather
        # than the workspace root. This keeps report.md + *_assets portable.
        HTML(string=full_html, base_url=str(output_path.parent)).write_pdf(
            str(output_path),
            stylesheets=[CSS(string=PDF_CSS)],
        )
        return f"PDF report successfully created and saved to {output_path}"
    except Exception as exc:
        return f"Error: failed to generate PDF report: {exc}"
