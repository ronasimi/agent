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
    margin: 2cm;
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
    line-height: 1.6;
    color: #2d3748;
}
.report-content h1 { font-size: 20pt; color: #1a202c; border-bottom: 2px solid #3182ce; padding-bottom: 6px; margin-top: 0; }
.report-content h2 { font-size: 14pt; color: #2b6cb0; margin-top: 1.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
.report-content h3 { font-size: 11pt; color: #2c5282; margin-top: 1.2em; }
.report-content p { margin-bottom: 1em; text-align: justify; }
.report-content code { font-family: monospace; background-color: #edf2f7; padding: 2px 4px; border-radius: 3px; font-size: 9pt; }
.report-content pre { background-color: #1a202c; color: #f7fafc; padding: 1em; border-radius: 5px; overflow-x: auto; font-size: 8.5pt; }
.report-content blockquote { border-left: 4px solid #3182ce; padding-left: 1em; color: #4a5568; font-style: italic; margin: 1em 0; background-color: #ebf8ff; padding: 0.8em 1em; }
.report-content img { max-width: 100%; height: auto; display: block; margin: 1.5em auto; }
.report-content table { width: 100%; border-collapse: collapse; margin: 1.5em 0; }
.report-content th, .report-content td { border: 1px solid #cbd5e0; padding: 8px 12px; text-align: left; font-size: 9pt; }
.report-content th { background-color: #ebf8ff; color: #2b6cb0; font-weight: bold; }
.report-content tr:nth-child(even) { background-color: #f7fafc; }
"""


def _safe_output_path(output_filename: str) -> Path:
    raw = str(output_filename).strip()
    if not raw:
        raw = "research_report.pdf"
    if not raw.lower().endswith(".pdf"):
        raw += ".pdf"
    path = (WORKSPACE / raw.lstrip("/"))
    resolved_parent = path.parent.resolve()
    if os.path.commonpath([str(WORKSPACE), str(resolved_parent)]) != str(WORKSPACE):
        raise ValueError("Output path must remain inside /app/workspace.")
    resolved_parent.mkdir(parents=True, exist_ok=True)
    return path


def generate_pdf_report(markdown_content: str, output_filename: str = "research_report.pdf") -> str:
    """Generate a PDF from Markdown under /app/workspace; nested relative paths are allowed."""
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
        HTML(string=full_html, base_url=str(WORKSPACE)).write_pdf(
            str(output_path),
            stylesheets=[CSS(string=PDF_CSS)],
        )
        return f"PDF report successfully created and saved to {output_path}"
    except Exception as exc:
        return f"Failed to generate PDF report: {exc}"
