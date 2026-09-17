# ==========================================
# FILE: tools/pdf_generator.py
# ==========================================
import os
import markdown
from weasyprint import HTML, CSS

# Clean, professional stylesheet for report generation
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
h1 {
    font-size: 20pt;
    color: #1a202c;
    border-bottom: 2px solid #3182ce;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 1em;
}
h2 {
    font-size: 14pt;
    color: #2b6cb0;
    margin-top: 1.5em;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}
h3 {
    font-size: 11pt;
    color: #2c5282;
    margin-top: 1.2em;
}
p {
    margin-bottom: 1em;
    text-align: justify;
}
code {
    font-family: monospace;
    background-color: #edf2f7;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 9pt;
}
pre {
    background-color: #1a202c;
    color: #f7fafc;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 8.5pt;
}
blockquote {
    border-left: 4px solid #3182ce;
    padding-left: 1em;
    color: #4a5568;
    font-style: italic;
    margin: 1em 0;
    background-color: #ebf8ff;
    padding: 0.8em 1em;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5em auto;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
}
th, td {
    border: 1px solid #cbd5e0;
    padding: 8px 12px;
    text-align: left;
    font-size: 9pt;
}
th {
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f7fafc;
}
a {
    color: #3182ce;
    text-decoration: none;
}
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2em 0;
}
ul, ol {
    margin-bottom: 1em;
    padding-left: 1.5em;
}
li {
    margin-bottom: 0.3em;
}
"""

def generate_pdf_report(markdown_content: str, output_filename: str = "research_report.pdf") -> str:
    """Converts Markdown text (including inline images, tables, code blocks, and references) into a styled PDF document.
    
    Args:
        markdown_content: Markdown-formatted report text.
        output_filename: Output filename (saved in /app/workspace/).
    """
    try:
        filename = os.path.basename(output_filename)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        output_path = os.path.join("/app/workspace", filename)

        # Convert Markdown to HTML with common extensions enabled
        html_body = markdown.markdown(
            markdown_content,
            extensions=['extra', 'tables', 'fenced_code', 'toc', 'nl2br', 'sane_lists']
        )

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Research Report</title>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """

        # Compile PDF using WeasyPrint
        HTML(string=full_html, base_url="/app/workspace").write_pdf(
            output_path, 
            stylesheets=[CSS(string=PDF_CSS)]
        )
        return f"PDF report successfully created and saved to {output_path}"
    except Exception as e:
        return f"Failed to generate PDF report: {str(e)}"
