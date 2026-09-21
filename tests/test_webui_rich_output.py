import json
import subprocess
from pathlib import Path


def _render(markdown: str) -> str:
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "rich_output.js"
    script = f"""
const rich = require({json.dumps(str(module))});
console.log(rich.renderMarkdown({json.dumps(markdown)}));
"""
    return subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout.strip()


def test_rich_output_asset_is_loaded_before_app():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'src="/static/rich_output.js"' in html
    assert html.index('src="/static/rich_output.js"') < html.index('src="/static/app.js"')


def test_markdown_tables_render_as_responsive_semantic_tables():
    html = _render(
        "| Time | Temp | Conditions | Rain |\n"
        "|:---|---:|:---|---:|\n"
        "| 2 PM | 18°C | Partly cloudy | 20% |\n"
        "| 3 PM | 19°C | Rain showers | 40% |"
    )
    assert '<div class="table-wrap"' in html
    assert '<table class="md-table">' in html
    assert '<thead>' in html and '<tbody>' in html
    assert '🌡️' in html
    assert '🌤️' in html
    assert '☔' in html
    assert '⛅' in html
    assert '🌧️' in html
    assert '<p>| Time |' not in html


def test_weather_lines_receive_semantic_icons_without_changing_raw_words():
    html = _render(
        "## Weather\n\n"
        "Temperature: 18°C\n"
        "Conditions: Partly cloudy\n"
        "Wind: 15 km/h\n"
        "Humidity: 72%\n"
        "Sunset: 7:31 PM"
    )
    assert 'class="weather-heading"' in html
    assert '🌤️' in html
    assert '🌡️' in html
    assert '⛅' in html
    assert '💨' in html
    assert '💧' in html
    assert '🌇' in html
    for text in ("Temperature: 18°C", "Conditions: Partly cloudy", "Wind: 15 km/h", "Humidity: 72%"):
        assert text in html


def test_rich_output_escapes_html_in_tables_and_weather_lines():
    html = _render(
        "Weather: <img src=x onerror=alert(1)>\n\n"
        "| Item | Value |\n|---|---|\n| <script>alert(1)</script> | safe |"
    )
    assert '<script>' not in html
    assert '<img src=x' not in html
    assert '&lt;script&gt;' in html
    assert '&lt;img src=x onerror=alert(1)&gt;' in html


def test_tool_activity_is_grouped_and_collapsed_by_default():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert "turn-activity" in js
    assert "Activity" in js
    assert "activity-body" in js
    assert "wrap.open" not in js  # outer <details> stays collapsed unless the user opens it
    assert ".turn-activity" in css
    assert ".activity-count" in css


def test_profile_media_reference_uses_dedicated_safe_endpoint():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert "profile-image://current" in js
    assert "/api/profile-image" in js
    assert "function addProfileMedia" in js


def test_email_header_payload_renders_as_a_structured_card():
    html = _render(
        "Here is the draft:\n\n"
        "From: Ron <ron@example.com>\n"
        "To: Support <support@example.com>\n"
        "Subject: Router follow-up\n\n"
        "Hello,\n\n**The network is stable now.**"
    )
    assert 'class="email-card"' in html
    assert 'class="email-card-meta"' in html
    assert 'class="email-card-body"' in html
    assert "Router follow-up" in html
    assert "<strong>The network is stable now.</strong>" in html
    assert "Here is the draft:" in html


def test_structured_email_json_is_rendered_and_escaped():
    html = _render(
        '```json\n{"type":"email_draft","to":["a@example.com","b@example.com"],'
        '"from":"me@example.com","subject":"<Quarterly update>",'
        '"body":"Hello <script>alert(1)</script>"}\n```'
    )
    assert 'class="email-card"' in html
    assert "a@example.com, b@example.com" in html
    assert "&lt;Quarterly update&gt;" in html
    assert "<script>" not in html


def test_email_object_payload_is_supported_directly():
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "rich_output.js"
    script = f"""
const rich = require({json.dumps(str(module))});
console.log(rich.renderMarkdown({{type:'email_preview',from:'sender@example.com',to:'me@example.com',subject:'Status',body:'All set.'}}));
"""
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert 'class="email-card"' in result.stdout
    assert "sender@example.com" in result.stdout


def test_workspace_attachment_markers_are_extracted_for_inline_rendering():
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "rich_output.js"
    value = (
        "Please inspect this.\n\n"
        "Attached text file `notes.txt` (/app/workspace/uploads/notes.txt):\n\n"
        "```text\nsecret contents\n```\n\n"
        "Attached file: /app/workspace/uploads/chart.png"
    )
    script = f"""
const rich = require({json.dumps(str(module))});
console.log(JSON.stringify(rich.extractWorkspaceAttachments({json.dumps(value)})));
"""
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    assert data["text"] == "Please inspect this."
    assert [item["path"] for item in data["attachments"]] == [
        "/app/workspace/uploads/notes.txt",
        "/app/workspace/uploads/chart.png",
    ]


def test_safe_workspace_markdown_image_renders_inline():
    html = _render("![Latency chart](/api/files/reports/latency.png)")
    assert 'class="inline-markdown-media"' in html
    assert 'src="/api/files/reports/latency.png"' in html
    assert 'alt="Latency chart"' in html
