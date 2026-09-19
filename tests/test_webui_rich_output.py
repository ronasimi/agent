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
