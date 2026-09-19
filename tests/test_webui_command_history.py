import json
import subprocess
from pathlib import Path


def test_chat_header_has_no_fake_agent_dropdown():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert 'class="panel-heading"' in html
    assert 'id="panelTitle">Al Agent<' in html
    assert "model-pill" not in html
    assert "chevron" not in html
    assert "chat:'Al Agent'" in js
    assert ".panel-heading" in css


def test_composer_history_controls_are_present():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'command_history.js' in html
    assert '↑/↓ history' in html
    assert "agent.webui.commandHistory.v1" in js
    assert "navigateCommandHistory(-1)" in js
    assert "navigateCommandHistory(1)" in js
    assert "e.key==='ArrowUp'" in js
    assert "e.key==='ArrowDown'" in js
    assert "e.key==='p'||e.key==='P'" in js
    assert "e.key==='n'||e.key==='N'" in js


def test_command_history_buffer_navigation_and_persistence():
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "command_history.js"
    script = f"""
const {{CommandHistoryBuffer}} = require({json.dumps(str(module))});
const values = new Map();
const storage = {{
  getItem: (k) => values.has(k) ? values.get(k) : null,
  setItem: (k, v) => values.set(k, v),
}};
const h = new CommandHistoryBuffer({{storage, key:'test', limit:3}});
h.push('one'); h.push('two'); h.push('three'); h.push('four');
const result = [];
result.push(h.move(-1, 'draft')); // four
result.push(h.move(-1, 'ignored')); // three
result.push(h.move(-1, 'ignored')); // two
result.push(h.move(-1, 'ignored')); // bounded at oldest => null
result.push(h.move(1, 'ignored')); // three
result.push(h.move(1, 'ignored')); // four
result.push(h.move(1, 'ignored')); // restored draft
const restored = new CommandHistoryBuffer({{storage, key:'test', limit:3}});
console.log(JSON.stringify({{result, items:restored.items}}));
"""
    proc = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    assert data["result"] == ["four", "three", "two", None, "three", "four", "draft"]
    assert data["items"] == ["two", "three", "four"]


def test_recipe_suggestion_uses_normal_composer_submit_path():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "sendMessage()" not in js
    assert "$('#composer').requestSubmit()" in js


def test_header_copy_control_and_inline_turn_status_contract():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert 'id="copyChat"' in html
    assert 'id="status"' not in html
    assert "/api/history/export" in js
    assert "copyEntireChat" in js
    assert "activeUserMessageEl" in js
    assert "turn-status" in js
    assert ".turn-status" in css
    assert ".message.user{display:flex;flex-direction:column;align-items:flex-end}" in css


def test_history_export_formats_complete_rows(monkeypatch):
    from webui import history

    monkeypatch.setattr(history, "_load_chat_history_from_db", lambda limit, include_compacted=False: [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "name": "current_time", "content": "{\"time\":\"12:00\"}"},
    ])
    exported = history._history_export()
    assert "User:\nhello" in exported
    assert "Assistant:\nhi" in exported
    assert "Tool [current_time]:" in exported
