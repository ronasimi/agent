import os
from pathlib import Path

import yaml


def test_web_profile_is_optional_and_localhost_first():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    web = compose["services"]["webui"]
    assert "web" in web["profiles"]
    assert web["network_mode"] == "host"
    env = "\n".join(web.get("environment", []))
    assert "WEBUI_HOST=${WEBUI_HOST:-127.0.0.1}" in env
    command = " ".join(web["command"])
    assert "uvicorn webui.server:app" in command


def test_webui_static_assets_exist():
    for name in ("index.html", "style.css", "command_history.js", "app.js"):
        assert (Path("webui/static") / name).is_file()


def test_frontend_event_context_routes_events():
    import agent

    received = []
    with agent.frontend_event_context(received.append):
        agent.emit_event("unit_test", value=7)
    assert received and received[0]["type"] == "unit_test"
    assert received[0]["value"] == 7


def test_webui_dependencies_declared():
    reqs = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "uvicorn", "websockets", "python-multipart"):
        assert package in reqs


def test_webui_uses_host_xresources_and_workspace_drawer():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    env = "\n".join(compose["services"]["webui"].get("environment", []))
    assert "WEBUI_XRESOURCES=/host${HOME}/.Xresources" in env
    html = Path("webui/static/index.html").read_text(encoding="utf-8")
    js = Path("webui/static/app.js").read_text(encoding="utf-8")
    assert "Files in workspace" in html
    assert 'id="workspaceDrawer"' in html
    assert "/api/workspace" in js
    assert "/api/theme" in js


def test_xresources_parser_accepts_only_palette_keys(tmp_path):
    from webui import server

    path = tmp_path / ".Xresources"
    path.write_text(
        "*.foreground: #abcdef\n*.background: #010203\n*.color4: #112233\nXft.dpi: 144\n*.color5: not-a-color\n",
        encoding="utf-8",
    )
    theme = server._read_xresources_theme(path)
    assert theme["foreground"] == "#abcdef"
    assert theme["background"] == "#010203"
    assert theme["color4"] == "#112233"
    assert theme["color5"] == server.DEFAULT_THEME["color5"]
    assert "dpi" not in theme


def test_workspace_listing_is_bounded_to_workspace(tmp_path, monkeypatch):
    from webui import server

    (tmp_path / "folder").mkdir()
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    listing = server._workspace_listing("")
    assert listing["path"] == ""
    assert listing["parent"] is None
    names = [item["name"] for item in listing["items"]]
    assert names == ["folder", "notes.md"]
    note = next(item for item in listing["items"] if item["name"] == "notes.md")
    assert note["path"].endswith("/notes.md")
    assert note["text"] is True


def test_both_sidebars_are_independently_collapsible_and_persisted():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert 'id="collapseSidebar"' in html
    assert 'id="sidebarExpand"' in html
    assert 'id="workspaceToggle"' in html
    assert 'id="workspaceClose"' in html
    assert "agent.webui.leftSidebarCollapsed" in js
    assert "agent.webui.workspaceOpen" in js
    assert "setLeftSidebarCollapsed" in js
    assert "toggleWorkspace" in js
    assert ".sidebar-collapsed .sidebar" in css
    assert ".workspace-open .workspace-drawer" in css


def test_workspace_add_file_and_composer_drag_drop_controls_exist():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert 'id="workspaceAddFile"' in html
    assert 'id="workspaceFileInput"' in html
    assert 'id="composerAddFile"' in html
    assert 'id="fileInput"' in html
    assert 'id="composerDropHint"' in html
    assert "/api/workspace/upload" in js
    assert "uploadChatFiles" in js
    assert "uploadWorkspaceFiles" in js
    assert "dragenter" in js and "drop" in js
    assert ".composer.drag-active" in css


def test_workspace_upload_writes_to_current_folder_without_overwrite(tmp_path, monkeypatch):
    import asyncio
    import io
    from starlette.datastructures import UploadFile
    from webui import server

    workspace = tmp_path.resolve()
    target_dir = workspace / "reports"
    target_dir.mkdir()
    (target_dir / "note.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(server, "WORKSPACE", workspace)

    upload = UploadFile(filename="note.txt", file=io.BytesIO(b"new"))
    result = asyncio.run(server.workspace_upload(upload, path="reports"))

    assert result["path"].endswith("/reports/note_2.txt")
    assert (target_dir / "note.txt").read_text(encoding="utf-8") == "old"
    assert (target_dir / "note_2.txt").read_text(encoding="utf-8") == "new"


def test_generated_artifact_preview_ui_and_event_contract_exist():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")

    assert "artifact_created" in server
    assert "/api/preview/{path:path}" in server
    assert "/api/pdf-preview/{path:path}" in server
    assert "artifact_created" in js
    assert "addArtifact" in js
    assert "↓ Download" in js
    assert ".artifact-card" in css
    assert ".artifact-preview" in css


def test_artifact_snapshot_detects_only_new_workspace_files(tmp_path, monkeypatch):
    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    (workspace / "existing.txt").write_text("old", encoding="utf-8")
    (workspace / ".agent_inference.lock").write_text("", encoding="utf-8")
    before = server._workspace_file_snapshot()

    report = workspace / "reports" / "result.md"
    report.parent.mkdir()
    report.write_text("# Result\n\nhello", encoding="utf-8")
    after = server._workspace_file_snapshot()
    artifacts = server._new_artifacts(before, after)

    assert ".agent_inference.lock" not in before
    assert [a["relative"] for a in artifacts] == ["reports/result.md"]
    assert artifacts[0]["preview_kind"] == "markdown"


def test_text_preview_is_bounded_and_download_metadata_is_available(tmp_path, monkeypatch):
    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    monkeypatch.setattr(server, "PREVIEW_TEXT_BYTES", 16)
    path = workspace / "notes.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    preview = server.workspace_preview("notes.txt")
    artifact = server._artifact_payload(path)

    assert preview["content"] == "abcdefghijklmnop"
    assert preview["truncated"] is True
    assert artifact["path"] == "/app/workspace/notes.txt"
    assert artifact["preview_kind"] == "text"
    assert artifact["size"] == 26


def test_websocket_turn_emits_generated_artifact_event(tmp_path, monkeypatch):
    import asyncio
    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    monkeypatch.setattr(server, "_load_chat_history_from_db", lambda limit=100: [])

    def fake_turn(messages, text, thinking):
        server.agent_runtime.emit_event("tool_start", name="write_file")
        (workspace / "created.txt").write_text("generated by agent", encoding="utf-8")
        server.agent_runtime.emit_event("tool_result", name="write_file", status="ok", content="created")
        server.agent_runtime.emit_event("assistant_final", content="done")

    monkeypatch.setattr(server.agent_runtime, "handle_user_turn", fake_turn)

    class FakeWebSocket:
        def __init__(self):
            self.events = []

        async def send_json(self, payload):
            self.events.append(payload)

    ws = FakeWebSocket()
    asyncio.run(server._run_turn(ws, {"turn_id": "unit-artifact", "content": "create a file", "attachments": []}))

    artifacts = [e for e in ws.events if e.get("type") == "artifact_created"]
    assert len(artifacts) == 1
    assert artifacts[0]["artifact"]["path"] == "/app/workspace/created.txt"
    assert artifacts[0]["artifact"]["preview_kind"] == "text"
    assert any(e.get("type") == "history_refresh" for e in ws.events)


def test_recipe_suggestion_ui_contract_exists():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    assert "recipe_suggestion" in js
    assert "addRecipeSuggestion" in js
    assert "Save recipe" in js
    assert ".recipe-suggestion" in css


def test_chat_pane_has_independent_scroll_container_and_fixed_composer():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="scrollLatest"' in html
    assert ".main{min-width:0;min-height:0;height:100%;overflow:hidden" in css
    assert ".chat-panel{display:flex;flex-direction:column;position:relative;overflow:hidden}" in css
    assert ".messages{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden" in css
    assert ".composer-wrap{position:relative;flex:0 0 auto" in css
    assert "isNearBottom" in js
    assert "followOutput" in js
    assert "messagesEl.addEventListener('scroll',updateScrollFollow" in js
    assert "grid-template-rows:minmax(0,1fr)" in css
    assert ".messages{height:0;scrollbar-width:thin;touch-action:pan-y" in css


def test_webui_reports_automatic_recipe_preflight_status():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "e.type==='recipe_check'" in js
    assert "Recipes checked" in js


def test_webui_static_assets_are_no_store_to_avoid_stale_rebuilds():
    root = Path(__file__).resolve().parents[1]
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")
    assert 'request.url.path.startswith("/static/")' in server
    assert 'response.headers["Cache-Control"] = "no-store, max-age=0"' in server


def test_webui_brand_assets_and_user_profile_footer_exist():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    for name in ("agent-logo.png", "favicon.png", "favicon.ico"):
        assert (root / "webui" / "static" / "assets" / name).is_file()
    assert '/static/assets/agent-logo.png' in html
    assert '/static/assets/favicon.ico' in html
    assert 'id="userProfileImage"' in html
    assert "/api/profile-image" in js
    assert "refreshProfileImage" in js
    assert "set_profile_image" in js


def test_profile_image_endpoint_uses_durable_user_picture(tmp_path, monkeypatch):
    from webui import server

    profile = tmp_path / "user_picture.png"
    profile.write_bytes(b"not-used-by-FileResponse-constructor")
    monkeypatch.setattr(server, "get_profile_image_path", lambda migrate_legacy=True: profile)
    response = server.profile_image()
    assert Path(response.path) == profile
    assert response.media_type == "image/png"
