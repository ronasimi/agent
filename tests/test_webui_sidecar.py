from pathlib import Path

import yaml


def test_webui_is_the_only_default_interface_and_localhost_first():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert "agent" not in compose["services"]
    web = compose["services"]["webui"]
    assert "profiles" not in web
    assert web["network_mode"] == "host"
    env = "\n".join(web.get("environment", []))
    assert "WEBUI_HOST=${WEBUI_HOST:-127.0.0.1}" in env
    assert web["command"] == ["python", "-m", "webui"]
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "-m", "webui"]' in dockerfile


def test_webui_module_entrypoint_honors_environment(monkeypatch):
    from webui import __main__ as web_main

    called = {}
    monkeypatch.setenv("WEBUI_HOST", "0.0.0.0")
    monkeypatch.setenv("WEBUI_PORT", "9090")
    monkeypatch.setattr(web_main.uvicorn, "run", lambda app, **kwargs: called.update(app=app, **kwargs))

    web_main.main()

    assert called == {
        "app": "webui.server:app",
        "host": "0.0.0.0",
        "port": 9090,
        "proxy_headers": True,
    }


def test_webui_static_assets_exist():
    for name in ("index.html", "style.css", "mdi_support.js", "command_history.js", "rich_output.js", "interaction_state.js", "app.js"):
        assert (Path("webui/static") / name).is_file()


def test_webui_uses_pictogrammers_mdi_icons_with_resilient_fallback():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    support = (root / "webui" / "static" / "mdi_support.js").read_text(encoding="utf-8")
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")

    assert "@mdi/font@7.4.47/css/materialdesignicons.min.css" in html
    for icon in (
        "mdi-square-edit-outline",
        "mdi-magnify",
        "mdi-account-circle-outline",
        "mdi-wrench-outline",
        "mdi-bell-outline",
        "mdi-folder-outline",
        "mdi-content-copy",
    ):
        assert icon in html
    assert "function mdiIcon" in js
    assert "file-image-outline" in js
    assert "data-fallback" in html
    assert "document.fonts.load" in support
    assert "html.mdi-ready .mdi-ui::before" in css
    assert "style-src 'self' https://cdn.jsdelivr.net" in server
    assert "font-src 'self' https://cdn.jsdelivr.net data:" in server


def test_frontend_event_context_routes_events():
    from al_agent import runtime as agent

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
    assert "mdiIcon('download','↓')" in js
    assert "mdiIcon('open-in-new','↗')" in js
    assert ".artifact-card" in css
    assert ".artifact-preview" in css


def test_user_and_agent_media_share_inline_responsive_renderer():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert 'class="message-media"' in js
    assert "renderMessageMedia" in js
    assert "addMessage('user',content,{forceScroll:true,media:items})" in js
    assert "kind==='video'" in js and "kind==='audio'" in js
    assert "kind==='document'" in js
    assert ".message-media" in css
    assert ".artifact-preview video" in css
    assert ".artifact-preview audio" in css


def test_inline_attachment_parser_filters_lock_files():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "rich_output.js").read_text(encoding="utf-8")
    assert "base.endsWith('.lock')" in js


def test_artifact_snapshot_detects_only_new_workspace_files(tmp_path, monkeypatch):
    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    (workspace / "existing.txt").write_text("old", encoding="utf-8")
    for lock_name in (
        ".agent_inference.lock",
        ".agent_model_maintenance.lock",
        ".agent_turn_deadbeef.lock",
        "custom.lock",
    ):
        (workspace / lock_name).write_text("", encoding="utf-8")
    before = server._workspace_file_snapshot()

    report = workspace / "reports" / "result.md"
    report.parent.mkdir()
    report.write_text("# Result\n\nhello", encoding="utf-8")
    after = server._workspace_file_snapshot()
    artifacts = server._new_artifacts(before, after)

    assert not any(name.endswith(".lock") for name in before)
    assert [a["relative"] for a in artifacts] == ["reports/result.md"]
    assert artifacts[0]["preview_kind"] == "markdown"


def test_generated_artifact_snapshot_hides_generalized_recipe_targets_fixture(tmp_path, monkeypatch):
    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    before = server._workspace_file_snapshot()

    fixture = workspace / "generalized_recipe_test" / "targets.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("example.com\nwww.iana.org\n", encoding="utf-8")
    visible = workspace / "reports" / "result.txt"
    visible.parent.mkdir(parents=True)
    visible.write_text("visible", encoding="utf-8")

    after = server._workspace_file_snapshot()
    artifacts = server._new_artifacts(before, after)

    assert "generalized_recipe_test/targets.txt" not in after
    assert [a["relative"] for a in artifacts] == ["reports/result.txt"]


def test_webui_defensively_hides_targets_fixture_from_inline_artifacts():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "INLINE_ARTIFACT_HIDDEN_RELATIVE" in js
    assert "generalized_recipe_test/targets.txt" in js
    assert "inlineArtifactHidden(item)" in js


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


def test_modern_document_preview_extracts_bounded_inline_text(tmp_path, monkeypatch):
    import zipfile

    from webui import server

    workspace = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE", workspace)
    path = workspace / "draft.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="urn:test"><w:body>'
            '<w:p><w:r><w:t>Inline document preview</w:t></w:r></w:p>'
            "</w:body></w:document>",
        )

    preview = server.workspace_preview("draft.docx")
    assert preview["preview_kind"] == "document"
    assert "Inline document preview" in preview["content"]
    assert preview["truncated"] is False


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


def test_recipe_interaction_can_defer_until_active_turn_finishes(monkeypatch):
    import asyncio

    from fastapi import WebSocketDisconnect
    from webui import chat

    order = []

    async def fake_run(_websocket, payload):
        order.append(f"start:{payload['content']}")
        await asyncio.sleep(0.01)
        order.append(f"end:{payload['content']}")

    monkeypatch.setattr(chat, "_run_turn", fake_run)

    class FakeWebSocket:
        def __init__(self):
            self.calls = 0

        async def accept(self):
            return None

        async def receive_json(self):
            self.calls += 1
            if self.calls == 1:
                return {"type": "message", "content": "first"}
            if self.calls == 2:
                return {"type": "message", "content": "save recipe", "defer_until_idle": True}
            await asyncio.sleep(0.02)
            raise WebSocketDisconnect()

        async def send_json(self, _payload):
            return None

    asyncio.run(chat.chat_socket(FakeWebSocket()))
    assert order == ["start:first", "end:first", "start:save recipe", "end:save recipe"]


def test_recipe_suggestion_ui_contract_exists():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    assert "recipe_suggestion" in js
    assert "addRecipeSuggestion" in js
    assert "recipe-vote-up" in js and "recipe-vote-down" in js
    assert "RecipeDecisionStore" in js
    assert "restoreRecipeSuggestions" in js
    assert ".recipe-suggestion" in css
    assert ".recipe-suggestion.decided-up" in css
    assert ".recipe-suggestion.decided-down" in css


def test_chat_pane_has_independent_scroll_container_and_fixed_composer():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="scrollLatest"' in html
    assert ".main{min-width:0;min-height:0;height:100%;overflow:hidden" in css
    assert ".chat-panel{display:flex;flex-direction:column;position:relative;overflow:hidden}" in css
    assert ".messages{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden" in css
    assert ".composer-wrap{position:absolute;left:0;right:0;bottom:0" in css
    assert ".messages{padding-top:38px;padding-bottom:178px}" in css
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


def test_webui_saved_conversations_restore_active_and_offer_row_actions():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    assert "recent-delete" in js
    assert "conversation-menu" in js
    assert "renameSavedConversation" in js
    assert "method:'DELETE'" in js
    assert "deleteSavedConversation" in js
    assert ".recent-delete" in css
    assert "async function bootstrapWebUi()" in js
    assert "readActiveConversation()" in js
    assert "agent.webui.activeConversation.v1" in js
    assert "active_conversation_id" in js
    assert "if(rows.length)setActiveConversation(rows[0].id)" in js


def test_empty_placeholder_conversations_are_not_listed(tmp_path, monkeypatch):
    from tools import memory

    db = str(tmp_path / "conversation-list.db")
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory.init_db()
    assert all(row["id"] != "default" for row in memory.list_conversations())
    fresh = memory.create_conversation()["id"]
    assert all(row["id"] != fresh for row in memory.list_conversations())
    active_rows = memory.list_conversations(active_conversation_id=fresh)
    assert active_rows[0]["id"] == fresh
    assert active_rows[0]["is_active"] is True

    memory._save_message_to_db({"role": "user", "content": "legacy thread"}, conversation_id="default")
    assert any(row["id"] == "default" for row in memory.list_conversations())

    assert memory.delete_conversation("default") is True
    assert all(row["id"] != "default" for row in memory.list_conversations())


def test_conversations_are_message_recent_with_active_conversation_pinned(tmp_path, monkeypatch):
    from tools import memory

    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "conversation-order.db"))
    memory.init_db()
    older = memory.create_conversation("Older thread")["id"]
    newer = memory.create_conversation("Newer thread")["id"]
    memory._save_message_to_db({"role": "user", "content": "first"}, conversation_id=older)
    memory._save_message_to_db({"role": "user", "content": "second"}, conversation_id=newer)

    assert [row["id"] for row in memory.list_conversations()][:2] == [newer, older]
    pinned = memory.list_conversations(active_conversation_id=older)
    assert [row["id"] for row in pinned][:2] == [older, newer]
    assert pinned[0]["is_active"] is True
    assert pinned[1]["is_active"] is False


def test_browser_history_reopens_compacted_conversation(tmp_path, monkeypatch):
    from tools import memory
    from webui import history as web_history

    db = str(tmp_path / "conversation-reopen.db")
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory.init_db()
    cid = memory.create_conversation("Saved thread")["id"]
    first = memory._save_message_to_db({"role": "user", "content": "old question"}, conversation_id=cid)
    second = memory._save_message_to_db({"role": "assistant", "content": "old answer"}, conversation_id=cid)
    memory._save_message_to_db({"role": "user", "content": "new question"}, conversation_id=cid)
    assert memory.apply_conversation_compaction("summary", second, conversation_id=cid)

    # Model context still excludes compacted rows.
    model_history = memory._load_chat_history_from_db(limit=20, conversation_id=cid)
    assert [row["content"] for row in model_history] == ["new question"]

    # The browser transcript must remain durable and show the whole saved chat.
    browser_history = web_history._history(limit=200, conversation_id=cid)
    assert [row["content"] for row in browser_history] == ["old question", "old answer", "new question"]


def test_sidebar_reopen_targets_clicked_conversation_and_highlights_only_it():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert "await loadHistory(target);" in js
    assert "loadState(target)" not in js
    assert "cid!==activeConversationId" in js
    assert "b.dataset.conversationId===activeConversationId" in js
    assert "document.querySelectorAll('.nav-item,.recent-item')" not in js


def test_health_endpoint_reports_current_model_roles_without_removed_micro_role():
    from webui import server

    payload = server.health()
    assert payload["ok"] is True
    assert payload["main_model"] == server.agent_runtime.MODEL
    assert payload["context"] == server.agent_runtime.MAX_CTX
    assert payload["model"] == server.agent_runtime.MODEL
    assert "report_model" not in payload
    assert "micro_model" not in payload


def test_webui_health_failure_isolated_from_chat_bootstrap():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert "async function loadHealth()" in js
    assert "Health load failed" in js
    assert "Promise.allSettled([loadTheme(),loadHealth(),loadSlashCommands(),loadJobs(),loadReminders(),loadWorkspace('')])" in js
    assert "Promise.allSettled([loadHistory()])" in js


def test_advanced_views_are_hidden_behind_wrench_menu():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="utilityMenuToggle"' in html
    assert 'mdi-wrench-outline' in html
    utility_start = html.index('id="utilityMenu"')
    utility_end = html.index('</div>', utility_start)
    utility = html[utility_start:utility_end]
    for label in ("Profile Setup", "Connections", "Generate Bug Report", "UI Benchmarks"):
        assert label in utility
    assert "Working State" not in utility
    sidebar_nav = html[html.index('<nav class="nav"'):html.index('</nav>')]
    assert "Working State" not in sidebar_nav
    assert "UI Benchmarks" not in sidebar_nav
    workspace_button = html[html.index('id="workspaceToggle"'):html.index('</button>', html.index('id="workspaceToggle"'))]
    assert "Files in workspace" not in workspace_button.replace('title="Files in workspace"', "").replace('aria-label="Files in workspace"', "")
    assert "mdi-folder-outline" in workspace_button
    assert "Jobs" in sidebar_nav and "Reminders" in sidebar_nav
    assert "toggleUtilityMenu" in js and "closeUtilityMenu" in js


def test_thinking_stream_has_dedicated_composer_host():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")

    host = html.index('id="thinkingStreamHost"')
    composer = html.index('id="composer"')
    assert host < composer
    assert "composer-thinking" in js
    assert ".composer-thinking" in css
    assert ".assistant-thinking" not in css


def test_turn_ack_and_workspace_scans_do_not_block_event_loop(monkeypatch):
    import asyncio
    import threading
    from webui import chat

    calls = []
    event_loop_thread = threading.get_ident()

    class Socket:
        async def send_json(self, packet):
            calls.append((packet['type'], threading.get_ident()))

    def scan():
        calls.append(('scan', threading.get_ident()))
        return {}

    def work(messages, text, thinking, *, refresh_history=False):
        assert refresh_history is True
        calls.append(('work', threading.get_ident()))

    monkeypatch.setattr(chat, 'ensure_conversation', lambda cid: None)
    monkeypatch.setattr(chat, '_workspace_file_snapshot', scan)
    monkeypatch.setattr(chat, '_new_artifacts', lambda *args, **kwargs: [])
    monkeypatch.setattr(chat.agent_runtime, 'handle_user_turn', work)
    asyncio.run(chat._run_turn(Socket(), {'content': 'hello'}))
    assert [name for name, _ in calls] == ['accepted', 'scan', 'work', 'scan', 'history_refresh']
    assert all(tid != event_loop_thread for name, tid in calls if name in {'scan', 'work'})


def test_hidden_diagnostic_panels_are_not_refreshed_after_every_turn():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    start = js.index("else if(e.type==='history_refresh')")
    end = js.index("else if(e.type==='error')", start)
    refresh = js[start:end]
    assert "loadState" not in refresh
    assert "if(!$('#jobsPanel').classList.contains('hidden'))loadJobs();" in refresh


def test_webui_forwards_thinking_only_when_turn_opted_in():
    from webui.chat import _should_forward_event

    event = {"type": "thinking_delta", "content": "reasoning"}
    assert _should_forward_event(event, thinking_enabled=False) is False
    assert _should_forward_event(event, thinking_enabled=True) is True
    assert _should_forward_event({"type": "assistant_delta", "content": "visible"}) is True
    assert _should_forward_event({"type": "tool_result"}) is True


def test_webui_think_toggle_streams_reasoning_and_defaults_off():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="thinking" type="checkbox"' in html
    assert 'id="thinking" type="checkbox" checked' not in html
    assert "const thinkingEnabled=Boolean($('#thinking').checked);activeThinkingEnabled=thinkingEnabled;" in js
    assert "thinking:thinkingEnabled" in js
    assert "e.type==='thinking_delta'" in js
    assert "appendThinking(e.content||'')" in js
    assert "requestAnimationFrame(paintThinkingStream)" in js




def test_reasoning_status_updates_do_not_force_scroll_layout():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    start = js.index("function setStatus(")
    end = js.index("function startTurnTimer", start)
    status_impl = js[start:end]
    assert "scrollBottom()" not in status_impl


def test_per_token_thinking_stdout_trace_is_opt_in():
    from al_agent import state

    assert state.LOG_THINKING_TRACE is False


def test_active_chat_can_be_reopened_while_turn_is_streaming():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    start = js.index("async function activateConversation(conversationId){")
    end = js.index("async function renameSavedConversation", start)
    impl = js[start:end]
    current_chat = impl.index("if(target===activeConversationId){")
    busy_guard = impl.index("if(activeTurn)return false;")
    assert current_chat < busy_guard
    assert "showPanel('chat');" in impl



def test_jobs_panel_surfaces_compute_progress_and_cancel_controls():
    root = Path(__file__).resolve().parents[1]
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")

    assert "jobProgress" in js
    assert "yield_count" in js and "tape_cells" in js and "recovery_failures" in js
    assert "/api/jobs/${encodeURIComponent(jobId)}/cancel" in js
    assert "stop-circle-outline" in js
    assert "if(panel&&!panel.classList.contains('hidden'))loadJobs()" in js
    assert ".job-progress" in css
    assert '@app.post("/api/jobs/{job_id}/cancel")' in server


def test_list_jobs_exposes_bounded_compute_progress_not_full_tape(tmp_path, monkeypatch):
    from tools import runtime

    db = str(tmp_path / "agent.db")
    monkeypatch.setenv("AGENT_DB_PATH", db)
    runtime.DB_PATH = db
    runtime.init_runtime_db()
    job_id = runtime.create_job("durable_compute", "demo", {"program": {}})
    runtime.claim_next_job("worker", ["durable_compute"])
    state = {
        "status": "yielded",
        "machine_state": "q1",
        "steps": 12,
        "yield_count": 3,
        "checkpoint_generation": 3,
        "head": 4,
        "tape_cells": 99,
        "tape": {str(i): "1" for i in range(99)},
    }
    assert runtime.checkpoint_and_defer_job(job_id, state, step=3, delay_seconds=0)
    row = runtime.list_jobs(limit=5)[0]
    assert row["progress"]["steps"] == 12
    assert row["progress"]["yield_count"] == 3
    assert row["progress"]["tape_cells"] == 99
    assert row["progress"]["recovery_failures"] == 0
    assert "state" not in row and "tape" not in row
