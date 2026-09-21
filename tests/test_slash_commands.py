import asyncio
from pathlib import Path


def test_slash_catalog_is_unique_described_and_complete():
    from al_agent.slash_commands import list_slash_commands

    rows = list_slash_commands()
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))
    assert all(name.startswith("/") for name in names)
    assert all(row["description"].strip() for row in rows)
    assert all(row["usage"].startswith(row["name"]) for row in rows)
    assert {
        "/help", "/forget", "/think", "/profile", "/tools", "/reload",
        "/research", "/jobs", "/job", "/cancel-job", "/reminders",
        "/optimize", "/optimizations", "/approve-optimization",
    } == set(names)


def test_slash_parser_uses_exact_command_tokens():
    from al_agent.slash_commands import parse_slash_command

    spec, args = parse_slash_command("/research London transit")
    assert spec and spec.name == "/research"
    assert args == "London transit"
    assert parse_slash_command("/researchfoo")[0] is None
    assert parse_slash_command("plain chat")[0] is None


def test_unknown_slash_command_is_consumed_and_never_treated_as_chat():
    from al_agent.slash_commands import execute_slash_command

    result = execute_slash_command("/does-not-exist")
    assert result.recognized is False
    assert result.ok is False
    assert "Unknown slash command" in result.message

def test_think_command_is_deterministic_and_validated():
    from al_agent.slash_commands import execute_slash_command

    on = execute_slash_command("/think on", thinking_enabled=False)
    assert on.ok is True
    assert on.action == "set_thinking"
    assert on.data["thinking"] is True

    toggled = execute_slash_command("/think", thinking_enabled=True)
    assert toggled.data["thinking"] is False

    invalid = execute_slash_command("/think perhaps", thinking_enabled=False)
    assert invalid.ok is False
    assert "Usage:" in invalid.message


def test_forget_command_calls_conversation_scoped_clear(monkeypatch):
    import tools
    from al_agent.slash_commands import execute_slash_command

    seen = []
    monkeypatch.setattr(tools, "clear_chat_history", lambda cid=None: seen.append(cid) or "cleared")
    result = execute_slash_command("/forget", conversation_id="chat-123")
    assert result.ok is True
    assert result.action == "clear_conversation"
    assert result.message == "cleared"
    assert seen == ["chat-123"]


def test_websocket_slash_command_bypasses_model(monkeypatch):
    from al_agent.slash_commands import SlashCommandResult
    from webui import chat

    monkeypatch.setattr(chat, "ensure_conversation", lambda cid: cid)
    monkeypatch.setattr(
        chat,
        "execute_slash_command",
        lambda *a, **k: SlashCommandResult(True, True, "/help", "command output"),
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("slash command reached model turn engine")

    monkeypatch.setattr(chat.agent_runtime, "handle_user_turn", should_not_run)

    class FakeWebSocket:
        def __init__(self):
            self.events = []

        async def send_json(self, payload):
            self.events.append(payload)

    ws = FakeWebSocket()
    asyncio.run(chat._run_turn(ws, {
        "turn_id": "slash-test",
        "conversation_id": "test",
        "content": "/help",
        "attachments": [],
        "thinking": False,
    }))

    assert [event["type"] for event in ws.events] == [
        "accepted", "command_result", "history_refresh", "turn_end"
    ]
    assert ws.events[0]["command"] is True
    assert ws.events[1]["message"] == "command output"


def test_webui_exposes_slash_command_catalog_and_dropdown_contract():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    css = (root / "webui" / "static" / "style.css").read_text(encoding="utf-8")
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")
    chat = (root / "webui" / "chat.py").read_text(encoding="utf-8")

    assert 'id="slashMenu"' in html
    assert '/api/commands' in js
    assert 'loadSlashCommands' in js
    assert 'matchingSlashCommands' in js
    assert 'selectSlashCommand' in js
    assert 'c.description' in js
    assert "e.key==='ArrowDown'||e.key==='ArrowUp'" in js
    assert "e.key==='Tab'" in js
    assert '.slash-menu' in css
    assert '.slash-command-main span' in css
    assert '@app.get("/api/commands")' in server
    assert 'raw_content.startswith("/")' in chat
    assert 'execute_slash_command' in chat


def test_webui_command_catalog_endpoint_uses_shared_registry():
    # Importing server is intentionally part of this contract: the endpoint must
    # expose exactly the same metadata used by the executor/autocomplete UI.
    from al_agent.slash_commands import list_slash_commands
    from webui import server

    assert server.commands() == list_slash_commands()


def test_all_registered_slash_commands_have_working_executor_paths(monkeypatch):
    import tools
    from tools import job_tools, reminders, runtime, self_optimization
    from al_agent.slash_commands import execute_slash_command

    monkeypatch.setattr(tools, "get_tools_prompt_summary", lambda: "tool inventory")
    monkeypatch.setattr(tools, "load_tools", lambda: (216, {}))
    monkeypatch.setattr(tools, "clear_chat_history", lambda cid=None: "cleared")
    monkeypatch.setattr(job_tools, "enqueue_research", lambda topic: '{"job_id":"abcdef123456","topic":"%s"}' % topic)
    monkeypatch.setattr(job_tools, "get_research_status", lambda job_id: '{"id":"%s","status":"running"}' % job_id)
    monkeypatch.setattr(job_tools, "cancel_background_job", lambda job_id: "Job cancelled.")
    monkeypatch.setattr(runtime, "list_jobs", lambda limit=25: [{"id":"abcdef123456","status":"running","job_type":"research","title":"Research: test"}])
    monkeypatch.setattr(reminders, "list_reminders", lambda: '[{"id":"wake","status":"scheduled"}]')
    monkeypatch.setattr(self_optimization, "enqueue_self_optimization", lambda objective: '{"job_id":"job123456","candidate_id":"cand123456"}')
    monkeypatch.setattr(self_optimization, "list_self_optimization_candidates", lambda: '[{"id":"cand123456"}]')
    monkeypatch.setattr(self_optimization, "approve_self_optimization", lambda candidate_id, digest: '{"status":"approved"}')

    commands = {
        "/help": True,
        "/forget": True,
        "/think on": True,
        "/profile": True,
        "/tools": True,
        "/reload": True,
        "/research test topic": True,
        "/jobs": True,
        "/job abcdef12": True,
        "/cancel-job abcdef12": True,
        "/reminders": True,
        "/optimize reduce TTFT": True,
        "/optimizations": True,
        "/approve-optimization cand123 " + "a" * 64: True,
    }
    results = {text: execute_slash_command(text, conversation_id="c1") for text in commands}
    assert all(result.ok for result in results.values())
    assert results["/profile"].action == "open_profile"
    assert results["/think on"].data["thinking"] is True
    assert results["/jobs"].action == "show_jobs"
    assert results["/reminders"].action == "show_reminders"
    assert results["/forget"].action == "clear_conversation"
