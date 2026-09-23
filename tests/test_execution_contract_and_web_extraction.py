from al_agent.prompts import SYSTEM_POLICY
from al_agent.turn_support import _bounded_tool_result_with_ref
from tools import web


def test_system_policy_requires_truncated_observation_read_and_execution_evidence():
    assert "If you receive a 'middle truncated' warning from the harness, you MUST execute `read_observation`" in SYSTEM_POLICY
    assert "Never confirm a task is complete unless you have successfully executed the corresponding tool and received an observation." in SYSTEM_POLICY


def test_extract_main_text_prefers_article_and_drops_site_chrome():
    html = """
    <html><body>
      <header><a>Home</a><a>Weather</a><a>Sports</a></header>
      <div class="advertisement">BUY THIS PRODUCT NOW</div>
      <main>
        <article>
          <h1>City council approves transit plan</h1>
          <p>London city council approved a new transit plan after a lengthy public meeting on Tuesday evening.</p>
          <p>The plan adds service on several major routes and is scheduled to begin next spring.</p>
          <p>Officials said implementation details will be published after the final budget review.</p>
        </article>
      </main>
      <section class="related-stories"><a>Celebrity story</a><a>Shopping guide</a></section>
      <footer>Privacy Cookies Careers Contact Us</footer>
    </body></html>
    """
    text = web.extract_main_text(html)
    assert "City council approves transit plan" in text
    assert "adds service on several major routes" in text
    assert "BUY THIS PRODUCT NOW" not in text
    assert "Celebrity story" not in text
    assert "Privacy Cookies" not in text
    assert "Home Weather Sports" not in text


def test_browse_url_returns_extracted_article_not_navigation(monkeypatch):
    html = """
    <html><body>
      <nav>HOME NEWS WEATHER SPORTS SHOP</nav>
      <article>
        <h1>Test headline</h1>
        <p>This is the first substantive paragraph of the test article and contains enough prose for extraction.</p>
        <p>This is the second substantive paragraph with additional details that belong to the article body.</p>
      </article>
      <aside>Sponsored links and recommendations</aside>
    </body></html>
    """
    monkeypatch.setattr(web, "fetch_text", lambda *a, **k: ("https://example.test/story", "text/html", html))
    result = web.browse_url("https://example.test/story")
    assert "Test headline" in result
    assert "first substantive paragraph" in result
    assert "HOME NEWS WEATHER SPORTS SHOP" not in result
    assert "Sponsored links" not in result


def test_middle_truncation_marker_requires_read_observation_with_missing_offset(monkeypatch):
    from al_agent import turn_support

    monkeypatch.setattr(turn_support, "MAX_TOOL_OUTPUT", 1200)
    monkeypatch.setattr(turn_support, "store_tool_observation", lambda _tool, _text: "abc123")
    original = "A" * 10000
    bounded, observation_id = _bounded_tool_result_with_ref("browse_url", original)
    assert observation_id == "abc123"
    assert "middle truncated" in bounded
    assert "You MUST use read_observation" in bounded
    assert "observation_id='abc123'" in bounded
    assert "offset=" in bounded
    assert "before summarizing" in bounded
    assert len(bounded) <= 1300  # allow a small marker-overhead margin for tiny test cap


def test_truncated_tool_result_forces_read_observation_before_summary(monkeypatch):
    import json
    from al_agent import turn_engine as te

    calls = []

    def fake_execute(name, args):
        calls.append(("tool", name, dict(args)))
        if name == "browse_url":
            return "ARTICLE " + ("middle-data " * 3000)
        if name == "read_observation":
            offset = int(args.get("offset", 0))
            returned = min(int(args.get("length", 10000)), max(0, 36000 - offset))
            return json.dumps({
                "observation_id": args["observation_id"],
                "offset": offset,
                "returned_chars": returned,
                "total_chars": 36000,
                "has_more": offset + returned < 36000,
                "content": "missing middle article data",
            })
        raise AssertionError((name, args))

    def fake_bound(name, text):
        if name == "browse_url":
            return (
                "HEAD\n\n[Harness: middle truncated; full 36000-character result stored as observation obs123. "
                "You MUST use read_observation(observation_id='obs123', offset=6000, length=3500) "
                "to retrieve missing middle data before summarizing.]\n\nTAIL",
                "obs123",
            )
        return text, ""

    class Model:
        def __init__(self):
            self.calls = 0
            self.requests = []

        def chat(self, **kwargs):
            self.calls += 1
            self.requests.append(kwargs)
            tools = {x.get("function", {}).get("name") for x in kwargs.get("tools", [])}
            if self.calls == 1:
                assert "browse_url" in tools
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "browse_url", "arguments": {"url": "https://example.test/story"}}}
                ]}}])
            # The harness must recover every omitted middle chunk itself before
            # asking the model to summarize. No extra model round-trip is needed
            # just to emit read_observation calls.
            return iter([{"done": True, "message": {
                "content": "The article summary uses the recovered middle data.", "tool_calls": []
            }}])

    model = Model()
    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)
    monkeypatch.setattr(te, "_bounded_tool_result_with_ref", fake_bound)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        "Use browse_url on https://example.test/story and summarize the article.",
        False,
        runtime_overrides={
            "OLLAMA": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    reads = [item for item in calls if item[0] == "tool" and item[1] == "read_observation"]
    assert [row[2]["offset"] for row in reads] == [6000, 9500, 13000, 16500, 20000, 23500, 27000, 30500, 34000]
    assert model.calls == 2
    assert messages[-1]["content"] == "The article summary uses the recovered middle data."


def test_generic_execution_timeout_schema_matches_runtime_clamp():
    from tools.system import execute_python, execute_shell
    from tools.tool_registry import function_schema

    for func in (execute_shell, execute_python):
        timeout = function_schema(func)["function"]["parameters"]["properties"]["timeout"]
        assert timeout["minimum"] == 1
        assert timeout["maximum"] == 120
