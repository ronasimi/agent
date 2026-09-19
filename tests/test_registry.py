
# This test is intended to run in the Docker image where the normal dependencies are installed.
def test_registry_contract():
    import tools
    count, errors = tools.load_tools()
    assert errors == {}
    assert count >= 30
    assert "schedule_reminder" in tools.AVAILABLE_TOOLS_MAP
    assert "enqueue_research" in tools.AVAILABLE_TOOLS_MAP
    assert "enqueue_self_optimization" in tools.AVAILABLE_TOOLS_MAP
    assert "approve_self_optimization" not in tools.AVAILABLE_TOOLS_MAP
    assert all("parameters" in schema["function"] for schema in tools.TOOL_SCHEMAS)


def test_tool_schema_selection():
    import tools
    schemas = tools.select_tool_schemas("check CPU and memory usage", max_tools=20)
    names = {schema["function"]["name"] for schema in schemas}
    assert len(schemas) <= 20
    assert "host_snapshot" in names
    assert "execute_shell" not in names
    assert "pressure_snapshot" in names or "pressure_info" in names


def test_small_core_and_file_bundle_are_stable():
    import tools
    generic = {schema["function"]["name"] for schema in tools.select_tool_schemas("hello", max_tools=12)}
    file_turn = {schema["function"]["name"] for schema in tools.select_tool_schemas("inspect this repo file", max_tools=12)}
    assert generic == tools._ALWAYS_TOOL_NAMES
    assert {"read_file", "path_stat", "find_paths", "read_text"} <= file_turn
    assert "write_file" not in file_turn
    assert "execute_python" not in file_turn
    assert len(file_turn) <= 12


def test_argument_normalization_performs_only_unambiguous_coercion():
    from tools.tool_registry import normalize_arguments

    def sample(count: int, enabled: bool, names: list[str] | None = None):
        return count, enabled, names

    args = normalize_arguments(sample, {"count": "3", "enabled": "false", "names": '["a","b"]'})
    assert args == {"count": 3, "enabled": False, "names": ["a", "b"]}


def test_optional_none_default_is_accepted_even_with_legacy_annotation():
    from tools.tool_registry import normalize_arguments

    def sample(hours: int = None):
        return hours

    assert normalize_arguments(sample, {"hours": None}) == {"hours": None}


def test_bare_list_annotation_generates_array_schema():
    from tools.tool_registry import function_schema

    def sample(tags: list = None):
        return tags

    schema = function_schema(sample)
    assert schema["function"]["parameters"]["properties"]["tags"]["type"] == "array"


def test_mixed_intent_keeps_lexically_relevant_tool_after_bundle_selection():
    import tools

    schemas = tools.select_tool_schemas("search the web and send a desktop notification", max_tools=12)
    names = {schema["function"]["name"] for schema in schemas}
    assert "web_search" in names
    assert "notify_desktop" in names


def test_generic_prompt_does_not_always_expose_privileged_shell():
    import tools

    names = {schema["function"]["name"] for schema in tools.select_tool_schemas("hello", max_tools=12)}
    assert "execute_shell" not in names


def test_semantically_required_empty_default_is_required_in_schema():
    import tools

    tools.load_tools()
    schema = next(item for item in tools.TOOL_SCHEMAS if item["function"]["name"] == "web_search")
    assert "query" in schema["function"]["parameters"]["required"]
    schema = next(item for item in tools.TOOL_SCHEMAS if item["function"]["name"] == "search_memory")
    assert "query" not in schema["function"]["parameters"]["required"]


def test_workspace_accepts_absolute_path_inside_workspace(tmp_path, monkeypatch):
    from tools import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_DIR", str(tmp_path.resolve()))
    target = tmp_path / "nested" / "file.txt"
    resolved = workspace._get_safe_path(str(target))
    assert resolved == str(target.resolve())


def test_short_followup_can_retain_tool_from_recent_context():
    import tools

    tools.load_tools()
    schemas = tools.select_tool_schemas(
        "yes, do that",
        max_tools=12,
        context_text="Please schedule a reminder for tomorrow. I can create it with schedule_reminder.",
    )
    names = {schema["function"]["name"] for schema in schemas}
    assert "schedule_reminder" in names


def test_custom_tool_decorator_defaults_to_bounded_timeout():
    from tools.tool_registry import agent_tool

    @agent_tool()
    def demo(value: str) -> str:
        return value

    assert demo._agent_tool_timeout == 60


def test_generic_turn_does_not_expose_mutating_research_job_tool():
    import tools

    tools.load_tools()
    schemas = tools.select_tool_schemas("hello there", max_tools=12)
    names = {schema["function"]["name"] for schema in schemas}
    assert "enqueue_research" not in names


def test_write_file_requires_explicit_content_to_avoid_accidental_truncation():
    import tools

    tools.load_tools()
    schema = tools.get_tool_schema("write_file")
    required = set(schema["function"]["parameters"]["required"])
    assert {"filename", "content"} <= required


def test_repeat_safe_artifact_tools_are_marked_separately_from_side_effecting_tools():
    import tools

    tools.load_tools()
    assert tools.TOOL_METADATA["take_web_screenshot"]["repeat_safe"] is True
    assert tools.TOOL_METADATA["generate_pdf_report"]["repeat_safe"] is True
    assert tools.TOOL_METADATA["write_file"]["repeat_safe"] is False


def test_custom_tool_decorator_defaults_to_mutating_for_safety():
    from tools.tool_registry import agent_tool

    @agent_tool()
    def demo_mutation(value: str) -> str:
        return value

    assert demo_mutation._agent_tool_readonly is False


def test_custom_tool_static_validator_rejects_top_level_side_effects_and_async():
    from tools.tool_manager import _validate_tool_code

    valid_code = '''\nfrom tools.tool_registry import agent_tool\n\n@agent_tool(readonly=True)\ndef inspect_value(value: str) -> str:\n    return value\n'''
    top_level_effect = '''\nfrom tools.tool_registry import agent_tool\nopen("/tmp/oops", "w").write("x")\n@agent_tool(readonly=True)\ndef inspect_value(value: str) -> str:\n    return value\n'''
    async_tool = '''\nfrom tools.tool_registry import agent_tool\n@agent_tool(readonly=True)\nasync def inspect_value(value: str) -> str:\n    return value\n'''

    assert _validate_tool_code(valid_code)[0] is True
    assert _validate_tool_code(top_level_effect)[0] is False
    assert _validate_tool_code(async_tool)[0] is False


def test_host_tool_default_urls_are_plain_urls_not_markdown_links():
    from pathlib import Path

    source = Path("tools/host_tools.py").read_text(encoding="utf-8")
    assert "[http://" not in source
    assert "[https://" not in source


def test_cancel_reminder_rejects_empty_id_before_slug_fallback():
    from tools.reminders import cancel_reminder

    assert cancel_reminder("").startswith("Error:")
