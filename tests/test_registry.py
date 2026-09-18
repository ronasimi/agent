
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
    assert "execute_shell" in names


def test_small_core_and_file_bundle_are_stable():
    import tools
    generic = {schema["function"]["name"] for schema in tools.select_tool_schemas("hello", max_tools=12)}
    file_turn = {schema["function"]["name"] for schema in tools.select_tool_schemas("inspect this repo file", max_tools=12)}
    assert generic == tools._ALWAYS_TOOL_NAMES
    assert {"read_file", "write_file", "read_observation", "execute_python"} <= file_turn
    assert len(file_turn) <= 12
