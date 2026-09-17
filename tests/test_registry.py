import sys
import types
from pathlib import Path

# This test is intended to run in the Docker image where the normal dependencies are installed.
def test_registry_contract():
    import tools
    count, errors = tools.load_tools()
    assert errors == {}
    assert count >= 30
    assert "schedule_reminder" in tools.AVAILABLE_TOOLS_MAP
    assert "enqueue_research" in tools.AVAILABLE_TOOLS_MAP
    assert all("parameters" in schema["function"] for schema in tools.TOOL_SCHEMAS)


def test_tool_schema_selection():
    import tools
    schemas = tools.select_tool_schemas("check CPU and memory usage", max_tools=20)
    names = {schema["function"]["name"] for schema in schemas}
    assert len(schemas) <= 20
    assert "host_snapshot" in names
    assert "execute_shell" in names
