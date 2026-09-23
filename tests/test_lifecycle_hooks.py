
def test_lifecycle_hooks_run_in_priority_order_and_isolate_errors():
    from tools.lifecycle_hooks import clear_hooks, emit_hooks, register_hook

    clear_hooks()
    seen = []
    register_hook("after_tool", lambda payload: seen.append(("low", payload["tool"])), priority=1, source="t:low")
    register_hook("after_tool", lambda payload: (_ for _ in ()).throw(RuntimeError("boom")), priority=5, source="t:bad")
    register_hook("after_tool", lambda payload: seen.append(("high", payload["tool"])), priority=10, source="t:high")
    results = emit_hooks("after_tool", {"tool": "demo"})
    assert seen == [("high", "demo"), ("low", "demo")]
    assert any(isinstance(item, dict) and "hook_error" in item for item in results)
    clear_hooks()


def test_before_tool_hook_can_deny_before_registry_lookup():
    from tools.lifecycle_hooks import clear_hooks, register_hook
    from tools.executor import execute_registered_tool

    clear_hooks()
    register_hook("before_tool", lambda payload: {"deny": "blocked for test"}, source="t:deny")
    # ``demo`` is intentionally not registered: a before hook must be able to
    # deny before registry lookup/execution.
    result = execute_registered_tool("demo", {"x": 1})
    assert "blocked by lifecycle hook" in result
    assert "blocked for test" in result
    clear_hooks()


def test_custom_extension_validator_accepts_agent_hook_module():
    from tools.tool_manager import _validate_tool_code

    ok, message = _validate_tool_code(
        "from tools.lifecycle_hooks import agent_hook\n\n"
        "@agent_hook('after_tool')\n"
        "def audit(payload: dict):\n"
        "    return None\n"
    )
    assert ok is True, message


def test_slow_hook_is_bounded_and_disabled():
    import time
    from tools.lifecycle_hooks import clear_hooks, emit_hooks, register_hook

    clear_hooks()
    def slow(_payload):
        time.sleep(0.2)
    register_hook("after_tool", slow, source="t:slow")
    started = time.monotonic()
    first = emit_hooks("after_tool", {"tool": "demo"})
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert "disabled" in first[0]["hook_error"]
    started = time.monotonic()
    second = emit_hooks("after_tool", {"tool": "demo"})
    assert time.monotonic() - started < 0.03
    assert "disabled" in second[0]["hook_error"]
    clear_hooks()
