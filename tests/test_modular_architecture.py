from pathlib import Path


def test_entrypoints_are_thin_compatibility_facades():
    assert Path("agent.py").stat().st_size < 8_000
    assert Path("worker.py").stat().st_size < 4_000
    assert Path("tools/primitive_ops.py").stat().st_size < 5_000


def test_builtin_provider_modules_are_discovered_without_duplicates():
    from tools.providers import BUILTINS
    assert len(BUILTINS) >= 170
    assert len(BUILTINS) == len(set(BUILTINS))
    assert any(module.startswith("primitive_modules.") for module, _ in BUILTINS)


def test_primitive_domains_are_split_and_legacy_facade_remains():
    expected = {"filesystem", "text", "structured", "process", "system", "network", "web", "documents", "media", "archive", "git", "database", "utility", "observation"}
    found = {p.stem for p in Path("tools/primitive_modules").glob("*.py")} - {"__init__", "common"}
    assert expected <= found
    import tools.primitive_ops as legacy
    assert callable(legacy.path_stat)
    assert callable(legacy.resolve_host)
    assert callable(legacy.calculate)


def test_background_jobs_are_provider_dispatched():
    from al_agent.background.handlers import JOB_HANDLERS
    assert set(JOB_HANDLERS) == {"research", "context_compaction", "self_optimization"}
    assert all(path.name.startswith("p") for path in Path("al_agent/background/job_providers").glob("p*.py"))


def test_cli_commands_are_registered_handlers():
    from al_agent.cli_commands import COMMANDS
    names = {command.name for command in COMMANDS}
    assert "/research" in names and "/tools" in names and "exit" in names
    assert len(names) == len(COMMANDS)


def test_webui_domain_helpers_are_modularized():
    for path in ("webui/chat.py", "webui/workspace_ops.py", "webui/theme.py", "webui/history.py", "webui/config.py"):
        assert Path(path).is_file()
