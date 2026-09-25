import json


def _prompt() -> str:
    sections = [
        ("A. DIRECT TOOL ROUTING", range(1, 7)),
        ("B. SIMILAR-TOOL DISAMBIGUATION", range(7, 12)),
        ("C. TOOL DISCOVERY / RECOVERY PATH", range(12, 15)),
        ("D. WORKSPACE PATH AND FILE-TOOL TEST", range(15, 18)),
        ("E. RECIPE DISCOVERY AND DUPLICATION CHECK", range(18, 20)),
        ("F. RECIPE CREATION", range(20, 22)),
        ("G. RECIPE REPLAY", range(22, 25)),
        ("H. RECIPE FAILURE / FALLBACK BEHAVIOR", range(25, 28)),
        ("I. OBSERVATION / TRUNCATION PATH", range(28, 30)),
        ("J. RECIPE STORAGE INTEGRITY", range(30, 33)),
        ("K. CLEANUP", range(33, 35)),
        ("L. REQUIREMENT / EVIDENCE AUDIT", range(35, 38)),
    ]
    lines = [
        "I want you to perform a tool-routing and recipe-system stress test.",
        "Treat every numbered item below as an independent requirement.",
        "GENERAL SAFETY RULES",
        "1. Do not modify system configuration.",
        "2. Do not install packages.",
    ]
    for heading, numbers in sections:
        lines += ["==================================================", heading, "=================================================="]
        for number in numbers:
            lines.append(f"{number}. Requirement {number} for deterministic tool/recipe validation.")
    lines += ["FINAL OUTPUT", "Return the requested report."]
    return "\n".join(lines)


def test_tool_recipe_stress_compiles_all_37_requirements():
    from tools.task_requirements import derive_requirements

    rows = derive_requirements(_prompt())
    assert len(rows) == 37
    assert [row.key for row in rows] == [f"tooltest:{i:02d}" for i in range(1, 38)]
    assert rows[0].tool == "current_time"
    assert rows[6].tool == "dns_query"
    assert rows[14].tool == "write_file"
    assert rows[17].tool == "search_recipes"
    assert rows[18].tool == "load_recipe"
    assert rows[19].tool == "save_recipe"
    assert rows[20].tool == "search_recipes"
    assert rows[21].tool == "run_recipe"
    assert rows[29].tool == "load_recipe"
    assert rows[32].tool == "remove_path"
    assert all("Do not install packages" not in str(row.scope.get("source_text") or "") for row in rows)


def test_key_scoped_requirement_recording_keeps_repeated_phase_pending():
    from tools.task_requirements import Requirement, TaskRequirementLedger

    ledger = TaskRequirementLedger([
        Requirement("tooltest:18", "search_recipes", "pre-search"),
        Requirement("tooltest:21", "search_recipes", "post-search"),
    ])
    ledger.record_tool_for_key(
        "tooltest:18", "search_recipes", status="ok", reason="ok",
        arguments={"query": "health"}, result_text="[]",
    )
    assert ledger.requirements[0].status == "satisfied"
    assert ledger.requirements[1].status == "pending"


def test_workspace_remove_path_is_workspace_bounded(tmp_path, monkeypatch):
    import tools.workspace as ws

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(root))
    nested = root / "harness_tool_recipe_test"
    nested.mkdir()
    (nested / "input.txt").write_text("x", encoding="utf-8")
    assert ws.remove_path("harness_tool_recipe_test", recursive=True).startswith("Successfully removed")
    assert not nested.exists()
    assert ws.remove_path("../outside", recursive=True).startswith("Error:")


def test_load_recipe_tool_exposes_pipeline(tmp_path, monkeypatch):
    import tools.recipe_store as store
    from tools.recipe_store import save_recipe
    from tools.pipeline import load_recipe

    db = tmp_path / "recipes.db"
    monkeypatch.setenv("AGENT_RECIPE_DB", str(db))
    save_recipe("r", "demo", [{"id": "s1", "tool": "calculate", "args": {"expression": "1+1"}}])
    payload = json.loads(load_recipe("r"))
    assert payload["name"] == "r"
    assert payload["pipeline"][0]["tool"] == "calculate"




def test_requirement_ledger_records_derived_discovery_provenance():
    from tools.task_requirements import Requirement, TaskRequirementLedger

    ledger = TaskRequirementLedger([
        Requirement(
            "tooltest:12", "__tooltest_observation_capability__",
            "observation retrieval capability discovery", scope={"derived": True},
        )
    ])
    ledger.record_evidence_for_key(
        "tooltest:12", source="tool_call", tool_name="tool_search", status="ok",
        reason="ok", fingerprint="fp", arguments_digest="digest",
        evidence_ref="obs-12", count_attempt=True,
    )
    row = ledger.requirements[0]
    assert row.attempts == 1
    assert row.evidence == [{
        "source": "tool_call", "tool": "tool_search", "status": "ok",
        "reason": "ok", "fingerprint": "fp", "arguments_digest": "digest",
        "evidence_ref": "obs-12",
    }]
    assert ledger.as_list()[0]["evidence"][0]["tool"] == "tool_search"
