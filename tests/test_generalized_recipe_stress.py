import json


def _prompt() -> str:
    sections = [
        ("A. LEDGER AND SYSTEM BASELINE", range(16, 21)),
        ("B. TOOL LAYER SEPARATION", range(21, 26)),
        ("C. CAPABILITY DISCOVERY AND PROVENANCE", range(26, 30)),
        ("D. DISPOSABLE WORKSPACE TEST", range(30, 33)),
        ("E. SEMANTIC RECIPE SEARCH", range(33, 36)),
        ("F. GENERALIZED RECIPE CREATION", range(36, 39)),
        ("G. FIRST PARAMETERIZED REPLAY", range(39, 43)),
        ("H. SECOND PARAMETERIZED REPLAY", range(43, 47)),
        ("I. GENERALIZATION AUDIT", range(47, 51)),
        ("J. ROUTING AND DISCOVERY AUDIT", range(51, 55)),
        ("K. OBSERVATION AND FAILURE AUDIT", range(55, 59)),
        ("L. WORKSPACE CLEANUP", range(59, 63)),
        ("M. PERSISTED LEDGER AUDIT", range(63, 68)),
        ("N. FINAL AUDITS", range(68, 73)),
    ]
    lines = [
        "I want a generalized recipe, parameterization, provenance, and requirement-ledger stress test.",
        "Treat every numbered item as an independent requirement.",
        "GENERAL RULES",
    ]
    for n in range(1, 16):
        lines.append(f"{n}. General safety and deterministic execution rule {n}.")
    for heading, numbers in sections:
        lines += ["==================================================", heading, "=================================================="]
        for n in numbers:
            text = f"Requirement {n}."
            if n == 53:
                text = "Verify tool_search provenance is persisted."
            lines.append(f"{n}. {text}")
    lines += ["FINAL OUTPUT", "Return the requested deterministic report."]
    return "\n".join(lines)


def test_generalized_recipe_stress_compiles_72_requirements():
    from tools.task_requirements import derive_requirements

    rows = derive_requirements(_prompt())
    assert len(rows) == 72
    assert [row.key for row in rows] == [f"genrecipe:{i:02d}" for i in range(1, 73)]
    assert rows[0].scope["policy_rule"] is True
    assert rows[15].tool == "current_time"
    assert rows[20].tool == "dns_query"
    assert rows[29].tool == "write_file"
    assert rows[35].tool == "save_recipe"
    assert rows[38].tool == "run_recipe"
    assert rows[42].tool == "run_recipe"
    assert rows[58].tool == "remove_path"
    assert rows[59].tool == "path_stat"


def test_pipeline_template_resolves_runtime_parameter():
    from tools.pipeline import _resolve

    resolved = _resolve(
        {"$template": "https://{hostname}", "vars": {"hostname": {"$param": "hostname"}}},
        {}, {"hostname": "www.iana.org"},
    )
    assert resolved == "https://www.iana.org"
