#!/usr/bin/env python3
"""Long-running runtime audit for every registered tool/primitive and recipe.

The controller inventories the live registry, synthesizes safe probe arguments,
runs each invocation in a short-lived child process, records one JSONL row per
probe, and checkpoints JSON/Markdown summaries throughout the run.

By default read-only tools are invoked and mutating tools are either exercised
against isolated audit state/workspace fixtures or contract-tested only.  Use
``--mutating-mode all`` only in a disposable harness/container: that mode may
install packages, create timers/jobs, modify the live work queue, or otherwise
change external state.
"""
from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import importlib.metadata as importlib_metadata
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from urllib.parse import urlsplit, urlunsplit
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _maybe_reexec_in_repo_venv(argv: list[str] | None = None) -> None:
    """Use the repo-local venv automatically when it has been bootstrapped."""
    if os.environ.get("AGENT_VENV_REEXEC") == "1" or not VENV_PYTHON.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        venv_python = VENV_PYTHON.resolve()
    except OSError:
        return
    if current == venv_python:
        return
    env = dict(os.environ)
    env["AGENT_VENV_REEXEC"] = "1"
    forwarded = list(sys.argv[1:] if argv is None else argv)
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *forwarded], env)


def _declared_distribution_names(path: Path = REQUIREMENTS_PATH) -> list[str]:
    """Return distribution names declared in requirements.txt without needing packaging."""
    if not path.is_file():
        return []
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-r ", "--requirement ", "-e ", "--editable ")):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            names.append(match.group(1))
    return names


def _missing_declared_dependencies(path: Path = REQUIREMENTS_PATH) -> list[str]:
    missing: list[str] = []
    for name in _declared_distribution_names(path):
        try:
            importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(name)
    return missing


def _dependency_preflight_message(missing: list[str]) -> str:
    bootstrap = ROOT / "scripts" / "bootstrap_venv.sh"
    listed = ", ".join(missing) if missing else "an undeclared Python module"
    return (
        "Tool soak cannot start because the repository Python environment is incomplete.\n"
        f"Missing Python package(s): {listed}\n\n"
        "Bootstrap the repo-local environment once:\n"
        f"  {bootstrap}\n\n"
        "Then rerun the same command. scripts/soak_test_tools.py will automatically "
        f"re-exec with {VENV_PYTHON} when it exists."
    )


def _preflight_repository_dependencies() -> tuple[bool, list[str]]:
    missing = _missing_declared_dependencies()
    return (not missing, missing)

def _runtime_context_preflight() -> tuple[bool, str]:
    """Require the production container namespace for meaningful runtime probes."""
    try:
        root = ROOT.resolve()
    except OSError:
        root = ROOT
    if root == Path("/app") and Path("/app/workspace").is_dir():
        return True, ""
    if os.environ.get("TOOL_SOAK_ALLOW_HOST_RUNTIME") == "1":
        return True, ""
    return False, (
        "Tool soak runtime probes must run inside the Compose worker container so /app, "
        "/app/workspace, /app/memory, /host, and the production PID namespace match the harness.\n"
        "Start the stack and run:\n"
        "  ./scripts/run_soak_test.sh\n"
        "or:\n"
        "  docker compose exec -w /app worker python scripts/soak_test_tools.py --workers 2 --mutating-mode isolated"
    )

# Mutators that can be pointed at the audit DB/profile or confined to an audit
# subdirectory under the workspace.  The remaining mutators are contract-only
# unless --mutating-mode all is explicitly selected.
ISOLATABLE_MUTATORS = {
    "remember", "remember_semantic", "set_user_identity", "set_research_preference",
    "set_profile_image", "write_file", "remove_path", "execute_shell", "execute_python",
    "render_document_page", "image_resize", "image_crop", "image_convert",
    "archive_extract", "page_diff", "generate_pdf_report",
    "save_recipe", "enqueue_research", "cancel_background_job",
    "start_computation", "cancel_computation",
    "set_goal", "clear_goal", "reload_tools",
}

# Deliberately never invoked in the default isolated mode.  Some affect absolute
# /app paths, OS timers/notifications, package state, or the live legacy queue.
NONISOLATABLE_MUTATORS = {
    "install_package", "schedule_reminder", "cancel_reminder", "notify_desktop",
    "queue_work", "update_work_status", "create_or_update_tool",
    "enqueue_self_optimization", "map_network", "take_web_screenshot", "browser_step",
}

# Expensive/active probes get a slightly larger child timeout when the global
# timeout is smaller.  This still remains bounded by the controller.
SLOW_TOOLS = {
    "repo_checks", "dependency_audit", "scan_subnet", "map_network", "scan_mdns",
    "network_path", "trace_route", "take_web_screenshot", "browser_step", "generate_pdf_report",
    "extract_document", "weather_forecast", "web_search", "news_search",
    "wiki_search", "market_quote", "browse_url", "discover_site", "page_diff",
}

ERROR_HINTS = {
    "credential": (
        "not configured", "not_connected", "not connected", "credential", "oauth",
        "unauthorized", "forbidden", "401", "403", "open connections",
    ),
    "dependency": (
        "not installed", "not available", "unavailable", "missing dependency",
        "command not found", "no such file or directory", "no module named",
        "try pulling it first", "embedding generation failed",
    ),
    "network": (
        "timed out", "timeout", "connection refused", "network is unreachable",
        "temporary failure", "name or service not known", "dns",
        "nameresolutionerror", "max retries exceeded", "failed to resolve",
        "failed to establish a new connection",
    ),
    "not_found": ("not found", "does not exist", "no results"),
}

# These classes normally describe the machine running the soak test rather than
# a defect in the harness implementation.  They remain visible as ordinary
# errors, but the summary also reports a harness-only success rate that excludes
# externally blocked probes.
ENVIRONMENT_BLOCKING_CLASSES = {"credential", "dependency", "network"}

WORKER_SENTINEL = "__TOOL_SOAK_RESULT__="

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)

# Tiny one-page PDF containing the word "audit".  Keeping the fixture inline
# avoids adding a binary test asset to the repository.
MINIMAL_PDF = b"""%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 41>>stream\nBT /F1 18 Tf 50 80 Td (audit) Tj ET\nendstream endobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000250 00000 n \n0000000340 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n410\n%%EOF\n"""



PASS_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "label": "baseline", "expression": "6*7", "hash_text": "audit-one",
        "wiki_query": "Linux", "web_query": "OpenAI", "news_query": "technology headlines",
        "endpoint_tls": False, "http_url": "http://example.com/", "base64_text": "audit-one",
    },
    {
        "label": "alternate", "expression": "84/2", "hash_text": "audit-two",
        "wiki_query": "Python (programming language)", "web_query": "Python programming language", "news_query": "science headlines",
        "endpoint_tls": True, "http_url": "https://example.com/", "base64_text": "audit-two",
    },
    {
        "label": "minimal", "expression": "1+1", "hash_text": "x",
        "wiki_query": "SQLite", "web_query": "SQLite database", "news_query": "business headlines",
        "endpoint_tls": False, "http_url": "http://example.com/", "base64_text": "x",
    },
    {
        "label": "unicode", "expression": "(5+7)*3", "hash_text": "audit-π",
        "wiki_query": "Unicode", "web_query": "Unicode standard", "news_query": "world headlines",
        "endpoint_tls": True, "http_url": "https://example.com/", "base64_text": "audit-π",
    },
    {
        "label": "boundary", "expression": "1000-958", "hash_text": "audit-five",
        "wiki_query": "Computer network", "web_query": "computer networking", "news_query": "health headlines",
        "endpoint_tls": True, "http_url": "https://example.com/", "base64_text": "audit-five",
    },
)

@dataclass
class Target:
    kind: str  # tool | primitive | recipe_builtin | recipe_saved
    name: str
    module: str = ""
    readonly: bool = True
    schema: dict[str, Any] | None = None
    recipe: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_duration(value: str | int | float) -> float:
    """Parse 90, 90s, 15m, 2h, or 1d into seconds."""
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([smhd]?)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid duration: {value!r}; use e.g. 30m, 8h, 1d")
    number = float(match.group(1))
    scale = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    return max(0.0, number * scale)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * pct
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return round(ordered[lo], 3)
    value = ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)
    return round(value, 3)


def _workspace_root() -> Path:
    configured = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace"))
    if configured.is_dir():
        return configured.resolve()
    fallback = ROOT / "workspace"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


def create_fixtures(base: Path, workspace: Path | None = None, pass_no: int = 1) -> dict[str, Any]:
    """Create deterministic, pass-specific fixtures for broader repeat coverage."""
    base.mkdir(parents=True, exist_ok=True)
    profile = dict(PASS_PROFILES[(max(1, int(pass_no)) - 1) % len(PASS_PROFILES)])
    label = str(profile["label"])
    value_b = max(2, int(pass_no) + 1)

    text_path = base / "sample.txt"
    text_path.write_text(f"alpha\nbeta\n{label}\ngamma {40 + value_b}\n", encoding="utf-8")
    json_path = base / "sample.json"
    json_path.write_text(json.dumps({"items": [{"name": "alpha", "value": 1}, {"name": "beta", "value": value_b}], "ok": True, "pass": int(pass_no)}), encoding="utf-8")
    yaml_path = base / "sample.yaml"
    yaml_path.write_text(f"ok: true\npass: {int(pass_no)}\nitems:\n  - name: alpha\n    value: 1\n", encoding="utf-8")
    csv_path = base / "sample.csv"
    csv_path.write_text(f"name,value\nalpha,1\nbeta,{value_b}\n", encoding="utf-8")
    jsonl_path = base / "sample.jsonl"
    jsonl_path.write_text(f'{{"name":"alpha","value":1}}\n{{"name":"beta","value":{value_b}}}\n', encoding="utf-8")
    html_path = base / "sample.html"
    html = f"""<!doctype html><html><head><title>Audit fixture {pass_no}</title><meta name=description content='audit {label}'></head>
<body><main><h1>Audit {pass_no}</h1><p>Tool soak fixture {label}.</p><a href='/next'>Next</a><img src='/image.png' alt='x'>
<script type='application/ld+json'>{{"@type":"Article","headline":"Audit {pass_no}"}}</script>
<table><tr><th>Name</th><th>Value</th></tr><tr><td>alpha</td><td>{value_b}</td></tr></table></main></body></html>"""
    html_path.write_text(html, encoding="utf-8")
    feed_path = base / "feed.xml"
    feed_path.write_text(f"""<?xml version='1.0'?><rss version='2.0'><channel><title>Audit {pass_no}</title><item><title>One {pass_no}</title><link>https://example.com/one</link><description>{label}</description></item></channel></rss>""", encoding="utf-8")
    image_path = base / "sample.png"
    try:
        from PIL import Image
        Image.new("RGB", (8 + (int(pass_no) % 3), 8 + (int(pass_no) % 2))).save(image_path, format="PNG")
    except Exception:
        image_path.write_bytes(PNG_1X1)
    pdf_path = base / "sample.pdf"
    pdf_path.write_bytes(MINIMAL_PDF)
    zip_path = base / "sample.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"inside-{int(pass_no)}.txt", f"audit archive {label}\n")
    db_path = base / "sample.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, name TEXT, value REAL)")
        conn.execute("DELETE FROM items")
        conn.executemany("INSERT INTO items(name,value) VALUES(?,?)", [("alpha", 1.0), ("beta", float(value_b))])
        conn.commit()
    workspace = (workspace or _workspace_root()).resolve()
    relative_base = str(base.resolve().relative_to(workspace)) if _is_relative_to(base, workspace) else str(base.resolve())
    remove_dir = base / "remove-me"
    remove_dir.mkdir(parents=True, exist_ok=True)
    (remove_dir / "sentinel.txt").write_text("remove-path soak fixture\n", encoding="utf-8")
    relative_remove_dir = str(remove_dir.resolve().relative_to(workspace)) if _is_relative_to(remove_dir, workspace) else str(remove_dir)
    base64_text = str(profile["base64_text"])
    return {
        "pass_no": int(pass_no), "profile": label,
        "expression": str(profile["expression"]), "hash_text": str(profile["hash_text"]),
        "wiki_query": str(profile["wiki_query"]), "web_query": str(profile["web_query"]),
        "news_query": str(profile["news_query"]), "endpoint_tls": bool(profile["endpoint_tls"]),
        "http_url": str(profile["http_url"]), "base64_text": base64_text,
        "base64_data": base64.b64encode(base64_text.encode("utf-8")).decode("ascii"),
        "feed_url": "https://feeds.bbci.co.uk/news/rss.xml", "host_file": "/etc/os-release",
        "base": str(base), "relative_base": relative_base, "text": str(text_path), "json": str(json_path), "yaml": str(yaml_path),
        "csv": str(csv_path), "jsonl": str(jsonl_path), "html_path": str(html_path),
        "html": html, "feed": feed_path.read_text(encoding="utf-8"), "image": str(image_path),
        "pdf": str(pdf_path), "zip": str(zip_path), "db": str(db_path), "relative_remove_dir": relative_remove_dir,
        "relative_text": str(text_path.resolve().relative_to(workspace)) if _is_relative_to(text_path, workspace) else str(text_path),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _seed_isolated_state(env: dict[str, str], pass_no: int = 1) -> dict[str, Any]:
    """Seed fresh IDs needed by stateful probes for one soak pass."""
    old = os.environ.copy()
    os.environ.update(env)
    ids: dict[str, Any] = {}
    suffix = f"pass-{max(1, int(pass_no)):03d}"
    try:
        # Modules read AGENT_DB_PATH at import time, so do this before importing.
        from tools.runtime import create_job, init_runtime_db
        init_runtime_db()
        ids["job_id"] = create_job("audit", f"Tool soak audit fixture {suffix}", {"audit": True, "pass": int(pass_no)}, max_attempts=1)
        ids["compute_job_id"] = create_job(
            "durable_compute",
            f"Tool soak durable compute fixture {suffix}",
            {"program": {"initial_state": "HALT", "halt_states": ["HALT"], "transitions": {}}},
            max_attempts=1,
        )
        from tools.memory import init_db, remember, store_tool_observation
        init_db()
        remember(f"tool-soak-audit-{suffix}", f"tool soak audit synthetic fact {suffix}")
        ids["observation_old"] = store_tool_observation("audit", json.dumps({"value": int(pass_no), "phase": "old"}))
        ids["observation_new"] = store_tool_observation("audit", json.dumps({"value": int(pass_no) + 1, "phase": "new"}))

        # Seed positive-path IDs for lookup tools so their normal success path is
        # exercised instead of deliberately asking for missing objects.
        from tools.task_manager import create_task, log_task
        ids["task_id"] = create_task(f"Tool soak audit task {suffix}", "Synthetic task fixture", priority=-100)
        log_task(f"tool soak audit log fixture {suffix}", ids["task_id"])

        from tools.runtime import create_optimization_candidate
        ids["candidate_id"] = create_optimization_candidate(
            ids["job_id"], f"Tool soak audit candidate {suffix}", "synthetic_metric"
        )

        # The legacy work queue predates AGENT_DB_PATH and keeps a module-global
        # absolute DB path. Patch that global only in the soak process, point it
        # at the isolated audit DB, and seed a completed result.
        from tools import work_queue
        work_queue.DB_PATH = env["AGENT_DB_PATH"]
        work_queue.init_work_queue_db()
        ids["work_id"] = work_queue.queue_work(
            f"Tool soak audit work {suffix}", "Synthetic work queue fixture", priority=-100,
            tags=["audit", suffix], estimated_hours=0.01,
        )
        work_queue.mark_work_complete(ids["work_id"], f"Synthetic audit result {suffix}", f"Synthetic audit full result {suffix}")

        # Seed a custom source file for the read/list custom-tool primitives.
        custom_dir = Path(env["TOOL_SOAK_CUSTOM_TOOLS_DIR"])
        custom_dir.mkdir(parents=True, exist_ok=True)
        (custom_dir / "audit_fixture_tool.py").write_text(
            "from tools.tool_registry import agent_tool\n\n"
            "@agent_tool(readonly=True)\n"
            "def audit_fixture_tool() -> str:\n"
            "    \"\"\"Synthetic soak-test fixture.\"\"\"\n"
            f"    return 'audit {suffix}'\n",
            encoding="utf-8",
        )
    except Exception as exc:
        ids["seed_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.environ.clear(); os.environ.update(old)
    return ids


def _start_process_fixture(base: Path, pass_no: int) -> subprocess.Popen:
    """Start a same-UID process whose metadata, I/O, and open files are probeable."""
    marker = base / "process-fixture.txt"
    code = (
        "import pathlib,sys,time; "
        "p=pathlib.Path(sys.argv[1]); "
        "f=p.open('a+', encoding='utf-8'); "
        "f.write('tool-soak-process\n'); f.flush(); "
        "time.sleep(3600)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(marker)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Give /proc enough time to expose the child before the first process probe.
    time.sleep(0.05)
    return proc


def _stop_process_fixture(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass


def build_env(report_dir: Path, workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    state_dir = report_dir / "isolated_state"
    profile_dir = state_dir / "profile"
    state_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONPATH": str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "AGENT_DB_PATH": str(state_dir / "knowledge.db"),
        "AGENT_RECIPE_DB": str(state_dir / "recipes.db"),
        "AGENT_PROFILE_DIR": str(profile_dir),
        "AGENT_WORKSPACE": str(workspace),
        "TOOL_SOAK_CUSTOM_TOOLS_DIR": str(state_dir / "custom_tools"),
        "TOOL_SOAK_ACTIVE": "1",
    })
    return env


def _apply_soak_isolation() -> None:
    """Redirect legacy module globals that do not yet honor AGENT_* paths.

    This runs only inside the dedicated soak worker. It does not alter normal
    harness behavior and keeps legacy read/write primitives from touching live
    state during an isolated audit.
    """
    if os.environ.get("TOOL_SOAK_ACTIVE") != "1":
        return
    isolated_db = os.environ.get("AGENT_DB_PATH", "").strip()
    workspace = os.environ.get("AGENT_WORKSPACE", "").strip()
    if isolated_db:
        try:
            from tools import work_queue
            work_queue.DB_PATH = isolated_db
            work_queue.init_work_queue_db()
        except Exception:
            pass
    custom_tools_dir = os.environ.get("TOOL_SOAK_CUSTOM_TOOLS_DIR", "").strip()
    if custom_tools_dir:
        try:
            from tools import tool_manager
            tool_manager.TOOLS_DIR = Path(custom_tools_dir).resolve()
            tool_manager.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _builtin_module_map() -> dict[str, str]:
    from tools.providers import BUILTINS
    return {str(function): str(module) for module, function in BUILTINS}


def read_saved_recipes_direct(database: str | Path, limit: int = 500) -> list[dict[str, Any]]:
    """Read user recipes without importing tools, so audit DB isolation stays intact."""
    path = Path(database).expanduser()
    if not path.is_file():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            columns = {row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()}
            if not {"name", "pipeline_json", "parameters_json"}.issubset(columns):
                return []
            rows = conn.execute("SELECT * FROM recipes ORDER BY id ASC LIMIT ?", (max(1, int(limit)),)).fetchall()
    except sqlite3.Error:
        return []
    recipes: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if str(item.get("origin") or "user") == "builtin":
            continue
        try:
            pipeline = json.loads(item.get("pipeline_json") or "[]")
            parameters = json.loads(item.get("parameters_json") or "{}")
            tags = json.loads(item.get("tags_json") or "[]") if "tags_json" in item else []
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        recipes.append({
            "id": item.get("id"), "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""), "pipeline": pipeline,
            "parameters": parameters, "tags": tags, "origin": str(item.get("origin") or "user"),
            "target_tool": str(item.get("target_tool") or ""),
        })
    return recipes


def discover_targets(saved_recipes: list[dict[str, Any]] | None = None) -> tuple[list[Target], dict[str, Any]]:
    from tools import load_tools, TOOL_SCHEMAS, TOOL_METADATA
    count, load_errors = load_tools()
    modules = _builtin_module_map()
    targets: list[Target] = []
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {})
        name = str(fn.get("name") or "")
        module = modules.get(name, "custom")
        kind = "primitive" if module.startswith("primitive_modules.") else "tool"
        targets.append(Target(kind=kind, name=name, module=module, readonly=bool(TOOL_METADATA.get(name, {}).get("readonly", True)), schema=schema))

    builtin_count = 0
    from tools.recipe_compat import discover_recipe_specs
    specs, native_only = discover_recipe_specs()
    for spec in specs:
        targets.append(Target(kind="recipe_builtin", name=str(spec["name"]), module="recipe_compat", readonly=True, recipe=dict(spec)))
        builtin_count += 1

    saved = list(saved_recipes or [])
    for recipe in saved:
        targets.append(Target(kind="recipe_saved", name=str(recipe.get("name") or "unnamed recipe"), module="recipe_store", readonly=True, recipe=recipe))
    return targets, {
        "registered_tools": count,
        "load_errors": load_errors,
        "builtin_recipes": builtin_count,
        "saved_recipes": len(saved),
        "native_only_recipe_targets": len(native_only),
    }


def _schema_parameters(target: Target) -> tuple[dict[str, Any], list[str]]:
    fn = (target.schema or {}).get("function", {})
    params = fn.get("parameters", {}) or {}
    return dict(params.get("properties") or {}), list(params.get("required") or [])


def _default_from_schema(spec: dict[str, Any]) -> Any:
    if "default" in spec:
        return spec["default"]
    enum = spec.get("enum")
    if enum:
        return enum[0]
    typ = spec.get("type")
    if typ == "boolean": return False
    if typ == "integer": return max(int(spec.get("minimum", 0)), 1)
    if typ == "number": return max(float(spec.get("minimum", 0.0)), 1.0)
    if typ == "array":
        item = dict(spec.get("items") or {})
        return [_default_from_schema(item)]
    if typ == "object": return {}
    return "audit"


def _required_value(name: str, spec: dict[str, Any], fixtures: dict[str, Any], ids: dict[str, Any]) -> Any:
    # Exact semantic fixtures first.
    values: dict[str, Any] = {
        "fact": "tool soak audit synthetic fact",
        "topic": "tool-soak-audit",
        "query": "OpenAI technology",
        "observation_id": ids.get("observation_new", "audit-observation"),
        "old_id": ids.get("observation_old", "audit-old"),
        "new_id": ids.get("observation_new", "audit-new"),
        "name": "example.com",
        "host": "example.com",
        "target": "example.com",
        "url": "https://example.com/",
        "path_or_url": fixtures["pdf"],
        "latitude": 42.9849,
        "longitude": -81.2453,
        "instruments": ["BZ=F", "CL=F"],
        "filename": fixtures["relative_text"],
        "path": fixtures["text"],
        "database": fixtures["db"],
        "table": "items",
        "pid": int(ids.get("process_pid") or os.getpid()),
        "port": 443,
        "expression": fixtures.get("expression", "6*7"),
        "value": 1,
        "from_unit": "km",
        "to_unit": "m",
        "text": "alpha\nbeta\nbeta\n",
        "data": {"items": [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}]},
        "a": "alpha",
        "b": "beta",
        "condition": True,
        "items": ["alpha", "beta"],
        "mapping": {"alpha": "one"},
        "network": "127.0.0.0/30",
        "address": "127.0.0.1",
        "base": "https://example.com/a/",
        "relative": "../b",
        "timestamp": "2026-09-21T12:00:00+00:00",
        "tool_name": "audit_generated_tool",
        "specification": "Create a read-only tool named audit_generated_tool that returns the string audit.",
        "package_name": "definitely-not-a-real-tool-soak-package-9f8e7d6c",
        "job_id": ids.get("job_id", "audit-job"),
        "candidate_id": "audit-candidate",
        "reminder_id": "audit-reminder",
        "work_id": "audit-work",
        "task_id": "audit-task",
        "message_id": "audit-message",
        "event_id": "audit-event",
        "markdown_content": "# Tool soak audit\n\nSynthetic report fixture.\n",
        "command": "printf 'tool-soak-audit\\n'",
        "code": "print('tool-soak-audit')",
        "title": "Tool soak audit",
        "description": "Synthetic tool soak audit fixture",
        "category": "audit",
        "key": "probe",
        "output": f"{fixtures['relative_base']}/rendered.png",
        "output_filename": f"{fixtures['relative_base']}/output.png",
        "destination": str(Path(fixtures["base"]) / "extracted"),
        "left": 0, "top": 0, "right": 1, "bottom": 1,
        "pattern": "beta",
        "replacement": "BETA",
        "key": "name",
        "operator": "eq",
        "html": fixtures["html"],
        "xml_text": fixtures["feed"],
        "links": ["https://example.com/a", "https://example.com/b"],
        "stages": [{"id": "s1", "tool": "calculate", "args": {"expression": "1+1"}}],
    }
    if name in values:
        return values[name]
    return _default_from_schema(spec)


def tool_args(target: Target, fixtures: dict[str, Any], ids: dict[str, Any]) -> dict[str, Any]:
    properties, required = _schema_parameters(target)
    args = {name: _required_value(name, properties.get(name, {}), fixtures, ids) for name in required}
    process_pid = int(ids.get("process_pid") or os.getpid())
    endpoint_tls = bool(fixtures.get("endpoint_tls", False))
    endpoint_port = 443 if endpoint_tls else 80

    # Tool-specific arguments improve meaningful path coverage while remaining bounded.
    exact: dict[str, dict[str, Any]] = {
        "search_memory": {"query": "tool soak audit", "limit": 5},
        "search_semantic_memory": {"query": "tool soak audit", "limit": 5},
        "read_observation": {"observation_id": ids.get("observation_new", "audit-observation")},
        "set_profile_image": {"path": fixtures["image"]},
        "set_user_identity": {"name": "Tool Soak Audit", "role": "test", "timezone": "America/Toronto", "interests": ["testing"]},
        "set_research_preference": {"category": "audit", "key": "probe", "value": "synthetic"},
        "search_repo_symbols": {"query": "load_tools", "limit": 5},
        "read_repo_symbol": {"path": "tools/catalog.py", "symbol": "load_tools", "max_lines": 80},
        "repo_checks": {"checks": ["compile"], "timeout": 60},
        "current_time": {"timezone_name": "America/Toronto"},
        "run_pipeline": {"stages": [{"id": "s1", "tool": "calculate", "args": {"expression": "1+1"}}]},
        "run_recipe": {"name": "__AUDIT_RECIPE__", "parameters": {}},
        "search_recipes": {"query": "audit calculate", "limit": 5},
        "save_recipe": {"name": "Tool soak synthetic", "description": "Synthetic audit recipe", "stages": [{"id": "s1", "tool": "calculate", "args": {"expression": "1+1"}}], "tags": ["audit"]},
        "read_file": {"filename": fixtures["relative_text"]},
        "write_file": {"filename": f"{fixtures['relative_base']}/write_file.txt", "content": "tool soak audit\n"},
        "remove_path": {"path": fixtures["relative_remove_dir"], "recursive": True},
        "execute_shell": {"command": "printf 'tool-soak-audit\\n'", "timeout": 5},
        "execute_python": {"code": "print('tool-soak-audit')", "timeout": 5},
        "path_stat": {"path": fixtures["text"]},
        "list_directory": {"path": fixtures["base"], "depth": 1, "limit": 20},
        "find_paths": {"root": fixtures["base"], "pattern": "*", "max_depth": 2, "limit": 20},
        "read_text": {"path": fixtures["text"], "limit": 2000},
        "read_bytes": {"path": fixtures["text"], "limit": 100},
        "tail_file": {"path": fixtures["text"], "lines": 3},
        "file_hash": {"path": fixtures["text"], "algorithm": "sha256"},
        "mime_type": {"path": fixtures["text"]},
        "disk_usage": {"path": fixtures["base"]},
        "read_lines": {"path": fixtures["text"], "start_line": 1, "end_line": 3},
        "directory_size": {"path": fixtures["base"], "max_depth": 2, "max_entries": 100},
        "text_search": {"pattern": "beta", "path": fixtures["text"], "limit": 10},
        "regex_extract": {"pattern": r"gamma\\s+(\\d+)", "path": fixtures["text"], "limit": 10},
        "text_diff": {"a": "alpha\n", "b": "beta\n"},
        "json_query": {"path_expr": "items", "path": fixtures["json"]},
        "json_filter": {"key": "name", "operator": "eq", "value": "alpha", "data": [{"name":"alpha","value":1},{"name":"beta","value":2}]},
        "json_sort": {"key": "value", "data": [{"name":"beta","value":2},{"name":"alpha","value":1}], "limit": 10},
        "json_diff": {"a": {"a": 1}, "b": {"a": 2}},
        "list_processes": {"limit": 5},
        "process_info": {"pid": process_pid},
        "resolve_host": {"host": "example.com"},
        "route_lookup": {"target": "1.1.1.1"},
        "tcp_connect": {"host": "example.com", "port": 443, "timeout": 3},
        "tls_handshake": {"host": "example.com", "port": 443, "timeout": 5},
        "http_request": {"url": "https://example.com/", "timeout": 8},
        "document_info": {"path": fixtures["pdf"]},
        "document_text": {"path": fixtures["pdf"], "max_chars": 2000},
        "render_document_page": {"path": fixtures["pdf"], "page": 1, "output": f"{fixtures['relative_base']}/rendered.png"},
        "image_info": {"path": fixtures["image"]},
        "archive_list": {"path": fixtures["zip"], "limit": 20},
        "calculate": {"expression": fixtures.get("expression", "6*7")},
        "parse_datetime": {"value": "2026-09-21T12:00:00+00:00"},
        "time_difference": {"a": "2026-09-21T12:00:00+00:00", "b": "2026-09-21T13:00:00+00:00"},
        "url_parse": {"url": "https://example.com/a?b=1"},
        "url_join": {"base": "https://example.com/a/", "relative": "../b"},
        "ip_parse": {"address": "192.0.2.1"},
        "subnet_contains": {"network": "192.0.2.0/24", "address": "192.0.2.1"},
        "command_available": {"name": "python"},
        "process_io": {"pid": process_pid},
        "process_threads": {"pid": process_pid, "limit": 10},
        "process_fds": {"pid": process_pid, "limit": 10},
        "pressure_info": {"resource": "cpu"},
        "interface_info": {"name": "lo"},
        "udp_probe": {"host": "127.0.0.1", "port": 9, "timeout": 1},
        "trace_route": {"target": "1.1.1.1", "max_hops": 5, "probes": 1},
        "fetch_url": {"url": "https://example.com/", "max_bytes": 100000},
        "extract_readable_text": {"html": fixtures["html"], "max_chars": 4000},
        "extract_links": {"html": fixtures["html"], "base_url": "https://example.com/", "limit": 10},
        "extract_metadata": {"html": fixtures["html"], "base_url": "https://example.com/"},
        "extract_images": {"html": fixtures["html"], "base_url": "https://example.com/", "limit": 10},
        "extract_jsonld": {"html": fixtures["html"], "limit": 10},
        "document_links": {"path": fixtures["pdf"], "limit": 10},
        "document_images": {"path": fixtures["pdf"], "limit": 10},
        "media_info": {"path": fixtures["image"]},
        "image_resize": {"path": fixtures["image"], "output": f"{fixtures['relative_base']}/resized.png", "max_width": 1, "max_height": 1},
        "image_crop": {"path": fixtures["image"], "output": f"{fixtures['relative_base']}/cropped.png", "left": 0, "top": 0, "right": 1, "bottom": 1},
        "image_convert": {"path": fixtures["image"], "output": f"{fixtures['relative_base']}/converted.png", "format": "png"},
        "archive_extract": {"path": fixtures["zip"], "destination": str(Path(fixtures["base"]) / "extracted"), "max_files": 10},
        "db_tables": {"database": fixtures["db"]},
        "db_schema": {"database": fixtures["db"], "table": "items"},
        "db_select": {"database": fixtures["db"], "table": "items", "limit": 5},
        "convert_units": {"value": 1, "from_unit": "km", "to_unit": "m"},
        "hash_text": {"text": fixtures.get("hash_text", "audit"), "algorithm": "sha256"},
        "base64_encode": {"text": fixtures.get("base64_text", "audit")},
        "base64_decode": {"data": fixtures.get("base64_data", "YXVkaXQ=")},
        "compare_values": {"a": 1, "b": 1},
        "compare_json": {"a": {"a": 1}, "b": {"a": 1}},
        "compare_text": {"a": "audit", "b": "audit"},
        "text_head": {"path": fixtures["text"], "lines": 2},
        "text_tail": {"path": fixtures["text"], "lines": 2},
        "text_count": {"path": fixtures["text"]},
        "text_sort": {"path": fixtures["text"], "limit": 10},
        "text_unique": {"path": fixtures["text"], "limit": 10},
        "regex_replace": {"pattern": "beta", "replacement": "BETA", "path": fixtures["text"], "count": 1},
        "text_split": {"text": "a,b,c", "delimiter": ",", "limit": 10},
        "json_head": {"data": [{"name":"alpha","value":1},{"name":"beta","value":2}], "count": 2},
        "json_count": {"path": fixtures["json"]},
        "json_keys": {"path": fixtures["json"], "limit": 10},
        "yaml_query": {"path": fixtures["yaml"], "path_expr": "items"},
        "csv_query": {"path": fixtures["csv"], "column": "name", "equals": "alpha", "limit": 10},
        "csv_summary": {"path": fixtures["csv"], "sample_rows": 2},
        "jsonl_summary": {"path": fixtures["jsonl"], "sample_rows": 2, "max_records": 10},
        "format_datetime": {"timestamp": "2026-09-21T12:00:00+00:00", "timezone_name": "America/Toronto"},
        "observation_get": {"observation_id": ids.get("observation_new", "audit-observation")},
        "observation_compare": {"old_id": ids.get("observation_old", "audit-old"), "new_id": ids.get("observation_new", "audit-new")},
        "compose_object": {"data": {"audit": True}},
        "compose_list": {"items": ["a", "b"]},
        "choose_value": {"condition": True, "if_true": "yes", "if_false": "no"},
        "map_value": {"value": "a", "mapping": {"a": "one"}, "default": "other"},
        "dns_query": {"name": "example.com", "record_type": "A"},
        "filesystem_usage": {"path": "/"},
        "host_read_text": {"path": "/etc/os-release", "max_chars": 2000},
        "url_endpoint": {"url": "https://example.com/a?b=1"},
        "filter_links": {"links": ["https://example.com/a", "https://openai.com/"], "base_url": "https://example.com/", "same_domain": True, "limit": 10},
        "parse_feed": {"xml_text": fixtures["feed"], "url": "https://example.com/feed.xml", "limit": 10},
        "fetch_json": {"url": "https://httpbin.org/json", "max_bytes": 100000},
        "extract_tables": {"html": fixtures["html"], "limit": 5, "max_rows": 10},
        "process_tree": {"pid": process_pid, "depth": 2, "limit": 20},
        "ping_host": {"host": "127.0.0.1", "count": 1, "timeout": 2},
        "diff_observations": {"old_id": ids.get("observation_old", "audit-old"), "new_id": ids.get("observation_new", "audit-new"), "max_diff_chars": 2000},
        "network_reachability": {"targets": ["1.1.1.1", "example.com"]},
        "read_host_file": {"filepath": fixtures.get("host_file", "/etc/os-release")},
        "read_host_journal": {"lines": 5},
        "tail_host_log": {"log_path": "/var/log/syslog", "lines": 5},
        "process_snapshot": {"limit": 5, "sort_by": "cpu"},
        "filesystem_snapshot": {"limit": 10},
        "service_health": {"service": "ollama", "lines": 5},
        "neighbor_snapshot": {"limit": 10},
        "connection_snapshot": {"limit": 10},
        "dns_diagnose": {"name": "example.com", "record_types": ["A", "AAAA"]},
        "network_path": {"target": "1.1.1.1", "max_hops": 5, "probes": 1},
        "endpoint_probe": {"host": "example.com", "port": endpoint_port, "tls": endpoint_tls, "timeout": 5},
        "http_probe": {"url": fixtures.get("http_url", "https://example.com/"), "timeout": 8},
        "local_subnets": {"include_virtual": False},
        "scan_subnet": {"network": "127.0.0.0/30", "max_detail_hosts": 2, "top_ports": 20},
        "map_network": {"network": "127.0.0.0/30", "output_filename": f"tool-soak-{Path(fixtures['base']).name}-network.png", "max_detail_hosts": 2, "top_ports": 20},
        "scan_mdns": {"timeout": 1},
        "geocode_location": {"query": "London, Ontario, Canada", "count": 1, "language": "en"},
        "weather_forecast": {"latitude": 42.9849, "longitude": -81.2453, "forecast_days": 2, "timezone_name": "America/Toronto"},
        "web_search": {"query": fixtures.get("web_query", "OpenAI")},
        "news_search": {"query": fixtures.get("news_query", "technology headlines"), "timelimit": "d", "max_results": 5},
        "wiki_search": {"query": fixtures.get("wiki_query", "Linux")},
        "market_quote": {"instruments": ["BZ=F", "CL=F"]},
        "browse_url": {"url": "https://example.com/"},
        "page_metadata": {"url": "https://example.com/"},
        "page_links": {"url": "https://example.com/", "limit": 10},
        "discover_site": {"url": "https://example.com/", "limit": 10},
        "read_feed": {"url": fixtures.get("feed_url", "https://feeds.bbci.co.uk/news/rss.xml"), "limit": 5},
        "extract_document": {"path_or_url": fixtures["pdf"], "max_pages": 2, "max_chars": 4000},
        "page_fingerprint": {"url": "https://example.com/"},
        "page_diff": {"url": "https://example.com/", "max_diff_chars": 2000},
        "take_web_screenshot": {"url": "https://example.com/", "output_filename": f"tool-soak-{Path(fixtures['base']).name}-example.png"},
        "attach_media": {"path": fixtures["image"], "context": "tool soak audit"},
        "generate_pdf_report": {"markdown_content": "# Tool soak audit\n\nSynthetic report.\n", "output_filename": f"{fixtures['relative_base']}/audit-report.pdf"},
        "search_packages": {"query": "python"},
        "install_package": {"package_name": "definitely-not-a-real-tool-soak-package-9f8e7d6c"},
        "gmail_search_messages": {"query": "newer_than:1d", "limit": 1},
        "gmail_read_message": {"message_id": "audit-invalid-message", "max_body_chars": 500},
        "google_calendar_list_events": {"limit": 1},
        "google_calendar_get_event": {"event_id": "audit-invalid-event"},
        "google_calendar_list_calendars": {"limit": 1},
        "enqueue_research": {"topic": "tool soak synthetic research", "priority": -100},
        "get_research_status": {"job_id": ids.get("job_id", "audit-job")},
        "list_background_jobs": {"limit": 5},
        "cancel_background_job": {"job_id": ids.get("job_id", "audit-job")},
        "start_computation": {
            "program": {
                "initial_state": "run",
                "halt_states": ["HALT"],
                "transitions": {"run": {"_": {"write": "1", "move": "N", "next": "HALT"}}},
            },
            "idempotency_key": f"tool-soak-{Path(fixtures['base']).name}",
        },
        "get_computation_status": {"job_id": ids.get("compute_job_id", "audit-compute"), "tape_cells": 8},
        "cancel_computation": {"job_id": ids.get("compute_job_id", "audit-compute")},
        "enqueue_self_optimization": {"objective": "tool soak synthetic no-op", "target_metric": "none", "priority": -100},
        "get_self_optimization_status": {"candidate_id": ids.get("candidate_id", "audit-candidate")},
        "list_self_optimization_candidates": {"limit": 5},
        "schedule_reminder": {"title": "Tool soak audit reminder", "message": "synthetic", "delay_seconds": 3600, "repeat": "once", "reminder_id": "tool-soak-audit"},
        "cancel_reminder": {"reminder_id": "tool-soak-audit"},
        "notify_desktop": {"title": "Tool soak audit", "message": "Synthetic notification"},
        "queue_work": {"title": "Tool soak audit", "description": "synthetic", "priority": -100, "tags": ["audit"]},
        "get_work_details": {"work_id": ids.get("work_id", "audit-work")},
        "get_work_result": {"work_id": ids.get("work_id", "audit-work")},
        "update_work_status": {"work_id": "audit-invalid-work", "status": "cancelled"},
        "get_task_info": {"task_id": ids.get("task_id", ids.get("job_id", "audit-task"))},
        "get_task_logs": {"task_id": ids.get("task_id", ids.get("job_id", "audit-task")), "limit": 5},
        "create_or_update_tool": {"tool_name": "audit_generated_tool", "specification": "Create a read-only tool that returns the string audit."},
        "read_tool_source": {"filename": "audit_fixture_tool.py"},
    }
    if target.name in exact:
        args = exact[target.name]
    return args


def _recipe_default_params(recipe: dict[str, Any], fixtures: dict[str, Any], ids: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    fixture_override_keys = {"old_id", "new_id", "path_or_url"}
    for key, spec in dict(recipe.get("parameters") or {}).items():
        key = str(key)
        if key in fixture_override_keys:
            params[key] = _required_value(key, spec if isinstance(spec, dict) else {}, fixtures, ids)
        elif isinstance(spec, dict) and "default" in spec:
            params[key] = spec["default"]
        else:
            params[key] = _required_value(key, spec if isinstance(spec, dict) else {}, fixtures, ids)

    # Catch $param references not declared in the recipe's parameters map.
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "$param" in value:
                key = str(value["$param"])
                if key not in params:
                    if "default" in value:
                        params[key] = value["default"]
                    else:
                        params[key] = _required_value(key, {}, fixtures, ids)
            for nested in value.values(): walk(nested)
        elif isinstance(value, list):
            for nested in value: walk(nested)
    walk(recipe.get("pipeline") or [])

    # Exercise meaningful alternate branches while keeping every pass valid.
    recipe_name = str(recipe.get("name") or "")
    if recipe_name == "compat.endpoint_probe":
        use_tls = bool(fixtures.get("endpoint_tls", False))
        params.update({"host": "example.com", "port": 443 if use_tls else 80, "tls": use_tls, "timeout": 5.0})
    elif recipe_name == "compat.http_probe":
        params.update({"url": fixtures.get("http_url", "https://example.com/"), "timeout": 8.0, "allow_private": False})
    elif recipe_name == "compat.read_feed":
        params.update({"url": fixtures.get("feed_url", "https://feeds.bbci.co.uk/news/rss.xml"), "limit": 5})
    elif recipe_name == "compat.read_host_file":
        params.update({"filepath": fixtures.get("host_file", "/etc/os-release")})
    return params


def contract_check(target: Target, fixtures: dict[str, Any], ids: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    if target.kind.startswith("recipe_"):
        recipe = target.recipe or {}
        pipeline = recipe.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            return False, "recipe_missing_pipeline", {}
        if len(pipeline) > 16:
            return False, "recipe_too_many_stages", {}
        return True, "ok", _recipe_default_params(recipe, fixtures, ids)

    schema = target.schema or {}
    function = schema.get("function")
    if not isinstance(function, dict) or str(function.get("name") or "") != target.name:
        return False, "invalid_schema", {}
    properties, required = _schema_parameters(target)
    missing_properties = [name for name in required if name not in properties]
    if missing_properties:
        return False, "required_not_in_properties:" + ",".join(missing_properties), {}
    args = tool_args(target, fixtures, ids)
    missing_args = [name for name in required if name not in args]
    if missing_args:
        return False, "probe_args_missing:" + ",".join(missing_args), args
    try:
        from tools import AVAILABLE_TOOLS_MAP
        from tools.tool_registry import normalize_arguments
        normalize_arguments(AVAILABLE_TOOLS_MAP[target.name], args)
    except Exception as exc:
        return False, f"argument_validation:{type(exc).__name__}:{exc}", args
    return True, "ok", args


def _mutation_action(target: Target, mode: str) -> str:
    if target.readonly:
        return "invoke"
    if mode == "all":
        return "invoke"
    if mode == "isolated" and target.name in ISOLATABLE_MUTATORS:
        return "invoke"
    return "contract_only"


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(token in lowered for token in ("password", "token", "secret", "api_key", "credential")):
        return "<redacted>"
    if isinstance(value, dict): return {k: _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list): return [_redact(v, key) for v in value]
    if isinstance(value, str):
        # Query strings frequently carry API keys/session identifiers. Preserve
        # enough URL shape for diagnostics without persisting their values.
        if "://" in value and "?" in value:
            try:
                parts = urlsplit(value)
                value = urlunsplit((parts.scheme, parts.netloc, parts.path, "<redacted-query>", ""))
            except ValueError:
                pass
        if len(value) > 500:
            return value[:500] + "…"
    return value


def inventory_target(target: Target) -> dict[str, Any]:
    """Return diagnostic inventory metadata without persisting recipe arguments/default secrets."""
    row = {"kind": target.kind, "name": target.name, "module": target.module, "readonly": target.readonly}
    if target.schema:
        fn = target.schema.get("function", {})
        params = fn.get("parameters", {}) or {}
        row["description"] = str(fn.get("description") or "")
        row["required"] = list(params.get("required") or [])
        row["parameters"] = sorted((params.get("properties") or {}).keys())
    if target.recipe:
        recipe = target.recipe
        row["recipe_origin"] = str(recipe.get("origin") or ("builtin" if target.kind == "recipe_builtin" else "user"))
        row["target_tool"] = str(recipe.get("target_tool") or "")
        row["stage_count"] = len(recipe.get("pipeline") or [])
        row["parameter_names"] = sorted(dict(recipe.get("parameters") or {}).keys())
    return row


def _worker_request(target: Target, args: dict[str, Any], fixtures: dict[str, Any], ids: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": {"kind": target.kind, "name": target.name, "module": target.module, "readonly": target.readonly, "recipe": target.recipe},
        "args": args, "fixtures": fixtures, "ids": ids,
    }


def _classify_environment(reason: str, preview: str) -> str:
    text = f"{reason} {preview}".lower()
    for category, hints in ERROR_HINTS.items():
        if any(hint in text for hint in hints):
            return category
    return "runtime"


def worker_main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    target = request.get("target") or {}
    name = str(target.get("name") or "")
    kind = str(target.get("kind") or "tool")
    args = dict(request.get("args") or {})
    started = time.perf_counter()
    result: dict[str, Any] = {"status": "error", "reason": "worker_failure", "output_preview": "", "output_bytes": 0}
    try:
        from tools import load_tools
        load_tools()
        _apply_soak_isolation()
        if kind.startswith("recipe_"):
            from tools.pipeline import execute_pipeline
            recipe = target.get("recipe") or {}
            payload = execute_pipeline(list(recipe.get("pipeline") or []), args)
            text = json.dumps(payload, ensure_ascii=False, default=str)
            stage_statuses = [str(row.get("status") or "") for row in payload.get("stages", []) if isinstance(row, dict)] if isinstance(payload, dict) else []
            if isinstance(payload, dict) and payload.get("ok"):
                status = "partial" if "partial" in stage_statuses or any(row.get("optional_failure") for row in payload.get("stages", []) if isinstance(row, dict)) else "success"
                reason = "recipe_ok" if status == "success" else "recipe_partial"
            else:
                status = "error"; reason = "recipe_failed"
            result.update(status=status, reason=reason, output_preview=text[:1200], output_bytes=len(text.encode("utf-8", errors="replace")))
        else:
            from tools import AVAILABLE_TOOLS_MAP
            from tools.executor import execute_registered_tool
            from tools.loop_validator import classify_tool_outcome
            # run_recipe needs a known recipe in the isolated DB so the primitive
            # itself is exercised rather than merely returning not-found.
            if name == "run_recipe" and args.get("name") == "__AUDIT_RECIPE__":
                from tools.recipe_store import save_recipe
                try:
                    save_recipe("__AUDIT_RECIPE__", "tool soak fixture", [{"id":"s1","tool":"calculate","args":{"expression":"1+1"}}])
                except ValueError:
                    pass
            normalized = args
            from tools.tool_registry import normalize_arguments
            normalized = normalize_arguments(AVAILABLE_TOOLS_MAP[name], args)
            raw = execute_registered_tool(name, normalized)
            if isinstance(raw, (dict, list)):
                text = json.dumps(raw, ensure_ascii=False, default=str)
            elif raw is None:
                text = ""
            else:
                text = str(raw)
            outcome = classify_tool_outcome(text, tool_name=name)
            status = "success" if outcome.get("status") == "ok" and outcome.get("success") else str(outcome.get("status") or "error")
            if status not in {"success", "partial", "error"}:
                status = "success" if outcome.get("success") else "error"
            result.update(status=status, reason=str(outcome.get("reason") or "ok"), output_preview=text[:1200], output_bytes=len(text.encode("utf-8", errors="replace")))
    except BaseException as exc:
        preview = f"{type(exc).__name__}: {exc}"
        result.update(status="error", reason="exception", output_preview=preview[:1200], output_bytes=len(preview.encode()))
        result["exception_type"] = type(exc).__name__
        result["traceback"] = traceback.format_exc(limit=8)[-4000:]
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    sys.stdout.write("\n" + WORKER_SENTINEL + json.dumps(result, ensure_ascii=False, default=str))
    return 0


def invoke_child(target: Target, args: dict[str, Any], fixtures: dict[str, Any], ids: dict[str, Any], env: dict[str, str], timeout: float) -> dict[str, Any]:
    request = _worker_request(target, args, fixtures, ids)
    started = time.perf_counter()
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=str(ROOT), start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(request, ensure_ascii=False), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Kill the whole process group, not just the Python worker. A tool may
        # have launched git/nmap/browser/PDF helpers that inherited our stdout
        # pipe; leaving one alive can otherwise wedge communicate() forever.
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            try: proc.kill()
            except OSError: pass
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return {
            "status": "timeout", "reason": "controller_timeout", "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "output_preview": str(stdout or "")[-1200:], "stderr_preview": str(stderr or "")[-1200:], "output_bytes": len(str(stdout or "").encode(errors="replace")),
        }
    if proc.returncode != 0:
        return {
            "status": "error", "reason": "worker_exit", "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "output_preview": stdout[-1200:], "stderr_preview": stderr[-1200:], "output_bytes": len(stdout.encode(errors="replace")), "worker_returncode": proc.returncode,
        }
    marker = stdout.rfind(WORKER_SENTINEL)
    if marker < 0:
        return {
            "status": "error", "reason": "worker_protocol", "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "output_preview": stdout[-1200:], "stderr_preview": stderr[-1200:], "output_bytes": len(stdout.encode(errors="replace")),
        }
    protocol_text = stdout[marker + len(WORKER_SENTINEL):].strip()
    try:
        payload = json.loads(protocol_text)
    except json.JSONDecodeError:
        return {
            "status": "error", "reason": "worker_protocol", "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "output_preview": stdout[-1200:], "stderr_preview": stderr[-1200:], "output_bytes": len(stdout.encode(errors="replace")),
        }
    incidental = stdout[:marker].strip()
    if incidental:
        payload["worker_stdout_preview"] = incidental[-1200:]
    if stderr.strip():
        payload["stderr_preview"] = stderr[-1200:]
    return payload


def _target_key(row: dict[str, Any]) -> str:
    return f"{row.get('kind')}:{row.get('name')}"


class MetricsAccumulator:
    """Compact exact aggregates for long-running soak tests.

    Raw rows remain append-only in results.jsonl; this object keeps only counters
    and latency samples needed for summaries, avoiding unbounded retention of
    output previews/arguments during 24-hour runs.
    """

    def __init__(self) -> None:
        self.probes = 0
        self.statuses: Counter[str] = Counter()
        self.latencies: list[float] = []
        self.reasons: Counter[str] = Counter()
        self.error_classes: Counter[str] = Counter()
        self.environment_blocked = 0
        self.kind_probes: Counter[str] = Counter()
        self.kind_statuses: dict[str, Counter[str]] = defaultdict(Counter)
        self.groups: dict[str, dict[str, Any]] = {}

    def add(self, row: dict[str, Any]) -> None:
        self.probes += 1
        status = str(row.get("status") or "unknown")
        kind = str(row.get("kind") or "unknown")
        self.statuses[status] += 1
        self.kind_probes[kind] += 1
        self.kind_statuses[kind][status] += 1
        if status in {"success", "partial", "error", "timeout"} and isinstance(row.get("latency_ms"), (int, float)):
            self.latencies.append(float(row["latency_ms"]))
        if status in {"error", "timeout", "contract_error"}:
            self.reasons[str(row.get("reason") or "unknown")] += 1
        error_class = str(row.get("error_class") or "")
        if error_class:
            self.error_classes[error_class] += 1
            if status == "error" and error_class in ENVIRONMENT_BLOCKING_CLASSES:
                self.environment_blocked += 1

        key = _target_key(row)
        group = self.groups.get(key)
        if group is None:
            group = {
                "target": key, "kind": row.get("kind"), "name": row.get("name"),
                "module": row.get("module", ""), "statuses": Counter(),
                "latencies": [], "last_reason": "",
            }
            self.groups[key] = group
        group["statuses"][status] += 1
        if status in {"success", "partial", "error", "timeout"} and isinstance(row.get("latency_ms"), (int, float)):
            group["latencies"].append(float(row["latency_ms"]))
        group["last_reason"] = str(row.get("reason") or "")

    def to_summary(self, inventory: dict[str, Any], started_at: str, elapsed_s: float) -> dict[str, Any]:
        attempted_statuses = {"success", "partial", "error", "timeout"}
        attempted = sum(self.statuses[s] for s in attempted_statuses)
        successish = self.statuses["success"] + self.statuses["partial"]
        harness_evaluable = max(0, attempted - self.environment_blocked)

        per_target: list[dict[str, Any]] = []
        for key, group in sorted(self.groups.items()):
            sc: Counter[str] = group["statuses"]
            attempted_n = sum(sc[s] for s in attempted_statuses)
            good = sc["success"] + sc["partial"]
            ls = list(group["latencies"])
            per_target.append({
                "target": key, "kind": group.get("kind"), "name": group.get("name"), "module": group.get("module", ""),
                "probes": sum(sc.values()), "attempted": attempted_n, "success": sc["success"], "partial": sc["partial"],
                "error": sc["error"], "timeout": sc["timeout"], "skipped": sc["skipped"], "contract_error": sc["contract_error"],
                "success_rate": round(good / attempted_n, 4) if attempted_n else None,
                "p50_ms": percentile(ls, .50), "p95_ms": percentile(ls, .95), "max_ms": round(max(ls), 3) if ls else None,
                "flaky": bool(good and (sc["error"] or sc["timeout"])),
                "last_reason": str(group.get("last_reason") or ""),
            })

        by_kind: dict[str, dict[str, Any]] = {}
        for kind in sorted(self.kind_probes):
            sc = self.kind_statuses[kind]
            at = sum(sc[s] for s in attempted_statuses)
            gd = sc["success"] + sc["partial"]
            by_kind[kind] = {
                "probes": self.kind_probes[kind], "attempted": at, "success": sc["success"], "partial": sc["partial"],
                "error": sc["error"], "timeout": sc["timeout"], "skipped": sc["skipped"],
                "success_rate": round(gd / at, 4) if at else None,
            }

        selected_targets = max(int(inventory.get("selected_targets") or 0), len(per_target))
        observed_targets = len(per_target)
        attempted_targets = sum(1 for item in per_target if item["attempted"] > 0)
        successful_targets = sum(1 for item in per_target if item["success"] + item["partial"] > 0)
        contract_failed_targets = sum(1 for item in per_target if item["contract_error"] > 0)
        contract_ok_targets = max(0, observed_targets - contract_failed_targets)
        never_succeeded = [x for x in per_target if x["attempted"] and x["success"] + x["partial"] == 0]
        flaky = [x for x in per_target if x["flaky"]]
        slowest = sorted([x for x in per_target if x["p95_ms"] is not None], key=lambda x: float(x["p95_ms"]), reverse=True)[:20]

        return {
            "generated_at": utc_now(), "started_at": started_at, "elapsed_seconds": round(elapsed_s, 3), "inventory": inventory,
            "totals": {
                "probes": self.probes, "attempted": attempted, **dict(self.statuses),
                "unique_targets": selected_targets, "observed_targets": observed_targets,
                "attempted_targets": attempted_targets, "successful_targets": successful_targets,
                "unprobed_targets": max(0, selected_targets - observed_targets),
                "runtime_target_coverage": round(attempted_targets / selected_targets, 4) if selected_targets else None,
                "contract_target_coverage": round(contract_ok_targets / selected_targets, 4) if selected_targets else None,
                "success_rate": round(successish / attempted, 4) if attempted else None,
                "environment_blocked": self.environment_blocked,
                "harness_evaluable": harness_evaluable,
                "harness_success_rate": round(successish / harness_evaluable, 4) if harness_evaluable else None,
                "latency_ms": {
                    "mean": round(statistics.fmean(self.latencies), 3) if self.latencies else None,
                    "p50": percentile(self.latencies, .5), "p95": percentile(self.latencies, .95),
                    "p99": percentile(self.latencies, .99), "max": round(max(self.latencies), 3) if self.latencies else None,
                },
            },
            "by_kind": by_kind,
            "failure_reasons": dict(self.reasons.most_common()),
            "error_classes": dict(self.error_classes.most_common()),
            "never_succeeded": never_succeeded, "flaky_targets": flaky, "slowest_targets": slowest, "targets": per_target,
        }


def summarize(rows: Iterable[dict[str, Any]], inventory: dict[str, Any], started_at: str, elapsed_s: float) -> dict[str, Any]:
    metrics = MetricsAccumulator()
    for row in rows:
        metrics.add(row)
    return metrics.to_summary(inventory, started_at, elapsed_s)

def render_markdown(summary: dict[str, Any]) -> str:
    total = summary["totals"]
    inv = summary.get("inventory", {})
    lines = [
        "# Tool / Primitive / Recipe Soak Summary", "",
        f"Generated: `{summary['generated_at']}`  ", f"Started: `{summary['started_at']}`  ", f"Elapsed: `{summary['elapsed_seconds']:.1f}s`", "",
        "## Overall", "",
        f"- Registered tools: **{inv.get('registered_tools', 0)}**",
        f"- Builtin recipes discovered: **{inv.get('builtin_recipes', 0)}**",
        f"- Saved user recipes discovered: **{inv.get('saved_recipes', 0)}**",
        f"- Probes recorded: **{total.get('probes', 0)}**; attempted runtime calls: **{total.get('attempted', 0)}**",
        f"- Contract/schema target coverage: **{(total.get('contract_target_coverage') or 0)*100:.2f}%**",
        f"- Runtime target coverage: **{total.get('attempted_targets', 0)}/{total.get('unique_targets', 0)} ({(total.get('runtime_target_coverage') or 0)*100:.2f}%)**",
        f"- Targets not yet probed: **{total.get('unprobed_targets', 0)}**",
        f"- Success: **{total.get('success', 0)}**; partial: **{total.get('partial', 0)}**; errors: **{total.get('error', 0)}**; timeouts: **{total.get('timeout', 0)}**; skipped/contract-only: **{total.get('skipped', 0)}**; contract errors: **{total.get('contract_error', 0)}**",
        f"- Runtime success rate: **{(total.get('success_rate') or 0)*100:.2f}%**",
        f"- Environment-blocked runtime probes: **{total.get('environment_blocked', 0)}**; harness-only success rate: **{(total.get('harness_success_rate') or 0)*100:.2f}%**",
        f"- Latency p50 / p95 / max: **{total.get('latency_ms',{}).get('p50')} / {total.get('latency_ms',{}).get('p95')} / {total.get('latency_ms',{}).get('max')} ms**", "",
        "## By kind", "", "| Kind | Attempted | Success | Partial | Error | Timeout | Success rate |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, row in summary.get("by_kind", {}).items():
        rate = "—" if row.get("success_rate") is None else f"{row['success_rate']*100:.2f}%"
        lines.append(f"| {kind} | {row['attempted']} | {row['success']} | {row['partial']} | {row['error']} | {row['timeout']} | {rate} |")

    lines += ["", "## Targets that never succeeded", "", "| Target | Attempts | Errors | Timeouts | Last reason |", "|---|---:|---:|---:|---|"]
    for row in summary.get("never_succeeded", [])[:50]:
        lines.append(f"| `{row['target']}` | {row['attempted']} | {row['error']} | {row['timeout']} | {row['last_reason']} |")
    if not summary.get("never_succeeded"): lines.append("| — | 0 | 0 | 0 | — |")

    lines += ["", "## Flaky targets", "", "| Target | Success rate | Errors | Timeouts | p95 ms |", "|---|---:|---:|---:|---:|"]
    for row in summary.get("flaky_targets", [])[:50]:
        lines.append(f"| `{row['target']}` | {(row['success_rate'] or 0)*100:.2f}% | {row['error']} | {row['timeout']} | {row['p95_ms']} |")
    if not summary.get("flaky_targets"): lines.append("| — | — | 0 | 0 | — |")

    lines += ["", "## Slowest targets by p95", "", "| Target | p50 ms | p95 ms | max ms |", "|---|---:|---:|---:|"]
    for row in summary.get("slowest_targets", [])[:20]:
        lines.append(f"| `{row['target']}` | {row['p50_ms']} | {row['p95_ms']} | {row['max_ms']} |")

    lines += ["", "## Failure reasons", ""]
    if summary.get("failure_reasons"):
        for reason, count in summary["failure_reasons"].items(): lines.append(f"- `{reason}`: {count}")
    else: lines.append("- None")
    lines += ["", "## Error classes", ""]
    if summary.get("error_classes"):
        for reason, count in summary["error_classes"].items(): lines.append(f"- `{reason}`: {count}")
    else: lines.append("- None")
    lines += ["", "Raw invocation records are in `results.jsonl`; the complete machine-readable aggregate is in `summary.json`.", ""]
    return "\n".join(lines)


def write_summary(report_dir: Path, metrics: MetricsAccumulator, inventory: dict[str, Any], started_at: str, started_mono: float, elapsed_offset: float = 0.0) -> dict[str, Any]:
    summary = metrics.to_summary(inventory, started_at, elapsed_offset + time.monotonic() - started_mono)
    temp = report_dir / ".summary.json.tmp"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(report_dir / "summary.json")
    (report_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    # Handy compact CSV for plotting in external tooling.
    with (report_dir / "targets.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["target","kind","name","module","probes","attempted","success","partial","error","timeout","skipped","contract_error","success_rate","p50_ms","p95_ms","max_ms","flaky","last_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in summary["targets"]: writer.writerow({k: row.get(k) for k in fields})
    return summary


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(fnmatch.fnmatch(name, pattern) or re.search(pattern, name) for pattern in include):
        return False
    if exclude and any(fnmatch.fnmatch(name, pattern) or re.search(pattern, name) for pattern in exclude):
        return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--duration", type=parse_duration, default=0.0, help="Optional hard outer duration; 0 disables the duration limit")
    parser.add_argument("--passes", type=int, default=5, help="Maximum complete passes; 0 means duration-only and requires --duration")
    parser.add_argument("--per-call-timeout", type=parse_duration, default=parse_duration("45s"), help="Controller timeout per child probe")
    parser.add_argument("--slow-call-timeout", type=parse_duration, default=parse_duration("90s"), help="Timeout used for known slow probes")
    parser.add_argument("--interval", type=float, default=0.05, help="Sleep between probe submissions/completions")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent isolated child probes; keep low for network/API tests")
    parser.add_argument("--checkpoint", type=parse_duration, default=parse_duration("60s"), help="Summary checkpoint interval")
    parser.add_argument("--mutating-mode", choices=["contract", "isolated", "all"], default="isolated", help="How mutating tools are tested")
    parser.add_argument("--no-recipes", action="store_true", help="Do not execute builtin/saved recipes")
    parser.add_argument("--no-saved-recipes", action="store_true", help="Do not include user-saved recipes from the selected recipe DB")
    parser.add_argument("--only", action="append", default=[], help="Glob/regex target-name filter; repeatable")
    parser.add_argument("--exclude", action="append", default=[], help="Glob/regex target-name exclusion; repeatable")
    parser.add_argument("--output-dir", default="", help="Report directory; defaults under workspace/tool_soak_reports")
    parser.add_argument("--resume", action="store_true", help="Resume metrics from an existing --output-dir/results.jsonl")
    parser.add_argument("--keep-output", action="store_true", help="Retain generated audit fixture/output files after completion")
    parser.add_argument("--fail-on-errors", action="store_true", help="Exit non-zero for harness/runtime errors or timeouts; credential/dependency/network blocks are excluded unless --fail-on-environment is set")
    parser.add_argument("--fail-on-environment", action="store_true", help="With --fail-on-errors, also fail for credential/dependency/network blocked probes")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def controller_main(args: argparse.Namespace) -> int:
    ok, missing = _preflight_repository_dependencies()
    if not ok:
        print(_dependency_preflight_message(missing), file=sys.stderr)
        return 2
    runtime_ok, runtime_message = _runtime_context_preflight()
    if not runtime_ok:
        print(runtime_message, file=sys.stderr)
        return 2

    workspace = _workspace_root()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (workspace / "tool_soak_reports" / stamp)
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / "results.jsonl"
    if results_path.exists() and not args.resume:
        print(f"Refusing to append to existing {results_path}; use --resume or choose another --output-dir.", file=sys.stderr)
        return 2

    # Workspace primitives still intentionally enforce /app/workspace. Keep all
    # file mutations inside one disposable audit subtree, with a fresh fixture
    # directory for every pass so state/file probes do not contaminate later runs.
    fixture_root = workspace / "tool_soak" / stamp
    fixture_root.mkdir(parents=True, exist_ok=True)
    env = build_env(report_dir, workspace)

    # Read user recipes directly before importing tools. This is important: the
    # tools package imports runtime DB settings at module-import time. Switching
    # to the isolated audit DB before that import guarantees probes cannot write
    # memories/jobs/profile state into the live harness database.
    active_recipe_db = os.environ.get("AGENT_RECIPE_DB", "/app/memory/recipes.db")
    saved_recipes = [] if args.no_saved_recipes else read_saved_recipes_direct(active_recipe_db)
    os.environ.update({k: v for k, v in env.items() if k.startswith("AGENT_") or k == "TOOL_SOAK_ACTIVE"})
    try:
        targets, inventory = discover_targets(saved_recipes=saved_recipes)
    except ModuleNotFoundError as exc:
        missing_module = str(exc.name or "unknown")
        print(
            _dependency_preflight_message([missing_module])
            + "\n\nThe missing import was raised while loading the live tool registry. "
              "If bootstrap_venv.sh succeeds but this remains missing, add the owning "
              "distribution to requirements.txt.",
            file=sys.stderr,
        )
        return 2
    inventory.update({
        "seed_state": {"strategy": "fresh_per_pass"}, "mutating_mode": args.mutating_mode, "report_dir": str(report_dir),
        "harness_workspace": str(workspace), "audit_workspace": str(fixture_root),
        "source_recipe_db": active_recipe_db, "workers": max(1, int(args.workers)),
        "pass_profiles": [str(profile["label"]) for profile in PASS_PROFILES],
    })

    if args.no_recipes:
        targets = [t for t in targets if not t.kind.startswith("recipe_")]
    targets = [t for t in targets if _matches(t.name, args.only, args.exclude)]
    targets.sort(key=lambda t: (0 if t.kind == "primitive" else 1 if t.kind == "tool" else 2, t.name.lower()))
    inventory["selected_targets"] = len(targets)
    inventory["selected_primitives"] = sum(1 for t in targets if t.kind == "primitive")
    inventory["selected_tools"] = sum(1 for t in targets if t.kind == "tool")
    inventory["selected_recipes"] = sum(1 for t in targets if t.kind.startswith("recipe_"))
    inventory["selected_target_keys"] = [f"{t.kind}:{t.name}" for t in targets]
    if not targets:
        print("No targets matched the requested filters.", file=sys.stderr)
        return 2

    (report_dir / "inventory.json").write_text(
        json.dumps({"inventory": inventory, "targets": [inventory_target(t) for t in targets]}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    metrics = MetricsAccumulator()
    prior_elapsed = 0.0
    started_at = utc_now()
    last_pass = 0
    completed_last: set[str] = set()
    if args.resume and results_path.is_file():
        # Stream prior JSONL instead of retaining every output/argument row in
        # memory.  This matters for all-day runs with many repeated passes.
        with results_path.open("r", encoding="utf-8", errors="replace") as prior:
            for line in prior:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed during its final append can leave one
                    # partial trailing line.  Earlier durable rows remain valid.
                    continue
                if not isinstance(row, dict):
                    continue
                metrics.add(row)
                try:
                    pass_no = int(row.get("pass") or 0)
                except (TypeError, ValueError):
                    pass_no = 0
                if pass_no > last_pass:
                    last_pass = pass_no
                    completed_last = set()
                if pass_no == last_pass and pass_no > 0:
                    # A child forcibly stopped only because the overall soak
                    # window expired was not a fair runtime probe. Leave it
                    # pending so --resume can test it normally.
                    deadline_deferred = row.get("status") == "skipped" and row.get("reason") == "soak_deadline"
                    if not deadline_deferred:
                        completed_last.add(_target_key(row))
        summary_path = report_dir / "summary.json"
        if summary_path.is_file():
            try:
                old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                started_at = str(old_summary.get("started_at") or started_at)
                prior_elapsed = float(old_summary.get("elapsed_seconds") or 0.0)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    if int(args.passes) < 0:
        print("--passes must be >= 0.", file=sys.stderr)
        return 2
    if int(args.passes) == 0 and float(args.duration) <= 0:
        print("At least one stopping condition is required: use --passes > 0 or --duration > 0.", file=sys.stderr)
        return 2

    started_mono = time.monotonic()
    duration_limit = float(args.duration)
    deadline = started_mono + duration_limit if duration_limit > 0 else None
    stop = {"requested": False, "signal": ""}

    def within_deadline() -> bool:
        return deadline is None or time.monotonic() < deadline

    def request_stop(signum, _frame):
        stop["requested"] = True
        stop["signal"] = signal.Signals(signum).name
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    workers = max(1, int(args.workers))
    duration_label = "none" if deadline is None else f"{duration_limit:.0f}s"
    pass_label = "duration-only" if not args.passes else str(int(args.passes))
    print(f"Tool soak: {len(targets)} targets; passes={pass_label}; duration={duration_label}; mutating={args.mutating_mode}; workers={workers}")
    print(f"Reports: {report_dir}")
    if metrics.probes:
        print(f"Resumed {metrics.probes} prior probe records.")
    if args.mutating_mode == "all":
        print("WARNING: --mutating-mode all may change package/system/external state.", file=sys.stderr)

    all_entries = list(enumerate(targets, start=1))
    next_pass = 1 if not last_pass else last_pass + 1
    resume_entries: list[tuple[int, Target]] | None = None
    if args.resume and last_pass:
        missing = [entry for entry in all_entries if f"{entry[1].kind}:{entry[1].name}" not in completed_last]
        if missing:
            next_pass = last_pass
            resume_entries = missing
            print(f"Resuming incomplete pass {last_pass}: {len(missing)} targets remain.")
    last_checkpoint = time.monotonic()

    with results_path.open("a", encoding="utf-8", buffering=1) as log:
        def record(row: dict[str, Any], target: Target, index: int) -> None:
            nonlocal last_checkpoint
            metrics.add(row)
            log.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            icon = {"success":"OK", "partial":"PART", "error":"ERR", "timeout":"TIME", "skipped":"SKIP", "contract_error":"CONTRACT"}.get(str(row.get("status")), "?")
            print(f"[{index:03d}/{len(targets):03d}] {icon:8s} {target.kind:14s} {target.name:32.32s} {float(row.get('latency_ms') or 0):8.1f} ms  {row.get('reason','')}")
            now = time.monotonic()
            if now - last_checkpoint >= float(args.checkpoint):
                checkpoint = write_summary(report_dir, metrics, inventory, started_at, started_mono, prior_elapsed)
                print(f"-- checkpoint: probes={checkpoint['totals']['probes']} success_rate={(checkpoint['totals'].get('success_rate') or 0)*100:.1f}% --")
                last_checkpoint = now

        while not stop["requested"] and within_deadline():
            pass_no = next_pass
            if args.passes and pass_no > args.passes:
                break
            entries = resume_entries if resume_entries is not None else all_entries
            resume_entries = None
            next_pass = pass_no + 1
            suffix = " (resume)" if len(entries) != len(all_entries) else ""

            pass_dir = fixture_root / f"pass-{pass_no:03d}"
            fixtures = create_fixtures(pass_dir, workspace, pass_no=pass_no)
            ids = _seed_isolated_state(env, pass_no=pass_no)
            process_fixture: subprocess.Popen | None = None
            try:
                process_fixture = _start_process_fixture(pass_dir, pass_no)
                ids["process_pid"] = int(process_fixture.pid)
            except Exception as exc:
                ids["process_fixture_error"] = f"{type(exc).__name__}: {exc}"

            print(f"\n=== pass {pass_no}{suffix} [{fixtures.get('profile', 'default')}] ===")
            cursor = 0
            try:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool-soak") as pool:
                    # Submit only one worker-sized batch at a time. This prevents a
                    # Ctrl-C or duration deadline from leaving hundreds of queued
                    # probes that still have to run before shutdown.
                    while cursor < len(entries) and not stop["requested"] and within_deadline():
                        batch = entries[cursor:cursor + workers]
                        cursor += len(batch)
                        pending: dict[Any, tuple[Target, int, dict[str, Any], bool]] = {}
                        for index, target in batch:
                            contract_ok, contract_reason, probe_args = contract_check(target, fixtures, ids)
                            base = {
                                "timestamp": utc_now(), "pass": pass_no, "fixture_profile": fixtures.get("profile", ""),
                                "ordinal": index, "kind": target.kind, "name": target.name, "module": target.module,
                                "readonly": target.readonly, "args": _redact(probe_args),
                            }
                            if not contract_ok:
                                record({**base, "status": "contract_error", "reason": contract_reason, "latency_ms": 0.0, "output_bytes": 0, "output_preview": ""}, target, index)
                            else:
                                action = "invoke" if target.kind.startswith("recipe_") else _mutation_action(target, args.mutating_mode)
                                if action == "contract_only":
                                    record({**base, "status": "skipped", "reason": "mutating_contract_only", "latency_ms": 0.0, "output_bytes": 0, "output_preview": ""}, target, index)
                                else:
                                    desired_timeout = max(float(args.per_call_timeout), float(args.slow_call_timeout) if target.name in SLOW_TOOLS else 0.0)
                                    if deadline is None:
                                        timeout = desired_timeout
                                        deadline_limited = False
                                    else:
                                        remaining = max(0.1, deadline - time.monotonic())
                                        timeout = min(desired_timeout, remaining)
                                        deadline_limited = timeout + 0.001 < desired_timeout
                                    future = pool.submit(invoke_child, target, probe_args, fixtures, ids, env, timeout)
                                    pending[future] = (target, index, base, deadline_limited)
                            if args.interval > 0:
                                time.sleep(args.interval)

                        for future in as_completed(pending):
                            target, index, base, deadline_limited = pending[future]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                result = {"status":"error", "reason":"controller_exception", "latency_ms":0.0, "output_bytes":0, "output_preview":f"{type(exc).__name__}: {exc}"}
                            row = {**base, **result}
                            if row.get("status") == "timeout" and deadline_limited:
                                row["status"] = "skipped"
                                row["reason"] = "soak_deadline"
                                row.pop("error_class", None)
                            elif row.get("status") == "timeout":
                                row["error_class"] = "timeout"
                            elif row.get("status") == "error":
                                row["error_class"] = _classify_environment(
                                    str(row.get("reason") or ""),
                                    str(row.get("output_preview") or "") + " " + str(row.get("stderr_preview") or ""),
                                )
                            record(row, target, index)
                        if args.interval > 0 and pending:
                            time.sleep(args.interval)
            finally:
                _stop_process_fixture(process_fixture)

            if args.passes and pass_no >= args.passes:
                break

    summary = write_summary(report_dir, metrics, inventory, started_at, started_mono, prior_elapsed)
    summary["stop_signal"] = stop["signal"]
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== final summary ===")
    print(render_markdown(summary))
    if not args.keep_output:
        shutil.rmtree(fixture_root, ignore_errors=True)
    if summary["totals"].get("contract_error", 0):
        return 1
    if args.fail_on_errors:
        timeout_count = int(summary["totals"].get("timeout", 0) or 0)
        error_count = int(summary["totals"].get("error", 0) or 0)
        blocked = int(summary["totals"].get("environment_blocked", 0) or 0)
        effective_errors = error_count if args.fail_on_environment else max(0, error_count - blocked)
        if timeout_count or effective_errors:
            return 1
    return 0

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _maybe_reexec_in_repo_venv(argv)
    if args._worker:
        # Internal children inherit the already-validated controller interpreter.
        # Avoid repeating distribution scans for every probe.
        return worker_main()
    return controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
