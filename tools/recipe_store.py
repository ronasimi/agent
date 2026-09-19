"""Dedicated semantic recipe storage for reusable successful workflows."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_LOCK = threading.RLock()
_DEFAULT_DB = "/app/memory/recipes.db"


def _db_path() -> str:
    return os.environ.get("AGENT_RECIPE_DB", _DEFAULT_DB)


def _connect() -> sqlite3.Connection:
    path = Path(_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]{2,}", str(text).lower()) if x not in {"the","and","for","with","from","this","that","into","using","then"}}


def _semantic_text(name: str, description: str, tags: list[str], tools: list[str], target_tool: str = "") -> str:
    return " ".join([name, description, target_tool, " ".join(tags), " ".join(tools)]).strip()


def pipeline_fingerprint(pipeline: list[dict[str, Any]]) -> str:
    # Include composition controls as well as tool/args so conditional/foreach
    # recipes do not collide with simpler linear pipelines.
    normalized = []
    for stage in pipeline:
        normalized.append({
            "tool": str(stage.get("tool") or ""),
            "args": stage.get("args") or {},
            "when": stage.get("when"),
            "foreach": stage.get("foreach"),
            "optional": bool(stage.get("optional", False)),
        })
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_recipe_store() -> None:
    with _DB_LOCK, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                semantic_text TEXT NOT NULL,
                pipeline_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                tags_json TEXT NOT NULL DEFAULT '[]',
                fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                use_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT,
                origin TEXT NOT NULL DEFAULT 'user',
                builtin_key TEXT,
                builtin_version INTEGER NOT NULL DEFAULT 0,
                target_tool TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_recipes_fingerprint ON recipes(fingerprint);
            CREATE TABLE IF NOT EXISTS recipe_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                objective TEXT NOT NULL,
                semantic_text TEXT NOT NULL,
                pipeline_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                tags_json TEXT NOT NULL DEFAULT '[]',
                fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_recipe_candidates_status ON recipe_candidates(status, created_at);
            """
        )
        # Migrate databases created before builtin compatibility recipes existed.
        _ensure_column(conn, "recipes", "origin", "TEXT NOT NULL DEFAULT 'user'")
        _ensure_column(conn, "recipes", "builtin_key", "TEXT")
        _ensure_column(conn, "recipes", "builtin_version", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "recipes", "target_tool", "TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_recipes_builtin_key ON recipes(builtin_key) WHERE builtin_key IS NOT NULL")
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(name, description, semantic_text, tags)")
        except sqlite3.OperationalError:
            conn.execute("CREATE TABLE IF NOT EXISTS recipes_fts(rowid INTEGER PRIMARY KEY, name TEXT, description TEXT, semantic_text TEXT, tags TEXT)")
        conn.commit()


def _row_to_recipe(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())
    return {
        "id": row["id"], "name": row["name"], "description": row["description"],
        "pipeline": json.loads(row["pipeline_json"]), "parameters": json.loads(row["parameters_json"]),
        "tags": json.loads(row["tags_json"]), "fingerprint": row["fingerprint"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "success_count": row["success_count"], "use_count": row["use_count"], "last_used_at": row["last_used_at"],
        "origin": row["origin"] if "origin" in keys else "user",
        "builtin_key": row["builtin_key"] if "builtin_key" in keys else None,
        "builtin_version": row["builtin_version"] if "builtin_version" in keys else 0,
        "target_tool": row["target_tool"] if "target_tool" in keys else "",
    }


def _write_fts(conn: sqlite3.Connection, rid: int, name: str, description: str, semantic: str, tags: list[str]) -> None:
    conn.execute("DELETE FROM recipes_fts WHERE rowid=?", (rid,))
    conn.execute("INSERT INTO recipes_fts(rowid,name,description,semantic_text,tags) VALUES(?,?,?,?,?)", (rid, name, description, semantic, " ".join(tags)))


def save_recipe(name: str, description: str, pipeline: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    init_recipe_store(); name = re.sub(r"[^a-zA-Z0-9 _.-]+", "", str(name)).strip()[:80]
    if not name: raise ValueError("recipe name is required")
    parameters = parameters or {}; tags = [str(x)[:40] for x in (tags or [])[:20]]
    tools = [str(s.get("tool") or "") for s in pipeline]
    semantic = _semantic_text(name, description, tags, tools); fp = pipeline_fingerprint(pipeline); now = _now()
    with _DB_LOCK, _connect() as conn:
        existing = conn.execute("SELECT id, origin FROM recipes WHERE name=?", (name,)).fetchone()
        if existing and existing["origin"] == "builtin":
            raise ValueError("builtin compatibility recipe names are reserved")
        if existing:
            rid = int(existing["id"])
            conn.execute("UPDATE recipes SET description=?, semantic_text=?, pipeline_json=?, parameters_json=?, tags_json=?, fingerprint=?, updated_at=?, origin='user', builtin_key=NULL, builtin_version=0, target_tool='' WHERE id=?",
                         (description, semantic, json.dumps(pipeline), json.dumps(parameters), json.dumps(tags), fp, now, rid))
        else:
            cur = conn.execute("INSERT INTO recipes(name,description,semantic_text,pipeline_json,parameters_json,tags_json,fingerprint,created_at,updated_at,origin,builtin_version,target_tool) VALUES(?,?,?,?,?,?,?,?,?,'user',0,'')",
                               (name, description, semantic, json.dumps(pipeline), json.dumps(parameters), json.dumps(tags), fp, now, now)); rid = int(cur.lastrowid)
        _write_fts(conn, rid, name, description, semantic, tags)
        conn.commit(); row = conn.execute("SELECT * FROM recipes WHERE id=?", (rid,)).fetchone()
    return _row_to_recipe(row)


def save_builtin_recipe(*, key: str, version: int, name: str, target_tool: str, description: str, pipeline: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    """Idempotently seed/update one harness-owned compatibility recipe."""
    init_recipe_store(); parameters = parameters or {}; tags = [str(x)[:40] for x in (tags or [])[:20]]
    tools = [str(s.get("tool") or "") for s in pipeline]
    semantic = _semantic_text(name, description, tags, tools, target_tool); fp = pipeline_fingerprint(pipeline); now = _now()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM recipes WHERE builtin_key=?", (key,)).fetchone()
        if row is None:
            # Do not overwrite a user recipe that happens to use the same display name.
            conflict = conn.execute("SELECT id FROM recipes WHERE name=?", (name,)).fetchone()
            if conflict:
                name = f"{name} [builtin]"[:80]
            cur = conn.execute("INSERT INTO recipes(name,description,semantic_text,pipeline_json,parameters_json,tags_json,fingerprint,created_at,updated_at,origin,builtin_key,builtin_version,target_tool) VALUES(?,?,?,?,?,?,?,?,?,'builtin',?,?,?)",
                               (name, description, semantic, json.dumps(pipeline), json.dumps(parameters), json.dumps(tags), fp, now, now, key, int(version), target_tool)); rid = int(cur.lastrowid)
        else:
            rid = int(row["id"])
            if int(row["builtin_version"] or 0) <= int(version) or row["fingerprint"] != fp:
                conn.execute("UPDATE recipes SET name=?, description=?, semantic_text=?, pipeline_json=?, parameters_json=?, tags_json=?, fingerprint=?, updated_at=?, origin='builtin', builtin_version=?, target_tool=? WHERE id=?",
                             (name, description, semantic, json.dumps(pipeline), json.dumps(parameters), json.dumps(tags), fp, now, int(version), target_tool, rid))
        _write_fts(conn, rid, name, description, semantic, tags)
        conn.commit(); row = conn.execute("SELECT * FROM recipes WHERE id=?", (rid,)).fetchone()
    return _row_to_recipe(row)


def list_recipes(limit: int = 50) -> list[dict[str, Any]]:
    init_recipe_store(); limit=max(1,min(int(limit),200))
    with _connect() as conn:
        return [_row_to_recipe(r) for r in conn.execute("SELECT * FROM recipes ORDER BY CASE origin WHEN 'user' THEN 0 ELSE 1 END, use_count DESC, updated_at DESC LIMIT ?",(limit,)).fetchall()]


def get_recipe(name_or_id: str | int) -> dict[str, Any] | None:
    init_recipe_store()
    with _connect() as conn:
        if str(name_or_id).isdigit(): row=conn.execute("SELECT * FROM recipes WHERE id=?",(int(name_or_id),)).fetchone()
        else: row=conn.execute("SELECT * FROM recipes WHERE lower(name)=lower(?)",(str(name_or_id),)).fetchone()
        return _row_to_recipe(row) if row else None


def search_recipes(query: str, limit: int = 8) -> list[dict[str, Any]]:
    init_recipe_store(); limit=max(1,min(int(limit),20)); words=list(_tokens(query))
    if not words: return []
    fts_query=" OR ".join(f'"{w}"' for w in words[:16])
    with _connect() as conn:
        try:
            rows=conn.execute("SELECT r.*, bm25(recipes_fts) AS rank FROM recipes_fts JOIN recipes r ON r.id=recipes_fts.rowid WHERE recipes_fts MATCH ? ORDER BY rank LIMIT ?",(fts_query,limit*3)).fetchall()
        except sqlite3.OperationalError:
            rows=conn.execute("SELECT * FROM recipes ORDER BY updated_at DESC LIMIT ?",(limit*3,)).fetchall()
    q=_tokens(query); scored=[]
    for row in rows:
        recipe=_row_to_recipe(row); r=_tokens(" ".join([recipe["name"],recipe["description"],recipe.get("target_tool", "")," ".join(recipe["tags"])]))
        overlap=len(q&r)/max(1,len(q|r)); recipe["semantic_score"]=round(overlap,3); scored.append(recipe)
    return sorted(scored,key=lambda x:(-x["semantic_score"], 0 if x.get("origin")=="user" else 1, -x["use_count"]))[:limit]


def check_recipes_for_task(query: str, threshold: float = 0.35, limit: int = 3) -> dict[str, Any]:
    """Deterministically search saved recipes before a model plans a user request."""
    report: dict[str, Any] = {
        "status": "checked", "checked": True, "query": str(query or "")[:1000],
        "threshold": float(threshold), "candidates": [], "relevant": [], "error": "",
    }
    try:
        matches = search_recipes(str(query or ""), limit=max(1, min(int(limit), 8)))
    except Exception as exc:
        report.update({"status": "error", "checked": False, "error": str(exc)[:500]})
        return report
    for item in matches:
        candidate = {
            "name": str(item.get("name") or "")[:120],
            "description": str(item.get("description") or "")[:500],
            "semantic_score": float(item.get("semantic_score") or 0.0),
            "origin": str(item.get("origin") or "user"),
            "target_tool": str(item.get("target_tool") or ""),
            "parameters": dict(item.get("parameters") or {}),
        }
        report["candidates"].append(candidate)
        if candidate["semantic_score"] >= float(threshold):
            report["relevant"].append(candidate)
    return report


def render_recipe_preflight(report: dict[str, Any]) -> str:
    """Render a compact, trusted control note that makes recipe consideration auditable."""
    if not report.get("checked"):
        detail = str(report.get("error") or "unknown recipe-store error")[:300]
        return (
            "[Harness recipe preflight]\n"
            f"The automatic saved-recipe check failed: {detail}. "
            "Use search_recipes before planning a multi-step workflow if that tool is supplied."
        )
    relevant = list(report.get("relevant") or [])
    if not relevant:
        return (
            "[Harness recipe preflight]\n"
            "Saved recipes were checked for this request and no candidate met the configured relevance threshold. "
            "Proceed with the smallest suitable primitives; do not repeat the same recipe search unless the user explicitly asks to browse recipes."
        )
    rows = []
    for item in relevant[:3]:
        name = json.dumps(str(item.get("name") or ""), ensure_ascii=False)
        description = json.dumps(str(item.get("description") or ""), ensure_ascii=False)
        parameters = json.dumps(item.get("parameters") or {}, ensure_ascii=False, separators=(",", ":"))[:500]
        rows.append(
            f"- name={name}; score={item['semantic_score']:.3f}; origin={item['origin']}; "
            f"parameters={parameters}; description={description}"
        )
    return (
        "[Harness recipe preflight]\n"
        "Saved recipes were checked before task planning. Consider the compatible candidates below before composing manual steps. "
        "Use run_recipe only when its workflow and parameters fit the current request and policy; otherwise use primitives.\n"
        + "\n".join(rows)
    )


def recipe_exists_for_task(objective: str, pipeline: list[dict[str, Any]]) -> bool:
    init_recipe_store(); fp=pipeline_fingerprint(pipeline)
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM recipes WHERE fingerprint=? LIMIT 1",(fp,)).fetchone(): return True
    matches=search_recipes(objective,limit=3)
    return bool(matches and matches[0].get("semantic_score",0)>=0.58)


def create_candidate(objective: str, pipeline: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> int:
    init_recipe_store(); fp=pipeline_fingerprint(pipeline); tools=[str(s.get("tool") or "") for s in pipeline]; tags=tags or []
    semantic=_semantic_text(objective, objective, tags, tools)
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE recipe_candidates SET status='expired' WHERE status='pending'")
        cur=conn.execute("INSERT INTO recipe_candidates(objective,semantic_text,pipeline_json,parameters_json,tags_json,fingerprint,created_at,status) VALUES(?,?,?,?,?,?,?,'pending')",
                         (objective,semantic,json.dumps(pipeline),json.dumps(parameters or {}),json.dumps(tags),fp,_now()))
        conn.commit(); return int(cur.lastrowid)


def pending_candidate() -> dict[str, Any] | None:
    init_recipe_store()
    with _connect() as conn:
        row=conn.execute("SELECT * FROM recipe_candidates WHERE status='pending' ORDER BY id DESC LIMIT 1").fetchone()
        if not row:return None
        return {"id":row["id"],"objective":row["objective"],"pipeline":json.loads(row["pipeline_json"]),"parameters":json.loads(row["parameters_json"]),"tags":json.loads(row["tags_json"]),"fingerprint":row["fingerprint"]}


def save_pending_candidate(name: str = "", description: str = "") -> dict[str, Any] | None:
    candidate=pending_candidate()
    if not candidate:return None
    default_name=re.sub(r"\s+"," ",candidate["objective"]).strip()[:60] or f"Recipe {candidate['id']}"
    recipe=save_recipe(name or default_name, description or candidate["objective"], candidate["pipeline"], candidate["parameters"], candidate["tags"])
    with _connect() as conn: conn.execute("UPDATE recipe_candidates SET status='saved' WHERE id=?",(candidate["id"],)); conn.commit()
    return recipe


def dismiss_pending_candidate() -> None:
    init_recipe_store()
    with _connect() as conn: conn.execute("UPDATE recipe_candidates SET status='dismissed' WHERE status='pending'"); conn.commit()


def mark_recipe_used(recipe_id: int, success: bool = True) -> None:
    init_recipe_store()
    with _connect() as conn:
        conn.execute("UPDATE recipes SET use_count=use_count+1, success_count=success_count+?, last_used_at=?, updated_at=? WHERE id=?",(1 if success else 0,_now(),_now(),int(recipe_id))); conn.commit()
