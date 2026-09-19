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


def _semantic_text(name: str, description: str, tags: list[str], tools: list[str]) -> str:
    return " ".join([name, description, " ".join(tags), " ".join(tools)]).strip()


def pipeline_fingerprint(pipeline: list[dict[str, Any]]) -> str:
    normalized = []
    for stage in pipeline:
        normalized.append({"tool": str(stage.get("tool") or ""), "args": stage.get("args") or {}})
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
                last_used_at TEXT
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
        # FTS5 gives local semantic-ish lookup without an embedding-model call.
        # Fall back to a normal mirror table on unusually small SQLite builds so
        # recipe persistence never prevents the agent from starting.
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(name, description, semantic_text, tags)")
        except sqlite3.OperationalError:
            conn.execute("CREATE TABLE IF NOT EXISTS recipes_fts(rowid INTEGER PRIMARY KEY, name TEXT, description TEXT, semantic_text TEXT, tags TEXT)")
        conn.commit()


def _row_to_recipe(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "name": row["name"], "description": row["description"],
        "pipeline": json.loads(row["pipeline_json"]), "parameters": json.loads(row["parameters_json"]),
        "tags": json.loads(row["tags_json"]), "fingerprint": row["fingerprint"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "success_count": row["success_count"], "use_count": row["use_count"], "last_used_at": row["last_used_at"],
    }


def save_recipe(name: str, description: str, pipeline: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    init_recipe_store(); name = re.sub(r"[^a-zA-Z0-9 _.-]+", "", str(name)).strip()[:80]
    if not name: raise ValueError("recipe name is required")
    parameters = parameters or {}; tags = [str(x)[:40] for x in (tags or [])[:20]]
    tools = [str(s.get("tool") or "") for s in pipeline]
    semantic = _semantic_text(name, description, tags, tools); fp = pipeline_fingerprint(pipeline); now = _now()
    with _DB_LOCK, _connect() as conn:
        existing = conn.execute("SELECT id FROM recipes WHERE name=?", (name,)).fetchone()
        if existing:
            rid = int(existing["id"])
            conn.execute("UPDATE recipes SET description=?, semantic_text=?, pipeline_json=?, parameters_json=?, tags_json=?, fingerprint=?, updated_at=? WHERE id=?",
                         (description, semantic, json.dumps(pipeline), json.dumps(parameters), json.dumps(tags), fp, now, rid))
            conn.execute("DELETE FROM recipes_fts WHERE rowid=?", (rid,))
        else:
            cur=conn.execute("INSERT INTO recipes(name,description,semantic_text,pipeline_json,parameters_json,tags_json,fingerprint,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                             (name,description,semantic,json.dumps(pipeline),json.dumps(parameters),json.dumps(tags),fp,now,now)); rid=int(cur.lastrowid)
        conn.execute("INSERT INTO recipes_fts(rowid,name,description,semantic_text,tags) VALUES(?,?,?,?,?)", (rid,name,description,semantic," ".join(tags)))
        conn.commit(); row=conn.execute("SELECT * FROM recipes WHERE id=?",(rid,)).fetchone()
    return _row_to_recipe(row)


def list_recipes(limit: int = 50) -> list[dict[str, Any]]:
    init_recipe_store(); limit=max(1,min(int(limit),200))
    with _connect() as conn:
        return [_row_to_recipe(r) for r in conn.execute("SELECT * FROM recipes ORDER BY use_count DESC, updated_at DESC LIMIT ?",(limit,)).fetchall()]


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
            rows=conn.execute("SELECT r.*, bm25(recipes_fts) AS rank FROM recipes_fts JOIN recipes r ON r.id=recipes_fts.rowid WHERE recipes_fts MATCH ? ORDER BY rank LIMIT ?",(fts_query,limit*2)).fetchall()
        except sqlite3.OperationalError:
            rows=conn.execute("SELECT * FROM recipes ORDER BY updated_at DESC LIMIT ?",(limit*2,)).fetchall()
    q=_tokens(query); scored=[]
    for row in rows:
        recipe=_row_to_recipe(row); r=_tokens(" ".join([recipe["name"],recipe["description"]," ".join(recipe["tags"])]))
        overlap=len(q&r)/max(1,len(q|r)); recipe["semantic_score"]=round(overlap,3); scored.append(recipe)
    return sorted(scored,key=lambda x:(-x["semantic_score"],-x["use_count"]))[:limit]


def recipe_exists_for_task(objective: str, pipeline: list[dict[str, Any]]) -> bool:
    init_recipe_store(); fp=pipeline_fingerprint(pipeline)
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM recipes WHERE fingerprint=? LIMIT 1",(fp,)).fetchone(): return True
    matches=search_recipes(objective,limit=3)
    # Conservative semantic duplicate cutoff: prompt only when no close workflow exists.
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
