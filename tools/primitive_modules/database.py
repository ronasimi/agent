from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def db_tables(database: str) -> str:
    """List tables/views in an allowed SQLite database under /app/memory or /app/workspace."""
    import sqlite3
    try:
        raw=Path(database).resolve(); allowed=[Path("/app/memory").resolve(),Path("/app/workspace").resolve()]
        if not any(os.path.commonpath([str(a),str(raw)])==str(a) for a in allowed):return "Error: database path is outside allowed roots."
        con=sqlite3.connect(f"file:{raw}?mode=ro",uri=True); rows=con.execute("SELECT name,type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name").fetchall(); con.close(); return _json([{"name":r[0],"type":r[1]} for r in rows])
    except Exception as exc:return f"Error: db_tables failed: {exc}"

def db_schema(database: str, table: str) -> str:
    """Return column metadata for one table in an allowed read-only SQLite database."""
    import sqlite3
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}",table):return "Error: invalid table name."
    try:
        raw=Path(database).resolve(); allowed=[Path("/app/memory").resolve(),Path("/app/workspace").resolve()]
        if not any(os.path.commonpath([str(a),str(raw)])==str(a) for a in allowed):return "Error: database path is outside allowed roots."
        con=sqlite3.connect(f"file:{raw}?mode=ro",uri=True); rows=con.execute(f'PRAGMA table_info("{table}")').fetchall(); con.close(); return _json([{"cid":r[0],"name":r[1],"type":r[2],"notnull":r[3],"default":r[4],"pk":r[5]} for r in rows])
    except Exception as exc:return f"Error: db_schema failed: {exc}"

def db_select(database: str, table: str, limit: int = 100) -> str:
    """Select bounded rows from one allowed SQLite table; no arbitrary SQL is accepted."""
    import sqlite3
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}",table):return "Error: invalid table name."
    try:
        raw=Path(database).resolve(); allowed=[Path("/app/memory").resolve(),Path("/app/workspace").resolve()]
        if not any(os.path.commonpath([str(a),str(raw)])==str(a) for a in allowed):return "Error: database path is outside allowed roots."
        con=sqlite3.connect(f"file:{raw}?mode=ro",uri=True); con.row_factory=sqlite3.Row; rows=con.execute(f'SELECT * FROM "{table}" LIMIT ?',(_bounded_int(limit,1,500),)).fetchall(); con.close(); return _json([dict(r) for r in rows])
    except Exception as exc:return f"Error: db_select failed: {exc}"
