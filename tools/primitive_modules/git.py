from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def git_log(limit: int = 20) -> str:
    """Return a bounded Git log for /app/source."""
    try:
        limit=_bounded_int(limit,1,100); proc=subprocess.run(["git","-C","/app/source","log",f"-{limit}","--pretty=format:%H%x09%ad%x09%s","--date=iso-strict"],capture_output=True,text=True,timeout=10)
        if proc.returncode:return f"Error: git log failed: {proc.stderr.strip()}"
        return proc.stdout
    except Exception as exc:return f"Error: git_log failed: {exc}"

def git_show(ref: str = "HEAD", path: str = "", max_chars: int = 50000) -> str:
    """Show a Git object/file from /app/source with bounded output."""
    try:
        spec=f"{ref}:{path}" if path else ref; proc=subprocess.run(["git","-C","/app/source","show","--no-ext-diff",spec],capture_output=True,text=True,timeout=10)
        if proc.returncode:return f"Error: git show failed: {proc.stderr.strip()}"
        return proc.stdout[:_bounded_int(max_chars,100,100000)]
    except Exception as exc:return f"Error: git_show failed: {exc}"

def git_files(pattern: str = "", limit: int = 500) -> str:
    """List tracked repository files, optionally filtered by substring."""
    try:
        proc=subprocess.run(["git","-C","/app/source","ls-files"],capture_output=True,text=True,timeout=10)
        if proc.returncode:return f"Error: git ls-files failed: {proc.stderr.strip()}"
        rows=[x for x in proc.stdout.splitlines() if not pattern or pattern.lower() in x.lower()][:_bounded_int(limit,1,2000)]; return _json(rows)
    except Exception as exc:return f"Error: git_files failed: {exc}"

def git_changed_files(base: str = "HEAD~1", head: str = "HEAD") -> str:
    """List files changed between two Git refs."""
    try:
        proc=subprocess.run(["git","-C","/app/source","diff","--name-status",base,head],capture_output=True,text=True,timeout=10)
        return proc.stdout if proc.returncode==0 else f"Error: git diff failed: {proc.stderr.strip()}"
    except Exception as exc:return f"Error: git_changed_files failed: {exc}"
