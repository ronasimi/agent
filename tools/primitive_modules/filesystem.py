from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def path_stat(path: str) -> str:
    """Return bounded metadata for one path inside the workspace."""
    try:
        p = _safe_workspace(path); st = p.stat()
        return _json({"path": str(p), "exists": True, "type": "directory" if p.is_dir() else "file" if p.is_file() else "other", "size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(), "mode": oct(st.st_mode & 0o777), "readable": os.access(p, os.R_OK), "writable": os.access(p, os.W_OK)})
    except FileNotFoundError:
        return _json({"path": path, "exists": False})
    except Exception as exc:
        return f"Error: path_stat failed: {exc}"

def list_directory(path: str = ".", depth: int = 1, limit: int = 200) -> str:
    """List workspace directory entries with bounded recursion."""
    try:
        root = _safe_workspace(path); depth = _bounded_int(depth, 0, 3); limit = _bounded_int(limit, 1, 500)
        if not root.is_dir(): return "Error: path is not a directory."
        rows=[]
        for current, dirs, files in os.walk(root):
            rel_depth = len(Path(current).relative_to(root).parts)
            if rel_depth >= depth: dirs[:] = []
            for name in sorted(dirs) + sorted(files):
                p=Path(current)/name
                try: st=p.stat()
                except OSError: continue
                rows.append({"path": str(p), "relative": str(p.relative_to(root)), "type": "directory" if p.is_dir() else "file", "size": st.st_size})
                if len(rows)>=limit: return _json({"root":str(root),"entries":rows,"truncated":True})
        return _json({"root":str(root),"entries":rows,"truncated":False})
    except Exception as exc: return f"Error: list_directory failed: {exc}"

def find_paths(root: str = ".", pattern: str = "*", kind: str = "any", max_depth: int = 6, limit: int = 100) -> str:
    """Find workspace paths by glob-like filename pattern without shell execution."""
    import fnmatch
    try:
        base=_safe_workspace(root); max_depth=_bounded_int(max_depth,0,12); limit=_bounded_int(limit,1,500)
        if kind not in {"any","file","directory"}: return "Error: kind must be any, file, or directory."
        out=[]
        for current, dirs, files in os.walk(base):
            depth=len(Path(current).relative_to(base).parts)
            if depth>=max_depth: dirs[:]=[]
            for name in [*dirs,*files]:
                p=Path(current)/name
                if not fnmatch.fnmatch(name, pattern): continue
                if kind=="file" and not p.is_file(): continue
                if kind=="directory" and not p.is_dir(): continue
                out.append(str(p))
                if len(out)>=limit: return _json({"matches":out,"truncated":True})
        return _json({"matches":out,"truncated":False})
    except Exception as exc: return f"Error: find_paths failed: {exc}"

def read_text(path: str, offset: int = 0, limit: int = 20000) -> str:
    """Read a bounded UTF-8 text slice from a workspace file."""
    try:
        p=_safe_workspace(path); offset=max(0,int(offset)); limit=_bounded_int(limit,1,MAX_TEXT)
        with p.open("r",encoding="utf-8",errors="replace") as f: f.seek(offset); data=f.read(limit)
        return _json({"path":str(p),"offset":offset,"text":data,"truncated":len(data)>=limit})
    except Exception as exc: return f"Error: read_text failed: {exc}"

def read_bytes(path: str, offset: int = 0, limit: int = 4096) -> str:
    """Read bounded workspace bytes as hexadecimal plus size metadata."""
    try:
        p=_safe_workspace(path); offset=max(0,int(offset)); limit=_bounded_int(limit,1,65536)
        with p.open("rb") as f: f.seek(offset); data=f.read(limit)
        return _json({"path":str(p),"offset":offset,"bytes_read":len(data),"hex":data.hex(),"truncated":len(data)>=limit})
    except Exception as exc: return f"Error: read_bytes failed: {exc}"

def tail_file(path: str, lines: int = 100) -> str:
    """Return the last N text lines from a workspace file."""
    from collections import deque
    try:
        p=_safe_workspace(path); lines=_bounded_int(lines,1,2000)
        with p.open("r",encoding="utf-8",errors="replace") as f: data=list(deque(f, maxlen=lines))
        return "".join(data)[-MAX_TEXT:]
    except Exception as exc: return f"Error: tail_file failed: {exc}"

def file_hash(path: str, algorithm: str = "sha256") -> str:
    """Hash a workspace file using sha256, sha1, or md5."""
    try:
        algorithm=algorithm.lower()
        if algorithm not in {"sha256","sha1","md5"}: return "Error: unsupported hash algorithm."
        h=hashlib.new(algorithm); p=_safe_workspace(path)
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
        return _json({"path":str(p),"algorithm":algorithm,"digest":h.hexdigest()})
    except Exception as exc: return f"Error: file_hash failed: {exc}"

def mime_type(path: str) -> str:
    """Return MIME type guessed from a workspace path and lightweight signature checks."""
    try:
        p=_safe_workspace(path); mime,encoding=mimetypes.guess_type(str(p))
        sig=p.read_bytes()[:16] if p.is_file() else b""
        if sig.startswith(b"%PDF"): mime="application/pdf"
        elif sig.startswith(b"\x89PNG"): mime="image/png"
        elif sig[:3]==b"\xff\xd8\xff": mime="image/jpeg"
        return _json({"path":str(p),"mime":mime or "application/octet-stream","encoding":encoding})
    except Exception as exc: return f"Error: mime_type failed: {exc}"

def disk_usage(path: str = ".") -> str:
    """Return disk usage for the filesystem containing a workspace path."""
    try:
        p=_safe_workspace(path); d=shutil.disk_usage(p)
        return _json({"path":str(p),"total":d.total,"used":d.used,"free":d.free,"used_percent":round(d.used/d.total*100,1) if d.total else 0})
    except Exception as exc: return f"Error: disk_usage failed: {exc}"
