from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def process_info(pid: int) -> str:
    """Return bounded metadata for one process PID."""
    try:
        p=psutil.Process(int(pid)); return _json({"pid":p.pid,"name":p.name(),"status":p.status(),"create_time":datetime.fromtimestamp(p.create_time()).astimezone().isoformat(),"cpu_percent":p.cpu_percent(interval=0.05),"memory":p.memory_info()._asdict(),"num_threads":p.num_threads(),"io":p.io_counters()._asdict() if hasattr(p,"io_counters") else {}})
    except Exception as exc:return f"Error: process_info failed: {exc}"

def process_io(pid: int) -> str:
    """Return I/O counters for one process."""
    try:
        process = psutil.Process(int(pid))
        counters = getattr(process, "io_counters", None)
        if counters is None:
            return _json({"ok": False, "supported": False, "error": "Process I/O counters are unavailable on this psutil platform"})
        return _json(counters()._asdict())
    except Exception as exc:return f"Error: process_io failed: {exc}"

def process_threads(pid: int, limit: int = 100) -> str:
    """Return bounded thread CPU counters for one process."""
    try:
        rows=[t._asdict() for t in psutil.Process(int(pid)).threads()[:_bounded_int(limit,1,500)]]; return _json(rows)
    except Exception as exc:return f"Error: process_threads failed: {exc}"

def process_fds(pid: int, limit: int = 100) -> str:
    """Return bounded open-file paths for one process when permissions permit."""
    try:
        files=psutil.Process(int(pid)).open_files()[:_bounded_int(limit,1,500)]; return _json([{"path":f.path,"fd":f.fd} for f in files])
    except Exception as exc:return f"Error: process_fds failed: {exc}"

def list_processes(limit: int = 100, include_command: bool = False, include_io: bool = False) -> str:
    """Return bounded process rows with optional command line and aggregate I/O counters."""
    procs=[]
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None); procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): pass
    time.sleep(0.12)
    rows=[]
    for p in procs:
        try:
            info=p.as_dict(attrs=["pid","name","status","memory_info","cmdline"] if include_command else ["pid","name","status","memory_info"])
            row={"pid":info["pid"],"name":info.get("name") or "","state":info.get("status") or "","cpu_percent":round(float(p.cpu_percent(None) or 0.0),1),"rss_mb":round((info.get("memory_info").rss if info.get("memory_info") else 0)/1048576,1)}
            if include_command:
                command=" ".join(info.get("cmdline") or [])[:1000]
                row["command"]=re.sub(r"(?i)(password|passwd|token|secret|api[_-]?key)=\S+",r"\1=[REDACTED]",command)
            if include_io:
                try:
                    io=p.io_counters(); row["io_bytes"]=int(getattr(io,"read_bytes",0))+int(getattr(io,"write_bytes",0)); row["read_mb"]=round(int(getattr(io,"read_bytes",0))/1048576,1); row["write_mb"]=round(int(getattr(io,"write_bytes",0))/1048576,1)
                except (psutil.NoSuchProcess,psutil.AccessDenied,AttributeError): row["io_bytes"]=0; row["read_mb"]=0.0; row["write_mb"]=0.0
            rows.append(row)
        except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess): pass
    return _json(rows[:_bounded_int(limit,1,500)])


def process_tree(pid: int = 1, depth: int = 3, limit: int = 100) -> str:
    """Return a bounded parent/child process tree rooted at one PID."""
    try:
        root = psutil.Process(int(pid)); depth = _bounded_int(depth, 0, 8); limit = _bounded_int(limit, 1, 500)
        rows = []

        def visit(proc, level: int) -> None:
            if len(rows) >= limit or level > depth:
                return
            try:
                rows.append({"pid": proc.pid, "ppid": proc.ppid(), "name": proc.name(), "status": proc.status(), "depth": level})
                for child in sorted(proc.children(), key=lambda item: item.pid):
                    visit(child, level + 1)
                    if len(rows) >= limit:
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return

        visit(root, 0)
        return _json({"root_pid": int(pid), "processes": rows, "truncated": len(rows) >= limit})
    except Exception as exc:
        return f"Error: process_tree failed: {exc}"
