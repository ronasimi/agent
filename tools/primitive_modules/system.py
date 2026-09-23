from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path
from ..subprocess_utils import run_argv

def kernel_info() -> str:
    """Return kernel/platform identity."""
    return _json({"system":platform.system(),"release":platform.release(),"version":platform.version(),"machine":platform.machine()})

def memory_info() -> str:
    """Return current system memory and swap counters."""
    m=psutil.virtual_memory(); s=psutil.swap_memory(); return _json({"memory":m._asdict(),"swap":s._asdict()})

def load_average() -> str:
    """Return 1/5/15 minute load averages and CPU count."""
    return _json({"load":os.getloadavg(),"cpu_count":os.cpu_count()})

def uptime() -> str:
    """Return boot time and uptime seconds."""
    boot=psutil.boot_time(); return _json({"boot_time":datetime.fromtimestamp(boot).astimezone().isoformat(),"uptime_seconds":round(time.time()-boot,1)})

def pressure_info(resource: str = "all") -> str:
    """Return Linux PSI pressure metrics for cpu, memory, io, or all."""
    resource=resource.lower()
    if resource not in {"cpu","memory","io","all"}:return "Error: resource must be cpu, memory, io, or all."
    roots=[Path("/host/proc/pressure"),Path("/proc/pressure")]; root=next((p for p in roots if p.exists()),None)
    if not root:return "Error: Linux PSI is unavailable."
    out={"source":str(root)}
    for kind in ("cpu","memory","io"):
        if resource not in {"all",kind}:continue
        rows=[]
        try:
            for line in (root/kind).read_text().splitlines():
                parts=line.split(); item={"scope":parts[0]}
                for token in parts[1:]:
                    k,v=token.split("=",1); item[k]=float(v) if k!="total" else int(v)
                rows.append(item)
        except (OSError, ValueError, IndexError) as exc: rows=[{"error":str(exc)}]
        out[kind]=rows
    return _json(out)

def mounts() -> str:
    """Return host mount table entries from the host PID-1 namespace when available."""
    source=Path("/host/proc/1/mountinfo") if Path("/host/proc/1/mountinfo").exists() else Path("/proc/1/mountinfo")
    try:
        rows=[]
        for line in source.read_text(errors="replace").splitlines()[:1000]:
            left,sep,right=line.partition(" - "); fields=left.split(); rf=right.split()
            if len(fields)>=6 and len(rf)>=3: rows.append({"mountpoint":fields[4],"options":fields[5],"fstype":rf[0],"source":rf[1]})
        return _json({"source":str(source),"mounts":rows})
    except Exception as exc:return f"Error: mounts failed: {exc}"

def block_devices() -> str:
    """Return lsblk JSON for host-visible block devices."""
    try:
        p=run_argv(["lsblk","-J","-b","-o","NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,ROTA"], timeout=8)
        return p.stdout if p.returncode==0 else f"Error: lsblk failed: {p.stderr.strip()}"
    except Exception as exc:return f"Error: block_devices failed: {exc}"

def temperature_sensors() -> str:
    """Return bounded temperature readings available through psutil/sysfs."""
    try:
        temps=psutil.sensors_temperatures(fahrenheit=False) or {}; return _json({k:[x._asdict() for x in v[:32]] for k,v in temps.items()})
    except Exception as exc:return f"Error: temperature_sensors failed: {exc}"

def command_available(name: str) -> str:
    """Report whether a command is installed and return a bounded version string."""
    if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}",name):return "Error: invalid command name."
    path=shutil.which(name)
    result={"name":name,"available":bool(path),"path":path or ""}
    if path:
        for flag in ("--version","-V","-v"):
            try:
                p=run_argv([path,flag], timeout=2, max_output_bytes=65536); text=(p.stdout or p.stderr).strip().splitlines()
                if text: result["version"]=text[0][:300]; break
            except Exception: pass
    return _json(result)

def filesystem_usage(path: str = "/", host: bool = True) -> str:
    """Return capacity and inode usage for one host/container filesystem path."""
    raw = str(path or "/")
    actual = raw
    if host and Path("/host").is_dir():
        try:
            root = Path("/host").resolve()
            target = root if raw == "/" else (root / raw.lstrip("/")).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                return "Error: path escapes /host boundary."
            actual = str(target)
        except (OSError, ValueError) as exc:
            return f"Error: filesystem_usage path validation failed: {exc}"
    try:
        usage = psutil.disk_usage(actual)
        result={"path":raw,"actual_path":actual,"total_gb":round(usage.total/1073741824,2),"free_gb":round(usage.free/1073741824,2),"used_percent":usage.percent}
        try:
            st=os.statvfs(actual); total=st.f_files; free=st.f_ffree
            result["inode_used_percent"] = round(((total-free)/total)*100,1) if total else None
        except OSError: result["inode_used_percent"] = None
        return _json(result)
    except Exception as exc:return f"Error: filesystem_usage failed: {exc}"


def gpu_info() -> str:
    """Return optional NVIDIA/AMD GPU telemetry without failing on unsupported hosts."""
    try:
        p=run_argv(["nvidia-smi","--query-gpu=name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"], timeout=3)
        if p.returncode==0 and p.stdout.strip():
            rows=[]
            for line in p.stdout.strip().splitlines():
                parts=[x.strip() for x in line.split(",")]
                if len(parts)>=4:
                    try: rows.append({"vendor":"nvidia","name":parts[0],"vram_total_mb":float(parts[1]),"vram_used_mb":float(parts[2]),"utilization_percent":float(parts[3])})
                    except ValueError: pass
            if rows:return _json({"gpus":rows})
    except Exception: pass
    try:
        p=run_argv(["rocm-smi","--showmeminfo","vram","--showuse","--json"], timeout=5)
        if p.returncode==0 and p.stdout.strip():
            try:return _json({"vendor":"amd","raw":json.loads(p.stdout)})
            except json.JSONDecodeError:return _json({"vendor":"amd","raw_text":p.stdout[:4000]})
    except Exception: pass
    return _json({"available":False})


def host_read_text(path: str = "/etc/resolv.conf", max_chars: int = 20000) -> str:
    """Read bounded text from the read-only /host mount without allowing path traversal."""
    raw=str(path or "/etc/resolv.conf")
    target=Path("/host")/raw.lstrip("/")
    try:
        safe=target.resolve(); root=Path("/host").resolve()
        if os.path.commonpath([str(root),str(safe)])!=str(root):return "Error: path escapes /host."
        text=safe.read_text(errors="replace"); limit=_bounded_int(max_chars,100,100000)
        return text[:limit] + ("\n[truncated]" if len(text)>limit else "")
    except Exception as exc:return f"Error: host_read_text failed: {exc}"


def cpu_info() -> str:
    """Return CPU topology, frequency, utilization, and bounded model identity."""
    try:
        source = Path("/host/proc/cpuinfo") if Path("/host/proc/cpuinfo").is_file() else Path("/proc/cpuinfo")
        models = []
        for line in source.read_text(errors="replace").splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            field = key.strip().lower()
            if field not in {"model name", "hardware", "processor", "cpu model", "machine"}:
                continue
            value = value.strip()
            # On x86 /proc/cpuinfo, `processor` is the logical CPU index (0, 1,
            # ...), not an identity string. Some ARM platforms use the same field
            # for a real model description, so retain only non-numeric values.
            if field == "processor" and re.fullmatch(r"\d+", value):
                continue
            if value and value not in models:
                models.append(value)
        frequency = psutil.cpu_freq()
        return _json({
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "frequency_mhz": frequency._asdict() if frequency else {},
            "utilization_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "models": models[:8],
            "source": str(source),
        })
    except Exception as exc:
        return f"Error: cpu_info failed: {exc}"


def os_release() -> str:
    """Return parsed host/container operating-system release metadata."""
    source = Path("/host/etc/os-release") if Path("/host/etc/os-release").is_file() else Path("/etc/os-release")
    try:
        values = {}
        for line in source.read_text(errors="replace").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")[:1000]
        return _json({"source": str(source), "release": values})
    except Exception as exc:
        return f"Error: os_release failed: {exc}"
