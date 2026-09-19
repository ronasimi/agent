from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

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
        except OSError as exc: rows=[{"error":str(exc)}]
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
        p=subprocess.run(["lsblk","-J","-b","-o","NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,ROTA"],capture_output=True,text=True,timeout=8)
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
                p=subprocess.run([path,flag],capture_output=True,text=True,timeout=2,stdin=subprocess.DEVNULL); text=(p.stdout or p.stderr).strip().splitlines()
                if text: result["version"]=text[0][:300]; break
            except Exception: pass
    return _json(result)
