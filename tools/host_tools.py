import os
import subprocess

def read_host_file(filepath: str = "/host/etc/resolv.conf") -> str:
    """Safely read the contents of a configuration or text file from the host filesystem."""
    if not filepath or not str(filepath).strip():
        filepath = "/host/etc/resolv.conf"
        
    if not filepath.startswith("/host"):
        filepath = os.path.join("/host", filepath.lstrip("/"))
        
    safe_path = os.path.abspath(filepath)
    if os.path.commonpath(["/host", safe_path]) != "/host":
        return "Error: Path traversal outside /host is forbidden."
        
    if not os.path.exists(safe_path):
        return f"Error: File {safe_path} not found on host."
    if not os.path.isfile(safe_path):
        return f"Error: {safe_path} is not a valid file."
        
    try:
        with open(safe_path, 'r', errors='ignore') as f:
            content = f.read(15000)
            if len(content) == 15000:
                content += "\n\n[Content truncated at 15,000 characters to prevent context overflow...]"
            return content
    except Exception as e:
        return f"Error reading host file: {str(e)}"

def read_host_journal(lines: int = 50, service: str = "", grep: str = "", priority: str = "") -> str:
    """Read systemd journal logs directly from the host machine."""
    cmd = ["journalctl", "-D", "/host_log/journal", "--no-pager", "-n", str(lines)]
    
    if service and str(service).strip():
        cmd.extend(["-u", str(service).strip()])
    if grep and str(grep).strip():
        cmd.extend(["-g", str(grep).strip()])
    if priority and str(priority).strip():
        cmd.extend(["-p", str(priority).strip()])
        
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 and not result.stdout:
            return f"Error reading journal: {result.stderr}"
        return result.stdout.strip() or "No logs found for the given criteria."
    except Exception as e:
        return f"Failed to execute journalctl: {str(e)}"

def tail_host_log(log_path: str = "syslog", lines: int = 50) -> str:
    """Read the last N lines of a host log file. Automatically falls back to system journal if file is missing."""
    if not log_path or not str(log_path).strip():
        log_path = "syslog"
        
    if log_path.startswith("/var/log/"):
        log_path = log_path.replace("/var/log/", "/host_log/", 1)
        
    if not log_path.startswith("/host_log"):
        log_path = os.path.join("/host_log", log_path.lstrip("/"))
        
    safe_path = os.path.abspath(log_path)
    if os.path.commonpath(["/host_log", safe_path]) != "/host_log":
        return "Error: Path traversal outside /host_log is forbidden."
        
    if not os.path.exists(safe_path):
        prio = "err" if "err" in log_path.lower() or "error" in log_path.lower() else ""
        return read_host_journal(lines=lines, priority=prio)
        
    try:
        with open(safe_path, 'r', errors='ignore') as f:
            all_lines = f.readlines()
            tail_slice = all_lines[-lines:]
            return "".join(tail_slice)
    except Exception:
        return read_host_journal(lines=lines)
