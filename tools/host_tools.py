import os
import subprocess

def read_host_file(filepath: str = "/host/etc/resolv.conf") -> str:
    """Safely read the contents of a configuration or text file from the host filesystem.
    
    Args:
        filepath: The absolute path of the file on the host (e.g., '/etc/fstab' or '/host/etc/mkinitcpio.conf').
    """
    if not filepath or not str(filepath).strip():
        filepath = "/host/etc/resolv.conf"
        
    # Ensure it maps to the /host mount
    if not filepath.startswith("/host"):
        filepath = os.path.join("/host", filepath.lstrip("/"))
        
    if not os.path.exists(filepath):
        return f"Error: File {filepath} not found on host."
    if not os.path.isfile(filepath):
        return f"Error: {filepath} is not a valid file."
        
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read(15000)
            if len(content) == 15000:
                content += "\n\n[Content truncated at 15,000 characters to prevent context overflow...]"
            return content
    except Exception as e:
        return f"Error reading host file: {str(e)}"

def read_host_journal(lines: int = 50, service: str = "", grep: str = "", priority: str = "") -> str:
    """Read systemd journal logs directly from the host machine.
    
    Args:
        lines: Number of recent log lines to retrieve (default: 50).
        service: Optional systemd service name to filter by (e.g., 'docker.service').
        grep: Optional text to search for in the logs.
        priority: Filter by priority level (e.g., 'err', 'warning', 'info'). Leave empty for all.
    """
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
    """Read the last N lines of a host log file. Automatically falls back to system journal if file is missing.
    
    Args:
        log_path: Path to the log file or keyword (e.g., /host_log/syslog, errors).
        lines: Number of trailing lines to return (default 50).
    """
    if not log_path or not str(log_path).strip():
        log_path = "syslog"
        
    # If the requested path doesn't exist, intelligently fallback to system journal
    full_path = log_path
    if not full_path.startswith("/host"):
        full_path = os.path.join("/host_log", full_path.lstrip("/"))
        
    if not os.path.exists(full_path):
        # Fall back to host journal automatically
        prio = "err" if "err" in log_path.lower() or "error" in log_path.lower() else ""
        return read_host_journal(lines=lines, priority=prio)
        
    try:
        with open(full_path, 'r', errors='ignore') as f:
            all_lines = f.readlines()
            tail_slice = all_lines[-lines:]
            return "".join(tail_slice)
    except Exception:
        # Final fallback to journal on any read exception
        return read_host_journal(lines=lines)
