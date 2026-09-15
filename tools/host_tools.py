import os
import subprocess

def read_host_file(filepath: str = "/host/etc/resolv.conf") -> str:
    """Read any file from the host system in read-only mode.
    
    Args:
        filepath: The path on the host, e.g. /host/etc/resolv.conf
    """
    if not filepath or not str(filepath).strip():
        filepath = "/host/etc/resolv.conf"
        
    if not filepath.startswith("/host"):
        filepath = os.path.join("/host", filepath.lstrip("/"))
        
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read(15000)
            if len(content) == 15000:
                content += "\n\n[Content truncated at 15,000 characters...]"
            return content
    except Exception as e:
        return f"Error reading host file: {str(e)}"

def read_host_journal(lines: int = 50, priority: str = "") -> str:
    """Read recent system logs directly from the host's systemd journal.
    
    Args:
        lines: Number of trailing log lines to return (default 50).
        priority: Filter by priority level (e.g., 'err', 'warning', 'info'). Leave empty for all.
    """
    cmd = ["journalctl", "--directory", "/host/var/log/journal", "-n", str(lines), "--no-pager"]
    if priority and str(priority).strip():
        cmd.extend(["-p", str(priority).strip()])
        
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            alt_cmd = ["journalctl", "-n", str(lines), "--no-pager"]
            if priority and str(priority).strip():
                alt_cmd.extend(["-p", str(priority).strip()])
            result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=15)
            
        output = result.stdout.strip()
        return output if output else "No journal log entries returned."
    except Exception as sub_err:
        return f"Error querying host journal: {str(sub_err)}"

def tail_host_log(log_path: str = "syslog", lines: int = 50) -> str:
    """Read the last N lines of a host log file. Automatically falls back to system journal if file is missing.
    
    Args:
        log_path: Path to the log file or keyword (e.g., /host_log/syslog, errors)
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
    except Exception as e:
        # Final fallback to journal on any read exception
        return read_host_journal(lines=lines)
