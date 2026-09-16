# ==========================================
# FILE: tools/notify.py
# ==========================================
import subprocess
import os

def notify_desktop(title: str, message: str = "") -> str:
    """Send a native desktop notification to the host machine.
    Must be used to alert the user when a long-running research task is fully completed.
    
    Args:
        title: The brief header or summary of the notification.
        message: The detailed body text of the notification.
    """
    if not title or not str(title).strip():
        return "Error: Missing required 'title' parameter."
        
    try:
        # Requires the host's DBus socket to be mapped into the container
        # e.g., -v /run/user/1000/bus:/run/user/1000/bus and -e DBUS_SESSION_BUS_ADDRESS
        result = subprocess.run(
            ["notify-send", str(title), str(message)],
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return f"Successfully sent desktop notification: '{title}'"
        return f"Failed to send notification. Exit code: {result.returncode}, Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Notification execution error: {str(e)}"
