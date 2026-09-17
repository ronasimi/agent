import re
import subprocess


def search_packages(query: str) -> str:
    """Search official Arch Linux repositories for packages matching a query."""
    try:
        result = subprocess.run(["pacman", "-Ss", str(query)], capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
        return result.stdout[:6000] if result.stdout else f"No official packages found matching '{query}'.\n{result.stderr[:2000]}"
    except Exception as exc:
        return f"Package search error: {exc}"


def install_package(package_name: str) -> str:
    """Install explicitly named official Arch Linux packages; flags and shell syntax are rejected."""
    raw = str(package_name).strip()
    if not raw or not re.fullmatch(r"[A-Za-z0-9@._+:/-]+(?:\s+[A-Za-z0-9@._+:/-]+)*", raw):
        return "Error: package_name must contain only package names separated by spaces; options and shell syntax are not allowed."
    packages = raw.split()
    try:
        result = subprocess.run(["pacman", "-S", "--needed", "--noconfirm", *packages], capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            return f"Package installation failed.\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
        return f"Installed/already present: {', '.join(packages)}\n{result.stdout[-4000:]}"
    except Exception as exc:
        return f"Package installation error: {exc}"
