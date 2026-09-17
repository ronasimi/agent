import subprocess

def search_packages(query: str) -> str:
    """Search official Arch Linux repositories for packages matching a query (excludes AUR)."""
    try:
        result = subprocess.run(
            ["pacman", "-Ss", query],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL
        )
        if result.returncode != 0 and not result.stdout:
            return f"No official packages found matching '{query}'."
        return result.stdout[:4000]
    except Exception as e:
        return str(e)

def install_package(package_name: str) -> str:
    """Install official Arch Linux packages using pacman (excludes AUR) and clear the package cache."""
    try:
        # Filter out flags to prevent parameter injection (e.g. pacman -Sy -Rns package)
        packages = [p for p in package_name.strip().split() if not p.startswith('-')]
        if not packages:
            return "Error: No valid packages specified. Flags are not allowed."
            
        cmd_install = ["pacman", "-Sy", "--noconfirm"] + packages
        result = subprocess.run(
            cmd_install,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL
        )
        
        if result.returncode == 0:
            subprocess.run(
                ["pacman", "-Scc", "--noconfirm"],
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL
            )
            return f"Successfully installed: {' '.join(packages)} and cleaned the package cache.\nSTDOUT:\n{result.stdout}"
        else:
            return f"Failed to install: {' '.join(packages)}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return str(e)
