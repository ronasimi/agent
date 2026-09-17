import ipaddress
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _ping_host(ip: str):
    try:
        result = subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True, text=True, timeout=2)
        if "bytes from" in result.stdout or "ttl=" in result.stdout.lower():
            return ip
    except Exception:
        pass
    return None


def _scan_hosts(network: str) -> list[str]:
    try:
        net = ipaddress.ip_network(network, strict=False)
    except ValueError:
        return []
    if net.num_addresses > 512:
        return []
    with ThreadPoolExecutor(max_workers=min(64, max(1, net.num_addresses - 2))) as executor:
        results = executor.map(_ping_host, (str(ip) for ip in net.hosts()))
        return [ip for ip in results if ip]


def _get_os_info(host: str) -> dict:
    cmd = ["nmap", "-sV", "-O", "--top-ports", "100", host]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        output = result.stdout
    except Exception as exc:
        output = str(exc)
    info = {"ip": host, "os": "Unknown", "os_family": "Unknown", "os_version": "Unknown", "hardware": "Unknown", "services": [], "open_ports": []}
    for line in output.splitlines():
        line_strip = line.strip()
        if "OS CPE:" in line_strip:
            info["os"] = line_strip.split("OS CPE:", 1)[1].strip()
        elif "OS details:" in line_strip:
            info["os_family"] = line_strip.split("OS details:", 1)[1].strip()
        elif "Hardware:" in line_strip:
            info["hardware"] = line_strip.split("Hardware:", 1)[1].strip()
        elif "Device type:" in line_strip:
            info["hardware"] = line_strip.split("Device type:", 1)[1].strip()
        if "/tcp" in line_strip or "/udp" in line_strip:
            parts = line_strip.split()
            if len(parts) >= 3 and "open" in parts[1]:
                info["open_ports"].append({"port": parts[0], "service": parts[2]})
    return info


def _classify_hardware(hardware: str, os_family: str) -> str:
    hw = (hardware or "").lower()
    os_f = (os_family or "").lower()
    if "raspberry" in hw or "raspberry" in os_f: return "Raspberry Pi"
    if "nvidia" in hw or "jetson" in hw: return "NVIDIA Jetson"
    if "router" in hw or "gateway" in os_f: return "Router / Gateway"
    if "switch" in hw: return "Network Switch"
    if "nas" in hw or "synology" in hw: return "NAS / Storage Server"
    if "apple" in hw or "mac" in hw or "darwin" in os_f: return "Apple Mac"
    if "linux" in os_f or "linux" in hw: return "Linux Device"
    if "windows" in os_f or "windows" in hw: return "Windows Device"
    return "Unknown Device"


def _generate_dot(hosts: list[dict]) -> str:
    dot = 'digraph NetworkMap {\n    graph [dpi=300, nodesep=1.0, ranksep=1.2];\n    rankdir=TB;\n'
    colors = {
        "Linux Device": "#4CAF50", "Windows Device": "#2196F3", "Router / Gateway": "#9C27B0",
        "Network Switch": "#607D8B", "NAS / Storage Server": "#FF5722", "Raspberry Pi": "#E91E63",
        "Apple Mac": "#FF9800", "Unknown Device": "#E0E0E0",
    }
    for host in hosts:
        hw_type = _classify_hardware(host["hardware"], host["os_family"])
        label = f"{host['ip']}\\n{hw_type}\\n{(host['os_family'] or 'Unknown')[:20]}"
        dot += f'    "{host["ip"]}" [label="{label}", style="filled,rounded", fillcolor="{colors.get(hw_type, colors["Unknown Device"])}"];\n'
    dot += '    "router" [label="Router / Gateway", shape=diamond, style="filled,rounded", fillcolor="#9C27B0"];\n'
    for host in hosts:
        dot += f'    "router" -> "{host["ip"]}" [label="reachable"];\n'
    return dot + '}\n'


def map_network(network: str = "192.168.1.0/24", output_filename: str = "network_map.png") -> str:
    """Scan one local subnet up to /23, perform limited OS discovery, and create a PNG topology map."""
    output_path = f"/app/workspace/{Path(output_filename).name}"
    hosts = _scan_hosts(network)
    if not hosts:
        try:
            net = ipaddress.ip_network(network, strict=False)
            if net.num_addresses > 512:
                return "Error: Network too large; use a subnet of 512 addresses or fewer."
        except ValueError:
            return "Error: Invalid CIDR network."
        return f"No active hosts discovered on subnet {network}."
    host_info = [_get_os_info(h) for h in hosts]
    dot_code = _generate_dot(host_info)
    try:
        import graphviz
        src = graphviz.Source(dot_code)
        src.format = "png"
        src.render(filename=Path(output_path).stem, directory=Path(output_path).parent, cleanup=True)
    except Exception:
        dot_file = Path(output_path).with_suffix(".dot")
        dot_file.write_text(dot_code, encoding="utf-8")
        try:
            subprocess.run(["dot", "-Tpng", str(dot_file), "-o", output_path], check=True, capture_output=True)
        except Exception as exc:
            return f"Error generating graph map: {exc}"
    return f"Network map successfully generated and saved to {output_path}"
