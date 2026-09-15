#!/usr/bin/env python3
"""
Network Mapper - Scans a local network and generates a visual topology diagram
showing each node with its OS and probable hardware type.

Usage:
    python network_mapper.py [--network NETWORK] [--output OUTPUT]
"""

import subprocess
import json
import socket
import argparse
import ipaddress
from datetime import datetime
from pathlib import Path
from typing import Optional


def scan_hosts(network: str = "192.168.1.0/24") -> list[str]:
    """Ping sweep to find live hosts on the network using stdlib ipaddress."""
    try:
        net = ipaddress.ip_network(network, strict=False)
    except ValueError as e:
        print(f"Invalid network CIDR '{network}': {e}")
        return []

    hosts = []
    # Limit scan to max 254 hosts to avoid long blocking sweeps
    for ip in list(net.hosts())[:254]:
        ip_str = str(ip)
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", ip_str],
                capture_output=True, text=True, timeout=2
            )
            if "bytes from" in result.stdout or "ttl=" in result.stdout.lower():
                hosts.append(ip_str)
        except Exception:
            pass
    return hosts


def get_os_info(host: str) -> dict:
    """Use nmap to detect OS and hardware of a host."""
    # Use -sT (Connect scan) if unprivileged, or -sS if root
    cmd = ["nmap", "-sV", "-O", "--top-ports", "100", host]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        output = result.stdout
    except Exception as e:
        output = str(e)

    info = {
        "ip": host,
        "os": "Unknown",
        "os_family": "Unknown",
        "os_version": "Unknown",
        "hardware": "Unknown",
        "services": [],
        "open_ports": []
    }
    
    for line in output.split("\n"):
        line_strip = line.strip()
        if "OS CPE:" in line_strip:
            info["os"] = line_strip.split("OS CPE:")[1].strip()
        elif "OS details:" in line_strip:
            info["os_family"] = line_strip.split("OS details:")[1].strip()
        elif "OS version:" in line_strip:
            info["os_version"] = line_strip.split("OS version:")[1].strip()
        elif "OS class:" in line_strip:
            info["os_family"] = line_strip.split("OS class:")[1].strip()
        elif "Hardware:" in line_strip:
            info["hardware"] = line_strip.split("Hardware:")[1].strip()
        elif "Device type:" in line_strip:
            info["hardware"] = line_strip.split("Device type:")[1].strip()
            
        if "/tcp" in line_strip or "/udp" in line_strip:
            parts = line_strip.split()
            if len(parts) >= 3 and "open" in parts[1]:
                info["open_ports"].append({"port": parts[0], "service": parts[2]})
    
    return info


def classify_hardware(hardware: str, os_family: str) -> str:
    """Classify hardware type based on detected info."""
    hw = hardware.lower() if hardware else ""
    os_f = os_family.lower() if os_family else ""
    
    if "raspberry" in hw or "raspberry" in os_f:
        return "Raspberry Pi"
    elif "nvidia" in hw or "jetson" in hw:
        return "NVIDIA Jetson"
    elif "router" in hw or "router" in os_f or "gateway" in os_f:
        return "Router / Gateway"
    elif "switch" in hw or "switch" in os_f:
        return "Network Switch"
    elif "nas" in hw or "synology" in hw or "qnap" in hw:
        return "NAS / Storage Server"
    elif "apple" in hw or "mac" in hw or "darwin" in os_f:
        return "Apple Mac"
    elif "linux" in os_f or "linux" in hw:
        return "Linux Device"
    elif "windows" in os_f or "windows" in hw:
        return "Windows Device"
    else:
        return "Unknown Device"


def generate_dot(hosts: list[dict]) -> str:
    """Generate a Graphviz DOT representation of the network."""
    dot = 'digraph NetworkMap {\n'
    dot += '    rankdir=TB;\n'
    dot += '    node [shape=box, style=filled, fontname="Helvetica", fontsize=10];\n'
    dot += '    edge [fontname="Helvetica", fontsize=8];\n\n'
    
    colors = {
        "Linux Device": "#4CAF50",
        "Windows Device": "#2196F3",
        "Router / Gateway": "#9C27B0",
        "Network Switch": "#607D8B",
        "NAS / Storage Server": "#FF5722",
        "Raspberry Pi": "#E91E63",
        "Apple Mac": "#FF5722",
        "Unknown Device": "#9E9E9E"
    }
    
    for host in hosts:
        ip = host["ip"]
        os_family = host["os_family"] or "Unknown"
        hardware = host["hardware"] or "Unknown"
        hw_type = classify_hardware(hardware, os_family)
        color = colors.get(hw_type, "#9E9E9E")
        
        os_display = os_family[:20] + "..." if len(os_family) > 20 else os_family
        label = f"{ip}\\n{hw_type}\\n{os_display}"
        dot += f'    "{ip}" [label="{label}", fillcolor="{color}"];\n'
    
    dot += '\n    "router" [label="Router / Gateway", shape=diamond, fillcolor="#9C27B0", style=filled];\n\n'
    
    for host in hosts:
        ip = host["ip"]
        dot += f'    "router" -> "{ip}" [label="ethernet", color="#666666", style=solid];\n'
    
    dot += '}\n'
    return dot


def generate_png(hosts: list[dict], output_path: str = "network_map.png") -> str:
    """Generate a PNG image of the network map."""
    try:
        import graphviz
    except ImportError:
        # Fallback to direct shell execution if python package isn't linked
        dot_code = generate_dot(hosts)
        dot_file = Path(output_path).with_suffix(".dot")
        with open(dot_file, "w") as f:
            f.write(dot_code)
        
        subprocess.run(["dot", "-Tpng", str(dot_file), "-o", output_path], check=True)
        return output_path

    dot_code = generate_dot(hosts)
    dot_file = Path(output_path).with_suffix(".dot")
    
    with open(dot_file, "w") as f:
        f.write(dot_code)
    
    src = graphviz.Source(dot_code)
    src.format = "png"
    src.render(filename=Path(output_path).stem, directory=Path(output_path).parent or ".", cleanup=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Network Mapper - Scan and visualize network topology")
    parser.add_argument("--network", default="192.168.1.0/24", help="Network to scan (CIDR notation)")
    parser.add_argument("--output", default="network_map.png", help="Output PNG filename")
    args = parser.parse_args()
    
    print(f"Scanning {args.network}...")
    hosts = scan_hosts(args.network)
    print(f"Found {len(hosts)} live hosts: {hosts}")
    
    if not hosts:
        return
    
    host_info = []
    for host in hosts:
        info = get_os_info(host)
        host_info.append(info)
    
    output_path = generate_png(host_info, args.output)
    print(f"Done! Network map saved to: {output_path}")


if __name__ == "__main__":
    main()
