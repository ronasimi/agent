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
from datetime import datetime
from pathlib import Path
from typing import Optional


def scan_hosts(network: str = "192.168.1.0/24") -> list[str]:
    """Ping sweep to find live hosts on the network."""
    network_prefix = network.split("/")[0]
    prefix_len = int(network.split("/")[1])
    mask = (256 - (256 - (256 >> prefix_len))) - 1
    mask_int = int(mask)
    network_int = int(network_prefix.replace(".", ""))
    mask_bits = mask_int.bit_length()
    mask = (1 << (32 - mask_bits)) - 1
    network_int &= mask
    network_prefix = f"{network_int // 256}.{(network_int >> 8) & 255}.{(network_int >> 4) & 15}.{network_int & 15}"
    
    hosts = []
    for i in range(1, 254):
        ip = f"{network_prefix}.{i}"
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", ip],
                capture_output=True, text=True, timeout=2
            )
            if "bytes from" in result.stdout:
                hosts.append(ip)
        except:
            pass
    return hosts


def get_os_info(host: str) -> dict:
    """Use nmap to detect OS and hardware of a host."""
    result = subprocess.run(
        ["nmap", "-sS", "-sV", "-O", "-A", host],
        capture_output=True, text=True, timeout=30
    )
    output = result.stdout
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
        if "OS CPE" in line:
            info["os"] = line.split("OS CPE:")[1].strip()
        elif "OS details:" in line:
            info["os_family"] = line.split("OS details:")[1].strip()
        elif "OS version:" in line:
            info["os_version"] = line.split("OS version:")[1].strip()
        elif "OS class:" in line:
            info["os_family"] = line.split("OS class:")[1].strip()
        elif "OS precision:" in line:
            info["os_family"] = line.split("OS precision:")[1].strip()
        elif "Hardware" in line:
            info["hardware"] = line.split("Hardware:")[1].strip()
        elif "Model:" in line:
            info["hardware"] = line.split("Model:")[1].strip()
        elif "Device:" in line:
            info["hardware"] = line.split("Device:")[1].strip()
    
    for line in output.split("\n"):
        if "open" in line.lower() and ":" in line:
            parts = line.split()
            if len(parts) >= 3:
                port = parts[1]
                service = parts[2] if len(parts) > 2 else "unknown"
                info["open_ports"].append({"port": port, "service": service})
    
    return info


def classify_hardware(hardware: str, os_family: str) -> str:
    """Classify hardware type based on detected info."""
    hw = hardware.lower() if hardware else ""
    os_family = os_family.lower() if os_family else ""
    
    if "raspberry" in hw or "raspberry pi" in hw:
        return "Raspberry Pi"
    elif "nvidia" in hw or "jetson" in hw:
        return "NVIDIA Jetson"
    elif "intel" in hw or "amd" in hw:
        if "server" in hw or "serverboard" in hw:
            return "Server (Intel/AMD)"
        elif "nuc" in hw:
            return "NUC / Mini PC"
        else:
            return "Desktop / Laptop (Intel/AMD)"
    elif "apple" in hw or "mac" in hw:
        return "Apple Mac"
    elif "linux" in os_family or "linux" in hw:
        return "Linux Device"
    elif "windows" in os_family or "windows" in hw:
        return "Windows Device"
    elif "android" in hw or "android" in os_family:
        return "Android Device"
    elif "ios" in hw or "iphone" in hw:
        return "iOS Device"
    elif "embedded" in hw or "embedded" in os_family:
        return "Embedded Device"
    elif "router" in hw or "router" in os_family:
        return "Router / Gateway"
    elif "switch" in hw or "switch" in os_family:
        return "Network Switch"
    elif "firewall" in hw or "firewall" in os_family:
        return "Firewall / Security Appliance"
    elif "printer" in hw or "printer" in os_family:
        return "Network Printer"
    elif "nas" in hw or "nas" in os_family:
        return "NAS / Storage Server"
    elif "server" in hw or "server" in os_family:
        return "Server"
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
        "Server": "#FF9800",
        "Router / Gateway": "#9C27B0",
        "Network Switch": "#607D8B",
        "Firewall / Security Appliance": "#F44336",
        "NAS / Storage Server": "#FF5722",
        "Raspberry Pi": "#E91E63",
        "NVIDIA Jetson": "#9C27B0",
        "Apple Mac": "#FF5722",
        "Embedded Device": "#795548",
        "Unknown Device": "#9E9E9E",
        "Unknown": "#BDBDBD"
    }
    
    for host in hosts:
        ip = host["ip"]
        os_family = host["os_family"] or "Unknown"
        hardware = host["hardware"] or "Unknown"
        hw_type = classify_hardware(hardware, os_family)
        color = colors.get(hw_type, colors.get("Unknown Device", "#BDBDBD"))
        
        os_display = os_family[:20] + "..." if len(os_family) > 20 else os_family
        hw_display = hardware[:25] + "..." if len(hardware) > 25 else hardware
        
        label = f"{ip}\n{os_display}\n{hw_display}"
        dot += f'    "{ip}" [label="{label}", fillcolor="{color}"];'
        dot += '\n'
    
    dot += '\n    // Central router node\n'
    dot += '    "router" [label="Router\\n/Gateway", shape=diamond, fillcolor="#4CAF50", style=filled];\n\n'
    
    for host in hosts:
        ip = host["ip"]
        dot += f'    "router" -> "{ip}" [label="ethernet", color="#4CAF50", style=solid];\n'
    
    dot += '}\n'
    return dot


def generate_png(hosts: list[dict], output_path: str = "network_map.png") -> str:
    """Generate a PNG image of the network map."""
    try:
        import graphviz
    except ImportError:
        subprocess.run(["pip", "install", "graphviz"], check=True)
        import graphviz
    
    dot = generate_dot(hosts)
    dot_file = Path(output_path).with_suffix(".dot")
    
    with open(dot_file, "w") as f:
        f.write(dot)
    
    graphviz.Source(dot).render(output_path, cleanup=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Network Mapper - Scan and visualize network topology")
    parser.add_argument("--network", default="192.168.1.0/24", help="Network to scan (CIDR notation)")
    parser.add_argument("--output", default="network_map.png", help="Output PNG filename")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"  Network Mapper - Scanning {args.network}")
    print(f"{'='*60}\n")
    
    print("[1/3] Scanning for live hosts...")
    hosts = scan_hosts(args.network)
    print(f"      Found {len(hosts)} live hosts: {hosts}")
    
    if not hosts:
        print("      No live hosts found. Exiting.")
        return
    
    print(f"\n[2/3] Gathering OS and hardware info...")
    host_info = []
    for host in hosts:
        info = get_os_info(host)
        host_info.append(info)
        print(f"      {host}: {info['os_family']} / {info['hardware']}")
    
    print(f"\n[3/3] Generating network map...")
    output_path = generate_png(host_info, args.output)
    print(f"      Saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"  Done! Network map saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()