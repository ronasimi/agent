"""Bounded local-network discovery and topology rendering."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .media import media_result

_VIRTUAL_PREFIXES = ("docker", "br-", "veth", "virbr", "vmnet", "tailscale", "zt", "podman")


def _run(argv: list[str], timeout: float = 10) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 1, "", str(exc)


def _local_ipv4_networks(include_virtual: bool = False) -> list[dict[str, Any]]:
    code, stdout, _ = _run(["ip", "-j", "address"], timeout=5)
    if code != 0:
        return []
    try:
        interfaces = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for interface in interfaces if isinstance(interfaces, list) else []:
        name = str(interface.get("ifname") or "")
        if not name or name == "lo":
            continue
        virtual = name.startswith(_VIRTUAL_PREFIXES)
        if virtual and not include_virtual:
            continue
        flags = {str(x).upper() for x in interface.get("flags") or []}
        operstate = str(interface.get("operstate") or "").upper()
        if "UP" not in flags and operstate not in {"UP", "UNKNOWN"}:
            continue
        for addr in interface.get("addr_info") or []:
            if str(addr.get("family") or "") != "inet":
                continue
            local = str(addr.get("local") or "")
            prefixlen = int(addr.get("prefixlen") or 32)
            try:
                ip = ipaddress.ip_address(local)
                network = ipaddress.ip_network(f"{local}/{prefixlen}", strict=False)
            except ValueError:
                continue
            if ip.is_loopback or not (ip.is_private or ip.is_link_local):
                continue
            cidr = str(network)
            marker = f"{name}|{cidr}"
            if marker in seen:
                continue
            seen.add(marker)
            rows.append({
                "interface": name,
                "address": local,
                "prefixlen": prefixlen,
                "network": cidr,
                "virtual": virtual,
                "operstate": operstate,
            })
    return rows


def local_subnets(include_virtual: bool = False) -> str:
    """Return active private/link-local IPv4 subnets suitable for local host discovery."""
    rows = _local_ipv4_networks(bool(include_virtual))
    return json.dumps({"subnets": rows, "count": len(rows)}, ensure_ascii=False, indent=2)


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
    if net.version != 4 or net.num_addresses > 512 or not (net.is_private or net.is_link_local):
        return []

    # Nmap's host-discovery pass is substantially faster and also populates the
    # kernel neighbor table. Fall back to bounded parallel ICMP when unavailable.
    if shutil.which("nmap"):
        code, stdout, _ = _run(["nmap", "-sn", "-n", "-oG", "-", str(net)], timeout=35)
        if stdout:
            hosts = []
            for line in stdout.splitlines():
                match = re.match(r"Host:\s+(\S+).*Status:\s+Up", line)
                if match:
                    hosts.append(match.group(1))
            if hosts:
                return hosts
        if code == 0:
            return []

    with ThreadPoolExecutor(max_workers=min(64, max(1, net.num_addresses - 2))) as executor:
        results = executor.map(_ping_host, (str(ip) for ip in net.hosts()))
        return [ip for ip in results if ip]


def _neighbor_metadata(host: str) -> dict[str, str]:
    code, stdout, _ = _run(["ip", "-j", "neigh", "show", str(host)], timeout=3)
    if code != 0 or not stdout.strip():
        return {}
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not rows:
        return {}
    row = rows[0]
    return {
        "mac": str(row.get("lladdr") or ""),
        "interface": str(row.get("dev") or ""),
        "neighbor_state": ",".join(str(x) for x in (row.get("state") or [])) if isinstance(row.get("state"), list) else str(row.get("state") or ""),
    }


def _get_os_info(host: str, top_ports: int = 50) -> dict[str, Any]:
    top_ports = max(10, min(int(top_ports), 200))
    info: dict[str, Any] = {
        "ip": host,
        "hostname": "",
        "mac": "",
        "vendor": "",
        "interface": "",
        "neighbor_state": "",
        "os": "Unknown",
        "os_family": "Unknown",
        "os_version": "Unknown",
        "hardware": "Unknown",
        "os_accuracy": None,
        "open_ports": [],
        "scan_error": "",
    }
    info.update(_neighbor_metadata(host))
    try:
        info["hostname"] = socket.gethostbyaddr(host)[0]
    except Exception:
        pass

    if not shutil.which("nmap"):
        info["scan_error"] = "nmap is not installed"
        return info

    cmd = ["nmap", "-sV", "--version-light", "--top-ports", str(top_ports), "-oX", "-"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.extend(["-O", "--osscan-limit"])
    cmd.append(host)
    code, stdout, stderr = _run(cmd, timeout=45)
    if not stdout.strip():
        info["scan_error"] = (stderr or f"nmap exited {code}").strip()[:500]
        return info
    try:
        root = ET.fromstring(stdout)
    except ET.ParseError:
        info["scan_error"] = "nmap returned malformed XML"
        return info
    host_node = root.find("host")
    if host_node is None:
        info["scan_error"] = (stderr or "nmap returned no host record").strip()[:500]
        return info

    hostname = host_node.find("./hostnames/hostname")
    if hostname is not None and hostname.attrib.get("name"):
        info["hostname"] = hostname.attrib["name"]
    for address in host_node.findall("address"):
        if address.attrib.get("addrtype") == "mac":
            info["mac"] = address.attrib.get("addr", info["mac"])
            info["vendor"] = address.attrib.get("vendor", "")

    for port in host_node.findall("./ports/port"):
        state = port.find("state")
        if state is None or state.attrib.get("state") != "open":
            continue
        service = port.find("service")
        service_attrs = service.attrib if service is not None else {}
        item = {
            "port": int(port.attrib.get("portid", "0") or 0),
            "protocol": port.attrib.get("protocol", ""),
            "service": service_attrs.get("name", ""),
            "product": service_attrs.get("product", ""),
            "version": service_attrs.get("version", ""),
            "extra_info": service_attrs.get("extrainfo", ""),
        }
        info["open_ports"].append(item)

    osmatch = host_node.find("./os/osmatch")
    if osmatch is not None:
        info["os"] = osmatch.attrib.get("name", "Unknown")
        try:
            info["os_accuracy"] = int(osmatch.attrib.get("accuracy", "0"))
        except ValueError:
            pass
        osclass = osmatch.find("osclass")
        if osclass is not None:
            info["os_family"] = osclass.attrib.get("osfamily", "Unknown")
            info["os_version"] = osclass.attrib.get("osgen", "Unknown")
            info["hardware"] = osclass.attrib.get("type", "Unknown")
            if not info["vendor"]:
                info["vendor"] = osclass.attrib.get("vendor", "")
    if code != 0 and stderr.strip():
        info["scan_error"] = stderr.strip()[:500]
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
        hw_type = _classify_hardware(str(host.get("hardware", "")), str(host.get("os_family", "")))
        label = f"{host.get('ip','')}\\n{hw_type}\\n{str(host.get('hostname') or host.get('os_family') or 'Unknown')[:28]}"
        dot += f'    "{host.get("ip","")}" [label="{label}", style="filled,rounded", fillcolor="{colors.get(hw_type, colors["Unknown Device"])}"];\n'
    dot += '    "router" [label="Router / Gateway", shape=diamond, style="filled,rounded", fillcolor="#9C27B0"];\n'
    for host in hosts:
        dot += f'    "router" -> "{host.get("ip","")}" [label="reachable"];\n'
    return dot + '}\n'


def _resolve_network(network: str) -> tuple[str, str]:
    value = str(network or "").strip()
    if not value:
        rows = _local_ipv4_networks(include_virtual=False)
        if not rows:
            return "", "No active private IPv4 subnet could be detected. Call local_subnets(include_virtual=true) if only virtual networks are available."
        value = str(rows[0]["network"])
    try:
        net = ipaddress.ip_network(value, strict=False)
    except ValueError:
        return "", "Invalid CIDR network."
    if net.version != 4:
        return "", "Only IPv4 local-subnet scanning is currently supported."
    if net.num_addresses > 512:
        return "", "Network too large; use a subnet of 512 addresses or fewer."
    if not (net.is_private or net.is_link_local):
        return "", "Local scanning is restricted to private/link-local networks."
    return str(net), ""


def _discover_network(network: str, max_detail_hosts: int, top_ports: int) -> tuple[dict[str, Any] | None, str]:
    """Return bounded host-discovery/fingerprint data without creating an artifact."""
    network, error = _resolve_network(network)
    if error:
        return None, error
    hosts = _scan_hosts(network)
    if not hosts:
        return None, f"No active hosts discovered on subnet {network}."

    max_detail_hosts = max(1, min(int(max_detail_hosts), 64))
    top_ports = max(10, min(int(top_ports), 200))
    detailed_ips = hosts[:max_detail_hosts]
    # Nmap service probes are independent per host; a small bounded worker pool
    # cuts wall-clock time without spawning an unbounded scan fan-out.
    with ThreadPoolExecutor(max_workers=min(4, len(detailed_ips))) as executor:
        host_info = list(executor.map(lambda h: _get_os_info(h, top_ports), detailed_ips))
    for host in hosts[max_detail_hosts:]:
        row: dict[str, Any] = {"ip": host, "detail_skipped": True, "open_ports": []}
        row.update(_neighbor_metadata(host))
        host_info.append(row)

    return {
        "network": network,
        "host_count": len(hosts),
        "detailed_hosts": min(len(hosts), max_detail_hosts),
        "detail_truncated": len(hosts) > max_detail_hosts,
        "top_ports_scanned": top_ports,
        "hosts": host_info,
    }, ""


def scan_subnet(
    network: str = "",
    max_detail_hosts: int = 32,
    top_ports: int = 50,
) -> str:
    """Discover and fingerprint hosts on one bounded private/link-local IPv4 subnet.

    With an empty network argument, the first active non-virtual private subnet is
    selected. Use local_subnets() first when more than one local subnet may need
    to be scanned. This is a read-only primitive; unlike map_network it creates
    no image artifact.
    """
    payload, error = _discover_network(network, max_detail_hosts, top_ports)
    if error:
        return f"Error: {error}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def map_network(
    network: str = "",
    output_filename: str = "network_map.png",
    max_detail_hosts: int = 32,
    top_ports: int = 50,
) -> dict | str:
    """Discover one local subnet, fingerprint bounded hosts, and create a PNG topology map plus structured host data."""
    payload, error = _discover_network(network, max_detail_hosts, top_ports)
    if error:
        return f"Error: {error}"
    assert payload is not None
    output_path = f"/app/workspace/{Path(output_filename).name}"
    dot_code = _generate_dot(payload["hosts"])
    try:
        import graphviz
        src = graphviz.Source(dot_code)
        src.format = "png"
        src.render(filename=Path(output_path).stem, directory=Path(output_path).parent, cleanup=True)
    except Exception:
        dot_file = Path(output_path).with_suffix(".dot")
        dot_file.write_text(dot_code, encoding="utf-8")
        try:
            subprocess.run(["dot", "-Tpng", str(dot_file), "-o", output_path], check=True, capture_output=True, timeout=15)
        except Exception as exc:
            return f"Error: generating graph map failed: {exc}"

    payload["map_path"] = output_path
    return media_result(json.dumps(payload, ensure_ascii=False, indent=2), [output_path])
