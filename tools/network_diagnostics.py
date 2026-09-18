"""Bounded network-analysis primitives that avoid arbitrary shell construction."""
from __future__ import annotations

import ipaddress
import json
import re
import shutil
import socket
import ssl
import subprocess
import time
from urllib.parse import urljoin, urlparse

import requests


def _run(argv: list[str], timeout: float = 10) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 1, "", str(exc)


def _safe_host(value: str) -> str:
    host = str(value or "").strip().rstrip(".")
    if not host or len(host) > 253:
        raise ValueError("host is required and must be <=253 characters")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if not re.fullmatch(r"(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", host):
        raise ValueError("host must be an IP address or DNS hostname")
    return host


def _safe_dns_name(value: str) -> str:
    name = str(value or "").strip().rstrip(".")
    if not name or len(name) > 253 or ".." in name:
        raise ValueError("DNS name is required and must be <=253 characters")
    try:
        ipaddress.ip_address(name)
        return name
    except ValueError:
        pass
    labels = name.split(".")
    if any(not label or len(label) > 63 or not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels):
        raise ValueError("invalid DNS query name")
    return name


def neighbor_snapshot(limit: int = 100) -> str:
    """Return the kernel ARP/NDP neighbor table as bounded structured JSON."""
    limit = max(1, min(int(limit), 300))
    code, stdout, stderr = _run(["ip", "-j", "neigh", "show"], timeout=5)
    if code != 0:
        return f"Error: ip neigh failed: {(stderr or stdout).strip()}"
    try:
        data = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return "Error: ip neigh returned malformed JSON."
    return json.dumps(data[:limit], ensure_ascii=False, indent=2)


def connection_snapshot(limit: int = 150, state: str = "") -> str:
    """Return established/listening TCP and UDP sockets with owning process data when permissions permit."""
    limit = max(1, min(int(limit), 500))
    state = str(state or "").strip().lower()
    allowed_states = {"", "established", "listen", "time-wait", "close-wait", "syn-sent", "syn-recv"}
    if state not in allowed_states:
        return "Error: unsupported state filter."
    argv = ["ss", "-H", "-tuna", "-p"]
    if state:
        argv += ["state", state]
    code, stdout, stderr = _run(argv, timeout=8)
    if code != 0 and not stdout.strip():
        return f"Error: ss failed: {stderr.strip()}"
    rows = []
    for line in stdout.splitlines()[:limit]:
        # ss columns are stable enough for a bounded split; process text is retained as opaque data.
        parts = line.split(None, 6)
        rows.append({
            "netid": parts[0] if len(parts) > 0 else "",
            "state": parts[1] if len(parts) > 1 else "",
            "recv_q": parts[2] if len(parts) > 2 else "",
            "send_q": parts[3] if len(parts) > 3 else "",
            "local": parts[4] if len(parts) > 4 else "",
            "peer": parts[5] if len(parts) > 5 else "",
            "process": parts[6] if len(parts) > 6 else "",
        })
    return json.dumps({"connections": rows, "process_info_may_be_limited": True}, ensure_ascii=False, indent=2)


def dns_diagnose(name: str, record_types: list[str] | None = None, resolver: str = "") -> str:
    """Run bounded DNS diagnostics for common record types and return answers, status, flags, and latency."""
    try:
        host = _safe_dns_name(name)
    except ValueError as exc:
        return f"Error: {exc}"
    record_types = record_types or ["A", "AAAA"]
    allowed = {"A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV", "PTR"}
    requested = []
    for value in record_types[:8]:
        rr = str(value).upper().strip()
        if rr not in allowed:
            return f"Error: unsupported DNS record type '{rr}'."
        if rr not in requested:
            requested.append(rr)
    result = {"name": host, "resolver": resolver or "system", "queries": []}
    for rr in requested:
        argv = ["dig", "+time=3", "+tries=1", "+comments", "+answer", "+authority", host, rr]
        if resolver:
            try:
                resolver_host = _safe_host(resolver)
            except ValueError as exc:
                return f"Error: invalid resolver: {exc}"
            argv.insert(1, "@" + resolver_host)
        started = time.monotonic()
        code, stdout, stderr = _run(argv, timeout=5)
        elapsed = round((time.monotonic() - started) * 1000, 1)
        status = "UNKNOWN"
        flags = []
        answers = []
        authority = []
        section = ""
        for line in stdout.splitlines():
            stripped = line.strip()
            if "HEADER" in stripped and "status:" in stripped:
                match = re.search(r"status:\s*([A-Z]+)", stripped)
                if match:
                    status = match.group(1)
            if stripped.startswith(";; flags:"):
                flags = stripped.split("flags:", 1)[1].split(";", 1)[0].split()
            if stripped == ";; ANSWER SECTION:":
                section = "answer"; continue
            if stripped == ";; AUTHORITY SECTION:":
                section = "authority"; continue
            if stripped.startswith(";;") or not stripped:
                continue
            if section == "answer":
                answers.append(stripped)
            elif section == "authority":
                authority.append(stripped)
        result["queries"].append({
            "type": rr, "ok": code == 0 and status in {"NOERROR", "NXDOMAIN"}, "status": status,
            "flags": flags, "dnssec_authenticated": "ad" in flags, "answers": answers[:50],
            "authority": authority[:20], "elapsed_ms": elapsed, "stderr": stderr.strip()[:300] if code else "",
        })
    return json.dumps(result, ensure_ascii=False, indent=2)


def network_path(target: str, max_hops: int = 20, probes: int = 3) -> str:
    """Run a bounded MTR path/loss probe to an IP or hostname and return JSON."""
    try:
        host = _safe_host(target)
    except ValueError as exc:
        return f"Error: {exc}"
    max_hops = max(1, min(int(max_hops), 30))
    probes = max(1, min(int(probes), 5))
    if not shutil.which("mtr"):
        return "Error: mtr is not installed."
    code, stdout, stderr = _run(["mtr", "--json", "--report", "--report-cycles", str(probes), "--max-ttl", str(max_hops), host], timeout=max(10, probes * 8))
    if code != 0 and not stdout.strip():
        return f"Error: mtr failed: {stderr.strip()}"
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return f"Error: mtr returned malformed JSON: {stdout[:500]}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def endpoint_probe(host: str, port: int, tls: bool = False, timeout: float = 5.0) -> str:
    """Measure DNS, TCP connect, and optional TLS handshake timing for one endpoint."""
    try:
        target = _safe_host(host)
    except ValueError as exc:
        return f"Error: {exc}"
    port = int(port)
    if port < 1 or port > 65535:
        return "Error: port must be between 1 and 65535."
    timeout = max(0.5, min(float(timeout), 15.0))
    result: dict = {"host": target, "port": port, "tls": bool(tls)}
    dns_start = time.monotonic()
    try:
        infos = socket.getaddrinfo(target, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        result.update({"ok": False, "stage": "dns", "error": str(exc), "dns_ms": round((time.monotonic()-dns_start)*1000, 1)})
        return json.dumps(result, indent=2)
    result["dns_ms"] = round((time.monotonic() - dns_start) * 1000, 1)
    result["addresses"] = list(dict.fromkeys(item[4][0] for item in infos if item[4]))[:8]
    last_error = ""
    for family, socktype, proto, _, sockaddr in infos[:8]:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            started = time.monotonic(); sock.connect(sockaddr)
            result["tcp_connect_ms"] = round((time.monotonic() - started) * 1000, 1)
            result["connected_address"] = sockaddr[0]
            if tls:
                context = ssl.create_default_context()
                tls_start = time.monotonic()
                wrapped = context.wrap_socket(sock, server_hostname=target)
                result["tls_handshake_ms"] = round((time.monotonic() - tls_start) * 1000, 1)
                cert = wrapped.getpeercert() or {}
                result["tls_version"] = wrapped.version()
                result["cipher"] = wrapped.cipher()[0] if wrapped.cipher() else ""
                result["certificate"] = {"subject": cert.get("subject"), "issuer": cert.get("issuer"), "notBefore": cert.get("notBefore"), "notAfter": cert.get("notAfter")}
                wrapped.close()
            else:
                sock.close()
            result.update({"ok": True, "stage": "complete"})
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            last_error = str(exc)
            try: sock.close()
            except OSError: pass
    result.update({"ok": False, "stage": "connect" if not tls else "connect_or_tls", "error": last_error})
    return json.dumps(result, ensure_ascii=False, indent=2)


def _looks_like_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value); return True
    except ValueError:
        return False


def http_probe(url: str, timeout: float = 8.0, allow_private: bool = False) -> str:
    """Probe DNS/TCP/TLS/HTTP stages for one URL and return timing, status, headers, and certificate metadata."""
    from .netutil import validate_public_url
    raw = str(url or "").strip()
    try:
        current = validate_public_url(raw, allow_private=bool(allow_private))
    except Exception as exc:
        return f"Error: URL validation failed: {exc}"
    parsed = urlparse(current)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        host = parsed.hostname or ""
        result = json.loads(endpoint_probe(host, port, tls=parsed.scheme == "https", timeout=min(float(timeout), 6.0)))
    except Exception as exc:
        return f"Error: endpoint probe failed: {exc}"
    if not result.get("ok"):
        result["url"] = current
        return json.dumps(result, ensure_ascii=False, indent=2)

    session = requests.Session()
    started = time.monotonic()
    response = None
    redirects = []
    try:
        for _ in range(4):
            current = validate_public_url(current, allow_private=bool(allow_private))
            response = session.get(
                current, timeout=max(1.0, min(float(timeout), 15.0)), stream=True,
                allow_redirects=False, headers={"User-Agent": "LocalAgent/1.0"},
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    break
                next_url = urljoin(current, location)
                validate_public_url(next_url, allow_private=bool(allow_private))
                redirects.append({"status": response.status_code, "from": current, "to": next_url})
                response.close(); response = None; current = next_url
                continue
            break
        if response is None:
            raise RuntimeError("redirect chain did not produce a final response")
        result.update({
            "url": raw, "final_url": current, "redirects": redirects,
            "http_status": response.status_code, "http_ok": response.ok,
            "time_to_headers_ms": round((time.monotonic() - started) * 1000, 1),
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "server": response.headers.get("Server", ""),
            "cache_control": response.headers.get("Cache-Control", ""),
        })
    except Exception as exc:
        result.update({"http_ok": False, "http_error": str(exc), "redirects": redirects})
    finally:
        if response is not None:
            response.close()
        session.close()
    return json.dumps(result, ensure_ascii=False, indent=2)

