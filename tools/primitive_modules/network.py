from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path
from ..subprocess_utils import run_argv

def interface_list() -> str:
    """Return network interfaces and addresses."""
    stats=psutil.net_if_stats(); addrs=psutil.net_if_addrs(); out=[]
    for name in sorted(set(stats)|set(addrs)):
        s=stats.get(name); out.append({"name":name,"up":bool(s.isup) if s else None,"mtu":s.mtu if s else None,"addresses":[{"family":str(a.family),"address":a.address,"netmask":a.netmask} for a in addrs.get(name,[])]})
    return _json(out)

def neighbor_list(limit: int = 100) -> str:
    """Return ARP/NDP neighbors through the existing structured network primitive."""
    from ..network_diagnostics import neighbor_snapshot
    return neighbor_snapshot(limit)

def socket_list(limit: int = 150, state: str = "") -> str:
    """Return active/listening sockets through the existing structured connection primitive."""
    from ..network_diagnostics import connection_snapshot
    return connection_snapshot(limit,state)

def resolve_host(host: str, record_type: str = "any") -> str:
    """Resolve a hostname to bounded IPv4/IPv6 address records."""
    try:
        family=socket.AF_UNSPEC if record_type.lower() in {"any","a+aaaa"} else socket.AF_INET if record_type.lower()=="a" else socket.AF_INET6
        infos=socket.getaddrinfo(host,None,family=family,type=socket.SOCK_STREAM)
        addresses=list(dict.fromkeys(i[4][0] for i in infos))[:16]
        return _json({"host":host,"record_type":record_type,"addresses":addresses})
    except Exception as exc:return f"Error: resolve_host failed: {exc}"

def route_lookup(target: str) -> str:
    """Return the kernel route selected for a target using ip route get."""
    try:
        proc=run_argv(["ip","-j","route","get",target], timeout=5)
        if proc.returncode:return f"Error: route lookup failed: {(proc.stderr or proc.stdout).strip()}"
        return _json(json.loads(proc.stdout))
    except Exception as exc:return f"Error: route_lookup failed: {exc}"

def tcp_connect(host: str, port: int, timeout: float = 5.0) -> str:
    """Resolve and attempt a bounded TCP connection."""
    from ..network_diagnostics import endpoint_probe
    return endpoint_probe(host,int(port),False,float(timeout))

def tls_handshake(host: str, port: int = 443, timeout: float = 5.0) -> str:
    """Perform DNS, TCP, and TLS handshake for one endpoint."""
    from ..network_diagnostics import endpoint_probe
    return endpoint_probe(host,int(port),True,float(timeout))

def http_request(url: str, method: str = "HEAD", timeout: float = 8.0, allow_private: bool = False) -> str:
    """Perform one bounded HTTP GET/HEAD request after SSRF-safe URL validation."""
    from ..netutil import validate_public_url
    try:
        safe=validate_public_url(url,allow_private=bool(allow_private)); method=method.upper()
        if method not in {"GET","HEAD"}:return "Error: method must be GET or HEAD."
        started=time.monotonic(); r=requests.request(method,safe,timeout=max(.5,min(float(timeout),15)),allow_redirects=False,stream=(method=="GET"))
        body=""
        if method=="GET": body=next(r.iter_content(chunk_size=8192),b"")[:8192].decode("utf-8",errors="replace")
        return _json({"url":safe,"status":r.status_code,"elapsed_ms":round((time.monotonic()-started)*1000,1),"headers":dict(list(r.headers.items())[:50]),"body_preview":body})
    except Exception as exc:return f"Error: http_request failed: {exc}"

def interface_info(name: str) -> str:
    """Return one network interface's status, addresses, and counters."""
    try:
        stats=psutil.net_if_stats().get(name); addrs=psutil.net_if_addrs().get(name); counters=psutil.net_io_counters(pernic=True).get(name)
        if not stats and not addrs:return "Error: interface not found."
        return _json({"name":name,"stats":stats._asdict() if stats else {},"addresses":[a._asdict() for a in (addrs or [])],"counters":counters._asdict() if counters else {}})
    except Exception as exc:return f"Error: interface_info failed: {exc}"

def udp_probe(host: str, port: int, timeout: float = 2.0) -> str:
    """Send an empty UDP datagram and report local socket behavior; lack of reply is not proof of failure."""
    try:
        infos=socket.getaddrinfo(host,int(port),type=socket.SOCK_DGRAM); family,socktype,proto,_,sockaddr=infos[0]
        s=socket.socket(family,socktype,proto); s.settimeout(max(.2,min(float(timeout),5.0))); started=time.monotonic(); s.connect(sockaddr); s.send(b""); local=s.getsockname(); elapsed=round((time.monotonic()-started)*1000,1); s.close()
        return _json({"host":host,"port":int(port),"sent":True,"local":local,"elapsed_ms":elapsed,"note":"UDP send success does not establish application-level reachability."})
    except Exception as exc:return f"Error: udp_probe failed: {exc}"

def trace_route(target: str, max_hops: int = 20, probes: int = 3) -> str:
    """Return a bounded structured network path using the existing MTR implementation."""
    from ..network_diagnostics import network_path
    return network_path(target,max_hops,probes)

def route_list(limit: int = 200) -> str:
    """Return the bounded kernel route table as structured JSON."""
    try:
        proc = run_argv(["ip", "-j", "route", "show"], timeout=5)
        if proc.returncode:
            return f"Error: route list failed: {(proc.stderr or proc.stdout).strip()}"
        rows = json.loads(proc.stdout or "[]")
        return _json(rows[:_bounded_int(limit, 1, 500)])
    except Exception as exc:
        return f"Error: route_list failed: {exc}"


def dns_servers() -> str:
    """Return configured DNS nameservers from the host resolver configuration when available."""
    candidates = [Path("/host/etc/resolv.conf"), Path("/etc/resolv.conf")]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        return _json({"source": "", "servers": []})
    try:
        servers = []
        for line in source.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == "nameserver" and parts[1] not in servers:
                servers.append(parts[1])
        return _json({"source": str(source), "servers": servers[:16]})
    except Exception as exc:
        return f"Error: dns_servers failed: {exc}"


def neighbor_list(limit: int = 100) -> str:
    """Return the ARP/NDP neighbor table as bounded structured JSON."""
    try:
        proc = run_argv(["ip", "-j", "neigh", "show"], timeout=5)
        if proc.returncode:
            return f"Error: ip neigh failed: {(proc.stderr or proc.stdout).strip()}"
        data = json.loads(proc.stdout or "[]")
        return _json(data[:_bounded_int(limit, 1, 300)])
    except Exception as exc:
        return f"Error: neighbor_list failed: {exc}"


def socket_list(limit: int = 150, state: str = "") -> str:
    """Return bounded TCP/UDP socket rows with owning process text when available."""
    limit = _bounded_int(limit, 1, 500)
    state = str(state or "").strip().lower()
    allowed = {"", "established", "listen", "time-wait", "close-wait", "syn-sent", "syn-recv"}
    if state not in allowed:
        return "Error: unsupported state filter."
    argv = ["ss", "-H", "-tuna", "-p"]
    if state:
        argv += ["state", "listening" if state == "listen" else state]
    try:
        proc = run_argv(argv, timeout=8)
        if proc.returncode and not proc.stdout.strip():
            return f"Error: ss failed: {proc.stderr.strip()}"
        rows = []
        for line in proc.stdout.splitlines()[:limit]:
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
        return _json({"connections": rows, "process_info_may_be_limited": True})
    except Exception as exc:
        return f"Error: socket_list failed: {exc}"


def dns_query(name: str, record_type: str = "A", resolver: str = "") -> str:
    """Run one bounded DNS query and return status, flags, answers, authority, and latency."""
    rr = str(record_type or "A").upper().strip()
    allowed = {"A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV", "PTR"}
    if rr not in allowed:
        return f"Error: unsupported DNS record type '{rr}'."
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,253}", str(name or "")):
        return "Error: invalid DNS query name."
    argv = ["dig", "+time=3", "+tries=1", "+comments", "+answer", "+authority", str(name), rr]
    if resolver:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,253}", str(resolver)):
            return "Error: invalid resolver."
        argv.insert(1, "@" + str(resolver))
    started = time.monotonic()
    try:
        proc = run_argv(argv, timeout=5)
    except Exception as exc:
        return f"Error: dns_query failed: {exc}"
    elapsed = round((time.monotonic() - started) * 1000, 1)
    status = "UNKNOWN"; flags = []; answers = []; authority = []; section = ""
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "HEADER" in stripped and "status:" in stripped:
            match = re.search(r"status:\s*([A-Z]+)", stripped)
            if match: status = match.group(1)
        if stripped.startswith(";; flags:"):
            flags = stripped.split("flags:", 1)[1].split(";", 1)[0].split()
        if stripped == ";; ANSWER SECTION:": section = "answer"; continue
        if stripped == ";; AUTHORITY SECTION:": section = "authority"; continue
        if stripped.startswith(";;") or not stripped: continue
        if section == "answer": answers.append(stripped)
        elif section == "authority": authority.append(stripped)
    return _json({
        "type": rr, "ok": proc.returncode == 0 and status in {"NOERROR", "NXDOMAIN"},
        "status": status, "flags": flags, "dnssec_authenticated": "ad" in flags,
        "answers": answers[:50], "authority": authority[:20], "elapsed_ms": elapsed,
        "stderr": proc.stderr.strip()[:300] if proc.returncode else "",
    })


def tcp_connect(host: str, port: int, timeout: float = 5.0) -> str:
    """Resolve and attempt a bounded TCP connection without invoking a higher-level probe tool."""
    timeout = max(0.5, min(float(timeout), 15.0)); port = int(port)
    if not 1 <= port <= 65535: return "Error: port must be between 1 and 65535."
    result = {"host": host, "port": port, "tls": False}
    dns_start = time.monotonic()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        result.update({"ok": False, "stage": "dns", "error": str(exc), "dns_ms": round((time.monotonic()-dns_start)*1000, 1)})
        return _json(result)
    result["dns_ms"] = round((time.monotonic()-dns_start)*1000, 1)
    result["addresses"] = list(dict.fromkeys(item[4][0] for item in infos if item[4]))[:8]
    last_error = ""
    for family, socktype, proto, _, sockaddr in infos[:8]:
        sock = socket.socket(family, socktype, proto); sock.settimeout(timeout)
        try:
            started = time.monotonic(); sock.connect(sockaddr)
            result.update({"tcp_connect_ms": round((time.monotonic()-started)*1000, 1), "connected_address": sockaddr[0], "ok": True, "stage": "complete"})
            sock.close(); return _json(result)
        except Exception as exc:
            last_error = str(exc)
            try: sock.close()
            except OSError: pass
    result.update({"ok": False, "stage": "connect", "error": last_error}); return _json(result)


def tls_handshake(host: str, port: int = 443, timeout: float = 5.0) -> str:
    """Perform DNS, TCP, and TLS handshake without invoking a higher-level probe tool."""
    timeout = max(0.5, min(float(timeout), 15.0)); port = int(port)
    if not 1 <= port <= 65535: return "Error: port must be between 1 and 65535."
    result = {"host": host, "port": port, "tls": True}
    dns_start = time.monotonic()
    try: infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        result.update({"ok": False, "stage": "dns", "error": str(exc), "dns_ms": round((time.monotonic()-dns_start)*1000, 1)}); return _json(result)
    result["dns_ms"] = round((time.monotonic()-dns_start)*1000, 1)
    result["addresses"] = list(dict.fromkeys(item[4][0] for item in infos if item[4]))[:8]
    last_error = ""
    for family, socktype, proto, _, sockaddr in infos[:8]:
        sock = socket.socket(family, socktype, proto); sock.settimeout(timeout)
        try:
            started=time.monotonic(); sock.connect(sockaddr); result["tcp_connect_ms"] = round((time.monotonic()-started)*1000,1); result["connected_address"] = sockaddr[0]
            context=ssl.create_default_context(); tls_start=time.monotonic(); wrapped=context.wrap_socket(sock,server_hostname=host)
            cert=wrapped.getpeercert() or {}; result.update({
                "tls_handshake_ms": round((time.monotonic()-tls_start)*1000,1), "tls_version": wrapped.version(),
                "cipher": wrapped.cipher()[0] if wrapped.cipher() else "",
                "certificate": {"subject": cert.get("subject"), "issuer": cert.get("issuer"), "notBefore": cert.get("notBefore"), "notAfter": cert.get("notAfter")},
                "ok": True, "stage": "complete",
            }); wrapped.close(); return _json(result)
        except Exception as exc:
            last_error=str(exc)
            try: sock.close()
            except OSError: pass
    result.update({"ok": False, "stage": "connect_or_tls", "error": last_error}); return _json(result)


def trace_route(target: str, max_hops: int = 20, probes: int = 3) -> str:
    """Return a bounded MTR path; parse JSON or the common report-text fallback."""
    max_hops = _bounded_int(max_hops, 1, 30); probes = _bounded_int(probes, 1, 5)
    if not shutil.which("mtr"): return "Error: mtr is not installed."
    try:
        proc = run_argv(["mtr", "--json", "--report", "--report-cycles", str(probes), "--max-ttl", str(max_hops), str(target)], timeout=max(10, probes*8))
    except Exception as exc: return f"Error: trace_route failed: {exc}"
    if proc.returncode and not proc.stdout.strip(): return f"Error: mtr failed: {proc.stderr.strip()}"
    try: return _json(json.loads(proc.stdout))
    except json.JSONDecodeError:
        from ..network_diagnostics import _parse_mtr_report_text
        payload = _parse_mtr_report_text(proc.stdout, str(target))
        if payload is None: return f"Error: mtr returned an unrecognized output format: {(proc.stdout or proc.stderr).strip()[:500]}"
        if proc.returncode: payload["warning"] = f"mtr exited with status {proc.returncode}: {proc.stderr.strip()[:300]}"
        return _json(payload)


def ping_host(host: str, count: int = 3, timeout: float = 2.0) -> str:
    """Run a bounded ICMP echo check and return packet-loss and latency output."""
    target = str(host or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,253}", target) or target.startswith("-"):
        return "Error: invalid host."
    binary = shutil.which("ping")
    if not binary:
        return "Error: ping is not installed."
    count = _bounded_int(count, 1, 8); timeout = max(0.2, min(float(timeout), 10.0))
    started = time.monotonic()
    try:
        proc = run_argv(
            [binary, "-n", "-c", str(count), "-W", str(max(1, int(timeout))), target],
            timeout=(count * timeout) + 3,
        )
        output = (proc.stdout or proc.stderr).strip()[:8000]
        loss = re.search(r"([0-9.]+)%\s*packet loss", output)
        timing = re.search(r"(?:rtt|round-trip).*?=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms", output)
        return _json({
            "host": target, "ok": proc.returncode == 0, "returncode": proc.returncode,
            "packet_loss_percent": float(loss.group(1)) if loss else None,
            "latency_ms": ({"min": float(timing.group(1)), "avg": float(timing.group(2)), "max": float(timing.group(3)), "mdev": float(timing.group(4))} if timing else {}),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1), "output": output,
        })
    except Exception as exc:
        return f"Error: ping_host failed: {exc}"
