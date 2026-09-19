from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def interface_list() -> str:
    """Return network interfaces and addresses."""
    stats=psutil.net_if_stats(); addrs=psutil.net_if_addrs(); out=[]
    for name in sorted(set(stats)|set(addrs)):
        s=stats.get(name); out.append({"name":name,"up":bool(s.isup) if s else None,"mtu":s.mtu if s else None,"addresses":[{"family":str(a.family),"address":a.address,"netmask":a.netmask} for a in addrs.get(name,[])]})
    return _json(out)

def neighbor_list(limit: int = 100) -> str:
    """Return ARP/NDP neighbors through the existing structured network primitive."""
    from .network_diagnostics import neighbor_snapshot
    return neighbor_snapshot(limit)

def socket_list(limit: int = 150, state: str = "") -> str:
    """Return active/listening sockets through the existing structured connection primitive."""
    from .network_diagnostics import connection_snapshot
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
        proc=subprocess.run(["ip","-j","route","get",target],capture_output=True,text=True,timeout=5,stdin=subprocess.DEVNULL)
        if proc.returncode:return f"Error: route lookup failed: {(proc.stderr or proc.stdout).strip()}"
        return _json(json.loads(proc.stdout))
    except Exception as exc:return f"Error: route_lookup failed: {exc}"

def tcp_connect(host: str, port: int, timeout: float = 5.0) -> str:
    """Resolve and attempt a bounded TCP connection."""
    from .network_diagnostics import endpoint_probe
    return endpoint_probe(host,int(port),False,float(timeout))

def tls_handshake(host: str, port: int = 443, timeout: float = 5.0) -> str:
    """Perform DNS, TCP, and TLS handshake for one endpoint."""
    from .network_diagnostics import endpoint_probe
    return endpoint_probe(host,int(port),True,float(timeout))

def http_request(url: str, method: str = "HEAD", timeout: float = 8.0, allow_private: bool = False) -> str:
    """Perform one bounded HTTP GET/HEAD request after SSRF-safe URL validation."""
    from .netutil import validate_public_url
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
    from .network_diagnostics import network_path
    return network_path(target,max_hops,probes)
