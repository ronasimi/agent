from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

_ALLOWED_BINOPS={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,ast.Div:lambda a,b:a/b,ast.FloorDiv:lambda a,b:a//b,ast.Mod:lambda a,b:a%b,ast.Pow:lambda a,b:a**b}
_ALLOWED_UNARY={ast.UAdd:lambda a:+a,ast.USub:lambda a:-a}

def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without eval or code execution."""
    try:
        tree=ast.parse(expression,mode="eval")
        def ev(n):
            if isinstance(n,ast.Expression):return ev(n.body)
            if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):return n.value
            if isinstance(n,ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:return _ALLOWED_BINOPS[type(n.op)](ev(n.left),ev(n.right))
            if isinstance(n,ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY:return _ALLOWED_UNARY[type(n.op)](ev(n.operand))
            raise ValueError("unsupported expression")
        value=ev(tree)
        if isinstance(value,float) and (math.isnan(value) or math.isinf(value)):raise ValueError("non-finite result")
        return _json({"expression":expression,"result":value})
    except Exception as exc:return f"Error: calculate failed: {exc}"

def parse_datetime(value: str) -> str:
    """Parse an ISO-8601 datetime and return normalized fields."""
    try:
        dt=datetime.fromisoformat(value.replace("Z","+00:00")); return _json({"iso":dt.isoformat(),"timestamp":dt.timestamp(),"timezone":str(dt.tzinfo or "naive")})
    except Exception as exc:return f"Error: parse_datetime failed: {exc}"

def time_difference(a: str, b: str) -> str:
    """Return the signed time difference b-a for two ISO-8601 datetimes."""
    try:
        x=datetime.fromisoformat(a.replace("Z","+00:00")); y=datetime.fromisoformat(b.replace("Z","+00:00")); sec=(y-x).total_seconds(); return _json({"seconds":sec,"minutes":sec/60,"hours":sec/3600,"days":sec/86400})
    except Exception as exc:return f"Error: time_difference failed: {exc}"

def url_parse(url: str) -> str:
    """Parse a URL into deterministic components."""
    p=urlparse(url); return _json({"scheme":p.scheme,"hostname":p.hostname,"port":p.port,"path":p.path,"query":p.query,"fragment":p.fragment})

def url_join(base: str, relative: str) -> str:
    """Resolve a relative URL against a base URL."""
    return _json({"url":urljoin(base,relative)})

def ip_parse(address: str) -> str:
    """Parse an IPv4/IPv6 address and return normalized properties."""
    try:
        ip=ipaddress.ip_address(address); return _json({"address":str(ip),"version":ip.version,"private":ip.is_private,"loopback":ip.is_loopback,"link_local":ip.is_link_local,"multicast":ip.is_multicast})
    except Exception as exc:return f"Error: ip_parse failed: {exc}"

def subnet_contains(network: str, address: str) -> str:
    """Return whether an address belongs to a CIDR network."""
    try:return _json({"network":network,"address":address,"contains":ipaddress.ip_address(address) in ipaddress.ip_network(network,strict=False)})
    except Exception as exc:return f"Error: subnet_contains failed: {exc}"

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert common length, mass, temperature, time, and byte units deterministically."""
    fu=from_unit.lower().strip(); tu=to_unit.lower().strip(); v=float(value)
    groups={
        "length":{"m":1,"km":1000,"cm":.01,"mm":.001,"mi":1609.344,"ft":.3048,"in":.0254},
        "mass":{"kg":1,"g":.001,"lb":.45359237,"oz":.028349523125},
        "time":{"s":1,"min":60,"h":3600,"day":86400},
        "bytes":{"b":1,"kb":1000,"mb":1000**2,"gb":1000**3,"kib":1024,"mib":1024**2,"gib":1024**3},
    }
    if fu in {"c","f","k"} and tu in {"c","f","k"}:
        c=v if fu=="c" else (v-32)*5/9 if fu=="f" else v-273.15; result=c if tu=="c" else c*9/5+32 if tu=="f" else c+273.15; return _json({"value":result,"unit":tu})
    for group in groups.values():
        if fu in group and tu in group:return _json({"value":v*group[fu]/group[tu],"unit":tu})
    return "Error: incompatible or unsupported units."

def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Hash text with sha256, sha1, or md5."""
    algorithm=algorithm.lower()
    if algorithm not in {"sha256","sha1","md5"}:return "Error: unsupported hash algorithm."
    h=hashlib.new(algorithm); h.update(str(text).encode()); return _json({"algorithm":algorithm,"digest":h.hexdigest()})

def base64_encode(text: str) -> str:
    """Base64-encode UTF-8 text."""
    import base64
    return base64.b64encode(str(text).encode()).decode()

def base64_decode(data: str) -> str:
    """Base64-decode UTF-8 text with strict validation."""
    import base64
    try:return base64.b64decode(str(data),validate=True).decode("utf-8",errors="replace")
    except Exception as exc:return f"Error: base64_decode failed: {exc}"

def format_datetime(timestamp: str, timezone_name: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format an ISO-8601 datetime in a named IANA timezone."""
    try:
        from zoneinfo import ZoneInfo
        dt=datetime.fromisoformat(timestamp.replace("Z","+00:00")); return dt.astimezone(ZoneInfo(timezone_name)).strftime(format)
    except Exception as exc:return f"Error: format_datetime failed: {exc}"

def url_endpoint(url: str) -> str:
    """Parse an HTTP(S) URL into host, effective port, scheme, and TLS boolean."""
    try:
        parsed=urlparse(str(url));
        if parsed.scheme not in {"http","https"} or not parsed.hostname:return "Error: URL must use http or https and include a host."
        port=parsed.port or (443 if parsed.scheme=="https" else 80)
        return _json({"url":str(url),"scheme":parsed.scheme,"host":parsed.hostname,"port":port,"tls":parsed.scheme=="https"})
    except Exception as exc:return f"Error: url_endpoint failed: {exc}"
