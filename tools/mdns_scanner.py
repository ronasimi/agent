import json
import time

try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
except ImportError:  # optional dependency; the rest of the agent must still load
    Zeroconf = None
    ServiceBrowser = None
    ServiceListener = object


class NetworkServiceListener(ServiceListener):
    def __init__(self):
        self.services = []

    def remove_service(self, zc, type_, name):
        pass

    def update_service(self, zc, type_, name):
        pass

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if not info:
            return
        addr_list = info.parsed_addresses() if hasattr(info, "parsed_addresses") else []
        if not addr_list and getattr(info, "addresses", None):
            addr_list = [".".join(map(str, addr)) for addr in info.addresses]
        props = {}
        for key, value in (info.properties or {}).items():
            key_str = key.decode("utf-8", "ignore") if isinstance(key, bytes) else str(key)
            val_str = value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
            props[key_str] = val_str
        self.services.append({
            "device_name": name.split(".")[0],
            "service_type": type_,
            "server": info.server,
            "ip_addresses": addr_list,
            "port": info.port,
            "properties": props,
        })


def scan_mdns(timeout: int = 5) -> str:
    """Listen briefly for local mDNS services and return discovered devices as JSON."""
    if Zeroconf is None or ServiceBrowser is None:
        return "Error: zeroconf is not installed."
    timeout = max(1, min(int(timeout), 30))
    browsers = []
    zc = None
    try:
        zc = Zeroconf()
        listener = NetworkServiceListener()
        service_types = [
            "_http._tcp.local.", "_ssh._tcp.local.", "_smb._tcp.local.", "_printer._tcp.local.",
            "_ipp._tcp.local.", "_googlecast._tcp.local.", "_workstation._tcp.local.", "_device-info._tcp.local.",
            "_afpovertcp._tcp.local.", "_nvstream._tcp.local.",
        ]
        browsers = [ServiceBrowser(zc, service_type, listener) for service_type in service_types]
        time.sleep(timeout)
        unique = {}
        for service in listener.services:
            unique[f"{service['device_name']}|{service['service_type']}|{service['server']}"] = service
        return json.dumps(list(unique.values()), ensure_ascii=False, indent=2) if unique else "No mDNS services discovered during the listening window."
    except Exception as exc:
        return f"Error: mDNS scan failed: {exc}"
    finally:
        for browser in browsers:
            try:
                browser.cancel()
            except Exception:
                pass
        if zc:
            try:
                zc.close()
            except Exception:
                pass
