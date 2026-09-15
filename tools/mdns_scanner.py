import time
import json

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
except ImportError:
    Zeroconf = None

class NetworkServiceListener(ServiceListener):
    def __init__(self):
        self.services = []

    def remove_service(self, zc, type_, name):
        pass

    def update_service(self, zc, type_, name):
        pass

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info:
            # Parse addresses cleanly
            addr_list = info.parsed_addresses() if hasattr(info, 'parsed_addresses') else []
            if not addr_list and info.addresses:
                addr_list = [".".join(map(str, addr)) for addr in info.addresses]

            # Decode properties safely
            props = {}
            if info.properties:
                for k, v in info.properties.items():
                    key_str = k.decode('utf-8', 'ignore') if isinstance(k, bytes) else str(k)
                    val_str = v.decode('utf-8', 'ignore') if isinstance(v, bytes) else str(v)
                    props[key_str] = val_str

            self.services.append({
                "device_name": name.split('.')[0],
                "service_type": type_,
                "server": info.server,
                "ip_addresses": addr_list,
                "port": info.port,
                "properties": props
            })

def scan_mdns(timeout: int = 5) -> str:
    """Scan the local network and cross-subnet repeaters using mDNS to discover live devices and their OS/hardware types.
    
    Args:
        timeout: Number of seconds to listen for mDNS broadcast packets.
    """
    if Zeroconf is None:
        return "Error: 'zeroconf' python package is missing. Use execute_shell to run: pip install zeroconf"

    try:
        zc = Zeroconf()
        listener = NetworkServiceListener()
        
        # Comprehensive list of service types that reveal OS and hardware details
        service_types = [
            "_http._tcp.local.", 
            "_ssh._tcp.local.", 
            "_smb._tcp.local.", 
            "_printer._tcp.local.", 
            "_ipp._tcp.local.", 
            "_googlecast._tcp.local.", 
            "_workstation._tcp.local.", 
            "_device-info._tcp.local.",
            "_afpovertcp._tcp.local.",
            "_nvstream._tcp.local."
        ]
        
        browsers = [ServiceBrowser(zc, st, listener) for st in service_types]
        
        time.sleep(timeout)
        zc.close()
        
        if not listener.services:
            return "No mDNS services discovered during the listening window."
            
        # Deduplicate services by device name and type
        unique_services = {f"{s['device_name']}-{s['service_type']}": s for s in listener.services}.values()
        return json.dumps(list(unique_services), indent=2)
        
    except Exception as e:
        return f"mDNS scan encountered an error: {str(e)}"
