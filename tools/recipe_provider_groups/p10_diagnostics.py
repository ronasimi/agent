"""Primitive compatibility recipes for high-level diagnostic tools."""
from __future__ import annotations

import os

P = lambda name, default, description: {"default": default, "description": description}

RECIPE_SPECS = [
    {
        "key": "compat.host_snapshot", "version": 1, "name": "compat.host_snapshot", "target_tool": "host_snapshot",
        "description": "Compose a host health snapshot from clock, identity, kernel, load, uptime, memory, filesystem, temperature, and GPU primitives.",
        "tags": ["compat", "host_snapshot", "host", "health", "primitive"], "parameters": {},
        "pipeline": [
            {"id":"clock","tool":"current_time","args":{}},
            {"id":"name","tool":"hostname","args":{}},
            {"id":"kernel","tool":"kernel_info","args":{}},
            {"id":"load","tool":"load_average","args":{}},
            {"id":"up","tool":"uptime","args":{}},
            {"id":"mem","tool":"memory_info","args":{}},
            {"id":"disk","tool":"filesystem_usage","args":{"path":"/","host":True}},
            {"id":"temps","tool":"temperature_sensors","args":{},"optional":True},
            {"id":"gpu","tool":"gpu_info","args":{},"optional":True},
            {"id":"result","tool":"compose_object","args":{"data":{
                "observed_at":{"$ref":"clock","path":"utc"},"observed_at_local":{"$ref":"clock","path":"local"},"timezone":{"$ref":"clock","path":"timezone"},
                "hostname":{"$ref":"name","path":"host_hostname"},"kernel":{"$ref":"kernel"},"load":{"$ref":"load"},"uptime":{"$ref":"up"},
                "memory":{"$ref":"mem"},"disk":{"$ref":"disk"},"temperatures":{"$ref":"temps"},"gpu":{"$ref":"gpu"}
            }}}
        ],
    },
    {
        "key":"compat.gpu_snapshot_dict","version":1,"name":"compat.gpu_snapshot_dict","target_tool":"gpu_snapshot_dict",
        "description":"Read optional GPU telemetry through the focused gpu_info primitive.","tags":["compat","gpu","telemetry"],"parameters":{},
        "pipeline":[{"id":"result","tool":"gpu_info","args":{}}],
    },
    {
        "key":"compat.network_snapshot","version":1,"name":"compat.network_snapshot","target_tool":"network_snapshot",
        "description":"Compose interfaces, routes, resolver settings, and listening sockets from focused network primitives.","tags":["compat","network_snapshot","network","routes","dns","sockets"],"parameters":{},
        "pipeline":[
            {"id":"ifaces","tool":"interface_list","args":{}},
            {"id":"routes","tool":"route_list","args":{"limit":200}},
            {"id":"dns","tool":"dns_servers","args":{}},
            {"id":"listening","tool":"socket_list","args":{"limit":200,"state":"listen"}},
            {"id":"result","tool":"compose_object","args":{"data":{"interfaces":{"$ref":"ifaces"},"routes":{"$ref":"routes"},"dns":{"$ref":"dns","path":"servers","default":[]},"listening":{"$ref":"listening","path":"connections","default":[]}}}},
        ],
    },
    {
        "key":"compat.network_reachability","version":1,"name":"compat.network_reachability","target_tool":"network_reachability",
        "description":"Probe a bounded list of public HTTP(S) targets using a foreach pipeline instead of a monolithic reachability helper.","tags":["compat","network","reachability","http"],
        "parameters":{"targets":P("targets",["https://www.cloudflare.com/cdn-cgi/trace","https://example.com/"],"Up to eight HTTP(S) targets")},
        "pipeline":[
            {"id":"checks","tool":"http_request","foreach":{"$param":"targets","default":["https://www.cloudflare.com/cdn-cgi/trace","https://example.com/"]},"args":{"url":{"$item":"$"},"method":"GET","timeout":5.0,"allow_private":False}},
            {"id":"result","tool":"compose_list","args":{"items":{"$ref":"checks"}}},
        ],
    },
    {
        "key":"compat.read_host_file","version":2,"name":"compat.read_host_file","target_tool":"read_host_file",
        "description":"Read bounded text from the read-only host mount using the focused host_read_text primitive.","tags":["compat","host","file","read"],
        "parameters":{"filepath":P("filepath","/etc/os-release","Host path under /host")},
        "pipeline":[{"id":"result","tool":"host_read_text","args":{"path":{"$param":"filepath","default":"/etc/os-release"},"max_chars":20000}}],
    },
    {
        "key":"compat.process_snapshot","version":1,"name":"compat.process_snapshot","target_tool":"process_snapshot",
        "description":"Build a top-process snapshot by mapping the requested sort mode, listing processes, sorting structured rows, and taking the requested head.","tags":["compat","process_snapshot","process","cpu","memory","io"],
        "parameters":{"limit":P("limit",10,"Number of process rows"),"sort_by":P("sort_by","cpu","cpu, memory, or io"),"include_command":P("include_command",False,"Include bounded process command lines")},
        "pipeline":[
            {"id":"key","tool":"map_value","args":{"value":{"$param":"sort_by","default":"cpu"},"mapping":{"cpu":"cpu_percent","memory":"rss_mb","io":"io_bytes"},"default":"cpu_percent"}},
            {"id":"rows","tool":"list_processes","args":{"limit":500,"include_command":{"$param":"include_command","default":False},"include_io":True}},
            {"id":"sorted","tool":"json_sort","args":{"key":{"$ref":"key"},"data":{"$ref":"rows"},"descending":True,"limit":500}},
            {"id":"top","tool":"json_head","args":{"data":{"$ref":"sorted"},"count":{"$param":"limit","default":10}}},
            {"id":"result","tool":"compose_object","args":{"data":{"sort_by":{"$param":"sort_by","default":"cpu"},"processes":{"$ref":"top"}}}},
        ],
    },
    {
        "key":"compat.pressure_snapshot","version":1,"name":"compat.pressure_snapshot","target_tool":"pressure_snapshot",
        "description":"Read Linux PSI pressure through the focused pressure_info primitive.","tags":["compat","pressure","psi","host"],"parameters":{},
        "pipeline":[{"id":"result","tool":"pressure_info","args":{"resource":"all"}}],
    },
    {
        "key":"compat.filesystem_snapshot","version":1,"name":"compat.filesystem_snapshot","target_tool":"filesystem_snapshot",
        "description":"Enumerate host mounts and collect bounded capacity/inode data for each visible mountpoint.","tags":["compat","filesystem","mounts","disk"],
        "parameters":{"limit":P("limit",20,"Maximum mount rows to inspect")},
        "pipeline":[
            {"id":"mounts","tool":"mounts","args":{}},
            {"id":"usage","tool":"filesystem_usage","foreach":{"$ref":"mounts","path":"mounts"},"args":{"path":{"$item":"mountpoint"},"host":True},"optional":True},
            {"id":"limited","tool":"json_head","args":{"data":{"$ref":"usage"},"count":{"$param":"limit","default":20}}},
            {"id":"result","tool":"compose_object","args":{"data":{"source":{"$ref":"mounts","path":"source","default":""},"filesystems":{"$ref":"limited"}}}},
        ],
    },
    {
        "key":"compat.neighbor_snapshot","version":1,"name":"compat.neighbor_snapshot","target_tool":"neighbor_snapshot",
        "description":"Read the ARP/NDP table through the independent neighbor_list primitive.","tags":["compat","network","neighbors","arp","ndp"],
        "parameters":{"limit":P("limit",100,"Maximum neighbor rows")},
        "pipeline":[{"id":"result","tool":"neighbor_list","args":{"limit":{"$param":"limit","default":100}}}],
    },
    {
        "key":"compat.connection_snapshot","version":1,"name":"compat.connection_snapshot","target_tool":"connection_snapshot",
        "description":"Read TCP/UDP sockets through the independent socket_list primitive.","tags":["compat","network","connections","sockets"],
        "parameters":{"limit":P("limit",150,"Maximum socket rows"),"state":P("state","","Optional socket state")},
        "pipeline":[{"id":"result","tool":"socket_list","args":{"limit":{"$param":"limit","default":150},"state":{"$param":"state","default":""}}}],
    },
    {
        "key":"compat.dns_diagnose","version":1,"name":"compat.dns_diagnose","target_tool":"dns_diagnose",
        "description":"Map a bounded set of DNS record types through dns_query and aggregate the results.","tags":["compat","dns","diagnose","dig"],
        "parameters":{"name":P("name","example.com","DNS name"),"record_types":P("record_types",["A","AAAA"],"Up to eight DNS record types"),"resolver":P("resolver","","Optional DNS resolver")},
        "pipeline":[
            {"id":"queries","tool":"dns_query","foreach":{"$param":"record_types","default":["A","AAAA"]},"args":{"name":{"$param":"name","default":"example.com"},"record_type":{"$item":"$"},"resolver":{"$param":"resolver","default":""}}},
            {"id":"result","tool":"compose_object","args":{"data":{"name":{"$param":"name","default":"example.com"},"resolver":{"$param":"resolver","default":""},"queries":{"$ref":"queries"}}}},
        ],
    },
    {
        "key":"compat.network_path","version":1,"name":"compat.network_path","target_tool":"network_path",
        "description":"Trace a bounded route through the focused trace_route primitive.","tags":["compat","network","path","mtr","traceroute"],
        "parameters":{"target":P("target","example.com","Target host/IP"),"max_hops":P("max_hops",20,"Maximum hops"),"probes":P("probes",3,"Probe cycles")},
        "pipeline":[{"id":"result","tool":"trace_route","args":{"target":{"$param":"target","default":"example.com"},"max_hops":{"$param":"max_hops","default":20},"probes":{"$param":"probes","default":3}}}],
    },
    {
        "key":"compat.endpoint_probe","version":2,"name":"compat.endpoint_probe","target_tool":"endpoint_probe",
        "description":"Choose a direct TCP or TLS primitive based on the tls parameter, preserving DNS/connect timing evidence.","tags":["compat","endpoint","tcp","tls","probe"],
        "parameters":{"host":P("host","example.com","Host/IP"),"port":P("port",443,"TCP port"),"tls":P("tls",False,"Perform TLS handshake"),"timeout":P("timeout",5.0,"Timeout seconds")},
        "pipeline":[
            {"id":"tcp","tool":"tcp_connect","when":{"$not":{"$param":"tls","default":False}},"args":{"host":{"$param":"host","default":"example.com"},"port":{"$param":"port","default":443},"timeout":{"$param":"timeout","default":5.0}}},
            {"id":"tls","tool":"tls_handshake","when":{"$param":"tls","default":False},"args":{"host":{"$param":"host","default":"example.com"},"port":{"$param":"port","default":443},"timeout":{"$param":"timeout","default":5.0}}},
            {"id":"result","tool":"choose_value","args":{"condition":{"$param":"tls","default":False},"if_true":{"$ref":"tls"},"if_false":{"$ref":"tcp"}}},
        ],
    },
    {
        "key":"compat.http_probe","version":2,"name":"compat.http_probe","target_tool":"http_probe",
        "description":"Decompose an HTTP(S) probe into URL parsing, TCP/TLS endpoint probing, and a bounded HTTP request.","tags":["compat","http","https","probe","tls"],
        "parameters":{"url":P("url","https://example.com","HTTP(S) URL"),"timeout":P("timeout",8.0,"Timeout seconds"),"allow_private":P("allow_private",False,"Allow private/local target")},
        "pipeline":[
            {"id":"endpoint","tool":"url_endpoint","args":{"url":{"$param":"url","default":"https://example.com"}}},
            {"id":"tcp","tool":"tcp_connect","when":{"$not":{"$ref":"endpoint","path":"tls"}},"args":{"host":{"$ref":"endpoint","path":"host"},"port":{"$ref":"endpoint","path":"port"},"timeout":{"$param":"timeout","default":8.0}}},
            {"id":"tls","tool":"tls_handshake","when":{"$ref":"endpoint","path":"tls"},"args":{"host":{"$ref":"endpoint","path":"host"},"port":{"$ref":"endpoint","path":"port"},"timeout":{"$param":"timeout","default":8.0}}},
            {"id":"transport","tool":"choose_value","args":{"condition":{"$ref":"endpoint","path":"tls"},"if_true":{"$ref":"tls"},"if_false":{"$ref":"tcp"}}},
            {"id":"http","tool":"http_request","args":{"url":{"$param":"url","default":"https://example.com"},"method":"GET","timeout":{"$param":"timeout","default":8.0},"allow_private":{"$param":"allow_private","default":False}}},
            {"id":"result","tool":"compose_object","args":{"data":{"url":{"$param":"url","default":"https://example.com"},"transport":{"$ref":"transport"},"http":{"$ref":"http"}}}},
        ],
    },
    {
        "key":"compat.ollama_runtime_snapshot","version":1,"name":"compat.ollama_runtime_snapshot","target_tool":"ollama_runtime_snapshot",
        "description":"Read Ollama /api/ps through the generic bounded JSON fetch primitive.","tags":["compat","ollama","runtime","models"],
        "parameters":{"url":P("url",os.environ.get("OLLAMA_HOST","http://127.0.0.1:11434").rstrip("/")+"/api/ps","Ollama /api/ps URL")},
        "pipeline":[{"id":"result","tool":"fetch_json","args":{"url":{"$param":"url","default":os.environ.get("OLLAMA_HOST","http://127.0.0.1:11434").rstrip("/")+"/api/ps"},"allow_private":True,"max_bytes":262144}}],
    },
]

NATIVE_ONLY = {
    "local_subnets": "derives active private IPv4 networks from live interface/routing state and remains an atomic read-only discovery primitive",
    "scan_subnet": "bounded active LAN discovery combines ARP/neighbor, DNS, service, and optional OS hints under one safety envelope; keep native rather than reproduce scanning policy in a recipe",
    "list_host_monitor_events": "durable monitor-event query; already a narrow storage accessor rather than a monolithic computation",
    "read_host_journal": "journalctl filtering and host journal namespace access are not yet decomposed into safe primitives",
    "tail_host_log": "host-log fallback logic spans file and journal backends and remains a narrow native accessor",
    "service_health": "multi-source systemd manager/journal/static fallback semantics are intentionally kept native",
    "map_network": "creates an artifact and performs active scanning; dynamic recipes are read-only by design",
    "scan_mdns": "active multicast discovery has no smaller stable primitive chain yet",
}
