"""Declarative tool policy plus auto-discovered builtin provider groups."""
from __future__ import annotations

import importlib
import pkgutil
from . import provider_groups

MUTATING_TOOLS = {'set_user_identity', 'set_research_preference', 'set_profile_image', 'image_crop', 'enqueue_self_optimization', 'write_file', 'execute_python', 'reload_tools', 'render_document_page', 'remember_semantic', 'cancel_reminder', 'create_or_update_tool', 'install_package', 'enqueue_research', 'cancel_background_job', 'schedule_reminder', 'queue_work', 'generate_pdf_report', 'save_recipe', 'image_convert', 'execute_shell', 'page_diff', 'take_web_screenshot', 'map_network', 'notify_desktop', 'archive_extract', 'update_work_status', 'remember', 'image_resize'}

REPEAT_SAFE_TOOLS = {'generate_pdf_report', 'take_web_screenshot', 'map_network'}

SAFE_ARTIFACT_TOOLS = {'generate_pdf_report', 'take_web_screenshot'}

def discover_builtin_specs() -> tuple[tuple[str, str], ...]:
    """Load provider-group manifests in deterministic module-name order."""
    specs: list[tuple[str, str]] = []
    prefix = provider_groups.__name__ + "."
    for info in sorted(pkgutil.iter_modules(provider_groups.__path__), key=lambda item: item.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(prefix + info.name)
        for spec in getattr(module, "TOOL_SPECS", ()):
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise RuntimeError(f"Invalid TOOL_SPECS entry in {module.__name__}: {spec!r}")
            specs.append((str(spec[0]), str(spec[1])))
    return tuple(specs)

BUILTINS = discover_builtin_specs()

TOOL_SELECTION_STOPWORDS = {'for', 'of', 'show', 'it', 'with', 'in', 'do', 'would', 'i', 'are', 'my', 'me', 'how', 'what', 'at', 'be', 'check', 'please', 'can', 'help', 'could', 'get', 'and', 'as', 'a', 'an', 'is', 'on', 'to', 'the', 'test', 'from', 'by', 'your', 'you'}

# Keep the default schema set empty. Relevant deterministic tools are selected
# by intent bundles/requirements, which reduces prompt prefill and prevents a
# small model from reaching for unrelated file/time tools on every turn.
ALWAYS_TOOL_NAMES: set[str] = set()

TOOL_BUNDLES = (
    ({'repo', 'script', 'project', 'python', 'file', 'files', 'coding', 'code'},
     ('read_file', 'path_stat', 'list_directory', 'find_paths', 'read_text', 'read_lines', 'directory_size', 'tail_file', 'file_hash', 'mime_type', 'text_search', 'regex_extract', 'regex_replace', 'json_query', 'run_pipeline', 'get_repo_map', 'search_repo_symbols', 'repo_status', 'repo_diff', 'repo_checks')),
    ({'edit', 'overwrite', 'save', 'create', 'write', 'modify'},
     ('write_file',)),
    ({'bash', 'shell', 'command', 'terminal'},
     ('execute_shell',)),
    ({'python3', 'execute code', 'run code', 'python'},
     ('execute_python',)),
    ({'benchmark', 'self', 'performance', 'optimization', 'refactor', 'optimize'},
     ('get_repo_map', 'search_repo_symbols', 'read_repo_symbol', 'repo_status', 'repo_diff', 'repo_checks', 'dependency_audit', 'tool_health', 'enqueue_self_optimization', 'get_self_optimization_status', 'list_self_optimization_candidates')),
    ({'source', 'url', 'sources', 'search', 'web', 'site', 'research', 'internet', 'news', 'headline', 'headlines', 'latest'},
     ('news_search', 'web_search', 'browse_url', 'page_metadata', 'page_links', 'discover_site', 'read_feed', 'extract_document', 'page_fingerprint', 'page_diff', 'fetch_url', 'extract_readable_text', 'extract_links', 'extract_metadata', 'extract_images', 'extract_jsonld', 'take_web_screenshot', 'enqueue_research', 'get_research_status', 'read_observation')),
    ({'weather', 'forecast', 'forecasts', 'precipitation', 'rainfall', 'snowfall'},
     ('geocode_location', 'weather_forecast', 'web_search', 'browse_url', 'read_observation')),
    ({'market', 'price', 'prices', 'quote', 'quotes', 'commodity', 'commodities', 'crude', 'oil', 'wti', 'brent', 'gold', 'silver'},
     ('market_quote', 'web_search', 'browse_url', 'read_observation')),
    ({'media', 'images', 'photo', 'screenshots', 'vision', 'image', 'screenshot', 'picture', 'visual', 'profile', 'avatar'},
     ('attach_media', 'image_info', 'mime_type', 'profile_image_info', 'set_profile_image')),
    ({'clock', 'time', 'local', 'today', 'utc', 'timezone', 'date'},
     ('current_time', 'hostname', 'environment_summary')),
    ({'convert', 'hash', 'calculate', 'calculation', 'encode', 'decode', 'conversion', 'base64', 'units', 'compare'},
     ('calculate', 'convert_units', 'hash_text', 'base64_encode', 'base64_decode', 'compare_values', 'compare_json', 'compare_text', 'run_pipeline')),
    ({'host', 'command', 'shell', 'cpu', 'disk', 'system', 'log', 'process', 'gpu', 'ram'},
     ('host_snapshot', 'process_snapshot', 'pressure_snapshot', 'filesystem_snapshot', 'service_health', 'kernel_info', 'cpu_info', 'os_release', 'memory_info', 'load_average', 'uptime', 'mounts', 'block_devices', 'temperature_sensors', 'list_processes', 'process_info', 'process_tree', 'read_host_file', 'read_host_journal', 'tail_host_log', 'read_observation')),
    ({'dns', 'mdns', 'network', 'wifi', 'lan', 'route', 'port', 'subnet', 'hosts'},
     ('local_subnets', 'scan_subnet', 'network_snapshot', 'neighbor_snapshot', 'connection_snapshot', 'dns_diagnose', 'network_path', 'endpoint_probe', 'http_probe', 'map_network', 'scan_mdns', 'resolve_host', 'route_lookup', 'tcp_connect', 'tls_handshake', 'ping_host', 'http_request', 'interface_list', 'interface_info', 'neighbor_list', 'socket_list')),
    ({'internet', 'online', 'external'},
     ('network_reachability', 'dns_diagnose', 'http_probe')),
    ({'tools', 'tooling', 'capabilities'},
     ('tool_health',)),
    ({'recipes', 'reuse', 'pipeline', 'workflow', 'recipe', 'reusable'},
     ('search_recipes', 'list_recipes', 'run_recipe', 'run_pipeline', 'save_recipe', 'recipe_coverage')),
    ({'path', 'hash', 'mime', 'folder', 'directory', 'find', 'json', 'tail', 'diff', 'regex', 'grep'},
     ('find_paths', 'list_directory', 'path_stat', 'read_text', 'read_lines', 'directory_size', 'tail_file', 'file_hash', 'mime_type', 'text_search', 'regex_extract', 'regex_replace', 'text_split', 'text_head', 'text_tail', 'text_count', 'text_sort', 'text_unique', 'json_query', 'json_filter', 'json_sort', 'json_head', 'json_count', 'json_keys', 'json_diff', 'yaml_query', 'csv_query', 'csv_summary', 'text_diff', 'run_pipeline')),
    ({'remember', 'preference', 'recall', 'profile', 'identity'},
     ('search_memory', 'remember', 'set_user_identity', 'set_research_preference', 'profile_image_info')),
    ({'timer', 'schedule', 'remind', 'reminder'},
     ('schedule_reminder', 'cancel_reminder', 'list_reminders')),
)
