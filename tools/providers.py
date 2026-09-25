"""Declarative tool policy plus auto-discovered builtin provider groups."""
from __future__ import annotations

import importlib
import pkgutil
from . import provider_groups

MUTATING_TOOLS = {'set_goal', 'clear_goal', 'set_user_identity', 'set_research_preference', 'set_profile_image', 'image_crop', 'enqueue_self_optimization', 'write_file', 'remove_path', 'execute_python', 'reload_tools', 'render_document_page', 'remember_semantic', 'cancel_reminder', 'create_or_update_tool', 'install_package', 'enqueue_research', 'cancel_background_job', 'start_computation', 'cancel_computation', 'schedule_reminder', 'queue_work', 'generate_pdf_report', 'save_recipe', 'image_convert', 'execute_shell', 'page_diff', 'browser_step', 'take_web_screenshot', 'map_network', 'notify_desktop', 'archive_extract', 'update_work_status', 'remember', 'image_resize'}

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

# Initial tool schemas are supplied by the per-turn discovery session.
ALWAYS_TOOL_NAMES: set[str] = set()
