import sys
import pkgutil
import importlib
import inspect
import doctest
import io
import contextlib
from pathlib import Path

ALL_TOOLS = []
AVAILABLE_TOOLS_MAP = {}

EXCLUDED_FUNCTIONS = {'init_db', 'load_tools', 'get_tools_prompt_summary', 'clear_chat_history', 'get_all_memories_prompt_summary'}

def load_tools():
    """Dynamically scan, test, and load all modules inside the tools directory."""
    ALL_TOOLS.clear()
    AVAILABLE_TOOLS_MAP.clear()
    errors = {}
    
    package_dir = Path(__file__).resolve().parent
    
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        full_module_name = f"{__name__}.{module_name}"
        
        try:
            if full_module_name in sys.modules:
                module = importlib.reload(sys.modules[full_module_name])
            else:
                module = importlib.import_module(f".{module_name}", package=__name__)
            
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                results = doctest.testmod(module)
            
            if results.failed > 0:
                errors[module_name] = capture.getvalue()
                continue
            
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__:
                    if not name.startswith('_') and name not in EXCLUDED_FUNCTIONS:
                        ALL_TOOLS.append(obj)
                        AVAILABLE_TOOLS_MAP[name] = obj
                        
        except Exception as e:
            errors[module_name] = str(e)
            
    return len(AVAILABLE_TOOLS_MAP), errors

def get_tools_prompt_summary() -> str:
    """Generate a clean markdown summary of all registered tools and their docstrings."""
    summary = "\n\n### Available Tools Inventory\n"
    for name, func in AVAILABLE_TOOLS_MAP.items():
        doc = inspect.getdoc(func) or "No description provided."
        first_line = doc.splitlines()[0]
        summary += f"- **{name}**: {first_line}\n"
    return summary

load_tools()

from .memory import (
    init_db, 
    _init_chat_db, 
    _init_checkpoint_db, 
    _load_chat_history_from_db, 
    _save_message_to_db, 
    clear_chat_history,
    get_all_memories_prompt_summary
)

__all__ = [
    'ALL_TOOLS',
    'AVAILABLE_TOOLS_MAP',
    'load_tools',
    'get_tools_prompt_summary',
    'init_db',
    '_init_chat_db',
    '_init_checkpoint_db',
    '_load_chat_history_from_db',
    '_save_message_to_db',
    'clear_chat_history',
    'get_all_memories_prompt_summary',
]
