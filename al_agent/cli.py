"""Prompt-toolkit CLI frontend for Al Agent."""
from __future__ import annotations
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
try:
    from prompt_toolkit.auto_suggestion import AutoSuggestFromHistory
    AUTO_SUGGEST = AutoSuggestFromHistory()
except ImportError:  # pragma: no cover
    AUTO_SUGGEST = None

from tools import _load_chat_history_from_db
from tools.job_tools import get_research_status
from tools.runtime import list_jobs
from .cli_commands import CliContext, dispatch_command
from .console import get_bottom_toolbar
from .prompts import build_system_prompt
from .state import FAST_MODEL, MAIN_OPTIONS, MAX_CTX, MODEL, OLLAMA, THINKING_DEFAULT
from .turn_engine import handle_user_turn


def print_jobs() -> None:
    jobs=list_jobs(limit=25)
    if not jobs: print('\n[System]: No durable background jobs.'); return
    print('\n[System]: Durable jobs:')
    for job in jobs: print(f"  {job['id'][:8]}  {job['status']:<10} {job['job_type']:<10} {job['title']}")


def print_job(job_id: str) -> None:
    print('\n'+get_research_status(job_id))


def main() -> None:
    history_file='/app/memory/.agent_history'; os.makedirs(os.path.dirname(history_file),exist_ok=True)
    style=Style.from_dict({'prompt':'ansigreen bold','input':'ansiwhite','toolbar':'bg:#1a202c #63b3ed bold','completion-menu.completion':'bg:#000000 #ffffff','completion-menu.completion.current':'bg:#005f5f #ffffff'})
    session=PromptSession(history=FileHistory(history_file),auto_suggest=AUTO_SUGGEST,style=style,bottom_toolbar=get_bottom_toolbar)
    messages=[{'role':'system','content':build_system_prompt()}]+_load_chat_history_from_db(limit=100)
    context=CliContext(messages=messages,thinking_enabled=THINKING_DEFAULT)
    print(f'Agent initialized with Main: {MODEL} | Fast: {FAST_MODEL} | Context: {MAX_CTX}')
    print(f'[System]: Preloading main model ({MODEL}) into VRAM...')
    try:
        OLLAMA.chat(model=MODEL,messages=[{'role':'user','content':'warmup'}],options=MAIN_OPTIONS,keep_alive=-1,think=False)
        print('[System]: Main model successfully loaded and pinned in VRAM.')
    except Exception as exc: print(f'[System]: Warning - failed to preload main model: {exc}')
    print('Commands: '+', '.join(command.name for command in __import__('al_agent.cli_commands',fromlist=['COMMANDS']).COMMANDS))
    print('Background research runs in the durable worker process and survives CLI restarts.')
    while True:
        try:
            with patch_stdout(): user_input=session.prompt('\nYou: ').strip()
            if not user_input: continue
            handled, should_exit=dispatch_command(user_input,context)
            if should_exit: break
            if handled: continue
            handle_user_turn(context.messages,user_input,context.thinking_enabled)
        except KeyboardInterrupt: print("\n[System]: Use 'exit' or 'quit' to close.")
        except EOFError: break
        except Exception as exc: print(f'\n[System] Unhandled frontend error: {exc}')
