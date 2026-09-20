"""Prompt-toolkit CLI frontend for Al Agent."""
from __future__ import annotations
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
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
from tools.user_profile import get_onboarding_state, run_terminal_onboarding
from .cli_commands import CliContext, dispatch_command
from .slash_commands import SLASH_COMMANDS
from .console import get_bottom_toolbar
from .prompts import build_system_prompt
from .model_protocol import warm_model_async
from .model_residency import schedule_fast_model_prewarm
from .state import (
    FAST_MODEL, MAIN_OPTIONS, MAX_CTX, MODEL, OLLAMA, THINKING_DEFAULT,
    WARMUP_FAST_MODEL, WARMUP_PRIME_PREFIX,
)
from .turn_engine import handle_user_turn


class SlashCommandCompleter(Completer):
    """Prompt-toolkit completion sourced from the shared slash registry."""
    def get_completions(self, document, complete_event):
        before = document.text_before_cursor.lstrip()
        if not before.startswith("/") or any(ch.isspace() for ch in before):
            return
        token = before.lower()
        for spec in SLASH_COMMANDS:
            if spec.name.startswith(token):
                yield Completion(
                    spec.name,
                    start_position=-len(before),
                    display=spec.name,
                    display_meta=spec.description,
                )


def print_jobs() -> None:
    jobs=list_jobs(limit=25)
    if not jobs: print('\n[System]: No durable background jobs.'); return
    print('\n[System]: Durable jobs:')
    for job in jobs: print(f"  {job['id'][:8]}  {job['status']:<10} {job['job_type']:<10} {job['title']}")


def print_job(job_id: str) -> None:
    print('\n'+get_research_status(job_id))


def main() -> None:
    if os.isatty(0) and not get_onboarding_state().get("completed"):
        run_terminal_onboarding(reset=False)
    history_file='/app/memory/.agent_history'; os.makedirs(os.path.dirname(history_file),exist_ok=True)
    style=Style.from_dict({'prompt':'ansigreen bold','input':'ansiwhite','toolbar':'bg:#1a202c #63b3ed bold','completion-menu.completion':'bg:#000000 #ffffff','completion-menu.completion.current':'bg:#005f5f #ffffff'})
    session=PromptSession(history=FileHistory(history_file),auto_suggest=AUTO_SUGGEST,style=style,bottom_toolbar=get_bottom_toolbar,completer=SlashCommandCompleter(),complete_while_typing=True)
    messages=[{'role':'system','content':build_system_prompt()}]+_load_chat_history_from_db(limit=100)
    context=CliContext(messages=messages,thinking_enabled=THINKING_DEFAULT)
    print(f'Agent initialized with Main: {MODEL} | Fast: {FAST_MODEL} | Context: {MAX_CTX}')
    print(f'[System]: Preloading main model ({MODEL}) into VRAM in the background...')
    # Non-blocking: the prompt is usable immediately. Weight preloading is
    # reliable; optional prefix priming is disabled by default and should only
    # be enabled after cache telemetry shows that the backend reuses it.
    def _main_warm_complete() -> None:
        print('\n[System]: Main model loaded and pinned in VRAM.')
        if WARMUP_FAST_MODEL:
            schedule_fast_model_prewarm("startup")

    warm_model_async(
        OLLAMA, MODEL, options=MAIN_OPTIONS, keep_alive=-1,
        system_prompt=build_system_prompt() if WARMUP_PRIME_PREFIX else "",
        on_success=_main_warm_complete,
        on_error=lambda exc: print(f'\n[System]: Warning - failed to preload main model: {exc}'),
    )
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
