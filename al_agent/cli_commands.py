"""Registry-driven CLI slash commands."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from tools import clear_chat_history, get_tools_prompt_summary, load_tools
from tools.job_tools import cancel_background_job, enqueue_research
from tools.reminders import list_reminders
from tools.self_optimization import approve_self_optimization, enqueue_self_optimization, list_self_optimization_candidates
from tools.user_profile import run_terminal_onboarding
from .prompts import build_system_prompt


@dataclass
class CliContext:
    messages: list[dict]
    thinking_enabled: bool


@dataclass(frozen=True)
class CliCommand:
    name: str
    matches: Callable[[str], bool]
    run: Callable[[str, CliContext], bool]


def _research(text: str, ctx: CliContext) -> bool:
    topic=text[len('/research'):].strip()
    if not topic: print('[System]: Usage: /research <topic>'); return False
    result=json.loads(enqueue_research(topic)); print(f"[System]: queued research job {result['job_id'][:8]} for '{topic}'"); return False

def _optimize(text: str, ctx: CliContext) -> bool:
    objective=text[len('/optimize'):].strip()
    if not objective: print('[System]: Usage: /optimize <bounded objective>'); return False
    result=json.loads(enqueue_self_optimization(objective))
    if result.get('job_id'): print(f"[System]: queued isolated candidate {result['candidate_id'][:8]} as job {result['job_id'][:8]}")
    else: print(f"[System]: {result.get('reason','Optimization was not queued.')}")
    return False

def _optimizations(text: str, ctx: CliContext) -> bool:
    print('\n'+list_self_optimization_candidates()); return False

def _approve(text: str, ctx: CliContext) -> bool:
    parts=text.split()
    if len(parts)!=3: print('[System]: Usage: /approve-optimization <candidate-id> <full-sha256>')
    else: print('\n'+approve_self_optimization(parts[1],parts[2]))
    return False

def _jobs(text: str, ctx: CliContext) -> bool:
    from .cli import print_jobs
    print_jobs(); return False

def _job(text: str, ctx: CliContext) -> bool:
    from .cli import print_job
    print_job(text.split(None,1)[1].strip()); return False

def _cancel_job(text: str, ctx: CliContext) -> bool:
    print(cancel_background_job(text.split(None,1)[1].strip())); return False

def _reminders(text: str, ctx: CliContext) -> bool:
    print('\n'+list_reminders()); return False

def _forget(text: str, ctx: CliContext) -> bool:
    clear_chat_history(); ctx.messages[:] = [{"role":"system","content":build_system_prompt()}]
    print('[System]: Conversation history and rolling summary cleared.'); return False

def _think(text: str, ctx: CliContext) -> bool:
    parts=text.lower().split(); ctx.thinking_enabled=(parts[1] in {'on','true','1'}) if len(parts)>1 else not ctx.thinking_enabled
    print(f"[System]: Thinking is {'ON' if ctx.thinking_enabled else 'OFF'}."); return False

def _tools(text: str, ctx: CliContext) -> bool:
    print(get_tools_prompt_summary()); return False

def _profile(text: str, ctx: CliContext) -> bool:
    run_terminal_onboarding(reset=True); return False

def _reload(text: str, ctx: CliContext) -> bool:
    count,errors=load_tools(); print(f'[System]: Reloaded {count} tools.')
    for key,value in errors.items(): print(f'  - {key}: {value}')
    return False

def _exit(text: str, ctx: CliContext) -> bool:
    return True

COMMANDS = (
    CliCommand('exit', lambda s:s.lower() in {'exit','quit'}, _exit),
    CliCommand('/research', lambda s:s.lower().startswith('/research'), _research),
    CliCommand('/approve-optimization', lambda s:s.lower().startswith('/approve-optimization '), _approve),
    CliCommand('/optimizations', lambda s:s.lower()=='/optimizations', _optimizations),
    CliCommand('/optimize', lambda s:s.lower().startswith('/optimize'), _optimize),
    CliCommand('/cancel-job', lambda s:s.lower().startswith('/cancel-job '), _cancel_job),
    CliCommand('/jobs', lambda s:s.lower()=='/jobs', _jobs),
    CliCommand('/job', lambda s:s.lower().startswith('/job '), _job),
    CliCommand('/reminders', lambda s:s.lower()=='/reminders', _reminders),
    CliCommand('/forget', lambda s:s.lower()=='/forget', _forget),
    CliCommand('/think', lambda s:s.lower().startswith('/think'), _think),
    CliCommand('/tools', lambda s:s.lower()=='/tools', _tools),
    CliCommand('/profile', lambda s:s.lower()=='/profile', _profile),
    CliCommand('/reload', lambda s:s.lower()=='/reload', _reload),
)


def dispatch_command(text: str, context: CliContext) -> tuple[bool, bool]:
    """Return ``(handled, should_exit)``."""
    for command in COMMANDS:
        if command.matches(text):
            return True, bool(command.run(text, context))
    return False, False
