"""Registry-driven CLI slash commands."""
from __future__ import annotations

from dataclasses import dataclass

from .slash_commands import SLASH_COMMANDS, execute_slash_command


@dataclass
class CliContext:
    messages: list[dict]
    thinking_enabled: bool


@dataclass(frozen=True)
class CliCommand:
    name: str
    description: str = ""


# Compatibility surface used by startup output and architecture tests.  The
# authoritative slash-command metadata lives in ``al_agent.slash_commands``.
COMMANDS = (
    CliCommand("exit", "Exit the terminal frontend."),
    *(CliCommand(spec.name, spec.description) for spec in SLASH_COMMANDS),
)


def dispatch_command(text: str, context: CliContext) -> tuple[bool, bool]:
    """Return ``(handled, should_exit)`` without sending slash commands to the LLM."""
    raw = str(text or "").strip()
    if raw.lower() in {"exit", "quit"}:
        return True, True
    if not raw.startswith("/"):
        return False, False

    result = execute_slash_command(
        raw,
        conversation_id="default",
        thinking_enabled=context.thinking_enabled,
    )
    if result.action == "clear_conversation" and result.ok:
        from .prompts import build_system_prompt
        context.messages[:] = [{"role": "system", "content": build_system_prompt()}]
    elif result.action == "set_thinking" and result.ok:
        context.thinking_enabled = bool(result.data.get("thinking"))
    elif result.action == "open_profile" and result.ok:
        from tools.user_profile import run_terminal_onboarding
        run_terminal_onboarding(reset=True)
    if result.message:
        print(f"\n[System]: {result.message}")
    return True, False
