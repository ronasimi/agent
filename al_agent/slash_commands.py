"""Shared slash-command registry and deterministic executor.

Slash commands are browser control operations, not model prompts. The Web UI
uses this registry for both autocomplete metadata and deterministic execution.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    description: str
    usage: str
    category: str = "General"
    argument_hint: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def accepts_arguments(self) -> bool:
        return bool(self.argument_hint)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["accepts_arguments"] = self.accepts_arguments
        return value


@dataclass
class SlashCommandResult:
    recognized: bool
    ok: bool
    command: str = ""
    message: str = ""
    action: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def event(self) -> dict[str, Any]:
        return {
            "type": "command_result",
            "recognized": self.recognized,
            "ok": self.ok,
            "command": self.command,
            "message": self.message,
            "action": self.action,
            "data": self.data,
        }


SLASH_COMMANDS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec("/help", "Show all available slash commands.", "/help", "General"),
    SlashCommandSpec("/forget", "Clear this conversation's history, summary, evidence, and working state.", "/forget", "Conversation"),
    SlashCommandSpec("/think", "Toggle extended thinking, or explicitly turn it on/off.", "/think [on|off]", "Conversation", "on|off"),
    SlashCommandSpec("/profile", "Open the user profile/onboarding editor and overwrite saved profile data when submitted.", "/profile", "Profile"),
    SlashCommandSpec("/tools", "Show the currently registered native tool inventory.", "/tools", "Agent"),
    SlashCommandSpec("/reload", "Reload dynamically discovered tools from disk.", "/reload", "Agent"),
    SlashCommandSpec("/research", "Queue a durable background research job.", "/research <topic>", "Research", "topic"),
    SlashCommandSpec("/jobs", "List recent durable background jobs.", "/jobs", "Jobs"),
    SlashCommandSpec("/job", "Show detailed status for one research/background job.", "/job <job-id>", "Jobs", "job-id"),
    SlashCommandSpec("/cancel-job", "Cancel a pending or running background job.", "/cancel-job <job-id>", "Jobs", "job-id"),
    SlashCommandSpec("/reminders", "List active/error reminders.", "/reminders", "Automation"),
    SlashCommandSpec("/optimize", "Queue a sandboxed self-optimization candidate for a bounded objective.", "/optimize <bounded objective>", "Optimization", "bounded objective"),
    SlashCommandSpec("/optimizations", "List persisted self-optimization candidates.", "/optimizations", "Optimization"),
    SlashCommandSpec("/approve-optimization", "Approve/export a validated optimization patch by candidate ID and full SHA-256.", "/approve-optimization <candidate-id> <full-sha256>", "Optimization", "candidate-id full-sha256"),
)

_COMMAND_BY_NAME: dict[str, SlashCommandSpec] = {}
for _spec in SLASH_COMMANDS:
    _COMMAND_BY_NAME[_spec.name.lower()] = _spec
    for _alias in _spec.aliases:
        _COMMAND_BY_NAME[_alias.lower()] = _spec


def list_slash_commands() -> list[dict[str, Any]]:
    """Return the public command catalog used by browser autocomplete."""
    return [spec.public_dict() for spec in SLASH_COMMANDS]


def parse_slash_command(text: str) -> tuple[SlashCommandSpec | None, str]:
    """Parse one slash command using exact command-token matching.

    Unknown slash-prefixed input intentionally does *not* fall through to the
    model; the Web UI can report it as an unknown command instead.
    """
    raw = str(text or "").strip()
    if not raw.startswith("/"):
        return None, ""
    parts = raw.split(None, 1)
    token = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""
    spec = _COMMAND_BY_NAME.get(token.lower())
    return spec, rest


def _usage(spec: SlashCommandSpec, detail: str = "") -> SlashCommandResult:
    suffix = f" {detail}" if detail else ""
    return SlashCommandResult(True, False, spec.name, f"Usage: `{spec.usage}`.{suffix}")


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _render_help() -> str:
    lines = ["### Available slash commands", ""]
    for spec in SLASH_COMMANDS:
        lines.append(f"- **`{spec.usage}`** — {spec.description}")
    return "\n".join(lines)


def _render_jobs(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No durable background jobs."
    lines = ["### Durable jobs", "", "| ID | Status | Type | Title |", "|---|---|---|---|"]
    for job in rows[:25]:
        lines.append(
            f"| `{str(job.get('id') or '')[:8]}` | {job.get('status') or '—'} | "
            f"{job.get('job_type') or '—'} | {str(job.get('title') or '').replace('|', '\\|')} |"
        )
    return "\n".join(lines)


def _execute_slash_command_impl(
    text: str,
    *,
    conversation_id: str = "default",
    thinking_enabled: bool = False,
) -> SlashCommandResult:
    """Execute a deterministic slash command without invoking the LLM."""
    raw = str(text or "").strip()
    spec, args = parse_slash_command(raw)
    if not raw.startswith("/"):
        return SlashCommandResult(False, False)
    if spec is None:
        token = raw.split(None, 1)[0]
        matches = [item.name for item in SLASH_COMMANDS if item.name.startswith(token.lower())][:5]
        hint = f" Did you mean: {', '.join(f'`{name}`' for name in matches)}?" if matches else " Type `/` to see available commands."
        return SlashCommandResult(False, False, token, f"Unknown slash command `{token}`.{hint}")

    name = spec.name
    if name == "/help":
        if args:
            return _usage(spec)
        return SlashCommandResult(True, True, name, _render_help())

    if name == "/forget":
        if args:
            return _usage(spec)
        from tools import clear_chat_history
        message = clear_chat_history(conversation_id)
        return SlashCommandResult(True, True, name, message, "clear_conversation")

    if name == "/think":
        value = args.lower()
        if value and value not in {"on", "off", "true", "false", "1", "0"}:
            return _usage(spec, "Accepted values are `on` or `off`.")
        enabled = (value in {"on", "true", "1"}) if value else (not bool(thinking_enabled))
        return SlashCommandResult(
            True, True, name,
            f"Thinking is **{'ON' if enabled else 'OFF'}**.",
            "set_thinking", {"thinking": enabled},
        )

    if name == "/profile":
        if args:
            return _usage(spec)
        return SlashCommandResult(True, True, name, "Opening profile editor.", "open_profile")

    if name == "/tools":
        if args:
            return _usage(spec)
        from tools import get_tools_prompt_summary
        return SlashCommandResult(True, True, name, get_tools_prompt_summary())

    if name == "/reload":
        if args:
            return _usage(spec)
        from tools import load_tools
        count, errors = load_tools()
        message = f"Reloaded **{count} tools**."
        if errors:
            message += "\n\n" + "\n".join(f"- `{key}`: {value}" for key, value in errors.items())
        return SlashCommandResult(True, not bool(errors), name, message, "tools_reloaded", {"count": count, "errors": errors})

    if name == "/research":
        if not args:
            return _usage(spec)
        from tools.job_tools import enqueue_research
        result = _json_object(enqueue_research(args))
        if not result.get("job_id"):
            return SlashCommandResult(True, False, name, result.get("reason") or "Research job was not queued.")
        return SlashCommandResult(
            True, True, name,
            f"Queued research job `{str(result['job_id'])[:8]}` for **{result.get('topic') or args}**.",
            "jobs_changed", result,
        )

    if name == "/jobs":
        if args:
            return _usage(spec)
        from tools.runtime import list_jobs
        rows = list_jobs(limit=25)
        return SlashCommandResult(True, True, name, _render_jobs(rows), "show_jobs", {"count": len(rows)})

    if name == "/job":
        if not args or " " in args:
            return _usage(spec)
        from tools.job_tools import get_research_status
        message = get_research_status(args)
        return SlashCommandResult(True, not message.startswith("Error:"), name, f"```json\n{message}\n```" if message.lstrip().startswith("{") else message)

    if name == "/cancel-job":
        if not args or " " in args:
            return _usage(spec)
        from tools.job_tools import cancel_background_job
        message = cancel_background_job(args)
        return SlashCommandResult(True, not message.startswith("Error:"), name, message, "jobs_changed")

    if name == "/reminders":
        if args:
            return _usage(spec)
        from tools.reminders import list_reminders
        message = list_reminders()
        if message.lstrip().startswith("["):
            message = f"```json\n{message}\n```"
        return SlashCommandResult(True, True, name, message, "show_reminders")

    if name == "/optimize":
        if not args:
            return _usage(spec)
        from tools.self_optimization import enqueue_self_optimization
        result = _json_object(enqueue_self_optimization(args))
        ok = bool(result.get("job_id"))
        if ok:
            message = f"Queued optimization candidate `{str(result.get('candidate_id') or '')[:8]}` as job `{str(result.get('job_id') or '')[:8]}`."
        else:
            message = str(result.get("reason") or "Optimization was not queued.")
        return SlashCommandResult(True, ok, name, message, "jobs_changed", result)

    if name == "/optimizations":
        if args:
            return _usage(spec)
        from tools.self_optimization import list_self_optimization_candidates
        message = list_self_optimization_candidates()
        return SlashCommandResult(True, not message.startswith("Error:"), name, f"```json\n{message}\n```" if message.lstrip().startswith("[") else message)

    if name == "/approve-optimization":
        parts = args.split()
        if len(parts) != 2:
            return _usage(spec)
        from tools.self_optimization import approve_self_optimization
        message = approve_self_optimization(parts[0], parts[1])
        ok = message.lstrip().startswith("{")
        return SlashCommandResult(True, ok, name, f"```json\n{message}\n```" if ok else message, "optimizations_changed")

    # Registry and executor are deliberately exhaustive; reaching this branch is
    # a developer error rather than something the model should attempt to fix.
    return SlashCommandResult(True, False, name, f"Command `{name}` has no registered executor.")


def execute_slash_command(
    text: str,
    *,
    conversation_id: str = "default",
    thinking_enabled: bool = False,
) -> SlashCommandResult:
    """Fail-closed public command executor. Slash input never reaches the LLM."""
    try:
        return _execute_slash_command_impl(
            text, conversation_id=conversation_id, thinking_enabled=thinking_enabled
        )
    except Exception as exc:
        raw = str(text or "").strip()
        spec, _ = parse_slash_command(raw)
        command = spec.name if spec else (raw.split(None, 1)[0] if raw.startswith("/") else "")
        return SlashCommandResult(
            recognized=bool(spec),
            ok=False,
            command=command,
            message=f"Command `{command or '/?'}` failed: {exc}",
        )
