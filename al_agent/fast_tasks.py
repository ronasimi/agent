"""Bounded fast-model classification/extraction helpers.

These helpers never authorize actions. They produce small advisory JSON that is
validated by deterministic harness code before use. Failure is always a cheap
fallback to deterministic behavior.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .model_capabilities import capability_chat_overrides


STRUCTURED_PLAN_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 32,
    "items": {
        "type": "string",
        "minLength": 1,
        "maxLength": 1200,
    },
}

_COMMAND_VERB_RE = re.compile(
    r"\b(?:check|read|scan|summari[sz]e|find|search|look\s+up|get|fetch|inspect|"
    r"list|show|compare|verify|test|run|execute|create|write|update|modify|fix|"
    r"analy[sz]e|review|download|upload|send|schedule|open|browse|probe|resolve|"
    r"measure|report|extract|convert|calculate|map|monitor)\b",
    re.I,
)
_COMMAND_BOUNDARY_RE = re.compile(
    r"(?:\n\s*(?:[-*•]|\d{1,3}[.)])\s+|\s*;\s*|\b(?:and\s+then|then|after\s+that|next)\b)",
    re.I,
)


def should_compile_structured_plan(
    user_text: str,
    *,
    min_chars: int = 900,
    min_commands: int = 3,
) -> bool:
    """Return whether a turn should be decomposed before main-model routing.

    This deliberately errs on the side of compiling long or genuinely compound
    operational prompts. The compiler has no tools, so a false positive costs a
    small fast-model classification call rather than exposing a broad schema set
    to the 4B execution model.
    """
    text = str(user_text or "").strip()
    if not text:
        return False
    if len(text) >= max(256, int(min_chars)):
        return True
    clauses = [part.strip() for part in _COMMAND_BOUNDARY_RE.split(text) if part.strip()]
    command_clauses = sum(1 for part in clauses if _COMMAND_VERB_RE.search(part))
    # Coordinated comma clauses are common in compact prompts such as
    # "check weather, read email, scan LAN, summarize the document".
    if command_clauses < min_commands and text.count(",") >= 2:
        comma_parts = [part.strip() for part in text.split(",") if part.strip()]
        command_clauses = max(command_clauses, sum(1 for part in comma_parts if _COMMAND_VERB_RE.search(part)))
    return command_clauses >= max(2, int(min_commands))


def _deterministic_plan_fallback(user_text: str, max_steps: int = 32) -> list[str]:
    """Best-effort non-model decomposition used only if structured output fails."""
    text = str(user_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    # Prefer explicit list items because they preserve user-authored boundaries.
    item_re = re.compile(r"(?ms)^\s*(?:[-*•]|\d{1,3}[.)])\s+(.+?)(?=^\s*(?:[-*•]|\d{1,3}[.)])\s+|\Z)")
    candidates = [re.sub(r"\s+", " ", match.group(1)).strip() for match in item_re.finditer(text)]
    candidates = [item for item in candidates if _COMMAND_VERB_RE.search(item)]
    if len(candidates) < 2:
        candidates = [
            re.sub(r"\s+", " ", part).strip(" ,.;:-")
            for part in _COMMAND_BOUNDARY_RE.split(text)
            if _COMMAND_VERB_RE.search(part or "")
        ]
    clean: list[str] = []
    seen: set[str] = set()
    for item in candidates[: max(2, int(max_steps))]:
        value = str(item or "").strip()
        marker = value.casefold()
        if not value or marker in seen:
            continue
        seen.add(marker)
        clean.append(value[:1200])
    return clean if len(clean) >= 2 else []


def structured_plan_compiler_prompt(user_text: str) -> list[dict[str, str]]:
    """Build the exact 2B compiler prompt used for structured plan generation."""
    system = (
        "You are a deterministic task-plan compiler. You do not execute tasks and you do not choose tools. "
        "Convert the user's actual requested work into a JSON array of short, sequential, self-contained task strings. "
        "Return ONLY the JSON array and no prose. Each array item must represent exactly one executable requirement. "
        "Preserve the user's order, targets, quantities, paths, locations, dates, and safety constraints. "
        "Do not invent work, tools, arguments, credentials, or missing facts. "
        "Do not turn examples, quoted text, explanations, source snippets, hypothetical scenarios, or lists of possible intents "
        "into tasks unless the user explicitly asks to execute them. Merge wording that belongs to the same atomic requirement. "
        "If a global constraint applies to multiple tasks, repeat only the minimal relevant constraint in those task strings."
    )
    # Keep the compiler request bounded even when the original prompt is huge;
    # the deterministic fallback remains available if the clipped view prevents
    # the model from producing a valid multi-step plan.
    source = str(user_text or "")
    if len(source) > 18000:
        source = source[:12000] + "\n...[middle omitted for compiler budget]...\n" + source[-6000:]
    user = (
        "Compile the following request. Output must validate as a JSON array of strings with 1-32 items. "
        "Use one item when the request contains only one executable requirement.\n\n"
        "USER REQUEST:\n" + source
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_structured_plan(
    client: Any,
    *,
    model: str,
    objective: str,
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
    min_chars: int = 900,
    min_commands: int = 3,
    max_steps: int = 32,
) -> list[str]:
    """Compile a complex request into validated atomic tasks using the fast role.

    The model receives no tool schemas and is constrained with Ollama structured
    output. Any malformed/failed response falls back to a deterministic splitter;
    the caller never needs to send a broad all-intents schema prompt to the main
    model merely because compilation failed.
    """
    if not should_compile_structured_plan(objective, min_chars=min_chars, min_commands=min_commands):
        return []
    fast_options = dict(options or {})
    fast_options["temperature"] = 0.0
    fast_options["num_predict"] = min(max(128, int(fast_options.get("num_predict") or 384)), 768)
    try:
        response = client.chat(
            model=model,
            messages=structured_plan_compiler_prompt(objective),
            stream=False,
            format=STRUCTURED_PLAN_SCHEMA,
            options=fast_options,
            keep_alive=keep_alive,
            **capability_chat_overrides(model, think=False, tools=[]),
        )
        parsed = _extract_json(_message_content(response))
    except Exception:
        parsed = None
    clean: list[str] = []
    seen: set[str] = set()
    if isinstance(parsed, list):
        for raw in parsed[: max(2, int(max_steps))]:
            if not isinstance(raw, str):
                continue
            task = re.sub(r"\s+", " ", raw).strip()
            marker = task.casefold()
            if not task or len(task) > 1200 or marker in seen:
                continue
            seen.add(marker)
            clean.append(task)
    if clean:
        return clean
    return _deterministic_plan_fallback(objective, max_steps=max_steps)


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        msg = response.get("message") or {}
        if isinstance(msg, dict):
            return str(msg.get("content") or "")
        return str(response.get("response") or "")
    msg = getattr(response, "message", None)
    if msg is not None:
        return str(getattr(msg, "content", "") or "")
    return str(getattr(response, "response", "") or "")


def _extract_json(text: str) -> Any:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(?:```json\s*)?(\{.*\}|\[.*\])(?:\s*```)?", value, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def infer_recipe_parameter_hints(
    client: Any,
    *,
    model: str,
    objective: str,
    trace: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
) -> list[dict[str, Any]]:
    """Ask the fast role for semantic names only; deterministic code decides use.

    The fast model is intentionally prevented from rewriting the pipeline. It may
    only point at literal values already present in the successful trace and
    suggest a semantic parameter name.
    """
    compact = []
    for entry in trace[:8]:
        if not entry.get("success") or not entry.get("readonly", True):
            continue
        compact.append({"tool": str(entry.get("tool") or ""), "args": entry.get("args") or {}})
    if len(compact) < 2:
        return []
    prompt = (
        "Classify reusable task inputs in this successful read-only workflow. "
        "Return JSON only: {\"parameters\":[{\"name\":\"...\",\"value\":...}]}. "
        "Only copy literal values that appear in the supplied tool arguments. "
        "Prefer task-defining values (hostname, location, query, path, symbol). "
        "Do not suggest timeouts, limits, booleans, port 443 unless user explicitly requested the port, "
        "credentials, tokens, secrets, timestamps, observation IDs, or tool outputs. "
        "When a hostname also appears inside URLs, suggest only the hostname; deterministic code will derive URLs.\n"
        f"Objective: {str(objective)[:1600]}\n"
        f"Workflow: {json.dumps(compact, ensure_ascii=False, default=str)[:5000]}"
    )
    fast_options = dict(options or {})
    fast_options["temperature"] = 0.0
    fast_options["num_predict"] = min(int(fast_options.get("num_predict") or 160), 192)
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options=fast_options,
            keep_alive=keep_alive,
            **capability_chat_overrides(model, think=False, tools=[]),
        )
    except Exception:
        return []
    parsed = _extract_json(_message_content(response))
    rows = parsed.get("parameters") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[:16]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or "value" not in row:
            continue
        out.append({"name": name[:48], "value": row.get("value")})
    return out
