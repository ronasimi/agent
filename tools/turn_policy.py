"""Deterministic per-turn tool restrictions derived only from explicit user constraints."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_ALIAS_TOOLS = {
    "shell": {"execute_shell"},
    "command": {"execute_shell"},
    "command execution": {"execute_shell", "execute_python"},
    "command-execution": {"execute_shell", "execute_python"},
    "python": {"execute_python"},
    "python execution": {"execute_python"},
}


@dataclass
class TurnToolPolicy:
    blocked: set[str] = field(default_factory=set)
    delayed: dict[str, int] = field(default_factory=dict)
    failed_iterations: int = 0
    readonly_only: bool = False

    def allowed(self, name: str, metadata: dict[str, Any]) -> bool:
        if name in self.blocked:
            return False
        threshold = self.delayed.get(name)
        if threshold is not None and self.failed_iterations < threshold:
            return False
        if self.readonly_only and not bool(metadata.get("readonly", True)):
            return False
        return True

    def record_iteration(self, made_progress: bool) -> bool:
        if made_progress:
            return False
        before = {name for name, threshold in self.delayed.items() if self.failed_iterations >= threshold}
        self.failed_iterations += 1
        after = {name for name, threshold in self.delayed.items() if self.failed_iterations >= threshold}
        return before != after

    def filter_schemas(self, schemas: list[dict], metadata_by_name: dict[str, dict]) -> list[dict]:
        return [
            schema for schema in schemas
            if self.allowed(str(schema.get("function", {}).get("name") or ""), metadata_by_name.get(str(schema.get("function", {}).get("name") or ""), {}))
        ]

    def note(self) -> str:
        parts = []
        if self.readonly_only:
            parts.append("This turn is read-only: mutating tools are disabled by the user's explicit constraint.")
        if self.blocked:
            parts.append("Disabled for this turn: " + ", ".join(sorted(self.blocked)) + ".")
        waiting = sorted(name for name, threshold in self.delayed.items() if self.failed_iterations < threshold)
        if waiting:
            parts.append("Conditionally disabled until an allowed approach fails: " + ", ".join(waiting) + ".")
        return " ".join(parts)


def _mentioned_tool_tokens(text: str, known_names: set[str]) -> set[str]:
    lower = text.lower()
    found = {name for name in known_names if re.search(rf"\b{re.escape(name.lower())}\b", lower)}
    for alias, names in _ALIAS_TOOLS.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            found.update(name for name in names if name in known_names)
    return found


def derive_turn_tool_policy(user_text: str, known_names: set[str], metadata_by_name: dict[str, dict]) -> TurnToolPolicy:
    """Parse narrow explicit restrictions such as 'do not use execute_shell unless the first approach fails'."""
    text = str(user_text or "")
    lower = text.lower()
    policy = TurnToolPolicy()

    # Broad explicit read-only constraints. Do not infer them from ordinary words
    # such as "inspect" or "check".
    explicit_readonly = (
        re.search(r"(?:^|\n)\s*read[- ]only(?:\s*[:,-]|\s*$)", lower)
        or re.search(r"\bkeep (?:this|the task|this task) read[- ]only\b", lower)
        or re.search(r"\b(?:do not|don't|never) (?:create|write|modify|change|delete) (?:any )?(?:files?|state)\b", lower)
    )
    if explicit_readonly:
        policy.readonly_only = True

    # Conditional restriction: the tool becomes available only after one failed
    # allowed tool iteration. This covers the common "unless first approach fails"
    # form without trying to interpret arbitrary natural-language conditions.
    conditional_pattern = re.compile(
        r"(?:do not|don't|never)\s+use\s+([a-z0-9_ -]+?)\s+unless\s+(?:your\s+)?(?:first|initial)\s+(?:approach|attempt)\s+fails",
        re.I,
    )
    conditional_spans = []
    for match in conditional_pattern.finditer(text):
        conditional_spans.append(match.span())
        phrase = match.group(1).strip().lower()
        names = _mentioned_tool_tokens(phrase, known_names)
        for name in names:
            policy.delayed[name] = 1

    # Permanent explicit tool bans. Remove conditional clauses before matching so
    # "do not use X unless..." is not accidentally converted into a permanent ban.
    scrubbed = text
    for start, end in reversed(conditional_spans):
        scrubbed = scrubbed[:start] + " " * (end - start) + scrubbed[end:]
    ban_pattern = re.compile(r"(?:do not|don't|never)\s+use\s+([a-z0-9_ -]+?)(?:[.,;\n]|$)", re.I)
    for match in ban_pattern.finditer(scrubbed):
        phrase = match.group(1).strip().lower()
        policy.blocked.update(_mentioned_tool_tokens(phrase, known_names))

    # Read-only always wins over conditional mutating access.
    if policy.readonly_only:
        for name, metadata in metadata_by_name.items():
            if not bool(metadata.get("readonly", True)):
                policy.blocked.add(name)
                policy.delayed.pop(name, None)
    return policy
