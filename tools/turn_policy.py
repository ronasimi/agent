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
    explicit_exemptions: set[str] = field(default_factory=set)
    all_tools_blocked: bool = False

    def allowed(self, name: str, metadata: dict[str, Any]) -> bool:
        if name in self.blocked:
            return False
        threshold = self.delayed.get(name)
        if threshold is not None and self.failed_iterations < threshold:
            return False
        if self.readonly_only and not bool(metadata.get("readonly", True)):
            if name not in self.explicit_exemptions or not bool(metadata.get("safe_artifact", False)):
                return False
        return True


    def allow_explicit_requirements(self, tool_names: set[str], metadata_by_name: dict[str, dict]) -> None:
        """Permit explicitly requested safe artifacts under broad read-only wording.

        This resolves requests such as "take a screenshot" plus "do not modify any
        files" where the latter is intended to protect source/system state, not to
        forbid the requested diagnostic artifact itself.
        """
        for name in set(tool_names or set()):
            metadata = metadata_by_name.get(name, {})
            if bool(metadata.get("safe_artifact", False)):
                self.explicit_exemptions.add(name)
                self.blocked.discard(name)

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
        if self.all_tools_blocked:
            parts.append("All tools are disabled for this turn by the user's explicit constraint.")
        elif self.blocked:
            names = sorted(self.blocked)
            shown = names[:20]
            suffix = f", +{len(names)-20} more" if len(names) > 20 else ""
            parts.append("Disabled for this turn: " + ", ".join(shown) + suffix + ".")
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

    # Explicit global tool prohibition. This must be stronger than all intent
    # bundles and harness-owned pre-grounding so contradictory requests such as
    # "give me the exact current time without using tools" fail closed instead
    # of silently violating the user's constraint.
    no_tools = bool(
        re.search(r"\b(?:do not|don't|never)\s+use\s+(?:any\s+)?tools?\b", lower)
        or re.search(r"\bwithout\s+(?:using\s+)?(?:any\s+)?tools?\b", lower)
        or re.search(r"(?:^|[.;:!?]\s*)no\s+tools?\s*(?:[.;:!?]|$)", lower)
    )
    if no_tools:
        policy.blocked.update(known_names)
        policy.all_tools_blocked = True

    # A file-modification prohibition is narrower than a globally read-only
    # turn. Block direct file/package mutators, but do not erase separately
    # stated conditional access to diagnostic shell/Python tools.
    if re.search(r"\b(?:do not|don't|never) (?:create|write|modify|change|delete) (?:any )?(?:files?|source|repository)\b", lower):
        policy.blocked.update(name for name in {
            "write_file", "create_or_update_tool", "install_package"
        } if name in known_names)

    # Broad explicit read-only constraints. Do not infer them from ordinary words
    # such as "inspect" or "check".
    explicit_readonly = (
        re.search(r"(?:^|\n)\s*read[- ]only(?:\s*[:,-]|\s*$)", lower)
        or re.search(r"\bkeep (?:this|the task|this task) read[- ]only\b", lower)
    )
    if explicit_readonly:
        policy.readonly_only = True

    # Conditional restrictions. Support both "unless the first approach fails"
    # and bounded counted forms such as "unless three structured-tool attempts
    # fail". The threshold is deterministic and never inferred from tool prose.
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    conditional_patterns = (
        re.compile(
            r"(?:do not|don't|never)\s+use\s+([a-z0-9_ -]+?)\s+unless\s+(?:your\s+)?(?:first|initial)\s+(?:approach|attempt)\s+fails",
            re.I,
        ),
        re.compile(
            r"(?:do not|don't|never)\s+use\s+([a-z0-9_ -]+?)\s+unless\s+(?:at\s+least\s+)?(\d+|one|two|three|four|five)\s+(?:structured[- ]tool\s+)?(?:approaches?|attempts?)\s+fail",
            re.I,
        ),
    )
    conditional_spans = []
    for pattern in conditional_patterns:
        for match in pattern.finditer(text):
            conditional_spans.append(match.span())
            phrase = match.group(1).strip().lower()
            threshold = 1
            if match.lastindex and match.lastindex >= 2:
                raw_count = str(match.group(2) or "1").lower()
                threshold = int(raw_count) if raw_count.isdigit() else number_words.get(raw_count, 1)
            threshold = max(1, min(threshold, 10))
            names = _mentioned_tool_tokens(phrase, known_names)
            for name in names:
                existing = policy.delayed.get(name)
                policy.delayed[name] = max(existing or 0, threshold)

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
