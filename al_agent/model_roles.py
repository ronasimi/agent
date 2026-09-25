"""Deterministic policy for the decision/executor/reasoning model hierarchy.

The harness intentionally resolves as much work as possible without inference.
When inference is still required, routine execution stays on the 1.5B coder executor;
4B reasoning is reserved for explicit/complex analysis and bounded recovery.
The 0.5B coder decision role is advisory only and never produces normal user prose.
"""
from __future__ import annotations

import re
from typing import Any

_COMPLEX_REASONING_RE = re.compile(
    r"\b(?:analy[sz]e|diagnos(?:e|is)|debug|root\s+cause|review|architect(?:ure)?|"
    r"design|evaluate|compare|trade[ -]?offs?|reason\s+about|prove|derive|refactor|"
    r"implement|formal(?:ly)?|correctness|security\s+review|performance\s+analysis)\b",
    re.I,
)
_CODE_OR_STRUCTURE_RE = re.compile(r"```|\b(?:stack\s*trace|source\s+code|codebase|schema|algorithm|runtime)\b", re.I)


def should_start_with_reasoning(
    request: str,
    *,
    thinking_enabled: bool = False,
    scheduler_finalizing: bool = False,
    tool_count: int = 0,
    enabled: bool = True,
    complex_direct: bool = True,
    min_complex_chars: int = 280,
) -> bool:
    """Return whether this call should bypass the 1.5B executor and use 4B.

    Tool-bearing turns default to the executor even when the user uses words
    such as "analyze": the deterministic harness and typed schemas already
    reduce that problem substantially. Direct no-tool reasoning gets escalated
    only when the request is both materially complex and sufficiently large.
    """
    if not enabled:
        return False
    if thinking_enabled or scheduler_finalizing:
        return True
    if tool_count:
        return False
    text = str(request or "").strip()
    if not complex_direct or len(text) < max(80, int(min_complex_chars)):
        return False
    return bool(_COMPLEX_REASONING_RE.search(text) and (_CODE_OR_STRUCTURE_RE.search(text) or len(text) >= max(600, min_complex_chars * 2)))


def validator_requests_reasoning(report: dict[str, Any] | None, *, low_confidence_enabled: bool = True) -> bool:
    """Interpret a constrained 0.5B validator report as an escalation request.

    High-confidence switch/retry decisions stay on the executor. Low-confidence
    reports or diagnoses that indicate the executor misunderstood the task are
    allowed to spend one bounded 4B call.
    """
    if not report:
        return False
    decision = str(report.get("decision") or "").strip()
    if decision not in {"retry", "switch_tool", "corrective_tool"}:
        return False
    confidence = str(report.get("confidence") or "medium").strip().lower()
    diagnosis = str(report.get("diagnosis") or "unknown").strip().lower()
    if low_confidence_enabled and confidence == "low":
        return True
    return diagnosis in {"wrong_tool", "bad_arguments"} and confidence != "high"
