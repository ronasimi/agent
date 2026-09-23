"""Progressive, lazy-loading textual skills.

A skill is domain guidance, not executable workflow logic. Only compact metadata
(name/description/tags) is scanned for relevance during prompt assembly; full
instructions are read only when the model explicitly calls ``load_skill``.
Recipes remain the executable deterministic workflow mechanism.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .tool_registry import agent_tool

SKILLS_DIR = Path(os.environ.get("AGENT_SKILLS_DIR", "/app/workspace/skills"))
_MAX_FILES = 128
_METADATA_READ_CHARS = 4096
_MAX_SKILL_FILE_CHARS = 100000
_STOP = {"the", "and", "for", "with", "from", "this", "that", "into", "your", "you", "use", "using", "skill", "skills", "how", "what"}
_METADATA_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9][a-z0-9_.+-]*", str(text or "").lower()) if len(tok) > 2 and tok not in _STOP}


def _safe_skill_path(name: str) -> Path | None:
    candidate = str(name or "").strip()
    if not candidate:
        return None
    try:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        base = SKILLS_DIR.resolve()
    except OSError:
        return None
    options = []
    if candidate.endswith(".md"):
        options.append(SKILLS_DIR / candidate)
    else:
        options.extend((SKILLS_DIR / f"{candidate}.md", SKILLS_DIR / candidate))
    for path in options:
        try:
            resolved = path.resolve()
            if resolved.parent == base and resolved.is_file() and resolved.suffix.lower() == ".md":
                return resolved
        except OSError:
            continue
    return None


def _parse_frontmatter(text: str, fallback_name: str) -> dict[str, Any]:
    name = fallback_name
    description = ""
    tags: list[str] = []
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            header = text[4:end]
            body = text[end + 5:]
            for raw in header.splitlines():
                key, sep, value = raw.partition(":")
                if not sep:
                    continue
                key = key.strip().lower(); value = value.strip().strip('"\'')
                if key == "name" and value:
                    name = value[:120]
                elif key == "description":
                    description = value[:500]
                elif key == "tags":
                    tags = [part.strip()[:64] for part in value.strip("[]").split(",") if part.strip()][:16]
    if not name:
        heading = re.search(r"^#\s+(.+)$", body, flags=re.M)
        name = (heading.group(1).strip() if heading else fallback_name)[:120]
    if not description:
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip() and not p.lstrip().startswith("#")]
        description = (paragraphs[0] if paragraphs else f"Instructions from {fallback_name}")[:500]
    return {"name": name, "description": description, "tags": tags}


def list_skill_metadata() -> list[dict[str, Any]]:
    try:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(SKILLS_DIR.glob("*.md"))[:_MAX_FILES]:
        try:
            stat = path.stat()
            cache_key = str(path)
            seen.add(cache_key)
            cached = _METADATA_CACHE.get(cache_key)
            if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
                meta = dict(cached[2])
            else:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read(_METADATA_READ_CHARS)
                meta = _parse_frontmatter(text, path.stem)
                meta.update({"id": path.stem, "file": path.name})
                _METADATA_CACHE[cache_key] = (int(stat.st_mtime_ns), int(stat.st_size), dict(meta))
            rows.append(meta)
        except OSError:
            continue
    for cache_key in list(_METADATA_CACHE):
        if cache_key.startswith(str(SKILLS_DIR)) and cache_key not in seen:
            _METADATA_CACHE.pop(cache_key, None)
    return rows


def relevant_skill_metadata(query: str, limit: int = 3) -> list[dict[str, Any]]:
    terms = _tokens(query)
    if not terms:
        return []
    scored = []
    for row in list_skill_metadata():
        name_tokens = _tokens(row.get("name", "")) | _tokens(row.get("id", ""))
        tag_tokens = _tokens(" ".join(row.get("tags") or []))
        desc_tokens = _tokens(row.get("description", ""))
        score = len(terms & name_tokens) * 5 + len(terms & tag_tokens) * 3 + len(terms & desc_tokens)
        if score > 0:
            scored.append((score, row.get("id", ""), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[: max(1, min(int(limit), 8))]]


def render_relevant_skill_index(query: str, limit: int = 3) -> str:
    rows = relevant_skill_metadata(query, limit=limit)
    if not rows:
        return ""
    lines = [
        "### Relevant skills (metadata only)",
        "Skills are textual guidance, not executable recipes. Call load_skill only if instructions are needed; request another offset only when has_more is true.",
    ]
    for row in rows:
        tags = ", ".join(row.get("tags") or [])
        suffix = f"; tags: {tags}" if tags else ""
        lines.append(f"- {row['id']}: {row['description']}{suffix}")
    return "\n".join(lines)


@agent_tool(readonly=True)
def search_skills(query: str = "", limit: int = 5) -> str:
    """Search installed textual skills by name, tags, and description without loading their full instructions."""
    rows = relevant_skill_metadata(query, limit=max(1, min(int(limit), 12)))
    return json.dumps(rows, ensure_ascii=False, indent=2) if rows else "No matching skills found."


@agent_tool(readonly=True)
def load_skill(name: str = "", offset: int = 0, length: int = 3500) -> str:
    """Load one bounded chunk of an installed Markdown skill by id/name."""
    path = _safe_skill_path(name)
    if path is None:
        return f"Error: skill '{str(name or '').strip()}' was not found."
    offset = max(0, int(offset))
    length = max(500, min(int(length), 5000))
    text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_SKILL_FILE_CHARS]
    chunk = text[offset:offset + length]
    return json.dumps({
        "skill": path.stem,
        "offset": offset,
        "returned_chars": len(chunk),
        "total_chars": len(text),
        "has_more": offset + len(chunk) < len(text),
        "content": chunk,
    }, ensure_ascii=False, indent=2)
