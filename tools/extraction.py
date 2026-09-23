"""Deterministic targeted extraction for large retrieved text."""
from __future__ import annotations

import re

_STOP = {
    "the","a","an","and","or","of","to","for","in","on","at","by","with","from",
    "is","are","was","were","be","been","being","this","that","these","those","find",
    "show","give","get","extract","exact","current","latest","please",
}


def _tokens(value: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9][a-z0-9._%$+-]*", str(value or "").lower()) if len(x) > 1 and x not in _STOP}


def targeted_extract(text: str, instruction: str, max_chars: int = 8000) -> str:
    """Return high-overlap source passages in original order, without synthesis."""
    source = str(text or "")
    query = " ".join(str(instruction or "").split())
    max_chars = max(500, min(int(max_chars), 20000))
    if not query or len(source) <= max_chars:
        return source[:max_chars]
    qtokens = _tokens(query)
    if not qtokens:
        return source[:max_chars]

    # Paragraph boundaries preserve useful table/list adjacency better than a
    # sentence-only extractor. Very long paragraphs are split into sentences.
    chunks: list[tuple[int, str]] = []
    cursor = 0
    for raw in re.split(r"\n{2,}|(?<=\.)\s+(?=[A-Z0-9])", source):
        piece = " ".join(raw.split())
        if not piece:
            cursor += len(raw) + 1
            continue
        if len(piece) > 2200:
            for sub in re.split(r"(?<=[.!?])\s+", piece):
                sub = sub.strip()
                if sub:
                    chunks.append((cursor, sub[:2200]))
                cursor += len(sub) + 1
        else:
            chunks.append((cursor, piece))
            cursor += len(raw) + 1

    ranked: list[tuple[float, int, str]] = []
    qlower = query.lower()
    for pos, chunk in chunks:
        lower = chunk.lower()
        ctokens = _tokens(lower)
        overlap = qtokens & ctokens
        if not overlap:
            continue
        score = float(len(overlap) * 4)
        score += sum(1.0 for token in overlap if lower.count(token) > 1)
        if qlower in lower:
            score += 10.0
        # Numbers/currency often matter in fact extraction.
        if re.search(r"[$€£¥]|\b\d+(?:\.\d+)?%?\b", chunk):
            score += 0.75
        ranked.append((score, pos, chunk))
    if not ranked:
        return source[:max_chars]

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, str]] = []
    used = 0
    for _score, pos, chunk in ranked:
        cost = len(chunk) + 2
        if selected and used + cost > max_chars:
            continue
        selected.append((pos, chunk))
        used += cost
        if used >= max_chars:
            break
        if len(selected) >= 18:
            break
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(chunk for _pos, chunk in selected)[:max_chars]
