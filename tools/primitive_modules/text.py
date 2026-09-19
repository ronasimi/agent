from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def text_search(pattern: str, text: str = "", path: str = "", regex: bool = False, ignore_case: bool = True, limit: int = 100) -> str:
    """Search text or a workspace text file and return matching lines."""
    try:
        data=_source_text(text,path); limit=_bounded_int(limit,1,500); out=[]
        flags=re.I if ignore_case else 0
        rx=re.compile(pattern if regex else re.escape(pattern),flags)
        for no,line in enumerate(data.splitlines(),1):
            if rx.search(line): out.append({"line":no,"text":line[:2000]})
            if len(out)>=limit: break
        return _json({"matches":out,"truncated":len(out)>=limit})
    except Exception as exc: return f"Error: text_search failed: {exc}"

def regex_extract(pattern: str, text: str = "", path: str = "", limit: int = 100) -> str:
    """Extract bounded regular-expression matches from text or a workspace file."""
    try:
        data=_source_text(text,path); rx=re.compile(pattern); matches=[]
        for m in rx.finditer(data):
            matches.append({"match":m.group(0),"groups":list(m.groups())})
            if len(matches)>=_bounded_int(limit,1,500): break
        return _json({"matches":matches})
    except Exception as exc: return f"Error: regex_extract failed: {exc}"

def text_diff(a: str = "", b: str = "", path_a: str = "", path_b: str = "", max_chars: int = 20000) -> str:
    """Return a bounded unified diff between two strings or workspace files."""
    import difflib
    try:
        left=_source_text(a,path_a); right=_source_text(b,path_b); max_chars=_bounded_int(max_chars,100,50000)
        diff="".join(difflib.unified_diff(left.splitlines(True),right.splitlines(True),fromfile=path_a or "a",tofile=path_b or "b"))
        return diff[:max_chars] + ("\n[truncated]" if len(diff)>max_chars else "")
    except Exception as exc: return f"Error: text_diff failed: {exc}"

def text_head(text: str = "", path: str = "", lines: int = 20) -> str:
    """Return the first N lines from text or a workspace file."""
    try:return "\n".join(_source_text(text,path).splitlines()[:_bounded_int(lines,1,2000)])
    except Exception as exc:return f"Error: text_head failed: {exc}"

def text_tail(text: str = "", path: str = "", lines: int = 20) -> str:
    """Return the last N lines from text or a workspace file."""
    try:return "\n".join(_source_text(text,path).splitlines()[-_bounded_int(lines,1,2000):])
    except Exception as exc:return f"Error: text_tail failed: {exc}"

def text_count(text: str = "", path: str = "") -> str:
    """Count characters, lines, and whitespace-delimited words."""
    try:
        data=_source_text(text,path); return _json({"characters":len(data),"lines":len(data.splitlines()),"words":len(data.split())})
    except Exception as exc:return f"Error: text_count failed: {exc}"

def text_sort(text: str = "", path: str = "", descending: bool = False, numeric: bool = False, limit: int = 5000) -> str:
    """Sort bounded text lines lexically or numerically."""
    try:
        rows=_source_text(text,path).splitlines()[:_bounded_int(limit,1,10000)]
        key=(lambda x:float(x.strip())) if numeric else (lambda x:x)
        rows=sorted(rows,key=key,reverse=bool(descending)); return "\n".join(rows)
    except Exception as exc:return f"Error: text_sort failed: {exc}"

def text_unique(text: str = "", path: str = "", limit: int = 5000) -> str:
    """Deduplicate bounded text lines while preserving first occurrence order."""
    try:
        seen=set(); out=[]
        for line in _source_text(text,path).splitlines():
            if line not in seen:seen.add(line);out.append(line)
            if len(out)>=_bounded_int(limit,1,10000):break
        return "\n".join(out)
    except Exception as exc:return f"Error: text_unique failed: {exc}"


def regex_replace(pattern: str, replacement: str, text: str = "", path: str = "", count: int = 0, ignore_case: bool = False) -> str:
    """Apply a bounded regular-expression replacement to text or a workspace file in memory."""
    try:
        source = _source_text(text, path); count = _bounded_int(count, 0, 10000)
        compiled = re.compile(pattern, re.I if ignore_case else 0)
        output, replacements = compiled.subn(replacement, source, count=count)
        return _json({"text": output[:MAX_TEXT], "replacements": replacements, "truncated": len(output) > MAX_TEXT})
    except Exception as exc:
        return f"Error: regex_replace failed: {exc}"


def text_split(text: str = "", path: str = "", delimiter: str = "", limit: int = 200) -> str:
    """Split text into a bounded JSON list using a delimiter, or lines when it is empty."""
    try:
        source = _source_text(text, path); limit = _bounded_int(limit, 1, 2000)
        parts = source.split(delimiter) if delimiter else source.splitlines()
        return _json({"parts": parts[:limit], "count": len(parts), "truncated": len(parts) > limit})
    except Exception as exc:
        return f"Error: text_split failed: {exc}"
