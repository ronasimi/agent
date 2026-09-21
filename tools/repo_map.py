"""Bounded, read-only repository mapping and symbol retrieval."""
from __future__ import annotations

import ast
import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .config import load_config

CONFIG = load_config()
OPT_CFG = CONFIG.get("self_optimization", {})
DEFAULT_EXCLUDES = (
    ".git/*", "workspace/*", "memory/*", "__pycache__/*", "*.pyc",
    ".pytest_cache/*", ".ruff_cache/*", ".mypy_cache/*", "*.db", "*.sqlite*",
)
DEFAULT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".sh"}


def source_root() -> Path:
    """Return the configured immutable source tree, with a source-tree fallback for tests."""
    configured = Path(str(OPT_CFG.get("source_root", "/app/source"))).resolve()
    if configured.is_dir():
        return configured
    return Path(__file__).resolve().parents[1]


def _allowed_prefixes() -> tuple[str, ...]:
    configured = OPT_CFG.get("allowed_paths") or [
        "al_agent", "webui", "worker.py", "tools", "tests", "config", "README.md",
        "Dockerfile", "docker-compose.yml", "requirements.txt", "ollama.env.example",
    ]
    return tuple(str(value).strip("/") for value in configured if str(value).strip("/"))


def is_allowed_relative(path: str | Path) -> bool:
    value = Path(path).as_posix().lstrip("./")
    if not value or value.startswith("../") or "/../" in f"/{value}/":
        return False
    if any(fnmatch.fnmatch(value, pattern) for pattern in DEFAULT_EXCLUDES):
        return False
    return any(value == prefix or value.startswith(prefix + "/") for prefix in _allowed_prefixes())


def resolve_source_path(relative_path: str) -> Path:
    root = source_root()
    relative = Path(str(relative_path).strip())
    if relative.is_absolute() or not is_allowed_relative(relative):
        raise ValueError("Path is outside the self-optimization source allowlist.")
    resolved = (root / relative).resolve()
    if os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise ValueError("Path escapes the source root.")
    if not resolved.is_file():
        raise FileNotFoundError(relative_path)
    return resolved


def iter_source_files(root: Path | None = None) -> Iterable[tuple[str, Path]]:
    root = (root or source_root()).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if not is_allowed_relative(relative):
            continue
        if path.suffix.lower() not in DEFAULT_SUFFIXES and path.name not in {"Dockerfile"}:
            continue
        yield relative, path


def _python_symbols(text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "line": int(node.lineno),
                "end_line": int(getattr(node, "end_lineno", node.lineno)),
            })
    return sorted(symbols, key=lambda item: (item["line"], item["name"]))


def build_repo_map(root: Path | None = None, max_files: int = 200) -> dict[str, Any]:
    """Build a deterministic, compact source inventory without loading whole files into a prompt."""
    root = (root or source_root()).resolve()
    files = []
    total_bytes = 0
    total_lines = 0
    for relative, path in iter_source_files(root):
        if len(files) >= max(1, min(int(max_files), 1000)):
            break
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        lines = text.count("\n") + bool(text)
        item: dict[str, Any] = {"path": relative, "bytes": len(raw), "lines": lines}
        if path.suffix == ".py":
            item["symbols"] = _python_symbols(text)
        files.append(item)
        total_bytes += len(raw)
        total_lines += int(lines)
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "lines": total_lines,
        "estimated_tokens": (total_bytes + 3) // 4,
        "files": files,
    }


def get_repo_map(max_files: int = 200) -> str:
    """Return a bounded repository map with file sizes, line counts, and Python symbols."""
    return json.dumps(build_repo_map(max_files=max_files), ensure_ascii=False, indent=2)


def search_repo_symbols(query: str, limit: int = 20) -> str:
    """Search Python symbol names and paths in the immutable source snapshot."""
    needle = str(query).strip().lower()
    if not needle:
        return "Error: query is required."
    results = []
    for item in build_repo_map().get("files", []):
        path = str(item["path"])
        if needle in path.lower():
            results.append({"path": path, "kind": "file"})
        for symbol in item.get("symbols", []):
            if needle in str(symbol["name"]).lower():
                results.append({"path": path, **symbol})
        if len(results) >= max(1, min(int(limit), 100)):
            break
    return json.dumps(results[:max(1, min(int(limit), 100))], ensure_ascii=False, indent=2)


def read_repo_symbol(path: str, symbol: str = "", start_line: int = 0, max_lines: int = 160) -> str:
    """Read one bounded symbol or line slice from the immutable source snapshot."""
    source = resolve_source_path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, int(start_line or 1))
    end = min(len(lines), start + max(1, min(int(max_lines), 400)) - 1)
    if symbol and source.suffix == ".py":
        match = next((entry for entry in _python_symbols(text) if entry["name"] == symbol), None)
        if not match:
            return f"Error: symbol '{symbol}' was not found in {path}."
        start = int(match["line"])
        end = min(int(match["end_line"]), start + max(1, min(int(max_lines), 400)) - 1)
    numbered = [f"{index}: {lines[index - 1]}" for index in range(start, end + 1)]
    return f"# {source.relative_to(source_root()).as_posix()} lines {start}-{end}\n" + "\n".join(numbered)
