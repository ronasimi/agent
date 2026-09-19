"""Xresources-backed Web UI theme loading."""
from __future__ import annotations
import re
from pathlib import Path
from .config import XRESOURCES_PATH

DEFAULT_THEME = {
    "foreground": "#c3c3c3", "background": "#272727", "cursorColor": "#c3c3c3",
    "color0": "#5d5d5d", "color1": "#ac4142", "color2": "#90a959", "color3": "#f4bf75",
    "color4": "#8ab4f8", "color5": "#aa759f", "color6": "#75b5aa", "color7": "#d0d0d0",
    "color8": "#818181", "color9": "#bc6667", "color10": "#a6ba7a", "color11": "#f6cb90",
    "color12": "#a1c3f9", "color13": "#bb90b2", "color14": "#90c3bb", "color15": "#e0e0e0",
}

def read_xresources_theme(path: Path | None = None) -> dict[str, str]:
    theme = dict(DEFAULT_THEME); source = path or XRESOURCES_PATH
    try: text = source.read_text(encoding="utf-8", errors="replace")
    except OSError: return theme
    allowed = set(theme)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!") or ":" not in line: continue
        key, value = (part.strip() for part in line.split(":", 1)); key = key.removeprefix("*.")
        if key in allowed and re.fullmatch(r"#[0-9A-Fa-f]{6}", value): theme[key] = value.lower()
    return theme
