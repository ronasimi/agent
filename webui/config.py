"""Web UI filesystem, upload, and preview configuration."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace")).resolve()
UPLOAD_DIR = (WORKSPACE / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = max(1024 * 1024, int(os.environ.get("WEBUI_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024))))
ALLOWED_TEXT_EXT = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log", ".py", ".js", ".ts", ".html", ".css", ".sh", ".toml", ".ini"}
IMAGE_PREVIEW_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_PREVIEW_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
VIDEO_PREVIEW_EXT = {".mp4", ".webm", ".mov", ".m4v"}
DOCUMENT_PREVIEW_EXT = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".rtf"}
ALLOWED_MEDIA_EXT = IMAGE_PREVIEW_EXT | AUDIO_PREVIEW_EXT | VIDEO_PREVIEW_EXT | DOCUMENT_PREVIEW_EXT | {".pdf"}
XRESOURCES_PATH = Path(os.environ.get("WEBUI_XRESOURCES", "/home/agent/.Xresources"))
WORKSPACE_LIST_LIMIT = max(25, min(int(os.environ.get("WEBUI_WORKSPACE_LIST_LIMIT", "250")), 1000))
PREVIEW_TEXT_BYTES = max(4096, min(int(os.environ.get("WEBUI_PREVIEW_TEXT_BYTES", str(48 * 1024))), 256 * 1024))
ARTIFACT_SCAN_LIMIT = max(250, min(int(os.environ.get("WEBUI_ARTIFACT_SCAN_LIMIT", "10000")), 50000))
ARTIFACT_MAX_PER_TURN = max(1, min(int(os.environ.get("WEBUI_ARTIFACT_MAX_PER_TURN", "24")), 100))
MARKDOWN_EXT = {".md", ".markdown"}
TEXT_PREVIEW_EXT = ALLOWED_TEXT_EXT | {".xml", ".rst", ".cfg", ".conf"}
ARTIFACT_IGNORE_RELATIVE = {".agent_inference.lock"}
ARTIFACT_IGNORE_SUFFIXES = {".lock"}
PDF_PREVIEW_DIR = Path(os.environ.get("WEBUI_PDF_PREVIEW_DIR", "/tmp/agent_webui_pdf_previews"))
