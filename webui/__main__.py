"""Executable entry point for the Al Agent Web UI."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("WEBUI_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("WEBUI_PORT", "8080"))
    except ValueError as exc:
        raise SystemExit("WEBUI_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("WEBUI_PORT must be between 1 and 65535")
    uvicorn.run("webui.server:app", host=host, port=port, proxy_headers=True)


if __name__ == "__main__":
    main()
