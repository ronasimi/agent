import os

from .browser_ui import capture_web_screenshot
from .media import media_result
from .netutil import validate_public_url


def take_web_screenshot(url: str, output_filename: str = "web_screenshot.png") -> dict | str:
    """Capture the current persistent browser page as PNG plus grounded semantic state."""
    try:
        safe_url = validate_public_url(str(url).strip())
    except Exception as exc:
        return f"Error: {exc}"
    output_filename = os.path.basename(str(output_filename))
    if not output_filename.endswith(".png"):
        output_filename += ".png"
    workspace = "/app/workspace"
    os.makedirs(workspace, exist_ok=True)
    output_path = os.path.join(workspace, output_filename)
    try:
        state = capture_web_screenshot(safe_url, output_path)
        elements = state.get("elements") if isinstance(state, dict) else []
        details = [
            f"Web screenshot saved to {output_path}",
            f"Final URL: {state.get('url', safe_url)}",
            f"Browser state version: {state.get('state_version', 0)}",
        ]
        if state.get("title"):
            details.append(f"Page title: {state['title']}")
        details.append(f"Interactive semantic elements: {len(elements or [])}")
        if state.get("text"):
            details.append("Visible page text:\n" + str(state["text"])[:5000])
        return media_result("\n".join(details), [output_path])
    except Exception as exc:
        return f"Error: failed to take screenshot: {exc}"
