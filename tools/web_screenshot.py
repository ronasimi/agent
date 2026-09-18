import os
import re

from .media import media_result
from .netutil import validate_public_url

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


def _clean_visible_text(text: str, max_chars: int = 7000) -> str:
    text = re.sub(r"[ \t]+", " ", str(text or ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[Visible page text truncated by screenshot tool.]"


def take_web_screenshot(url: str, output_filename: str = "web_screenshot.png") -> dict | str:
    """Render a public webpage, save a PNG, and attach it with visible page text for visual analysis."""
    if not sync_playwright:
        return "Error: playwright package is not installed."
    try:
        safe_url = validate_public_url(str(url).strip())
    except Exception as exc:
        return f"Error: {exc}"
    output_filename = os.path.basename(str(output_filename))
    if not output_filename.endswith(".png"):
        output_filename += ".png"
    output_path = os.path.join("/app/workspace", output_filename)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1080})

            def guard_request(route, request):
                request_url = request.url
                if request_url.startswith(("http://", "https://")):
                    try:
                        validate_public_url(request_url)
                    except Exception:
                        route.abort()
                        return
                route.continue_()

            page.route("**/*", guard_request)
            page.goto(safe_url, wait_until="domcontentloaded", timeout=15000)
            # Give client-rendered sites a bounded opportunity to paint without
            # waiting indefinitely for network-idle on pages with live feeds.
            page.wait_for_timeout(1500)
            final_url = page.url
            title = page.title().strip()
            try:
                visible_text = _clean_visible_text(page.locator("body").inner_text(timeout=3000))
            except Exception:
                visible_text = ""
            page.screenshot(path=output_path, full_page=False)
            browser.close()

        details = [
            f"Web screenshot saved to {output_path}",
            f"Final URL: {final_url}",
        ]
        if title:
            details.append(f"Page title: {title}")
        if visible_text:
            details.append("Visible page text:\n" + visible_text)
        else:
            details.append("Visible page text: [none extracted]")
        return media_result("\n".join(details), [output_path])
    except Exception as exc:
        return f"Error: failed to take screenshot: {exc}"
