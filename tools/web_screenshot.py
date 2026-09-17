import os

from .netutil import validate_public_url

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


def take_web_screenshot(url: str, output_filename: str = "web_screenshot.png") -> str:
    """Render a public webpage and save a bounded PNG screenshot in the workspace."""
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
            page.screenshot(path=output_path, full_page=False)
            browser.close()
        return f"Web screenshot saved to {output_path}"
    except Exception as exc:
        return f"Failed to take screenshot: {exc}"
