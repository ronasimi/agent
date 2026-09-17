import os

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

def take_web_screenshot(url: str, output_filename: str = "web_screenshot.png") -> str:
    """Navigate to a URL in a headless browser, wait for it to load, and save a screenshot image."""
    if not sync_playwright:
        return "Error: playwright package is not installed."
        
    if not url.startswith("http"):
        url = "http://" + url
        
    # Unconditionally sanitize filename to prevent path traversal
    output_filename = os.path.join("/app/workspace", os.path.basename(output_filename))
        
    if not output_filename.endswith('.png'):
        output_filename += '.png'
        
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
            page = browser.new_page(viewport={"width": 1440, "height": 1080})
            page.goto(url, wait_until="networkidle", timeout=15000)
            page.screenshot(path=output_filename, full_page=False)
            browser.close()
        return f"Successfully rendered webpage and saved screenshot to {output_filename}"
    except Exception as e:
        return f"Failed to take screenshot of {url}: {str(e)}"
