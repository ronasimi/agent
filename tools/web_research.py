"""Structured web/document research helpers optimized for small-model context budgets."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .netutil import fetch_bytes, fetch_text, validate_public_url
from .extraction import targeted_extract
from .runtime import get_monitor_state, record_monitor_state, utc_now

WORKSPACE = Path("/app/workspace").resolve()


def _page(url: str, max_bytes: int = 2 * 1024 * 1024) -> tuple[str, str, BeautifulSoup, str]:
    final_url, content_type, body = fetch_text(str(url), max_bytes=max_bytes)
    soup = BeautifulSoup(body, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    readable = "\n".join(soup.stripped_strings)
    return final_url, content_type, soup, readable


def page_metadata(url: str) -> str:
    """Extract title, canonical URL, authorship/date hints, OpenGraph, Twitter, and JSON-LD metadata from a public page."""
    try:
        final_url, content_type, soup, _ = _page(url)
    except Exception as exc:
        return f"Error: page metadata fetch failed: {exc}"
    metadata: dict = {"url": final_url, "content_type": content_type, "title": soup.title.get_text(" ", strip=True)[:500] if soup.title else ""}
    canonical = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if canonical and canonical.get("href"):
        metadata["canonical"] = urljoin(final_url, canonical.get("href"))
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name") or tag.get("itemprop")
        content = tag.get("content")
        if key and content:
            key = str(key).strip().lower()
            if key in {
                "description", "author", "date", "article:published_time", "article:modified_time",
                "og:title", "og:description", "og:image", "og:type", "og:site_name",
                "twitter:title", "twitter:description", "twitter:image", "twitter:card",
            }:
                meta[key] = str(content).strip()[:2000]
    metadata["meta"] = meta
    json_ld = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"})[:10]:
        raw = script.string or script.get_text()
        if not raw or len(raw) > 100000:
            continue
        try:
            value = json.loads(raw)
            json_ld.append(json.dumps(value, ensure_ascii=False)[:3000])
        except json.JSONDecodeError:
            continue
    metadata["json_ld"] = json_ld[:10]
    return json.dumps(metadata, ensure_ascii=False, indent=2)


def page_links(url: str, same_domain: bool = True, limit: int = 50) -> str:
    """Extract and categorize bounded HTTP(S) links from a public page without dumping page prose."""
    limit = max(1, min(int(limit), 200))
    try:
        final_url, _, soup, _ = _page(url)
    except Exception as exc:
        return f"Error: page link extraction failed: {exc}"
    origin = urlparse(final_url).hostname or ""
    rows = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(final_url, anchor.get("href"))
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        if same_domain and parsed.hostname.lower() != origin.lower():
            continue
        clean = parsed._replace(fragment="").geturl()
        if clean in seen:
            continue
        seen.add(clean)
        rows.append({"url": clean, "text": anchor.get_text(" ", strip=True)[:300], "same_domain": parsed.hostname.lower() == origin.lower()})
        if len(rows) >= limit:
            break
    return json.dumps({"source": final_url, "links": rows}, ensure_ascii=False, indent=2)


def _sitemap_urls(xml_text: str, limit: int) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    indexes: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return urls, indexes
    root_name = root.tag.split("}")[-1].lower()
    for elem in root.iter():
        if elem.tag.split("}")[-1].lower() != "loc" or not elem.text:
            continue
        value = elem.text.strip()
        if root_name == "sitemapindex":
            indexes.append(value)
        else:
            urls.append(value)
        if len(urls) + len(indexes) >= limit:
            break
    return urls, indexes


def discover_site(url: str, limit: int = 100) -> str:
    """Inspect robots.txt and standard sitemap locations and return a bounded site URL inventory."""
    limit = max(1, min(int(limit), 300))
    try:
        safe = validate_public_url(url)
    except Exception as exc:
        return f"Error: URL validation failed: {exc}"
    parsed = urlparse(safe)
    base = f"{parsed.scheme}://{parsed.netloc}/"
    sitemap_candidates = [urljoin(base, "sitemap.xml"), urljoin(base, "sitemap_index.xml")]
    robots_url = urljoin(base, "robots.txt")
    robots_text = ""
    try:
        _, _, robots_text = fetch_text(robots_url, max_bytes=512 * 1024, allowed_types={"text/plain", "text/html"})
        for line in robots_text.splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate and candidate not in sitemap_candidates:
                    sitemap_candidates.append(candidate)
    except Exception:
        pass
    urls: list[str] = []
    checked = []
    queue = sitemap_candidates[:6]
    while queue and len(urls) < limit and len(checked) < 10:
        sitemap = queue.pop(0)
        if sitemap in checked:
            continue
        checked.append(sitemap)
        try:
            final, ctype, body = fetch_text(sitemap, max_bytes=2 * 1024 * 1024, allowed_types={"application/xml", "text/xml", "text/plain", "application/octet-stream"})
        except Exception:
            continue
        found, indexes = _sitemap_urls(body, limit - len(urls))
        for value in found:
            if value not in urls:
                urls.append(value)
                if len(urls) >= limit:
                    break
        for nested in indexes[:6]:
            if nested not in checked and nested not in queue:
                queue.append(nested)
    return json.dumps({
        "base": base,
        "robots_url": robots_url,
        "robots_excerpt": robots_text[:4000],
        "sitemaps_checked": checked,
        "urls": urls[:limit],
    }, ensure_ascii=False, indent=2)


def read_feed(url: str, limit: int = 20) -> str:
    """Read a public RSS/Atom feed into bounded structured entries."""
    limit = max(1, min(int(limit), 50))
    try:
        final, _, body = fetch_text(url, max_bytes=2 * 1024 * 1024, allowed_types={"application/rss+xml", "application/atom+xml", "application/xml", "text/xml", "text/plain", "text/html"})
        import feedparser
        feed = feedparser.parse(body)
    except Exception as exc:
        return f"Error: feed read failed: {exc}"
    entries = []
    for item in feed.entries[:limit]:
        summary = BeautifulSoup(str(item.get("summary") or item.get("description") or ""), "html.parser").get_text(" ", strip=True)
        entries.append({
            "title": str(item.get("title") or "")[:500],
            "url": str(item.get("link") or "")[:2000],
            "published": str(item.get("published") or item.get("updated") or "")[:200],
            "author": str(item.get("author") or "")[:300],
            "summary": summary[:1500],
        })
    return json.dumps({"url": final, "feed_title": str(feed.feed.get("title") or "")[:500], "entries": entries}, ensure_ascii=False, indent=2)


def _safe_workspace_path(value: str) -> Path:
    raw = Path(str(value).strip())
    candidate = raw.resolve() if raw.is_absolute() else (WORKSPACE / raw).resolve()
    if os.path.commonpath([str(WORKSPACE), str(candidate)]) != str(WORKSPACE):
        raise ValueError("local document path must stay inside /app/workspace")
    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    return candidate


def _pdf_extract(path: Path, max_pages: int, max_chars: int) -> dict:
    info: dict = {"format": "pdf"}
    try:
        proc = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=10)
        metadata = {}
        for line in proc.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in {"Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate", "Pages", "Encrypted", "Page size"}:
                    metadata[key.strip()] = value.strip()
        info["metadata"] = metadata
    except Exception as exc:
        info["metadata_error"] = str(exc)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    try:
        cmd = ["pdftotext", "-layout", "-f", "1", "-l", str(max_pages), str(path), str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            info["error"] = (proc.stderr or proc.stdout).strip()[:1000]
            return info
        text = out.read_text(errors="replace")[:max_chars]
        info["text"] = text
        info["truncated"] = len(text) >= max_chars
        return info
    finally:
        try: out.unlink()
        except OSError: pass


def extract_document(path_or_url: str, max_pages: int = 30, max_chars: int = 30000, extract: str = "") -> str:
    """Extract bounded text and metadata from a workspace document or public PDF/text/HTML URL."""
    value = str(path_or_url or "").strip()
    if not value:
        return "Error: path_or_url is required."
    max_pages = max(1, min(int(max_pages), 100))
    max_chars = max(1000, min(int(max_chars), 100000))
    temp_path: Path | None = None
    try:
        if value.startswith(("http://", "https://")):
            final, response, body = fetch_bytes(value, max_bytes=12 * 1024 * 1024, allowed_types={
                "application/pdf", "text/plain", "text/html", "application/xhtml+xml", "application/json", "application/xml", "text/xml"
            })
            ctype = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if ctype == "application/pdf":
                fd, name = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
                temp_path = Path(name); temp_path.write_bytes(body)
                payload = _pdf_extract(temp_path, max_pages, max_chars); payload["url"] = final
            else:
                encoding = response.encoding or "utf-8"
                text = body.decode(encoding, errors="replace")
                if "html" in ctype:
                    soup = BeautifulSoup(text, "html.parser")
                    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]): element.decompose()
                    text = "\n".join(soup.stripped_strings)
                payload = {"url": final, "format": ctype, "text": text[:max_chars], "truncated": len(text) > max_chars}
        else:
            path = _safe_workspace_path(value)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                payload = _pdf_extract(path, max_pages, max_chars); payload["path"] = str(path)
            elif suffix in {".txt", ".md", ".json", ".xml", ".html", ".htm", ".csv", ".log", ".yaml", ".yml"}:
                text = path.read_text(errors="replace")
                if suffix in {".html", ".htm"}:
                    soup = BeautifulSoup(text, "html.parser"); text = "\n".join(soup.stripped_strings)
                payload = {"path": str(path), "format": suffix.lstrip("."), "text": text[:max_chars], "truncated": len(text) > max_chars}
            else:
                return "Error: supported local document types are PDF and text/HTML/JSON/XML/CSV/Markdown/YAML/log files."
        extraction = " ".join(str(extract or "").split())
        if extraction and isinstance(payload, dict) and isinstance(payload.get("text"), str):
            original = payload["text"]
            payload["text"] = targeted_extract(original, extraction, max_chars=min(max_chars, 10000))
            payload["extraction"] = extraction[:500]
            payload["source_chars"] = len(original)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: document extraction failed: {exc}"
    finally:
        if temp_path is not None:
            try: temp_path.unlink()
            except OSError: pass


def _normalized_page_text(url: str) -> tuple[str, str]:
    final, _, soup, _ = _page(url)
    for element in soup(["nav", "footer", "header", "aside", "form"]):
        element.decompose()
    text = "\n".join(line.strip() for line in soup.stripped_strings if line.strip())
    text = re.sub(r"[ \t]+", " ", text)
    return final, text[:60000]


def page_fingerprint(url: str) -> str:
    """Return a stable hash and bounded metadata for normalized public-page text without storing state."""
    try:
        final, text = _normalized_page_text(url)
    except Exception as exc:
        return f"Error: page fingerprint failed: {exc}"
    return json.dumps({"url": final, "sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text), "sample": text[:1200]}, ensure_ascii=False, indent=2)


def page_diff(url: str, max_diff_chars: int = 12000) -> str:
    """Compare normalized page text with the previously stored version, persist the new version, and return only meaningful changes."""
    max_diff_chars = max(1000, min(int(max_diff_chars), 30000))
    try:
        final, text = _normalized_page_text(url)
    except Exception as exc:
        return f"Error: page diff failed: {exc}"
    key = "page.diff." + hashlib.sha256(final.encode()).hexdigest()[:24]
    previous = get_monitor_state(key)
    old_text = ""
    old_hash = ""
    if isinstance(previous, dict):
        old_text = str(previous.get("text") or "")
        old_hash = str(previous.get("sha256") or "")
    new_hash = hashlib.sha256(text.encode()).hexdigest()
    changed = bool(previous) and new_hash != old_hash
    diff_text = ""
    if changed:
        lines = difflib.unified_diff(old_text.splitlines(), text.splitlines(), fromfile="previous", tofile="current", lineterm="", n=2)
        diff_text = "\n".join(lines)[:max_diff_chars]
    record_monitor_state(key, {"url": final, "sha256": new_hash, "text": text, "captured_at": utc_now()})
    return json.dumps({"url": final, "baseline_created": previous is None, "changed": changed, "previous_sha256": old_hash, "sha256": new_hash, "diff": diff_text}, ensure_ascii=False, indent=2)
