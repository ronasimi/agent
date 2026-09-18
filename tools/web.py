"""Safe web search and browsing helpers."""
from __future__ import annotations

import json

import requests
from bs4 import BeautifulSoup

from .netutil import fetch_text


def web_search(query: str = "") -> str:
    """Search the public web for current information and return a small JSON result set."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=5))
        compact = [
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
            for item in results
        ]
        return json.dumps(compact, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: web search failed: {exc}"


def wiki_search(query: str = "") -> str:
    """Search Wikipedia for encyclopedic background information."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=4, auto_suggest=True)
    except Exception as exc:
        return f"Error: Wikipedia search failed: {exc}"


def browse_url(url: str = "") -> str:
    """Fetch a public HTTP(S) URL with redirect, size, and private-network protections and extract readable text."""
    if not str(url).strip():
        return "Error: Missing required 'url' parameter."
    try:
        final_url, content_type, body = fetch_text(url, max_bytes=2 * 1024 * 1024)
        if content_type in {"application/json", "application/xml", "text/xml", "text/plain"}:
            text = body
        else:
            soup = BeautifulSoup(body, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
                element.decompose()
            text = "\n".join(soup.stripped_strings)
        text = text[:20000]
        return f"URL: {final_url}\nContent-Type: {content_type}\n\n{text or 'The page returned no readable text content.'}"
    except Exception as exc:
        return f"Error: browsing URL failed: {exc}"
