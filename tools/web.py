import json
import requests
from bs4 import BeautifulSoup
import wikipedia
from ddgs import DDGS

def web_search(query: str = "") -> str:
    """Search the web for current real-time information and return search snippets."""
    if not query or not str(query).strip():
        return "Error: Missing required 'query' parameter."
    try:
        # Wrap DDGS().text generator in list() to ensure valid JSON serialization
        results = list(DDGS().text(query, max_results=3))
        return json.dumps(results)
    except Exception as e:
        return f"Web search error: {str(e)}"

def wiki_search(query: str = "") -> str:
    """Search Wikipedia for encyclopedic summaries and factual background."""
    if not query or not str(query).strip():
        return "Error: Missing required 'query' parameter."
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"

def browse_url(url: str = "") -> str:
    """Fetch and extract clean text content from a live URL in real-time."""
    if not url or not str(url).strip():
        return "Error: Missing required 'url' parameter."
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        
        if len(text) > 12000:
            return text[:12000] + "\n[Content truncated due to length...]"
        return text.strip() or "The page returned no readable text content."
    except Exception as e:
        return f"Error browsing URL: {str(e)}"
