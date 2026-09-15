import json
import requests
from bs4 import BeautifulSoup
import wikipedia
from ddgs import DDGS

def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return str(e)

def wiki_search(query: str) -> str:
    """Search Wikipedia for encyclopedic knowledge."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return str(e)

def browse_url(url: str) -> str:
    """Fetch and extract clean text content from a live URL in real-time.
    
    Args:
        url: The full http:// or https:// URL to browse and read.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy layout elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        
        # Cap text length to prevent flooding the context window
        if len(text) > 12000:
            return text[:12000] + "\n[Content truncated due to length...]"
        return text.strip() or "The page returned no readable text content."
    except Exception as e:
        return f"Error browsing URL: {str(e)}"
