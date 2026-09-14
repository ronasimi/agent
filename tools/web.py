import json
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
