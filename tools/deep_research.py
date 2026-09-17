# ==========================================
# FILE: tools/deep_research.py
# ==========================================
import sqlite3
import os
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ollama import Client

FAST_MODEL = "qwen2.5-coder:1.5b"
DB_PATH = "/app/memory/knowledge.db"

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_buffer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            url TEXT,
            title TEXT,
            summary TEXT,
            raw_content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def deep_search_and_scrape(query: str, max_results: int = 3) -> str:
    """Executes web search, scrapes URLs, stores raw text in SQLite buffer, and returns distilled reflections."""
    conn = _get_db()
    cursor = conn.cursor()
    
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search execution failed: {str(e)}"

    if not results:
        return f"No search results found for query: '{query}'."

    reflections = []
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=ollama_host)

    for r in results:
        url = r.get("href")
        title = r.get("title", "")
        
        # Deduplication check
        cursor.execute("SELECT id FROM research_buffer WHERE url = ?", (url,))
        if cursor.fetchone():
            continue
            
        # Scrape and clean page content
        try:
            resp = requests.get(url, headers={"User-Agent": "DeepResearchAgent/1.0"}, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                element.decompose()
            text = ' '.join(soup.stripped_strings)[:8000]
        except Exception as e:
            text = f"Scraping failed: {str(e)}"

        # Distill findings using fast secondary model
        prompt = (
            f"Analyze the following text regarding the query: '{query}'.\n"
            "Extract 3-5 distinct, key factual insights as concise bullet points.\n\n"
            f"Text:\n{text[:4000]}"
        )
        try:
            distill_response = client.generate(model=FAST_MODEL, prompt=prompt)
            distilled_notes = distill_response.get('response', '').strip()
        except Exception as e:
            distilled_notes = f"Distillation error: {str(e)}"

        # Save to SQLite buffer
        cursor.execute(
            "INSERT INTO research_buffer (query, url, title, summary, raw_content) VALUES (?, ?, ?, ?, ?)",
            (query, url, title, distilled_notes, text)
        )
        conn.commit()
        
        reflections.append(f"### Source: {title}\nURL: {url}\nReflections:\n{distilled_notes}")

    conn.close()

    if not reflections:
        return f"All URLs for query '{query}' were already processed in buffer."

    return "\n\n---\n\n".join(reflections)

def read_research_buffer(topic: str = "") -> str:
    """Retrieves all distilled research summaries from the SQLite buffer for final report generation."""
    conn = _get_db()
    cursor = conn.cursor()
    if topic:
        cursor.execute("SELECT title, url, summary FROM research_buffer WHERE query LIKE ? OR summary LIKE ?", (f"%{topic}%", f"%{topic}%"))
    else:
        cursor.execute("SELECT title, url, summary FROM research_buffer")
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "Research buffer is currently empty."
        
    compiled_notes = []
    for title, url, summary in rows:
        compiled_notes.append(f"#### {title}\n**URL:** {url}\n**Key Findings:**\n{summary}\n")
        
    return "\n".join(compiled_notes)

def clear_research_buffer() -> str:
    """Clears all records stored in the research buffer database."""
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM research_buffer")
    conn.commit()
    conn.close()
    return "Research buffer successfully cleared."
