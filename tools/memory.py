import json
import sqlite3

DB_PATH = "/app/memory/knowledge.db"

def init_db():
    """Ensure the SQLite knowledge database and table exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS memory 
                        (topic TEXT PRIMARY KEY, fact TEXT)''')

def remember(topic: str, fact: str) -> str:
    """Save a learned fact or user preference to the persistent knowledge base."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO memory (topic, fact) VALUES (?, ?)", 
                     (topic, fact))
    return f"Successfully committed '{topic}' to long-term memory."

def search_memory(query: str) -> str:
    """Search the knowledge base, returning all records if queried with 'memory' or empty string."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query_str = query.strip().lower()
        
        if not query_str or query_str in ["memory", "all", "everything", "what's in your memory"]:
            cursor.execute("SELECT topic, fact FROM memory")
        else:
            keywords = query_str.split()
            conditions = ["(topic LIKE ? OR fact LIKE ?)" for _ in keywords]
            params = []
            for kw in keywords:
                params.extend([f'%{kw}%', f'%{kw}%'])
            sql = "SELECT topic, fact FROM memory WHERE " + " OR ".join(conditions)
            cursor.execute(sql, params)
            
        rows = cursor.fetchall()
    
    if not rows:
        return "No related memories found."
    return json.dumps([{"topic": r[0], "fact": r[1]} for r in rows])
