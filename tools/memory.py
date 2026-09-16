# ==========================================
# FILE: tools/memory.py
# ==========================================
import json
import sqlite3
import threading
import math
import subprocess
import os
import yaml

DB_PATH = "/app/memory/knowledge.db"

with open('/app/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
EMBED_MODEL = config.get('agent', {}).get('embed_model', 'nomic-embed-text')

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute('''CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS semantic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, fact TEXT, embedding TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS background_tasks (task_name TEXT PRIMARY KEY, status TEXT, output TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

def _init_chat_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT)''')

def _init_checkpoint_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute('''CREATE TABLE IF NOT EXISTS checkpoints (task_name TEXT PRIMARY KEY, state_data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

def _save_message_to_db(msg: dict):
    role = msg.get('role')
    content = msg.get('content', '')
    name = msg.get('name')
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 1")
        last = cursor.fetchone()
        if last and last[0] == role and last[1] == content:
            return
            
    extra_data = {}
    if 'tool_calls' in msg: extra_data['tool_calls'] = msg['tool_calls']
    extra = json.dumps(extra_data) if extra_data else None
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO chat_history (role, content, name, extra) VALUES (?, ?, ?, ?)", (role, content, name, extra))

def _load_chat_history_from_db(limit: int = 20) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content, name, extra FROM chat_history ORDER BY id DESC LIMIT ?", (limit * 2,))
        rows = cursor.fetchall()
    
    rows.reverse()
    messages = []
    last_msg = None
    for r in rows:
        role, content, name, extra = r
        content = content or ''
        if last_msg and last_msg['role'] == role and last_msg['content'] == content: continue
            
        msg = {'role': role, 'content': content}
        if name: msg['name'] = name
        if extra:
            try:
                extra_data = json.loads(extra)
                if 'tool_calls' in extra_data: msg['tool_calls'] = extra_data['tool_calls']
            except json.JSONDecodeError: pass
        messages.append(msg)
        last_msg = msg
        
    return messages[-limit:] if len(messages) > limit else messages

def clear_chat_history() -> str:
    with sqlite3.connect(DB_PATH) as conn: conn.execute("DELETE FROM chat_history")
    return "Chat history cleared."

def remember(topic: str = "general_knowledge", fact: str = "Recorded by agent action") -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO memory (topic, fact) VALUES (?, ?)", (str(topic), str(fact)))
    return f"Successfully committed '{topic}' to long-term memory."

def search_memory(query: str = "") -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query_str = str(query).strip().lower()
        if not query_str or query_str in ["memory", "all", "everything"]:
            cursor.execute("SELECT topic, fact FROM memory")
        else:
            keywords = query_str.split()
            conditions = ["(topic LIKE ? OR fact LIKE ?)" for _ in keywords]
            params = [f'%{kw}%' for kw in keywords for _ in range(2)]
            cursor.execute("SELECT topic, fact FROM memory WHERE " + " OR ".join(conditions), params)
        rows = cursor.fetchall()
    return json.dumps([{"topic": r[0], "fact": r[1]} for r in rows]) if rows else "No related memories found."

def remember_semantic(topic: str = "general_knowledge", fact: str = "") -> str:
    if not fact or not str(fact).strip(): return "Error: Missing required 'fact' parameter."
    embedding_json = "[]"
    try:
        import ollama
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        res = client.embeddings(model=EMBED_MODEL, prompt=fact)
        if 'embedding' in res: embedding_json = json.dumps(res['embedding'])
    except Exception as e: return f"Embedding generation failed: {e}"
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO semantic_memory (topic, fact, embedding) VALUES (?, ?, ?)", (str(topic), str(fact), embedding_json))
    return f"Successfully stored semantic memory under topic '{topic}'."

def _cosine_similarity(vec1, vec2):
    if not vec1 or not vec2 or len(vec1) != len(vec2): return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1, norm2 = math.sqrt(sum(a * a for a in vec1)), math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def search_semantic_memory(query: str = "", limit: int = 5) -> str:
    if not query or not str(query).strip(): return search_memory(query)
    try:
        import ollama
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        res = client.embeddings(model=EMBED_MODEL, prompt=query)
        query_embedding = res.get('embedding', [])
    except Exception: return search_memory(query)
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT topic, fact, embedding FROM semantic_memory")
        rows = cursor.fetchall()
        
    scored = []
    for topic, fact, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            scored.append((_cosine_similarity(query_embedding, emb), topic, fact))
        except Exception: continue
            
    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:limit]
    return json.dumps([{"similarity": round(score, 4), "topic": t, "fact": f} for score, t, f in top_results]) if top_results else search_memory(query)

def _run_background_task(task_name: str, python_code: str, timeout: int):
    workspace = "/app/workspace"
    try:
        result = subprocess.run(["python", "-c", python_code], cwd=workspace, capture_output=True, text=True, timeout=timeout)
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        status = "completed" if result.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        output = f"Error: Background task timed out after {timeout} seconds."
        status = "failed"
    except Exception as e:
        output = f"Error executing background task: {str(e)}"
        status = "failed"
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO background_tasks (task_name, status, output) VALUES (?, ?, ?)", (task_name, status, output))

def start_background_task(task_name: str = "", python_code: str = "", timeout: int = 3600) -> str:
    """Start a long-running python task in a background thread.
    
    Args:
        task_name: Unique name identifier for the task.
        python_code: Python code string to execute in the background workspace.
        timeout: Maximum execution time in seconds (default: 3600).
    """
    if not task_name or not python_code: return "Error: Missing required 'task_name' or 'python_code' parameter."
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO background_tasks (task_name, status, output) VALUES (?, ?, ?)",
                     (task_name, "running", "Task is currently executing in the background..."))
                     
    thread = threading.Thread(target=_run_background_task, args=(task_name, python_code, timeout), daemon=True)
    thread.start()
    return f"Successfully started background task '{task_name}' with a timeout of {timeout}s."

def check_background_task(task_name: str = "") -> str:
    """Check the status and output of a background task."""
    if not task_name: return "Error: Missing required 'task_name' parameter."
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, output, timestamp FROM background_tasks WHERE task_name = ?", (task_name,))
        row = cursor.fetchone()
    return json.dumps({"task_name": task_name, "status": row[0], "output": row[1], "timestamp": row[2]}) if row else f"No background task found with name '{task_name}'."

# [The rest of the SQLite custom table functions remain unchanged, but they inherit the WAL concurrency benefits from the DB initialization above.]
