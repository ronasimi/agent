import json
import sqlite3

DB_PATH = "/app/memory/knowledge.db"

def init_db():
    """Ensure the SQLite knowledge database and table exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS memory 
                        (topic TEXT PRIMARY KEY, fact TEXT)''')

def _init_chat_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT)''')

def _init_checkpoint_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS checkpoints 
                        (task_name TEXT PRIMARY KEY, state_data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

def _save_message_to_db(msg: dict):
    """Save a message to SQLite, dropping exact consecutive duplicates."""
    role = msg.get('role')
    content = msg.get('content', '')
    name = msg.get('name')
    
    # Check if this exact message is already the latest one in the DB (deduplication)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 1")
        last = cursor.fetchone()
        if last and last[0] == role and last[1] == content:
            return  # Skip saving consecutive duplicate
    
    extra_data = {}
    if 'tool_calls' in msg:
        extra_data['tool_calls'] = msg['tool_calls']
    extra = json.dumps(extra_data) if extra_data else None
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO chat_history (role, content, name, extra) VALUES (?, ?, ?, ?)", 
                     (role, content, name, extra))

def _load_chat_history_from_db(limit: int = 20) -> list:
    """Load recent chat messages from SQLite with built-in consecutive deduplication."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Fetch slightly more rows to account for filtered duplicates
        cursor.execute("SELECT role, content, name, extra FROM chat_history ORDER BY id DESC LIMIT ?", (limit * 2,))
        rows = cursor.fetchall()
    
    rows.reverse()
    
    messages = []
    last_msg = None
    for r in rows:
        role, content, name, extra = r
        content = content or ''
        
        # Filter out consecutive duplicate entries in loaded history
        if last_msg and last_msg['role'] == role and last_msg['content'] == content:
            continue
            
        msg = {'role': role, 'content': content}
        if name:
            msg['name'] = name
        if extra:
            try:
                extra_data = json.loads(extra)
                if 'tool_calls' in extra_data:
                    msg['tool_calls'] = extra_data['tool_calls']
            except json.JSONDecodeError:
                pass
        messages.append(msg)
        last_msg = msg
        
    # Trim down to the requested limit after deduplication filtering
    if len(messages) > limit:
        messages = messages[-limit:]
        
    return messages

def clear_chat_history() -> str:
    """Clear all chat history from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history")
    return "Chat history successfully cleared."

def remember(topic: str = "general_knowledge", fact: str = "Recorded by agent action") -> str:
    """Save a learned fact or user preference to the persistent knowledge base.
    
    Args:
        topic: The subject category or key name.
        fact: The factual information or instruction to store.
    """
    if not topic or not str(topic).strip():
        topic = "general_knowledge"
    if not fact or not str(fact).strip():
        fact = "Recorded by agent action"
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO memory (topic, fact) VALUES (?, ?)", 
                     (str(topic), str(fact)))
    return f"Successfully committed '{topic}' to long-term memory."

def search_memory(query: str = "") -> str:
    """Search the knowledge base, returning all records if queried with 'memory' or empty string.
    
    Args:
        query: Search keywords or term.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query_str = str(query).strip().lower()
        
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

def search_chat_history(query: str = "") -> str:
    """Search older archived chat history for past discussions or code snippets.
    
    Args:
        query: Keywords to search for in past conversations.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query_str = f"%{str(query).strip().lower()}%"
        cursor.execute("SELECT role, content FROM chat_history WHERE LOWER(content) LIKE ? ORDER BY id DESC LIMIT 10", (query_str,))
        rows = cursor.fetchall()
        
    if not rows:
        return "No matching past messages found."
    return json.dumps([{"role": r[0], "content": r[1]} for r in rows])

def save_checkpoint(task_name: str = "default_task", state_data: str = "{}") -> str:
    """Save or update the state variables and progress of a long-running task.
    
    Args:
        task_name: A unique identifier for the task.
        state_data: JSON string or summary of current variables, loops, or progress.
    """
    if not task_name or not str(task_name).strip():
        task_name = "default_task"
    if not state_data or not str(state_data).strip():
        state_data = "{}"
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO checkpoints (task_name, state_data) VALUES (?, ?)", 
                     (str(task_name), str(state_data)))
    return f"Checkpoint successfully saved for task '{task_name}'."

def load_checkpoint(task_name: str = "default_task") -> str:
    """Retrieve the saved state of a long-running task to resume execution.
    
    Args:
        task_name: The unique identifier of the task to load.
    """
    if not task_name or not str(task_name).strip():
        task_name = "default_task"
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT state_data, timestamp FROM checkpoints WHERE task_name = ?", (str(task_name),))
        row = cursor.fetchone()
    
    if not row:
        return f"No checkpoint found for task '{task_name}'."
    return json.dumps({"task_name": task_name, "state_data": row[0], "timestamp": row[1]})
