# ==========================================
# FILE: tools/task_manager.py
# Task State Management with Checkpoints
# ==========================================
import sqlite3
import json
import uuid
from enum import Enum
from datetime import datetime
import os
import yaml

# Load config
try:
    with open('/app/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except:
    config = {}

DB_PATH = "/app/memory/knowledge.db"
DB_TIMEOUT = 10.0


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def init_task_manager_db():
    """Initialize task persistence layer with checkpoint support."""
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        
        # Main tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                status TEXT DEFAULT 'pending',
                state_json TEXT,
                priority INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                started_at DATETIME,
                completed_at DATETIME,
                error TEXT,
                dependencies TEXT
            )
        """)
        
        # Checkpoints for resumable execution
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_checkpoints (
                task_id TEXT,
                checkpoint_num INTEGER,
                state_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_id, checkpoint_num),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)
        
        # Task execution logs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                level TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)
        
        conn.commit()
    finally:
        conn.close()


def create_task(
    name: str,
    description: str = "",
    priority: int = 0,
    dependencies: list = None,
    state: dict = None
) -> str:
    """
    Create a new long-running task with state management.
    
    Args:
        name: Unique task identifier
        description: Human-readable task description
        priority: 0=normal, >0=higher, <0=lower
        dependencies: List of task IDs this task waits for
        state: Initial state dict (e.g., {"step": 1, "context": {...}})
    
    Returns: Task ID
    
    Example:
        >>> task_id = create_task("research_ai_trends", "Research AI trends", priority=2)
        >>> isinstance(task_id, str) and len(task_id) > 0
        True
    """
    task_id = str(uuid.uuid4())
    state = state or {"step": 0, "progress": 0}
    
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("""
            INSERT INTO tasks 
            (id, name, status, state_json, priority, dependencies)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            task_id, name, TaskState.PENDING.value, json.dumps(state),
            priority, json.dumps(dependencies or [])
        ))
        conn.commit()
        log_task(f"Task created: {name}", task_id, "info")
        return task_id
    except sqlite3.IntegrityError:
        return f"Error: Task name '{name}' already exists"
    finally:
        conn.close()


def save_checkpoint(task_id: str, state: dict) -> bool:
    """
    Save a checkpoint for resumable execution.
    
    Args:
        task_id: Task identifier
        state: State dictionary to save
    
    Returns: True if successful
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> save_checkpoint(task_id, {"progress": 50})
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        # Get next checkpoint number
        cursor.execute("SELECT MAX(checkpoint_num) FROM task_checkpoints WHERE task_id = ?", (task_id,))
        last_num = (cursor.fetchone()[0] or 0)
        
        # Save checkpoint
        conn.execute("""
            INSERT INTO task_checkpoints (task_id, checkpoint_num, state_json)
            VALUES (?, ?, ?)
        """, (task_id, last_num + 1, json.dumps(state)))
        
        # Update main task state
        conn.execute("UPDATE tasks SET state_json = ? WHERE id = ?", (json.dumps(state), task_id))
        conn.commit()
        return True
    except Exception as e:
        log_task(f"Checkpoint save failed: {e}", task_id, "error")
        return False
    finally:
        conn.close()


def get_task_state(task_id: str) -> dict:
    """
    Retrieve current task state (hydrates from checkpoint on startup).
    
    Args:
        task_id: Task identifier
    
    Returns: Task state dictionary
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> state = get_task_state(task_id)
        >>> isinstance(state, dict)
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT state_json FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else {}
    except:
        return {}
    finally:
        conn.close()


def update_task_status(task_id: str, status: TaskState, error: str = None) -> bool:
    """
    Update task status with error tracking.
    
    Args:
        task_id: Task identifier
        status: New task status (TaskState enum)
        error: Error message if failed
    
    Returns: True if successful
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> update_task_status(task_id, TaskState.RUNNING)
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        timestamp = datetime.now().isoformat()
        
        if status == TaskState.COMPLETED:
            conn.execute(
                "UPDATE tasks SET status = ?, completed_at = ?, error = ? WHERE id = ?",
                (status.value, timestamp, error, task_id)
            )
        elif status == TaskState.RUNNING:
            conn.execute(
                "UPDATE tasks SET status = ?, started_at = ?, error = ? WHERE id = ?",
                (status.value, timestamp, error, task_id)
            )
        else:
            conn.execute(
                "UPDATE tasks SET status = ?, error = ? WHERE id = ?",
                (status.value, error, task_id)
            )
        
        conn.commit()
        if error:
            log_task(f"Status → {status.value}: {error}", task_id, "warning")
        return True
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False
    finally:
        conn.close()


def log_task(message: str, task_id: str, level: str = "info") -> None:
    """
    Append structured logs to task execution history.
    
    Args:
        message: Log message
        task_id: Task identifier
        level: Log level (info, warning, error)
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> log_task("Test message", task_id, "info")
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("""
            INSERT INTO task_logs (task_id, level, message)
            VALUES (?, ?, ?)
        """, (task_id, level, message))
        conn.commit()
    finally:
        conn.close()


def list_tasks(status: str = None) -> str:
    """
    List all tasks with filtering.
    
    Args:
        status: Filter by status (pending, running, completed, failed, etc.)
    
    Returns: JSON string of tasks
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> result = list_tasks("pending")
        >>> isinstance(result, str)
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        if status:
            cursor.execute("""
                SELECT id, name, status, priority, created_at, completed_at
                FROM tasks WHERE status = ?
                ORDER BY priority DESC, created_at ASC
            """, (status,))
        else:
            cursor.execute("""
                SELECT id, name, status, priority, created_at, completed_at
                FROM tasks
                ORDER BY priority DESC, created_at ASC
            """)
        
        rows = cursor.fetchall()
        tasks = []
        for r in rows:
            tasks.append({
                "id": r[0][:8],  # Truncate UUID for readability
                "name": r[1],
                "status": r[2],
                "priority": r[3],
                "created": r[4][:10] if r[4] else None,
                "completed": r[5][:10] if r[5] else None
            })
        
        return json.dumps(tasks, indent=2) if tasks else f"No {status or 'any'} tasks"
    finally:
        conn.close()


def get_task_logs(task_id: str, limit: int = 20) -> str:
    """
    Retrieve logs for a specific task.
    
    Args:
        task_id: Task identifier
        limit: Maximum number of logs to return
    
    Returns: JSON string of logs
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT level, message, timestamp
            FROM task_logs
            WHERE task_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (task_id, limit))
        
        rows = cursor.fetchall()
        logs = [{"level": r[0], "message": r[1], "time": r[2]} for r in rows]
        return json.dumps(logs[::-1], indent=2) if logs else "No logs for this task"
    finally:
        conn.close()


def resume_interrupted_task(task_id: str) -> dict:
    """
    Restore task state from latest checkpoint after crash.
    
    Args:
        task_id: Task identifier
    
    Returns: Restored state dictionary
    
    Example:
        >>> task_id = create_task("test", "test")
        >>> save_checkpoint(task_id, {"progress": 75})
        True
        >>> state = resume_interrupted_task(task_id)
        >>> state.get("progress")
        75
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        # Get latest checkpoint
        cursor.execute("""
            SELECT state_json FROM task_checkpoints
            WHERE task_id = ?
            ORDER BY checkpoint_num DESC LIMIT 1
        """, (task_id,))
        
        row = cursor.fetchone()
        if row:
            state = json.loads(row[0])
            update_task_status(task_id, TaskState.RUNNING)
            log_task(f"Resumed from checkpoint", task_id, "info")
            return state
        else:
            return get_task_state(task_id)
    finally:
        conn.close()


def get_task_info(task_id: str) -> str:
    """
    Get detailed information about a task.
    
    Args:
        task_id: Task identifier
    
    Returns: JSON string with task details
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, status, priority, created_at, started_at, 
                   completed_at, error, state_json
            FROM tasks WHERE id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        if row:
            return json.dumps({
                "id": row[0][:8],
                "name": row[1],
                "status": row[2],
                "priority": row[3],
                "created": row[4],
                "started": row[5],
                "completed": row[6],
                "error": row[7],
                "state": json.loads(row[8]) if row[8] else {}
            }, indent=2)
        else:
            return f"Task {task_id} not found"
    finally:
        conn.close()


def delete_task(task_id: str) -> str:
    """Delete a task and its checkpoints."""
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        # Delete checkpoints
        conn.execute("DELETE FROM task_checkpoints WHERE task_id = ?", (task_id,))
        # Delete logs
        conn.execute("DELETE FROM task_logs WHERE task_id = ?", (task_id,))
        # Delete task
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return f"Task {task_id[:8]} deleted"
    finally:
        conn.close()


# Initialize on import
try:
    init_task_manager_db()
except:
    pass
