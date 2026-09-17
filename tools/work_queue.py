# ==========================================
# FILE: tools/work_queue.py
# Autonomous Work Delegation and Scheduling
# ==========================================
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
import yaml

DB_PATH = "/app/memory/knowledge.db"
DB_TIMEOUT = 10.0


def init_work_queue_db():
    """Initialize work queue tables."""
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_queue (
                id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                assigned_to TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                due_at DATETIME,
                estimated_hours REAL,
                tags TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_results (
                work_id TEXT PRIMARY KEY,
                result_summary TEXT,
                result_full TEXT,
                completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (work_id) REFERENCES work_queue(id)
            )
        """)
        
        conn.commit()
    finally:
        conn.close()


def queue_work(
    title: str,
    description: str = "",
    priority: int = 0,
    due_in_hours: int = None,
    tags: list = None,
    estimated_hours: float = 1.0
) -> str:
    """
    Add a durable work-list item. This legacy queue is a planning list; the dedicated
    runtime job queue is used for executable background research.
    
    Args:
        title: Work title
        description: Detailed description
        priority: Priority level (0=normal, >0=higher, <0=lower)
        due_in_hours: Hours from now when due
        tags: List of tags for categorization
        estimated_hours: Estimated time to complete
    
    Returns: Work ID
    
    Example:
        >>> work_id = queue_work("Research AI trends", priority=2, due_in_hours=24)
        >>> isinstance(work_id, str) and len(work_id) > 0
        True
    """
    work_id = str(uuid.uuid4())
    
    due_at = None
    if due_in_hours:
        due_at = (datetime.now() + timedelta(hours=due_in_hours)).isoformat()
    
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("""
            INSERT INTO work_queue 
            (id, title, description, priority, due_at, tags, estimated_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (work_id, title, description, priority, due_at, json.dumps(tags or []), estimated_hours))
        conn.commit()
        return work_id
    finally:
        conn.close()


def get_next_work_item() -> dict:
    """
    Retrieve the highest-priority pending work-list item.
    
    Returns: Work item dictionary or None
    
    Example:
        >>> queue_work("Test work", priority=1)
        '...'
        >>> next_item = get_next_work_item()
        >>> next_item is not None
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        # Priority: urgent overdue > due soon > high priority
        cursor.execute("""
            SELECT id, title, description, priority, due_at, estimated_hours, tags
            FROM work_queue
            WHERE status = 'pending'
            ORDER BY 
                CASE WHEN due_at < datetime('now') THEN -999 ELSE 0 END,
                CASE WHEN due_at < datetime('now', '+1 hour') THEN -100 ELSE 0 END,
                priority DESC,
                created_at ASC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "priority": row[3],
                "due_at": row[4],
                "estimated_hours": row[5],
                "tags": json.loads(row[6])
            }
        return None
    finally:
        conn.close()


def mark_work_complete(work_id: str, summary: str, full_result: str = "") -> str:
    """
    Mark a work item complete and store results.
    
    Args:
        work_id: Work item ID
        summary: Summary of results
        full_result: Full detailed results
    
    Returns: Confirmation message
    
    Example:
        >>> work_id = queue_work("Test work")
        >>> mark_work_complete(work_id, "Completed", "Full details")
        "Work item '...' marked complete."
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute(
            "UPDATE work_queue SET status = 'completed' WHERE id = ?",
            (work_id,)
        )
        conn.execute(
            "INSERT INTO work_results (work_id, result_summary, result_full) VALUES (?, ?, ?)",
            (work_id, summary, full_result or summary)
        )
        conn.commit()
        return f"Work item '{work_id[:8]}' marked complete."
    finally:
        conn.close()


def list_work_queue(status: str = "pending") -> str:
    """
    List work items by status.
    
    Args:
        status: Filter by status (pending, running, completed, cancelled)
    
    Returns: JSON string of work items
    
    Example:
        >>> queue_work("Work 1", priority=1)
        '...'
        >>> queue_work("Work 2", priority=2)
        '...'
        >>> result = list_work_queue("pending")
        >>> "Work" in result
        True
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, priority, due_at, estimated_hours, status
            FROM work_queue
            WHERE status = ?
            ORDER BY priority DESC, created_at ASC
        """, (status,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return f"No {status} work items."
        
        items = []
        for r in rows:
            items.append({
                "id": r[0][:8],
                "title": r[1],
                "priority": r[2],
                "due": r[3][:16] if r[3] else "no deadline",
                "hours": r[4],
                "status": r[5]
            })
        
        return json.dumps(items, indent=2)
    finally:
        conn.close()


def get_work_details(work_id: str) -> str:
    """
    Get detailed information about a work item.
    
    Args:
        work_id: Work item ID
    
    Returns: JSON string with work details
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        # Get work item
        cursor.execute("""
            SELECT id, title, description, priority, status, due_at, 
                   estimated_hours, tags, created_at
            FROM work_queue WHERE id = ?
        """, (work_id,))
        
        work_row = cursor.fetchone()
        if not work_row:
            return f"Work item {work_id[:8]} not found"
        
        # Get results if completed
        cursor.execute("""
            SELECT result_summary, result_full, completed_at
            FROM work_results WHERE work_id = ?
        """, (work_id,))
        
        result_row = cursor.fetchone()
        
        details = {
            "id": work_row[0][:8],
            "title": work_row[1],
            "description": work_row[2],
            "priority": work_row[3],
            "status": work_row[4],
            "due": work_row[5],
            "estimated_hours": work_row[6],
            "tags": json.loads(work_row[7]),
            "created": work_row[8]
        }
        
        if result_row:
            details["result_summary"] = result_row[0]
            details["result_full"] = result_row[1]
            details["completed"] = result_row[2]
        
        return json.dumps(details, indent=2)
    finally:
        conn.close()


def get_work_result(work_id: str) -> str:
    """
    Get the result/output from a completed work item.
    
    Args:
        work_id: Work item ID
    
    Returns: Result text or error message
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT result_full FROM work_results WHERE work_id = ?",
            (work_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else f"No results found for work {work_id[:8]}"
    finally:
        conn.close()


def update_work_status(work_id: str, status: str) -> str:
    """
    Update the status of a work item.
    
    Args:
        work_id: Work item ID
        status: New status (pending, running, completed, cancelled)
    
    Returns: Confirmation message
    """
    valid_statuses = ["pending", "running", "completed", "cancelled"]
    if status not in valid_statuses:
        return f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
    
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute(
            "UPDATE work_queue SET status = ? WHERE id = ?",
            (status, work_id)
        )
        conn.commit()
        return f"Work {work_id[:8]} status updated to '{status}'"
    finally:
        conn.close()


def delete_work(work_id: str) -> str:
    """
    Delete a work item and its results.
    
    Args:
        work_id: Work item ID
    
    Returns: Confirmation message
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("DELETE FROM work_results WHERE work_id = ?", (work_id,))
        conn.execute("DELETE FROM work_queue WHERE id = ?", (work_id,))
        conn.commit()
        return f"Work {work_id[:8]} deleted"
    finally:
        conn.close()


def get_work_statistics() -> str:
    """
    Get statistics about work queue.
    
    Returns: JSON string with statistics
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM work_queue WHERE status = 'pending'")
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM work_queue WHERE status = 'running'")
        running = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM work_queue WHERE status = 'completed'")
        completed = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(estimated_hours) FROM work_queue WHERE status IN ('pending', 'running')")
        total_hours = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM work_queue")
        total = cursor.fetchone()[0]
        
        stats = {
            "total": total,
            "pending": pending,
            "running": running,
            "completed": completed,
            "estimated_hours_remaining": total_hours,
            "completion_rate": round((completed / total * 100) if total > 0 else 0, 1)
        }
        
        return json.dumps(stats, indent=2)
    finally:
        conn.close()


# Initialize on import
try:
    init_work_queue_db()
except:
    pass
