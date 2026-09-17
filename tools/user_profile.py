# ==========================================
# FILE: tools/user_profile.py
# User Identity and Preferences Management
# ==========================================
import sqlite3
import json
import yaml

DB_PATH = "/app/memory/knowledge.db"
DB_TIMEOUT = 10.0


def init_user_profile_db():
    """Initialize user profile tables."""
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                category TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (category, key)
            )
        """)
        
        conn.commit()
    finally:
        conn.close()


def set_user_identity(
    name: str = "",
    role: str = "",
    timezone: str = "UTC",
    email: str = "",
    interests: list = None
) -> str:
    """
    Configure who 'you' are for the agent to act as.
    
    Args:
        name: Your name
        role: Your role/title
        timezone: Your timezone (e.g., America/New_York)
        email: Your email address
        interests: List of interests/topics
    
    Returns: Confirmation message
    
    Example:
        >>> result = set_user_identity(name="Alice", role="Researcher")
        >>> "User profile set" in result
        True
    """
    profile = {
        "name": name,
        "role": role,
        "timezone": timezone,
        "email": email,
        "interests": interests or []
    }
    
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        for k, v in profile.items():
            value = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            conn.execute(
                "INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)",
                (f"identity.{k}", value)
            )
        conn.commit()
        return f"User profile set: {name} ({role})"
    finally:
        conn.close()


def get_user_identity() -> dict:
    """
    Retrieve user identity information.
    
    Returns: Dictionary with user identity
    
    Example:
        >>> set_user_identity(name="Bob", role="Engineer")
        'User profile set: Bob (Engineer)'
        >>> identity = get_user_identity()
        >>> identity.get("name")
        'Bob'
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM user_profile WHERE key LIKE 'identity.%'")
        
        identity = {}
        for k, v in cursor.fetchall():
            key = k.replace("identity.", "")
            try:
                identity[key] = json.loads(v)
            except:
                identity[key] = v
        
        return identity
    finally:
        conn.close()


def set_research_preference(category: str, key: str, value: str) -> str:
    """
    Set research behavior preferences.
    
    Categories:
    - search_depth: light, balanced, deep
    - output_format: summary, detailed, bullets
    - academic_weight: 0.0-1.0
    - recency_weight: 0.0-1.0
    - max_results: number
    
    Args:
        category: Preference category
        key: Preference key
        value: Preference value
    
    Returns: Confirmation message
    
    Example:
        >>> set_research_preference("search_depth", "depth", "deep")
        'Preference set: search_depth.depth = deep'
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO user_preferences (category, key, value)
            VALUES (?, ?, ?)
        """, (category, key, value))
        conn.commit()
        return f"Preference set: {category}.{key} = {value}"
    finally:
        conn.close()


def get_user_preferences() -> dict:
    """
    Retrieve all user preferences.
    
    Returns: Dictionary of preferences by category
    
    Example:
        >>> set_research_preference("search_depth", "depth", "balanced")
        'Preference set: search_depth.depth = balanced'
        >>> prefs = get_user_preferences()
        >>> prefs.get("search_depth", {}).get("depth")
        'balanced'
    """
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT category, key, value FROM user_preferences")
        
        prefs = {}
        for cat, k, v in cursor.fetchall():
            if cat not in prefs:
                prefs[cat] = {}
            prefs[cat][k] = v
        
        return prefs
    finally:
        conn.close()


def get_user_prompt_context() -> str:
    """
    Load user identity & preferences into system prompt format.
    
    Returns: Formatted string for system prompt injection
    
    Example:
        >>> set_user_identity(name="Charlie", role="Manager")
        'User profile set: Charlie (Manager)'
        >>> context = get_user_prompt_context()
        >>> "Charlie" in context
        True
    """
    identity = get_user_identity()
    prefs = get_user_preferences()
    
    context = "\n### User Context\n"
    
    if identity:
        name = identity.get('name', 'Unknown')
        role = identity.get('role', 'N/A')
        tz = identity.get('timezone', 'UTC')
        
        context += f"**Name**: {name}\n"
        context += f"**Role**: {role}\n"
        context += f"**Timezone**: {tz}\n"
        
        interests = identity.get('interests', [])
        if interests:
            context += f"**Interests**: {', '.join(interests)}\n"
    else:
        context += "[No user profile configured yet]\n"
    
    if prefs:
        context += "\n**Preferences**:\n"
        for cat, items in prefs.items():
            for k, v in items.items():
                context += f"- {cat}.{k}: {v}\n"
    
    return context


def get_user_research_style() -> dict:
    """
    Get research style preferences for decision-making.
    
    Returns: Dictionary with research configuration
    
    Example:
        >>> style = get_user_research_style()
        >>> isinstance(style, dict)
        True
        >>> "depth" in style
        True
    """
    prefs = get_user_preferences()
    search_prefs = prefs.get("search_depth", {})
    
    style = {
        "depth": search_prefs.get("depth", "balanced"),  # light, balanced, deep
        "academic_weight": float(search_prefs.get("academic_weight", 0.5)),  # 0.0-1.0
        "recency_weight": float(search_prefs.get("recency_weight", 0.5)),  # 0.0-1.0
        "max_results": int(search_prefs.get("max_results", 5))
    }
    
    return style


def clear_user_profile() -> str:
    """Clear all user profile data."""
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    try:
        conn.execute("DELETE FROM user_profile")
        conn.execute("DELETE FROM user_preferences")
        conn.commit()
        return "User profile cleared"
    finally:
        conn.close()


def export_user_profile() -> str:
    """
    Export user profile as JSON string.
    
    Returns: JSON string of full profile
    """
    identity = get_user_identity()
    prefs = get_user_preferences()
    
    profile = {
        "identity": identity,
        "preferences": prefs
    }
    
    return json.dumps(profile, indent=2)


def import_user_profile(profile_json: str) -> str:
    """
    Import user profile from JSON string.
    
    Args:
        profile_json: JSON string with identity and preferences
    
    Returns: Confirmation message
    """
    try:
        data = json.loads(profile_json)
        
        # Import identity
        if "identity" in data:
            identity = data["identity"]
            set_user_identity(
                name=identity.get("name", ""),
                role=identity.get("role", ""),
                timezone=identity.get("timezone", "UTC"),
                email=identity.get("email", ""),
                interests=identity.get("interests", [])
            )
        
        # Import preferences
        if "preferences" in data:
            prefs = data["preferences"]
            for category, items in prefs.items():
                for key, value in items.items():
                    set_research_preference(category, key, value)
        
        return "User profile imported successfully"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error importing profile: {e}"


# Initialize on import
try:
    init_user_profile_db()
except:
    pass
