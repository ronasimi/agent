"""User identity, preferences, and Web UI profile-image persistence."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .runtime import DB_PATH, DB_TIMEOUT

PROFILE_DIR = Path(os.environ.get("AGENT_PROFILE_DIR", "/app/memory/profile")).resolve()
PROFILE_IMAGE_PATH = PROFILE_DIR / "user_picture.png"
WORKSPACE_ROOT = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace")).resolve()
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_IMAGE_PATH_RE = re.compile(r"(?:Attached file:\s*)?(/[^\s\"'<>]+\.(?:png|jpe?g|webp))", re.I)
_SELF_PHOTO_RE = re.compile(r"\b(?:photo|picture|image)\s+of\s+me\b|\bmy\s+(?:photo|picture|image)\b|\bthis\s+is\s+(?:a\s+photo\s+of\s+)?me\b", re.I)


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_user_profile_db() -> None:
    """Initialize user-profile tables."""
    with _connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS user_profile (key TEXT PRIMARY KEY, value TEXT)")
        # The OOBE owns a small set of durable memory topics (for example the
        # user's declared location), so profile initialization must also work on
        # a completely fresh database before tools.memory has been imported.
        conn.execute("CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS user_preferences (
                   category TEXT, key TEXT, value TEXT,
                   PRIMARY KEY (category, key)
               )"""
        )


def _safe_workspace_image(path: str) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Missing required image path.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("Profile images must come from the agent workspace.") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Image not found: {candidate}")
    if candidate.suffix.lower() not in _ALLOWED_IMAGE_EXT:
        raise ValueError("Profile image must be PNG, JPEG, or WebP.")
    return candidate


def _install_profile_image(source: Path, *, update_memory: bool = True) -> Path:
    """Normalize a workspace image into the durable shared profile location."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_DIR / ".user_picture.tmp.png"
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            # Keep enough detail for future UI sizes without storing giant camera originals.
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(tmp, format="PNG", optimize=True)
        os.replace(tmp, PROFILE_IMAGE_PATH)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    if update_memory:
        try:
            with _connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT INTO memory(topic, fact, updated_at) VALUES ('user_picture', ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(topic) DO UPDATE SET fact=excluded.fact, updated_at=CURRENT_TIMESTAMP",
                    (f"Current Web UI profile image: {PROFILE_IMAGE_PATH}",),
                )
        except sqlite3.Error:
            pass
    return PROFILE_IMAGE_PATH


def set_profile_image(path: str) -> str:
    """Set the user's Web UI profile image from an existing workspace image after explicit user approval."""
    source = _safe_workspace_image(path)
    destination = _install_profile_image(source)
    return json.dumps(
        {"ok": True, "source": str(source), "profile_image": str(destination)},
        ensure_ascii=False,
    )


def _legacy_profile_candidate() -> Path | None:
    """Recover a previously identified user photo from memory/chat history when possible."""
    try:
        with _connect() as conn:
            # First prefer an explicit durable path saved under the newer or legacy topic.
            try:
                rows = conn.execute(
                    "SELECT topic, fact FROM memory WHERE topic IN ('user_picture','user_photo') "
                    "ORDER BY CASE topic WHEN 'user_picture' THEN 0 ELSE 1 END"
                ).fetchall()
            except sqlite3.Error:
                rows = []
            for _, fact in rows:
                match = _IMAGE_PATH_RE.search(str(fact or ""))
                if match:
                    try:
                        return _safe_workspace_image(match.group(1))
                    except (OSError, ValueError, FileNotFoundError):
                        pass

            # Older chats often stored only "this is a photo of me" in memory.
            # Recover the nearest prior attached image path without doing face recognition.
            try:
                history = conn.execute(
                    "SELECT role, content FROM ("
                    "SELECT id, role, content FROM chat_history ORDER BY id DESC LIMIT 1000"
                    ") ORDER BY id ASC"
                ).fetchall()
            except sqlite3.Error:
                history = []
            last_image: Path | None = None
            for role, content in history:
                if str(role) != "user":
                    continue
                text = str(content or "")
                matches = list(_IMAGE_PATH_RE.finditer(text))
                if matches:
                    try:
                        last_image = _safe_workspace_image(matches[-1].group(1))
                    except (OSError, ValueError, FileNotFoundError):
                        last_image = None
                if last_image is not None and _SELF_PHOTO_RE.search(text):
                    return last_image
    except sqlite3.Error:
        return None
    return None


def get_profile_image_path(*, migrate_legacy: bool = True) -> Path | None:
    """Return the durable profile image, optionally migrating a legacy remembered photo."""
    if PROFILE_IMAGE_PATH.is_file():
        return PROFILE_IMAGE_PATH
    if not migrate_legacy:
        return None
    candidate = _legacy_profile_candidate()
    if candidate is None:
        return None
    try:
        return _install_profile_image(candidate)
    except (OSError, ValueError):
        return None


def profile_image_info() -> str:
    """Report whether a durable Web UI profile image is currently available."""
    path = get_profile_image_path(migrate_legacy=True)
    return json.dumps({"present": bool(path), "profile_image": str(path) if path else ""}, ensure_ascii=False)


def set_user_identity(
    name: str = "",
    role: str = "",
    timezone: str = "UTC",
    email: str = "",
    interests: list | None = None,
) -> str:
    """Configure stable user identity fields for the local agent."""
    profile = {"name": name, "role": role, "timezone": timezone, "email": email, "interests": interests or []}
    with _connect() as conn:
        for key, value in profile.items():
            encoded = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            conn.execute("INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)", (f"identity.{key}", encoded))
    return f"User profile set: {name} ({role})"


def get_user_identity() -> dict[str, Any]:
    """Retrieve configured user identity information."""
    init_user_profile_db()
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM user_profile WHERE key LIKE 'identity.%'").fetchall()
    identity: dict[str, Any] = {}
    for key, value in rows:
        short = str(key).replace("identity.", "", 1)
        try:
            identity[short] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            identity[short] = value
    return identity




def get_user_location() -> str:
    """Return the explicitly configured durable user location, if any."""
    init_user_profile_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT fact FROM memory WHERE topic IN ('user_location','location','city') "
            "ORDER BY CASE topic WHEN 'user_location' THEN 0 WHEN 'location' THEN 1 ELSE 2 END LIMIT 1"
        ).fetchone()
    return str(row[0]).strip() if row and str(row[0] or '').strip() else ""

def set_research_preference(category: str, key: str, value: str) -> str:
    """Set a durable research preference."""
    init_user_profile_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (category, key, value) VALUES (?, ?, ?)",
            (category, key, value),
        )
    return f"Preference set: {category}.{key} = {value}"


def get_user_preferences() -> dict[str, dict[str, str]]:
    """Retrieve all configured user preferences."""
    init_user_profile_db()
    with _connect() as conn:
        rows = conn.execute("SELECT category, key, value FROM user_preferences").fetchall()
    prefs: dict[str, dict[str, str]] = {}
    for category, key, value in rows:
        prefs.setdefault(str(category), {})[str(key)] = str(value)
    return prefs


def get_user_prompt_context() -> str:
    """Render user identity and preferences as prompt context."""
    identity = get_user_identity()
    prefs = get_user_preferences()
    lines = ["\n### User Context"]
    if identity:
        lines += [
            f"**Name**: {identity.get('name', 'Unknown')}",
            f"**Role**: {identity.get('role', 'N/A')}",
            f"**Timezone**: {identity.get('timezone', 'UTC')}",
        ]
        location = get_user_location()
        if location:
            lines.append(f"**Location**: {location}")
        interests = identity.get("interests", [])
        if interests:
            lines.append(f"**Interests**: {', '.join(map(str, interests))}")
    else:
        lines.append("[No user profile configured yet]")
    if prefs:
        lines.append("\n**Preferences**:")
        for category, items in prefs.items():
            for key, value in items.items():
                lines.append(f"- {category}.{key} = {value}")
    return "\n".join(lines) + "\n"


def get_relevant_user_prompt_context(user_text: str) -> str:
    """Return only profile fields materially relevant to the current request.

    The full profile is intentionally *not* injected into every turn.  Besides
    saving prompt tokens, this prevents ordinary greetings or unrelated questions
    from causing the model to volunteer the user's name, location, interests, or
    other stored profile details without a reason.
    """
    text = " ".join(str(user_text or "").lower().split())
    identity = get_user_identity()
    prefs = get_user_preferences()
    lines = ["\n### Relevant User Context"]

    explicit_profile = bool(re.search(
        r"\b(?:my profile|about me|what do you know about me|who am i|my identity|my preferences?)\b",
        text,
    ))
    needs_location = bool(re.search(
        r"\b(?:weather|forecast|near me|nearby|local(?: news| weather| forecast)?|around me)\b",
        text,
    ))
    needs_timezone = bool(re.search(
        r"\b(?:time|date|timezone|remind|reminder|schedule|calendar|today|tomorrow|tonight)\b",
        text,
    ))
    research_request = bool(re.search(r"\b(?:research|study|investigate|deep dive)\b", text))

    if explicit_profile:
        if identity:
            for key in ("name", "role", "timezone", "email"):
                value = identity.get(key)
                if value:
                    lines.append(f"**{key.title()}**: {value}")
            interests = identity.get("interests") or []
            if interests:
                lines.append(f"**Interests**: {', '.join(map(str, interests))}")
        location = get_user_location()
        if location:
            lines.append(f"**Location**: {location}")
    else:
        if needs_location:
            location = get_user_location()
            if location:
                lines.append(f"**Location**: {location}")
        if needs_timezone and identity.get("timezone"):
            lines.append(f"**Timezone**: {identity['timezone']}")
        if research_request:
            interests = identity.get("interests") or []
            if interests:
                lines.append(f"**Interests**: {', '.join(map(str, interests))}")

    if prefs:
        for category, items in prefs.items():
            if not explicit_profile and category not in {"response", "research"}:
                continue
            if not explicit_profile and category == "research" and not research_request:
                continue
            for key, value in items.items():
                lines.append(f"- {category}.{key} = {value}")

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""



def get_onboarding_state() -> dict[str, Any]:
    """Return whether the first-run user-profile questionnaire has completed."""
    init_user_profile_db()
    with _connect() as conn:
        row = conn.execute("SELECT value FROM user_profile WHERE key='onboarding.completed'").fetchone()
    return {
        "completed": bool(row and str(row[0]).lower() in {"1", "true", "yes"}),
        "identity": get_user_identity(),
        "preferences": get_user_preferences(),
        "profile_image": str(get_profile_image_path(migrate_legacy=True) or ""),
    }


def complete_onboarding_profile(
    *,
    name: str = "",
    role: str = "",
    timezone: str = "UTC",
    location: str = "",
    email: str = "",
    interests: list[str] | None = None,
    response_style: str = "concise",
    research_depth: str = "balanced",
    profile_image_path: str = "",
    reset: bool = False,
) -> dict[str, Any]:
    """Populate/replace durable user context from the first-run questionnaire."""
    init_user_profile_db()
    if reset:
        with _connect() as conn:
            conn.execute("DELETE FROM user_profile")
            conn.execute("DELETE FROM user_preferences")
            conn.execute("DELETE FROM memory WHERE topic IN ('user_location','location','city','user_picture','user_photo')")
        try:
            PROFILE_IMAGE_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    set_user_identity(name=name, role=role, timezone=timezone, email=email, interests=interests or [])
    if response_style:
        set_research_preference("response", "style", response_style)
    if research_depth:
        set_research_preference("search_depth", "depth", research_depth)
    if location:
        with _connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO memory(topic, fact, updated_at) VALUES ('user_location', ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(topic) DO UPDATE SET fact=excluded.fact, updated_at=CURRENT_TIMESTAMP",
                (str(location).strip(),),
            )
    if profile_image_path:
        set_profile_image(profile_image_path)
    with _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO user_profile(key,value) VALUES('onboarding.completed','true')")
    return get_onboarding_state()


def reset_onboarding_profile() -> dict[str, Any]:
    """Clear questionnaire-owned profile/preferences so the OOBE can run again."""
    init_user_profile_db()
    with _connect() as conn:
        conn.execute("DELETE FROM user_profile")
        conn.execute("DELETE FROM user_preferences")
        conn.execute("DELETE FROM memory WHERE topic IN ('user_location','location','city','user_picture','user_photo')")
    try:
        PROFILE_IMAGE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    return get_onboarding_state()


def run_terminal_onboarding(*, reset: bool = False, input_fn=input, output_fn=print) -> dict[str, Any]:
    """Run the local first-run questionnaire in a terminal; safe to invoke again with reset=True."""
    state = get_onboarding_state()
    if state.get("completed") and not reset:
        return state
    current = state.get("identity") or {}
    output_fn("\n[Profile setup] Configure stable context for this local agent. Press Enter to keep a field blank.")
    name = input_fn(f"Name [{current.get('name','')}]: ").strip() or str(current.get("name") or "")
    role = input_fn(f"Role / occupation [{current.get('role','')}]: ").strip() or str(current.get("role") or "")
    timezone = input_fn(f"Timezone [{current.get('timezone','UTC')}]: ").strip() or str(current.get("timezone") or "UTC")
    location = input_fn("Location (city/region/country): ").strip()
    email = input_fn(f"Email (optional) [{current.get('email','')}]: ").strip() or str(current.get("email") or "")
    interests_raw = input_fn("Interests (comma-separated): ").strip()
    interests = [item.strip() for item in interests_raw.split(",") if item.strip()] or list(current.get("interests") or [])
    style = input_fn("Response style [concise/balanced/detailed] (concise): ").strip().lower() or "concise"
    depth = input_fn("Research depth [quick/balanced/deep] (balanced): ").strip().lower() or "balanced"
    photo = input_fn("Profile picture workspace path (optional): ").strip()
    result = complete_onboarding_profile(
        name=name, role=role, timezone=timezone, location=location, email=email, interests=interests,
        response_style=style, research_depth=depth, profile_image_path=photo, reset=reset,
    )
    output_fn("[Profile setup] Saved.")
    return result

def get_user_research_style() -> dict[str, Any]:
    """Get research style preferences with defaults."""
    search_prefs = get_user_preferences().get("search_depth", {})
    return {
        "depth": search_prefs.get("depth", "balanced"),
        "academic_weight": float(search_prefs.get("academic_weight", 0.5)),
        "recency_weight": float(search_prefs.get("recency_weight", 0.5)),
        "max_results": int(search_prefs.get("max_results", 5)),
    }


def clear_user_profile() -> str:
    """Clear configured profile fields and preferences (not the profile image)."""
    init_user_profile_db()
    with _connect() as conn:
        conn.execute("DELETE FROM user_profile")
        conn.execute("DELETE FROM user_preferences")
    return "User profile cleared"


def export_user_profile() -> str:
    """Export identity and preferences as JSON."""
    return json.dumps({"identity": get_user_identity(), "preferences": get_user_preferences()}, indent=2)


def import_user_profile(profile_json: str) -> str:
    """Import identity and preferences from JSON."""
    try:
        data = json.loads(profile_json)
        if "identity" in data:
            identity = data["identity"]
            set_user_identity(
                name=identity.get("name", ""), role=identity.get("role", ""),
                timezone=identity.get("timezone", "UTC"), email=identity.get("email", ""),
                interests=identity.get("interests", []),
            )
        for category, items in data.get("preferences", {}).items():
            for key, value in items.items():
                set_research_preference(category, key, value)
        return "User profile imported successfully"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as exc:
        return f"Error importing profile: {exc}"


try:
    init_user_profile_db()
except (OSError, sqlite3.Error):
    pass
