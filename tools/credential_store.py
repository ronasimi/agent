"""Provider-neutral encrypted storage for local integration credentials.

Secret payloads are authenticated-encrypted before SQLite sees them.  Record
metadata is deliberately non-secret so connection status can be rendered
without decrypting or exposing tokens.  For stronger separation, operators can
inject ``AGENT_CREDENTIAL_KEY`` from a password manager or container secret;
otherwise a mode-0600 local key is created beside the vault.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


class CredentialStoreError(RuntimeError):
    """Raised when the credential vault cannot safely complete an operation."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._@+:-]{1,160}$")
_SENSITIVE_METADATA_RE = re.compile(
    r"(?:token|secret|password|credential|authorization|code[_-]?verifier|private[_-]?key)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _identifier(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise CredentialStoreError(f"Invalid credential {label}.")
    return normalized


def _validate_metadata(value: Any, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if _SENSITIVE_METADATA_RE.search(key_text):
                raise CredentialStoreError(f"Secret-like field is not allowed in {path}.")
            _validate_metadata(child, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_metadata(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise CredentialStoreError(f"Unsupported value in {path}.")


class LocalCredentialStore:
    """Small encrypted credential vault with provider/account/kind namespacing."""

    def __init__(self, root: str | Path | None = None, *, key: str | bytes | None = None):
        configured_root = root or os.environ.get("AGENT_CREDENTIAL_DIR", "/app/memory/credentials")
        unresolved_root = Path(configured_root).expanduser()
        if unresolved_root.is_symlink():
            raise CredentialStoreError("Credential directory must not be a symbolic link.")
        self.root = unresolved_root.resolve()
        self.database_path = self.root / "vault.db"
        self.key_path = self.root / "master.key"
        self._ensure_directory()
        self._fernet = Fernet(self._load_key(key))
        self._init_database()

    @contextmanager
    def _private_umask(self) -> Iterator[None]:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)

    def _ensure_directory(self) -> None:
        with self._private_umask():
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise CredentialStoreError("Credential directory must be a real directory.")
        try:
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise CredentialStoreError("Could not secure the credential directory.") from exc

    def _load_key(self, explicit: str | bytes | None) -> bytes:
        configured = explicit or os.environ.get("AGENT_CREDENTIAL_KEY")
        key_file = str(os.environ.get("AGENT_CREDENTIAL_KEY_FILE") or "").strip()
        if not configured and key_file:
            source = Path(key_file).expanduser()
            if source.is_symlink() or not source.is_file():
                raise CredentialStoreError("AGENT_CREDENTIAL_KEY_FILE must be a regular, non-symlink file.")
            try:
                if source.stat().st_mode & 0o077:
                    raise CredentialStoreError("AGENT_CREDENTIAL_KEY_FILE permissions are too broad.")
                configured = source.read_bytes().strip()
            except OSError as exc:
                raise CredentialStoreError("Could not read AGENT_CREDENTIAL_KEY_FILE.") from exc
        if configured:
            key = configured.encode("ascii") if isinstance(configured, str) else bytes(configured)
            try:
                Fernet(key)
            except (ValueError, TypeError) as exc:
                raise CredentialStoreError("AGENT_CREDENTIAL_KEY is not a valid Fernet key.") from exc
            return key

        if self.key_path.is_symlink():
            raise CredentialStoreError("Credential key file must not be a symbolic link.")
        if self.key_path.exists():
            try:
                key = self.key_path.read_bytes().strip()
                mode = self.key_path.stat().st_mode & 0o777
            except OSError as exc:
                raise CredentialStoreError("Could not read the credential key.") from exc
            if mode & 0o077:
                raise CredentialStoreError("Credential key permissions are too broad; expected mode 0600.")
            try:
                Fernet(key)
            except (ValueError, TypeError) as exc:
                raise CredentialStoreError("Credential key file is invalid.") from exc
            return key

        key = Fernet.generate_key()
        try:
            with self._private_umask():
                descriptor = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(key + b"\n")
            os.chmod(self.key_path, 0o600)
        except FileExistsError:
            return self._load_key(None)
        except OSError as exc:
            raise CredentialStoreError("Could not create the credential key.") from exc
        return key

    def _connect(self) -> sqlite3.Connection:
        if self.database_path.is_symlink():
            raise CredentialStoreError("Credential database must not be a symbolic link.")
        if self.database_path.exists() and not self.database_path.is_file():
            raise CredentialStoreError("Credential database path must be a regular file.")
        try:
            with self._private_umask():
                conn = sqlite3.connect(self.database_path, timeout=15)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=15000")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
            os.chmod(self.database_path, 0o600)
            return conn
        except (OSError, sqlite3.Error) as exc:
            raise CredentialStoreError("Could not open the credential database safely.") from exc

    def _init_database(self) -> None:
        with self._private_umask(), self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS credential_records (
                    provider TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ciphertext BLOB NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(provider, account_id, kind)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_credential_records_provider_kind "
                "ON credential_records(provider, kind, updated_at)"
            )
        try:
            os.chmod(self.database_path, 0o600)
        except OSError as exc:
            raise CredentialStoreError("Could not secure the credential database.") from exc

    def _identity(self, provider: str, account_id: str, kind: str) -> tuple[str, str, str]:
        return (
            _identifier(provider, "provider"),
            _identifier(account_id, "account"),
            _identifier(kind, "kind"),
        )

    def _encrypt(self, provider: str, account_id: str, kind: str, payload: dict[str, Any]) -> bytes:
        if not isinstance(payload, dict):
            raise CredentialStoreError("Credential payload must be an object.")
        envelope = {
            "version": 1,
            "provider": provider,
            "account_id": account_id,
            "kind": kind,
            "payload": payload,
        }
        try:
            raw = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CredentialStoreError("Credential payload is not JSON serializable.") from exc
        return self._fernet.encrypt(raw)

    def _decrypt(self, provider: str, account_id: str, kind: str, ciphertext: bytes) -> dict[str, Any]:
        try:
            envelope = json.loads(self._fernet.decrypt(bytes(ciphertext)).decode("utf-8"))
        except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CredentialStoreError("Credential record failed authentication.") from exc
        if not isinstance(envelope, dict) or any(
            envelope.get(field) != expected
            for field, expected in (("provider", provider), ("account_id", account_id), ("kind", kind))
        ):
            raise CredentialStoreError("Credential record identity mismatch.")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise CredentialStoreError("Credential record payload is invalid.")
        return payload

    def put_secret(
        self,
        provider: str,
        account_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        provider, account_id, kind = self._identity(provider, account_id, kind)
        public_metadata = dict(metadata or {})
        _validate_metadata(public_metadata)
        metadata_json = json.dumps(public_metadata, ensure_ascii=False, separators=(",", ":"))
        if len(metadata_json.encode("utf-8")) > 16_384:
            raise CredentialStoreError("Credential metadata is too large.")
        ciphertext = self._encrypt(provider, account_id, kind, payload)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO credential_records(
                    provider, account_id, kind, ciphertext, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, account_id, kind) DO UPDATE SET
                    ciphertext=excluded.ciphertext,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (provider, account_id, kind, ciphertext, metadata_json, now, now),
            )

    def get_secret(self, provider: str, account_id: str, kind: str) -> dict[str, Any] | None:
        provider, account_id, kind = self._identity(provider, account_id, kind)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ciphertext FROM credential_records WHERE provider=? AND account_id=? AND kind=?",
                (provider, account_id, kind),
            ).fetchone()
        return self._decrypt(provider, account_id, kind, row["ciphertext"]) if row else None

    def pop_secret(self, provider: str, account_id: str, kind: str) -> dict[str, Any] | None:
        """Atomically consume a one-time secret such as an OAuth transaction."""
        provider, account_id, kind = self._identity(provider, account_id, kind)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT ciphertext FROM credential_records WHERE provider=? AND account_id=? AND kind=?",
                (provider, account_id, kind),
            ).fetchone()
            if row:
                conn.execute(
                    "DELETE FROM credential_records WHERE provider=? AND account_id=? AND kind=?",
                    (provider, account_id, kind),
                )
        return self._decrypt(provider, account_id, kind, row["ciphertext"]) if row else None

    def delete_secret(self, provider: str, account_id: str, kind: str) -> bool:
        provider, account_id, kind = self._identity(provider, account_id, kind)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM credential_records WHERE provider=? AND account_id=? AND kind=?",
                (provider, account_id, kind),
            )
        return bool(cursor.rowcount)

    def list_records(self, *, provider: str = "", kind: str = "") -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[str] = []
        if provider:
            clauses.append("provider=?")
            values.append(_identifier(provider, "provider"))
        if kind:
            clauses.append("kind=?")
            values.append(_identifier(kind, "kind"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT provider, account_id, kind, metadata_json, created_at, updated_at "
                f"FROM credential_records{where} ORDER BY updated_at DESC, provider, account_id, kind",
                values,
            ).fetchall()
        result = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            result.append({
                "provider": row["provider"],
                "account_id": row["account_id"],
                "kind": row["kind"],
                "metadata": metadata if isinstance(metadata, dict) else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return result
