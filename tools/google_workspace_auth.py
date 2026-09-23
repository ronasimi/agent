"""Google Workspace OAuth 2.0 orchestration with encrypted local persistence."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode, urlparse

import requests

from .credential_store import CredentialStoreError, LocalCredentialStore

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
DRIVE_METADATA_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.metadata.readonly"
GOOGLE_WORKSPACE_SCOPES = (GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE, DRIVE_METADATA_READONLY_SCOPE)
GOOGLE_WORKSPACE_SCOPE_SET = frozenset(GOOGLE_WORKSPACE_SCOPES)

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_REVOCATION_ENDPOINT = "https://oauth2.googleapis.com/revoke"
GOOGLE_GMAIL_PROFILE_ENDPOINT = "https://gmail.googleapis.com/gmail/v1/users/me/profile"

_CLIENT_PROVIDER = "google_workspace"
_PENDING_PROVIDER = "google_oauth"
_CLIENT_KIND = "oauth_client"
_TOKEN_KIND = "oauth_token"
_PENDING_KIND = "pending"
_DEFAULT_ACCOUNT = "default"
_CLIENT_ID_SUFFIX = ".apps.googleusercontent.com"
_STATE_TTL_SECONDS = 10 * 60
_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9._@+:-]{1,160}$")


class GoogleWorkspaceAuthError(RuntimeError):
    """Safe integration error whose code can be returned without leaking secrets."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _utc_timestamp() -> int:
    return int(time.time())


def _utc_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(value, tz=UTC).isoformat(timespec="seconds")


def _state_key(state: str) -> str:
    return hashlib.sha256(state.encode("ascii", errors="strict")).hexdigest()


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _bounded_provider_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return f"HTTP {response.status_code}"
    if not isinstance(payload, dict):
        return f"HTTP {response.status_code}"
    error = payload.get("error")
    if isinstance(error, dict):
        value = error.get("message") or error.get("status")
    else:
        value = payload.get("error_description") or error
    cleaned = " ".join(str(value or "").split())[:240]
    return cleaned or f"HTTP {response.status_code}"


def _expires_in(payload: dict[str, Any], error_code: str) -> int:
    try:
        seconds = int(payload.get("expires_in") or 3600)
    except (TypeError, ValueError) as exc:
        raise GoogleWorkspaceAuthError(error_code, "Google returned an invalid token lifetime.") from exc
    if seconds <= 0:
        raise GoogleWorkspaceAuthError(error_code, "Google returned an invalid token lifetime.")
    return max(60, min(seconds, 86_400))


def google_oauth_redirect_uri() -> str:
    configured = str(os.environ.get("GOOGLE_OAUTH_REDIRECT_URI") or "").strip()
    if not configured:
        port = str(os.environ.get("WEBUI_PORT") or "8080").strip()
        if not port.isdigit() or not (1 <= int(port) <= 65535):
            port = "8080"
        configured = f"http://127.0.0.1:{port}/api/integrations/google/callback"
    parsed = urlparse(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise GoogleWorkspaceAuthError("invalid_redirect", "GOOGLE_OAUTH_REDIRECT_URI is invalid.")
    if parsed.scheme == "http" and (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise GoogleWorkspaceAuthError("insecure_redirect", "HTTP OAuth redirects must use a loopback host.")
    return configured


class GoogleWorkspaceOAuth:
    """Owns client configuration, one-time OAuth state, token refresh, and revocation."""

    def __init__(
        self,
        store: LocalCredentialStore | None = None,
        *,
        http: Any = requests,
    ):
        self.store = store or LocalCredentialStore()
        self.http = http
        self._refresh_lock = threading.RLock()

    @staticmethod
    def _account(account: str) -> str:
        value = str(account or _DEFAULT_ACCOUNT).strip()
        if not _ACCOUNT_RE.fullmatch(value):
            raise GoogleWorkspaceAuthError("invalid_account", "Invalid Google Workspace account selector.")
        return value

    @staticmethod
    def _parse_client_config(raw: bytes | str | dict[str, Any]) -> dict[str, str]:
        try:
            payload = json.loads(raw.decode("utf-8")) if isinstance(raw, bytes) else json.loads(raw) if isinstance(raw, str) else raw
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GoogleWorkspaceAuthError("invalid_client_config", "OAuth client JSON is invalid.") from exc
        if not isinstance(payload, dict):
            raise GoogleWorkspaceAuthError("invalid_client_config", "OAuth client JSON must contain an object.")
        client_type = "web" if isinstance(payload.get("web"), dict) else "installed" if isinstance(payload.get("installed"), dict) else ""
        if not client_type:
            raise GoogleWorkspaceAuthError("invalid_client_config", "Expected a Google OAuth web or desktop client JSON file.")
        data = payload[client_type]
        client_id = str(data.get("client_id") or "").strip()
        client_secret = str(data.get("client_secret") or "").strip()
        project_id = str(data.get("project_id") or "").strip()[:160]
        if not client_id.endswith(_CLIENT_ID_SUFFIX) or len(client_id) > 512 or not client_secret or len(client_secret) > 1024:
            raise GoogleWorkspaceAuthError("invalid_client_config", "OAuth client ID or secret is invalid.")
        auth_uri = str(data.get("auth_uri") or "https://accounts.google.com/o/oauth2/auth").strip()
        token_uri = str(data.get("token_uri") or GOOGLE_TOKEN_ENDPOINT).strip()
        if auth_uri not in {"https://accounts.google.com/o/oauth2/auth", GOOGLE_AUTHORIZATION_ENDPOINT} or token_uri != GOOGLE_TOKEN_ENDPOINT:
            raise GoogleWorkspaceAuthError("invalid_client_config", "OAuth endpoints in the client file are not trusted Google endpoints.")
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "project_id": project_id,
            "client_type": client_type,
        }

    def store_client_config(self, raw: bytes | str | dict[str, Any], *, account: str = _DEFAULT_ACCOUNT) -> dict[str, Any]:
        account = self._account(account)
        config = self._parse_client_config(raw)
        previous = self.store.get_secret(_CLIENT_PROVIDER, account, _CLIENT_KIND)
        if previous and previous != config:
            # Tokens are bound to the OAuth client that minted them. Revoke and
            # remove any old token before accepting a replacement client.
            self.disconnect(account=account)
        self.store.put_secret(
            _CLIENT_PROVIDER,
            account,
            _CLIENT_KIND,
            config,
            metadata={
                "client_type": config["client_type"],
                "project_id": config["project_id"],
                "configured_at": _utc_iso(),
            },
        )
        return self.status(account=account)

    def _client_config(self, account: str) -> dict[str, Any]:
        config = self.store.get_secret(_CLIENT_PROVIDER, account, _CLIENT_KIND)
        if not config:
            raise GoogleWorkspaceAuthError("client_not_configured", "Upload a Google OAuth client JSON file first.")
        return config

    def _token(self, account: str) -> dict[str, Any] | None:
        return self.store.get_secret(_CLIENT_PROVIDER, account, _TOKEN_KIND)

    def _cleanup_pending(self) -> None:
        now = _utc_timestamp()
        for record in self.store.list_records(provider=_PENDING_PROVIDER, kind=_PENDING_KIND):
            expires_at = int((record.get("metadata") or {}).get("expires_at") or 0)
            if expires_at <= now:
                self.store.delete_secret(_PENDING_PROVIDER, str(record["account_id"]), _PENDING_KIND)

    def authorization_url(self, *, account: str = _DEFAULT_ACCOUNT) -> str:
        account = self._account(account)
        config = self._client_config(account)
        redirect_uri = google_oauth_redirect_uri()
        self._cleanup_pending()
        state = secrets.token_urlsafe(32)
        verifier, challenge = _pkce_pair()
        state_id = _state_key(state)
        expires_at = _utc_timestamp() + _STATE_TTL_SECONDS
        self.store.put_secret(
            _PENDING_PROVIDER,
            state_id,
            _PENDING_KIND,
            {
                "account": account,
                "client_id": config["client_id"],
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
                "scopes": list(GOOGLE_WORKSPACE_SCOPES),
                "expires_at": expires_at,
            },
            metadata={"expires_at": expires_at, "created_at": _utc_iso()},
        )
        params = {
            "client_id": config["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_WORKSPACE_SCOPES),
            "access_type": "offline",
            # Preserve previously granted Gmail/Calendar permissions when an
            # existing connection is upgraded with a newly added read-only
            # capability such as Drive metadata.
            "include_granted_scopes": "true",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{urlencode(params)}"

    def cancel_authorization(self, state: str) -> None:
        try:
            self.store.delete_secret(_PENDING_PROVIDER, _state_key(str(state or "")), _PENDING_KIND)
        except (CredentialStoreError, UnicodeEncodeError):
            return

    def _request_token(self, data: dict[str, str], error_code: str) -> dict[str, Any]:
        try:
            response = self.http.post(GOOGLE_TOKEN_ENDPOINT, data=data, timeout=20)
        except requests.RequestException as exc:
            raise GoogleWorkspaceAuthError(error_code, "Google's token endpoint could not be reached.") from exc
        if response.status_code != 200:
            raise GoogleWorkspaceAuthError(error_code, f"Google rejected the token request: {_bounded_provider_error(response)}")
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise GoogleWorkspaceAuthError(error_code, "Google returned an invalid token response.") from exc
        access_token = str(payload.get("access_token") or "") if isinstance(payload, dict) else ""
        if not access_token or len(access_token) > 32_768:
            raise GoogleWorkspaceAuthError(error_code, "Google returned an incomplete token response.")
        return payload

    def _profile_email(self, access_token: str) -> str:
        try:
            response = self.http.get(
                GOOGLE_GMAIL_PROFILE_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15,
            )
            if response.status_code == 200:
                payload = response.json()
                return str(payload.get("emailAddress") or "").strip()[:320] if isinstance(payload, dict) else ""
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            pass
        return ""

    def complete_authorization(self, state: str, code: str) -> dict[str, Any]:
        state = str(state or "").strip()
        code = str(code or "").strip()
        if not 20 <= len(state) <= 512 or not 4 <= len(code) <= 8192:
            raise GoogleWorkspaceAuthError("invalid_callback", "OAuth callback parameters are invalid.")
        try:
            pending = self.store.pop_secret(_PENDING_PROVIDER, _state_key(state), _PENDING_KIND)
        except (CredentialStoreError, UnicodeEncodeError) as exc:
            raise GoogleWorkspaceAuthError("invalid_state", "OAuth state validation failed.") from exc
        if not pending or int(pending.get("expires_at") or 0) < _utc_timestamp():
            raise GoogleWorkspaceAuthError("expired_state", "OAuth state is missing or expired. Start the connection again.")
        account = self._account(str(pending.get("account") or ""))
        config = self._client_config(account)
        if config.get("client_id") != pending.get("client_id"):
            raise GoogleWorkspaceAuthError("client_changed", "OAuth client configuration changed during authorization.")
        token_payload = self._request_token(
            {
                "code": code,
                "client_id": str(config["client_id"]),
                "client_secret": str(config["client_secret"]),
                "redirect_uri": str(pending["redirect_uri"]),
                "grant_type": "authorization_code",
                "code_verifier": str(pending["code_verifier"]),
            },
            "token_exchange_failed",
        )
        previous = self._token(account) or {}
        refresh_token = str(token_payload.get("refresh_token") or previous.get("refresh_token") or "")
        if not refresh_token or len(refresh_token) > 32_768:
            raise GoogleWorkspaceAuthError("missing_refresh_token", "Google did not return offline access. Revoke access and connect again.")
        returned_scope = str(token_payload.get("scope") or "").split()
        scopes = set(returned_scope or pending.get("scopes") or [])
        requested_scopes = set(pending.get("scopes") or GOOGLE_WORKSPACE_SCOPES)
        if scopes - GOOGLE_WORKSPACE_SCOPE_SET or not requested_scopes.issubset(scopes):
            raise GoogleWorkspaceAuthError(
                "scope_mismatch",
                "Google did not return the requested read-only Workspace scopes.",
            )
        expires_in = _expires_in(token_payload, "token_exchange_failed")
        access_token = str(token_payload["access_token"])
        expires_at = _utc_timestamp() + expires_in
        email = self._profile_email(access_token)
        self.store.put_secret(
            _CLIENT_PROVIDER,
            account,
            _TOKEN_KIND,
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "scopes": sorted(scopes),
                "token_type": "Bearer",
            },
            metadata={
                "email": email,
                "scopes": sorted(scopes),
                "connected_at": _utc_iso(),
                "expires_at": expires_at,
            },
        )
        return self.status(account=account)

    def get_access_token(
        self,
        *,
        account: str = _DEFAULT_ACCOUNT,
        force_refresh: bool = False,
        required_scopes: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
    ) -> str:
        account = self._account(account)
        required = set(required_scopes or ())
        if required - GOOGLE_WORKSPACE_SCOPE_SET:
            raise GoogleWorkspaceAuthError("invalid_scope", "Unsupported Google Workspace scope requested by the harness.")
        with self._refresh_lock:
            token = self._token(account)
            if not token:
                raise GoogleWorkspaceAuthError("not_connected", "Google Workspace is not connected. Open Connections in the Web UI.")
            scopes = set(token.get("scopes") or [])
            if not scopes or scopes - GOOGLE_WORKSPACE_SCOPE_SET:
                raise GoogleWorkspaceAuthError(
                    "scope_mismatch",
                    "Stored Google credentials contain an invalid or unsupported scope set.",
                )
            missing = required - scopes
            if missing:
                raise GoogleWorkspaceAuthError(
                    "scope_upgrade_required",
                    "Google Workspace is connected, but this capability needs an additional read-only permission. Reconnect Google from Connections to grant it.",
                )
            expires_at = int(token.get("expires_at") or 0)
            access_token = str(token.get("access_token") or "")
            if not force_refresh and expires_at > _utc_timestamp() + 60 and 0 < len(access_token) <= 32_768:
                return access_token
            config = self._client_config(account)
            refresh_token = str(token.get("refresh_token") or "")
            if not refresh_token or len(refresh_token) > 32_768:
                raise GoogleWorkspaceAuthError("reauthorization_required", "Google Workspace must be reconnected.")
            refreshed = self._request_token(
                {
                    "client_id": str(config["client_id"]),
                    "client_secret": str(config["client_secret"]),
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                "refresh_failed",
            )
            returned_scope = str(refreshed.get("scope") or "").split()
            refreshed_scopes = set(returned_scope or scopes)
            if not refreshed_scopes or refreshed_scopes - GOOGLE_WORKSPACE_SCOPE_SET:
                raise GoogleWorkspaceAuthError(
                    "scope_mismatch",
                    "Refreshed Google credentials contain an invalid or unsupported scope set.",
                )
            missing = required - refreshed_scopes
            if missing:
                raise GoogleWorkspaceAuthError(
                    "scope_upgrade_required",
                    "Google Workspace is connected, but this capability needs an additional read-only permission. Reconnect Google from Connections to grant it.",
                )
            token.update({
                "access_token": str(refreshed["access_token"]),
                "expires_at": _utc_timestamp() + _expires_in(refreshed, "refresh_failed"),
                "scopes": sorted(refreshed_scopes),
                "token_type": "Bearer",
            })
            # Preserve stable public metadata without calling status(), which can
            # intentionally report a scope-upgrade state for an older token.
            previous_record = next((
                item for item in self.store.list_records(provider=_CLIENT_PROVIDER, kind=_TOKEN_KIND)
                if item.get("account_id") == account
            ), None)
            previous_metadata = (previous_record or {}).get("metadata") or {}
            self.store.put_secret(
                _CLIENT_PROVIDER,
                account,
                _TOKEN_KIND,
                token,
                metadata={
                    "email": str(previous_metadata.get("email") or ""),
                    "scopes": sorted(refreshed_scopes),
                    "connected_at": str(previous_metadata.get("connected_at") or _utc_iso()),
                    "expires_at": int(token["expires_at"]),
                },
            )
            return str(token["access_token"])

    def status(self, *, account: str = _DEFAULT_ACCOUNT) -> dict[str, Any]:
        account = self._account(account)
        redirect_error: dict[str, str] | None = None
        connection_error: dict[str, str] | None = None
        try:
            redirect_uri = google_oauth_redirect_uri()
        except GoogleWorkspaceAuthError as exc:
            redirect_uri = ""
            redirect_error = {"code": exc.code, "message": str(exc)}
        client_record = next((
            item for item in self.store.list_records(provider=_CLIENT_PROVIDER, kind=_CLIENT_KIND)
            if item.get("account_id") == account
        ), None)
        token_record = next((
            item for item in self.store.list_records(provider=_CLIENT_PROVIDER, kind=_TOKEN_KIND)
            if item.get("account_id") == account
        ), None)
        client_metadata = (client_record or {}).get("metadata") or {}
        token_metadata = (token_record or {}).get("metadata") or {}
        token: dict[str, Any] | None = None
        if token_record:
            try:
                token = self._token(account)
            except CredentialStoreError:
                connection_error = {
                    "code": "credential_unreadable",
                    "message": "The saved Google credential vault cannot be decrypted with the current key.",
                }
        scopes = set((token or {}).get("scopes") or token_metadata.get("scopes") or [])
        invalid_scopes = sorted(scopes - GOOGLE_WORKSPACE_SCOPE_SET)
        missing_scopes = sorted(GOOGLE_WORKSPACE_SCOPE_SET - scopes)
        refresh_token = str((token or {}).get("refresh_token") or "")
        usable = bool(token and refresh_token and not invalid_scopes and scopes)
        if token and invalid_scopes and connection_error is None:
            connection_error = {
                "code": "scope_mismatch",
                "message": "The saved Google credential contains scopes outside the harness read-only allowlist.",
            }
            usable = False
        return {
            "provider": "google_workspace",
            "account": account,
            "configured": bool(client_record),
            "connected": usable,
            "token_present": bool(token_record),
            "refreshable": bool(usable and refresh_token),
            "email": str(token_metadata.get("email") or ""),
            "scopes": sorted(scopes),
            "required_scopes": list(GOOGLE_WORKSPACE_SCOPES),
            "missing_scopes": missing_scopes,
            "scope_upgrade_required": bool(usable and missing_scopes),
            "client_type": str(client_metadata.get("client_type") or ""),
            "project_id": str(client_metadata.get("project_id") or ""),
            "connected_at": str(token_metadata.get("connected_at") or ""),
            "expires_at": int((token or {}).get("expires_at") or token_metadata.get("expires_at") or 0),
            "redirect_uri": redirect_uri,
            "configuration_error": redirect_error,
            "connection_error": connection_error,
            "capabilities": {
                "gmail": {"available": GMAIL_READONLY_SCOPE in scopes, "scope": GMAIL_READONLY_SCOPE},
                "calendar": {"available": CALENDAR_READONLY_SCOPE in scopes, "scope": CALENDAR_READONLY_SCOPE},
                "drive": {"available": DRIVE_METADATA_READONLY_SCOPE in scopes, "scope": DRIVE_METADATA_READONLY_SCOPE},
            },
        }

    def disconnect(self, *, account: str = _DEFAULT_ACCOUNT) -> dict[str, Any]:
        account = self._account(account)
        token = self._token(account)
        revoked = False
        if token:
            value = str(token.get("refresh_token") or token.get("access_token") or "")
            if value:
                try:
                    response = self.http.post(GOOGLE_REVOCATION_ENDPOINT, data={"token": value}, timeout=15)
                    revoked = response.status_code in {200, 400}
                except requests.RequestException:
                    revoked = False
            self.store.delete_secret(_CLIENT_PROVIDER, account, _TOKEN_KIND)
        result = self.status(account=account)
        result["revoked"] = revoked
        return result

    def remove_client_config(self, *, account: str = _DEFAULT_ACCOUNT) -> dict[str, Any]:
        account = self._account(account)
        self.disconnect(account=account)
        self.store.delete_secret(_CLIENT_PROVIDER, account, _CLIENT_KIND)
        return self.status(account=account)


_SERVICE: GoogleWorkspaceOAuth | None = None
_SERVICE_LOCK = threading.Lock()


def get_google_workspace_oauth() -> GoogleWorkspaceOAuth:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = GoogleWorkspaceOAuth()
    return _SERVICE
