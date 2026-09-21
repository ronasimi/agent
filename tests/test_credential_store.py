import os
import sqlite3

import pytest
from cryptography.fernet import Fernet

from tools.credential_store import CredentialStoreError, LocalCredentialStore


def test_credentials_are_encrypted_namespaced_and_private(tmp_path):
    root = tmp_path / "credentials"
    store = LocalCredentialStore(root)
    secret = "refresh-token-that-must-not-be-plaintext"
    store.put_secret(
        "example_service",
        "default",
        "oauth_token",
        {"refresh_token": secret, "access_token": "access-secret"},
        metadata={"account": "person@example.com", "connected_at": "now"},
    )

    assert store.get_secret("example_service", "default", "oauth_token")["refresh_token"] == secret
    assert secret.encode() not in store.database_path.read_bytes()
    records = store.list_records(provider="example_service", kind="oauth_token")
    assert records[0]["metadata"]["account"] == "person@example.com"
    assert "ciphertext" not in records[0]

    if os.name == "posix":
        assert root.stat().st_mode & 0o777 == 0o700
        assert store.key_path.stat().st_mode & 0o777 == 0o600
        assert store.database_path.stat().st_mode & 0o777 == 0o600


def test_one_time_secret_is_consumed_atomically(tmp_path):
    store = LocalCredentialStore(tmp_path / "credentials")
    store.put_secret("oauth", "state-id", "pending", {"verifier": "abc"})

    assert store.pop_secret("oauth", "state-id", "pending") == {"verifier": "abc"}
    assert store.pop_secret("oauth", "state-id", "pending") is None


def test_tampered_ciphertext_is_rejected(tmp_path):
    store = LocalCredentialStore(tmp_path / "credentials")
    store.put_secret("service", "default", "token", {"token": "secret"})
    with sqlite3.connect(store.database_path) as conn:
        conn.execute(
            "UPDATE credential_records SET ciphertext=? WHERE provider=? AND account_id=? AND kind=?",
            (b"not-a-valid-fernet-token", "service", "default", "token"),
        )

    with pytest.raises(CredentialStoreError, match="failed authentication"):
        store.get_secret("service", "default", "token")


def test_secret_like_metadata_and_symlink_roots_are_rejected(tmp_path):
    store = LocalCredentialStore(tmp_path / "credentials")
    with pytest.raises(CredentialStoreError, match="Secret-like"):
        store.put_secret("service", "default", "token", {"token": "inside"}, metadata={"access_token": "outside"})

    if hasattr(os, "symlink"):
        target = tmp_path / "actual"
        target.mkdir()
        link = tmp_path / "linked"
        link.symlink_to(target, target_is_directory=True)
        with pytest.raises(CredentialStoreError, match="symbolic link"):
            LocalCredentialStore(link)


def test_external_key_does_not_create_local_master_key(tmp_path):
    root = tmp_path / "credentials"
    store = LocalCredentialStore(root, key=Fernet.generate_key())
    store.put_secret("service", "default", "token", {"value": "secret"})

    assert not store.key_path.exists()
    assert store.get_secret("service", "default", "token") == {"value": "secret"}
