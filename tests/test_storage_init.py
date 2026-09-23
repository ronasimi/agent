import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "init_storage.sh"


def _run_initializer(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "STORAGE_ROOT": str(root),
            "AGENT_UID": str(os.geteuid()),
            "AGENT_GID": str(os.getegid()),
        },
    )


def test_storage_initializer_creates_tree_and_is_idempotent(tmp_path: Path):
    state = tmp_path / "state"
    first = _run_initializer(state)
    assert first.returncode == 0, first.stderr

    expected = (
        state / "memory",
        state / "workspace" / "custom_tools",
        state / "workspace" / "skills",
        state / "workspace" / "research",
        state / "workspace" / "self_optimization" / "candidates",
        state / "workspace" / "self_optimization" / "approved",
        state / "workspace" / "self_optimization" / "validation" / "inbox",
        state / "workspace" / "self_optimization" / "validation" / "outbox",
    )
    assert all(path.is_dir() for path in expected)

    marker = state / "memory" / "knowledge.db"
    marker.write_text("preserve-existing-state", encoding="utf-8")
    second = _run_initializer(state)
    assert second.returncode == 0, second.stderr
    assert marker.read_text(encoding="utf-8") == "preserve-existing-state"



def test_storage_initializer_migrates_legacy_credentials_to_persistent_root(tmp_path: Path):
    state = tmp_path / "state"
    legacy = state / "memory" / "credentials"
    legacy.mkdir(parents=True)
    (legacy / "master.key").write_text("legacy-key", encoding="utf-8")
    (legacy / "vault.db").write_text("legacy-vault", encoding="utf-8")
    credentials = tmp_path / "persistent-credentials"
    result = subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "STORAGE_ROOT": str(state),
            "CREDENTIAL_ROOT": str(credentials),
            "AGENT_UID": str(os.geteuid()),
            "AGENT_GID": str(os.getegid()),
        },
    )
    assert result.returncode == 0, result.stderr
    assert (credentials / "master.key").read_text(encoding="utf-8") == "legacy-key"
    assert (credentials / "vault.db").read_text(encoding="utf-8") == "legacy-vault"

    # Existing persistent state wins on later starts; legacy data is not recopied.
    (credentials / "vault.db").write_text("persistent-vault", encoding="utf-8")
    second = subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "STORAGE_ROOT": str(state),
            "CREDENTIAL_ROOT": str(credentials),
            "AGENT_UID": str(os.geteuid()),
            "AGENT_GID": str(os.getegid()),
        },
    )
    assert second.returncode == 0, second.stderr
    assert (credentials / "vault.db").read_text(encoding="utf-8") == "persistent-vault"

def test_storage_initializer_rejects_root_target():
    result = _run_initializer(Path("/"))
    assert result.returncode == 2
    assert "Refusing" in result.stderr
