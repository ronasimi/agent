#!/usr/bin/env python3
"""Apply one human-approved, digest-pinned optimization patch to a Git repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(argv: list[str], repo: Path, patch: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=repo, input=patch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--confirm", required=True, help="Must equal APPLY_APPROVED_PATCH")
    args = parser.parse_args()

    repo = args.repo.resolve()
    patch_path = args.patch.resolve()
    if args.confirm != "APPLY_APPROVED_PATCH":
        raise SystemExit("Refusing promotion: confirmation phrase did not match.")
    if not (repo / ".git").exists():
        raise SystemExit("Target must be a Git repository.")
    patch = patch_path.read_bytes()
    digest = hashlib.sha256(patch).hexdigest()
    if digest != args.sha256.lower():
        raise SystemExit("Patch SHA-256 does not match --sha256.")

    manifest_path = patch_path.with_suffix(".json")
    if not manifest_path.is_file():
        raise SystemExit("Approved candidate manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "approved" or manifest.get("patch_sha256") != digest:
        raise SystemExit("Candidate manifest is not approved or does not match the patch.")

    status = run(["git", "status", "--porcelain"], repo)
    if status.returncode or status.stdout.strip():
        raise SystemExit("Target repository must have a clean worktree.")
    check = run(["git", "apply", "--check", "--whitespace=error-all", "-"], repo, patch)
    if check.returncode:
        raise SystemExit(check.stdout.decode("utf-8", errors="replace"))
    applied = run(["git", "apply", "--whitespace=error-all", "-"], repo, patch)
    if applied.returncode:
        raise SystemExit(applied.stdout.decode("utf-8", errors="replace"))
    print(f"Applied approved candidate {manifest.get('id')} ({digest}). Review and commit it normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
