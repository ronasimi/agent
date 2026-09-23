#!/bin/sh
set -eu

storage_root=${STORAGE_ROOT:-/state}
runtime_uid=${AGENT_UID:-1000}
runtime_gid=${AGENT_GID:-1000}

case "$storage_root" in
  /*) ;;
  *) echo "STORAGE_ROOT must be an absolute path." >&2; exit 2 ;;
esac

if [ "$storage_root" = "/" ]; then
  echo "Refusing to initialize the filesystem root." >&2
  exit 2
fi

case "$runtime_uid:$runtime_gid" in
  *[!0-9:]*|:*|*:|*:*:*)
    echo "AGENT_UID and AGENT_GID must be numeric." >&2
    exit 2
    ;;
esac

memory_dir="$storage_root/memory"
workspace_dir="$storage_root/workspace"
credential_dir=${CREDENTIAL_ROOT:-"$memory_dir/credentials"}
legacy_credential_dir="$memory_dir/credentials"

case "$credential_dir" in
  /*) ;;
  *) echo "CREDENTIAL_ROOT must be an absolute path." >&2; exit 2 ;;
esac

install -d -m 0750 -o "$runtime_uid" -g "$runtime_gid" \
  "$memory_dir" \
  "$memory_dir/huggingface" \
  "$workspace_dir" \
  "$workspace_dir/custom_tools" \
  "$workspace_dir/skills" \
  "$workspace_dir/research" \
  "$workspace_dir/self_optimization/candidates" \
  "$workspace_dir/self_optimization/approved" \
  "$workspace_dir/self_optimization/validation/inbox" \
  "$workspace_dir/self_optimization/validation/outbox"

# Keep OAuth credentials in a stable mount independent of the source checkout.
# On the first run after this migration, preserve an existing repo-local vault.
install -d -m 0700 -o "$runtime_uid" -g "$runtime_gid" "$credential_dir"
if [ "$credential_dir" != "$legacy_credential_dir" ] && [ -d "$legacy_credential_dir" ]; then
  if [ -z "$(find "$credential_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    if [ -n "$(find "$legacy_credential_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
      cp -a "$legacy_credential_dir"/. "$credential_dir"/
      echo "Migrated legacy credential vault to persistent credential storage."
    fi
  fi
fi

# Existing databases, WAL files, reports, and candidate state may have been
# created by an earlier root-owned bind mount. Preserve them and repair access.
chown -R "$runtime_uid:$runtime_gid" "$memory_dir" "$workspace_dir" "$credential_dir"
chmod -R u+rwX "$memory_dir" "$workspace_dir"
chmod 0700 "$credential_dir"
find "$credential_dir" -type f -exec chmod 0600 {} + 2>/dev/null || true

echo "Initialized writable state and persistent credentials for UID:GID $runtime_uid:$runtime_gid."
