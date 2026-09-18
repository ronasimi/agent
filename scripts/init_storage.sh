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

install -d -m 0750 -o "$runtime_uid" -g "$runtime_gid" \
  "$memory_dir" \
  "$workspace_dir" \
  "$workspace_dir/custom_tools" \
  "$workspace_dir/research" \
  "$workspace_dir/self_optimization/candidates" \
  "$workspace_dir/self_optimization/approved" \
  "$workspace_dir/self_optimization/validation/inbox" \
  "$workspace_dir/self_optimization/validation/outbox"

# Existing databases, WAL files, reports, and candidate state may have been
# created by an earlier root-owned bind mount. Preserve them and repair access.
chown -R "$runtime_uid:$runtime_gid" "$memory_dir" "$workspace_dir"
chmod -R u+rwX "$memory_dir" "$workspace_dir"

echo "Initialized writable state for UID:GID $runtime_uid:$runtime_gid."
