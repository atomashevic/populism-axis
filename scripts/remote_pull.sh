#!/usr/bin/env bash
# Fast-forward the GPU checkout without discarding local work.
# Run inside the GPU checkout; local launch-host selection belongs to SSH.
# Tracked edits and conflicting untracked/ignored files require manual review.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

if ! branch="$(git symbolic-ref --quiet --short HEAD)"; then
  echo "cannot update a detached HEAD; inspect git status and select the intended branch manually" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "cannot update: tracked files have local changes; preserve them and review git status before retrying" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

git fetch -q origin "$branch"
if ! git merge --ff-only --no-overwrite-ignore FETCH_HEAD; then
  echo "update stopped; local work was not discarded. Inspect git status and the Git error above; back up conflicting local files or resolve branch divergence manually before retrying" >&2
  exit 1
fi
echo "updated $(git rev-parse --short HEAD) on $branch; local files preserved"
