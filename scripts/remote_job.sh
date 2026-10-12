#!/usr/bin/env bash
# Launch one GPU job on the shared GPU host, politely.
#
# Usage (run inside the repo on the GPU host):
#   scripts/remote_job.sh <job-name> -- <command> [args...]
# Example:
#   scripts/remote_job.sh step1_encode -- python scripts/encode_sentences.py --sentences ... --set-id ...
#
# Guard: refuses to start (exit 3) when another job is still running, when
# the GPU already has more than GPU_MEM_LIMIT_MB in use, or when the 1-minute
# load average exceeds LOAD_LIMIT. Runs the command under nohup, nice 10, two
# CPU threads, via `uv run --extra cu126`, logging to logs/<job-name>.log.
# The command may be `bash -c '...'` to chain several steps sequentially.
set -euo pipefail

GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-4096}"
LOAD_LIMIT="${LOAD_LIMIT:-2.5}"

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: $0 <job-name> -- <command> [args...]" >&2
  exit 2
fi
name="$1"; shift 2

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
mkdir -p logs

for pidfile in logs/*.pid; do
  [[ -e "$pidfile" ]] || continue
  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "busy: job $(basename "$pidfile" .pid) is still running; postpone" >&2
    exit 3
  fi
  rm -f "$pidfile"
done

used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( used_mb > GPU_MEM_LIMIT_MB )); then
  echo "busy: GPU has ${used_mb} MiB in use (limit ${GPU_MEM_LIMIT_MB}); postpone" >&2
  exit 3
fi

load1="$(cut -d' ' -f1 /proc/loadavg)"
if awk -v l="$load1" -v lim="$LOAD_LIMIT" 'BEGIN { exit !(l > lim) }'; then
  echo "busy: 1-minute load ${load1} exceeds ${LOAD_LIMIT}; postpone" >&2
  exit 3
fi

log="logs/${name}.log"
{
  echo "# job ${name} started $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  echo "# commit $(git rev-parse HEAD) dirty=$([[ -n "$(git status --porcelain --untracked-files=no)" ]] && echo yes || echo no)"
  echo "# gpu before: ${used_mb} MiB used, load ${load1}"
  echo "# command: $*"
} > "$log"

nohup nice -n 10 env \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  uv run --extra cu126 "$@" >> "$log" 2>&1 &
pid=$!
echo "$pid" > "logs/${name}.pid"
echo "started ${name} pid=${pid} log=${log}"
