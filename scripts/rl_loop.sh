#!/usr/bin/env bash
# Alternate RL (attack generator) and SFT (classifier), in short rounds:
#   1. GRPO: the generator learns to write attacks the current classifier misses (25 steps)
#   2. harvest: judged misses + matched benign look-alikes -> data/v2/raw/rl_r<N>.parquet
#   3. SFT: train from the START classifier (never the previous round's: chaining drifts and
#      forgets) on all RL rows so far plus a 40k replay of the rest
#   4. benchmark it on the full held-out suite next to the live model (results/bench-*.jsonl)
#   5. next round attacks the new classifier
#
#   scripts/rl_loop.sh <generator-adapter> <classifier-dir> <first-round> <last-round>
#   START=<dir> sets the classifier each round trains from (default: models/release).
#   SKIP_RL=1 skips step 1 of the first round (its runs already exist in $RUNS/rl-r<N>).
set -euo pipefail
cd "$(dirname "$0")/.."
adapter=$1; clf=$2; first=$3; last=$4
start=${START:-models/release}
RUNS=${RUNS:-/data/codex-scratch/user/codex-job-manual-promptscan/rl-loop}
mkdir -p "$RUNS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_gpu() {   # the GPU is shared with other jobs
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 20; done
}

gpu_run() {   # another job can grab the GPU between our check and our start: wait and retry
  for _ in 1 2 3 4 5; do wait_gpu; "$@" && return 0; sleep 60; done
  return 1
}

benchmark() {   # $1 = classifier dir, $2 = name. Full held-out suite, at the live threshold (0.97).
  uv run python scripts/eval_suite.py --scorer "v2:$1" --out "results/bench-$2.jsonl" > "$RUNS/bench-$2.log" 2>&1 || return 1
  uv run python scripts/bench.py "$2"
}

[ -f "results/bench-live.jsonl" ] || gpu_run benchmark models/release live

for r in $(seq "$first" "$last"); do
  run=$RUNS/rl-r$r
  if [ "$r" != "$first" ] || [ -z "${SKIP_RL:-}" ]; then
    gpu_run uv run python scripts/rl_attacker.py --steps 25 --lr 3e-5 --adapter "$adapter" --classifier "$clf" \
      --judge-below 0.995 --out "$run" > "$run.log" 2>&1
    echo "ROUND $r RL: $(grep -oE 'step [0-9]+: valid [0-9]+/16, fooled [0-9]+/16' "$run.log" | awk '{f+=$6} END {print f " fooled in " NR " steps"}')"
    adapter=$run
  fi
  if [ ! -f "data/v2/raw/rl_r$r.parquet" ]; then
    uv run python scripts/rl_attacker.py --harvest "$run" --classifier "$clf" --tag "r$r" --out "$run" > "$run.harvest.log" 2>&1
    echo "ROUND $r harvest: $(tail -1 "$run.harvest.log")"
  fi
  uv run python scripts/build_dataset.py --only none > "$run.build.log" 2>&1
  gpu_run uv run python scripts/train_v2.py --base "$start" --out "models/rlc-r$r" --limit 40000 --epochs 1 --lr 2e-5 \
    --eval-steps 500 > "$run.train.log" 2>&1
  gpu_run benchmark "models/rlc-r$r" "rlc-r$r"
  clf=models/rlc-r$r
done
echo "LOOP DONE"
