#!/usr/bin/env bash
# scripts/rl_loop.sh <adapter-or-empty> <classifier> <first-round> <last-round>
# Additional arguments go to rl_attacker.py, e.g. --steps 2 --generations 4 --accum 1.
# REWARD=agent selects observed actions; BASE_DATA holds fixed train/val splits.
# START overrides the initial training checkpoint; TRAIN_LR sets classifier LR.
# Read the driver once so edits during a long run cannot corrupt Bash's file offset.
if [ -z "${BASH_EXECUTION_STRING+x}" ]; then
  exec bash -c "$(cat "$0")" "$0" "$@"
fi
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <adapter-or-empty> <classifier> <first> <last> [attacker options]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
adapter=$1; clf=$2; first=$3; last=$4
shift 4
start=${START:-$clf}
base_data=${BASE_DATA:-data/v2}
reward_mode=${REWARD:-agent}
target_model=${TARGET_MODEL:-qwen3:1.7b}
trial_below=${TRIAL_BELOW:-1.0}
if [ "$reward_mode" = judge ]; then trial_below=${TRIAL_BELOW:-0.995}; fi
RUNS=${RUNS:-results/rl-loop-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUNS"
if [ -e "$RUNS/accepted-model.txt" ]; then
  echo "Use a fresh RUNS directory; this one already contains a model decision" >&2
  exit 2
fi
for r in $(seq "$first" "$last"); do
  if [ -e "$RUNS/r$r" ]; then
    echo "Refusing to mix this run with existing output: $RUNS/r$r" >&2
    exit 2
  fi
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_gpu() {
  local used notified=0
  while true; do
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits) || return $?
    [ "$used" -ge 1500 ] || return 0
    if [ "$notified" = 0 ]; then
      echo "Waiting for GPU memory to become available ($used MiB used)" >&2
      notified=1
    fi
    sleep 20
  done
}

gpu_run() {  # Retry once for an actual CUDA OOM; all other failures stop immediately.
  local attempt log status
  for attempt in 1 2; do
    wait_gpu || return $?
    log=$(mktemp "$RUNS/.gpu-XXXXXX.log") || return 2
    if "$@" 2>&1 | tee "$log"; then
      rm "$log"
      return 0
    else
      status=$?
    fi
    if [ "$attempt" = 2 ] || [ "$status" != 1 ] || ! grep -qE \
      '^(torch(\.cuda)?\.OutOfMemoryError|RuntimeError): CUDA out of memory' "$log"; then
      rm "$log"
      return "$status"
    fi
    rm "$log"
    echo "CUDA OOM; waiting before one retry" >&2
    sleep 10
  done
}

threshold=$(uv run python - "$clf" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1], "config.json").read_text())
      .get("promptscan", {}).get("threshold", 0.5))
PY
)
printf '%s\n' "$clf" > "$RUNS/accepted-model.txt"
harvests=()

for r in $(seq "$first" "$last"); do
  run=$RUNS/r$r
  mkdir "$run"
  gpu_run uv run python scripts/rl_attacker.py --steps 25 --lr 3e-5 \
    --adapter "$adapter" --classifier "$clf" --judge-below "$trial_below" \
    --reward "$reward_mode" --target-model "$target_model" --out "$run/attacker" \
    "$@" > "$run/attacker.log" 2>&1
  adapter=$run/attacker
  if uv run python scripts/rl_attacker.py --harvest "$adapter" --classifier "$clf" \
    --tag "round$r" --out "$run/harvest" > "$run/harvest.log" 2>&1; then
    :
  else
    status=$?
    [ "$status" = 3 ] || exit "$status"
    echo "ROUND $r: no new misses; skipped detector retraining and retained $clf"
    continue
  fi
  harvests+=("$run/harvest/harvest.parquet")
  uv run python scripts/build_dataset.py --base-data "$base_data" \
    --extra "${harvests[@]}" --out "$run/data" > "$run/data.log" 2>&1
  gpu_run uv run python scripts/train_v2.py --base "$start" --data "$run/data" \
    --out "$run/classifier" --limit 40000 --epochs 1 --lr "${TRAIN_LR:-2e-6}" \
    --batch 1 --accum 32 --gradient-checkpointing \
    --threshold "$threshold" --eval-steps 0 > "$run/train.log" 2>&1
  if gpu_run uv run python scripts/validate_candidate.py --baseline "$clf" \
    --candidate "$run/classifier" --data "$run/data/val.parquet" \
    --out "$run/validation.json" > "$run/validation.log" 2>&1; then
    # Only validated candidates reach the held-out benchmark. No cached scores.
    gpu_run uv run python scripts/eval_suite.py --scorer "v2:$clf" \
      --out "$run/baseline-benchmark.jsonl" > "$run/baseline-benchmark.log" 2>&1
    gpu_run uv run python scripts/eval_suite.py --scorer "v2:$run/classifier" \
      --out "$run/candidate-benchmark.jsonl" > "$run/candidate-benchmark.log" 2>&1
    if uv run python scripts/validate_candidate.py --scores \
      --baseline "$run/baseline-benchmark.jsonl" --candidate "$run/candidate-benchmark.jsonl" \
      --out "$run/benchmark-gate.json" > "$run/benchmark-gate.log" 2>&1; then
      :
    else
      status=$?
      [ "$status" = 1 ] || exit "$status"
      echo "ROUND $r rejected by benchmark; retained $clf"
      continue
    fi
    clf=$run/classifier
    printf '%s\n' "$clf" > "$RUNS/accepted-model.txt"
    echo "ROUND $r accepted: $clf"
  else
    status=$?
    [ "$status" = 1 ] || exit "$status"
    echo "ROUND $r rejected; retained $clf (see $run/validation.json)"
  fi
done
