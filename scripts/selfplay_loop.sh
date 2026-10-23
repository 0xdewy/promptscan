#!/usr/bin/env bash
# Self-play loop: each round attacks the latest model, adds the judged rows, retrains, evaluates.
#   scripts/selfplay_loop.sh <start-model-dir> <first-round> <last-round>
#   scripts/selfplay_loop.sh models/s5-small-60k-selfplay 2 4
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; first=$2; last=$3
logs=/data/codex-scratch/user/codex-job-manual-promptscan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for r in $(seq "$first" "$last"); do
  out=models/sp-r$r
  if [ ! -f "data/v2/raw/selfplay_r$r.parquet" ]; then   # resume: keep already generated rounds
    uv run python scripts/selfplay.py --model "$model" --tag "r$r" --batches 60 --per-batch 15 ${FOCUS:+--focus $FOCUS} > "$logs/selfplay_r$r.log" 2>&1
    echo "ROUND $r generated: $(grep -E '^round|judge agreed' "$logs/selfplay_r$r.log" | tail -2 | tr '\n' ' ')"
  fi
  uv run python scripts/build_dataset.py --only none > "$logs/build_r$r.log" 2>&1
  # the GPU is shared with other jobs: wait until it is nearly empty (<1.5 GiB used), retry if another job grabs it
  for attempt in $(seq 10); do
    free=0   # require 3 consecutive free minutes, so we don't jump into the gap between another job's runs
    while [ $free -lt 3 ]; do
      if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; then free=$((free + 1)); else free=0; fi
      sleep 60
    done
    uv run python scripts/train_v2.py --base microsoft/deberta-v3-small --out "$out" --limit 60000 --epochs 1 \
      --eval-steps 400 > "$logs/train_r$r.log" 2>&1 && break
    echo "ROUND $r training attempt $attempt failed, retrying"
    [ "$attempt" = 10 ] && exit 1
  done
  uv run python scripts/eval_suite.py --scorer "v2:$out" --limit 300 --out "results/sp-r$r.jsonl" > "$logs/eval_r$r.log" 2>&1
  uv run python - "$r" <<'EOF'
import sys, pandas as pd
d = pd.read_json(f"results/sp-r{sys.argv[1]}.jsonl", lines=True)
n = d.label == 0
for t in (d.threshold[0], 0.9, 0.97):
    p = d.score >= t
    print(f"ROUND {sys.argv[1]} thr {t:.2f}: recall {p[~n].mean():.3f} fpr {p[n].mean():.3f} "
          f"real {p[n & (d.set == 'toxicchat')].mean():.3f} notinject {p[d.set == 'notinject'].mean():.3f}")
EOF
  model=$out
done
echo "LOOP DONE"
