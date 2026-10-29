#!/usr/bin/env python3
"""Global benchmark line using the threshold recorded by each evaluation run.

  python scripts/bench.py live rlc-r3      # one line per model
"""
import sys

import pandas as pd

for name in sys.argv[1:]:
    d = pd.read_json(f"results/bench-{name}.jsonl", lines=True)
    thresholds = d.threshold.unique()
    if len(thresholds) != 1:
        raise ValueError(f"{name}: mixed decision thresholds")
    n, p = d.label == 0, d.score >= thresholds[0]
    print(f"BENCH {name}: recall {p[~n].mean():.1%}, FP {p[n].mean():.2%} ({int(p[n].sum())}/{int(n.sum())}), "
          f"real-traffic FP {p[n & (d.set == 'toxicchat')].mean():.2%}, notinject FP {p[d.set == 'notinject'].mean():.1%}")
