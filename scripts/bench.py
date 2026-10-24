#!/usr/bin/env python3
"""Global benchmark line for results/bench-<name>.jsonl, at the live threshold (0.97).

  python scripts/bench.py live rlc-r3      # one line per model
"""
import sys

import pandas as pd

for name in sys.argv[1:]:
    d = pd.read_json(f"results/bench-{name}.jsonl", lines=True)
    n, p = d.label == 0, d.score >= 0.97
    print(f"BENCH {name}: recall {p[~n].mean():.1%}, FP {p[n].mean():.2%} ({int(p[n].sum())}/{int(n.sum())}), "
          f"real-traffic FP {p[n & (d.set == 'toxicchat')].mean():.2%}, notinject FP {p[d.set == 'notinject'].mean():.1%}")
