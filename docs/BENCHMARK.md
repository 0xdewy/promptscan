# Benchmark (2026-10-21)

Command: `python scripts/benchmark.py --samples 300`. CPU only, batch size 1, one sample at a time.

| dataset | model | n | accuracy | precision | recall | F1 | ms/sample |
|---|---|---|---|---|---|---|---|
| deepset/prompt-injections (test) | promptscan ensemble | 116 | 0.966 | 1.000 | 0.933 | 0.966 | 99.6 |
| deepset/prompt-injections (test) | deepset/deberta-v3-base-injection | 116 | 0.991 | 1.000 | 0.983 | 0.992 | 161.6 |
| xTRam1/safe-guard-prompt-injection (test) | promptscan ensemble | 300 | 0.657 | 0.477 | 0.874 | 0.617 | 100.1 |
| xTRam1/safe-guard-prompt-injection (test) | deepset/deberta-v3-base-injection | 300 | 0.497 | 0.385 | 0.989 | 0.555 | 237.7 |
| jackhhao/jailbreak-classification (test) | promptscan ensemble | 262 | 0.641 | 0.604 | 0.942 | 0.736 | 100.5 |
| jackhhao/jailbreak-classification (test) | deepset/deberta-v3-base-injection | 262 | 0.557 | 0.545 | 1.000 | 0.706 | 474.7 |

Notes:
- deepset/prompt-injections is in promptscan's training data and is what the deberta model was fine-tuned on, so that row is in-distribution for both.
- The other two sets are not in promptscan's training data. jailbreak-classification labels jailbreaks, a neighbouring task.
- Latency is per-sample wall time on a 2-vCPU Hetzner box, tokenization included, no batching.
