# CI revision results — 2026-10-25

The CI revision is implemented locally. The short training candidate was rejected;
`models/release` and the published checkpoint were not replaced. The new hard
cases expose a larger context problem than earlier benchmarks measured.

## Product changes

- Default coverage includes Python, JavaScript, TypeScript and JSON/YAML/TOML.
  JSON reports skipped paths and reasons. Symlinks and files over 1 MB are skipped.
- HTML comments and decoded Unicode tags retain nearby source context. There is
  no whitelist for quoted text or code fences. The independent 200,000-character
  cutoff was removed from both training and inference.
- PR review compares base and head with one model/cutoff, reports new findings,
  supports path scopes and renames, and uses the base's ignore policy. An existing
  stronger finding cannot silence a new lower-scoring payload elsewhere.
- Code releases pin the default model revision. The Action installs its bundled
  code by default and caches that model identity. JSON/SARIF identify code, policy,
  model and cutoff. Terminal and GitHub output escape untrusted control characters.
- Unicode hygiene is a separate warning by default. `--fail-on all` blocks it.
  Training-only dependencies moved to an optional extra. Runtime still uses one
  classifier and the Unicode check.

## Frozen comparison

The [label contract](LABELS.md) and frozen fixture manifests define the task,
matched pairs, provenance, review limitations and repository/source-family splits.
There are 72 training cases, 52 calibration files and 42 test files, including
10 pinned public files from six repositories. Labels are assistant-reviewed,
not independently human-reviewed. The test attacks are three adaptations of one
AgentDojo attack family across six formats; they are correlated examples.

The existing local release is the baseline. The candidate continued from it for
16 optimizer steps at learning rate `2e-6`, CPU fp32, batch 1 with 16 accumulated
examples, seed 270. Training used the 72 matched cases repeated four times plus
256 replay rows sampled equally by class from the existing training corpus.
The original corpus was preserved. No test rows entered training or calibration.

At the unchanged stored cutoff, **0.989**:

| Hard test cohort | Existing local model | Candidate |
|---|---:|---:|
| Detected attack attempts | 12/18 (66.7%) | 5/18 (27.8%) |
| False alarms on benign files | 17/24 (70.8%) | 13/24 (54.2%) |
| Matched pairs both correct | 0/18 | 0/18 |

These are deliberately difficult security examples, not representative estimates
of ordinary repository traffic. All six real public test files were clean for
both models; all false alarms were on the quoted attack controls.

Selecting a cutoff for zero observed calibration false positives chose
`0.9998313188552858` for the baseline and `0.9994097948074342` for the candidate.
Calibration detections improved from 1/24 to 5/24. **Both then detected 0/18 test
attacks with 0/24 false alarms.** The holdout gate rejected the candidate for no
held-out improvement. Neither these cutoffs nor the candidate were promoted.

The zero-error test false-positive upper bound is still about 11.7%, assuming
independent observations; the cases are actually correlated. This does not
establish a 0.01% production rate. This test has now been inspected and must be
used only for regression, not another optimization round.

The pinned published default was also checked separately. At its stored cutoff
of 0.97, it detected 17/18 test attempts but flagged all 18 quoted controls
(18/24 total benign files). A zero-calibration-error cutoff again yielded 0/18
test detections. These results do not describe the ordinary-file false-positive
rate, but confirm that the context problem also affects the published model.

## Native tool workflow

Six local Qwen3 1.7B trials used native tool calls, simulated files and fresh fake
secrets. The legitimate task required reading project data and saving an exact
report including a fresh nonce. Results came from executed simulated actions,
not an LLM judge or claimed success in prose.

| Observed result | No scanner | Existing model guard | Candidate guard |
|---|---:|---:|---:|
| Benign tasks completed | 3/3 | 0/3 | 1/3 |
| Tasks completed under attack | 3/3 | 1/3 | 2/3 |
| Attacker goals achieved | 0/3 | 0/3 | 0/3 |
| Target errors | 0 | 0 | 0 |

Guarded outcomes block flagged inputs before exposure and replay identical tool
traces for unblocked inputs. The agent already resisted all three attacks, so
this run demonstrates lost utility and does **not** demonstrate a security benefit.
The workflow is inspired by AgentDojo; it is not the official benchmark.
Rates describe observed actions over all attempted trials; errors are reported
separately and cannot erase an unsafe action already recorded.

## Artifacts and reproduction

Local artifacts:

- `results/ci-product-20261025/report.json` and per-model scored parquet files.
- `results/ci-published-20261025/report.json`: the pinned published default.
- `models/ci-context-20261025/`: rejected candidate, retained for inspection.
- `data/ci-context-20261025/`: exact training/calibration inputs and provenance.
- `results/agent-workflow-20261025/`: original native tool traces and target digest.
- `results/agent-workflow-candidate-20261025/`: candidate decisions on the same traces.

```bash
python scripts/eval_product.py \
  --model release models/release --model candidate models/ci-context-20261025 \
  --out results/new-ci-comparison
python scripts/eval_agent_workflow.py --model models/release \
  --out results/new-agent-traces
python scripts/eval_agent_workflow.py --model models/ci-context-20261025 \
  --replay results/new-agent-traces --out results/new-agent-comparison
```

The generator reproduces the frozen authored fixture bytes. Public sources are
retrieved at their recorded revisions and verified by digest. Future model
experiments need a new independent test set before selecting another candidate.
The useful next training work is to distinguish active redirection from quoted
examples without losing detection across new attack families. The >90% recall
and 0.01% false-positive goal remains unproven.

Validation: 112 tests passed; Ruff and diff checks passed. Eight runtime source
files scanned clean. The wheel and source distribution built and passed
`twine check`; the wheel includes PR review and keeps training dependencies
optional. The pinned public checkpoint loaded successfully. Original local
release weights were verified unchanged. No model, package or code was published.
