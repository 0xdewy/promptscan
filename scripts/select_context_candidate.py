#!/usr/bin/env python3
"""Select contextual SFT weights on development data at fixed false-alarm budgets.

The sealed test is deliberately not an input. Passing this gate permits one final
holdout comparison; it never publishes or replaces a model by itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptscan.classifier import Classifier  # noqa: E402
from scripts.context_baseline import legacy_segments  # noqa: E402
from scripts.eval_ci import measure, scan_scores  # noqa: E402


def cutoff_for_budget(scores, allowed):
    values = np.sort(np.asarray(scores, dtype=float))[::-1]
    if not len(values) or not np.isfinite(values).all() or allowed < 0:
        raise ValueError("invalid calibration cohort or budget")
    if allowed >= len(values):
        return 0.0
    # CPU probes showed drift above 3e-6; leave 1e-5, then round upward.
    cutoff = math.ceil((float(values[allowed]) + 1e-5) * 1e6) / 1e6
    return cutoff if cutoff <= 1 else None


def counts(data, cutoff):
    return {name: measure(group, cutoff) for name, group in data.groupby("cohort")}


def recall_preserving_cutoff(candidate, before):
    """Prefer a conservative cutoff while retaining the baseline's detections."""
    if not candidate.score.between(0, 1).all():
        raise ValueError("candidate needs finite scores in [0, 1]")
    ceilings = [1.0]
    for group, required in [
        (candidate[candidate.cohort.eq("general")], before["general"]["tp"]),
        (candidate, sum(row["tp"] for row in before.values())),
    ]:
        if required:
            scores = np.sort(group.loc[group.label.eq(1), "score"])[::-1]
            if len(scores) < required:
                return -1.0
            ceilings.append(
                math.floor((float(scores[required - 1]) - 1e-5) * 1e6) / 1e6
            )
    return min(ceilings)


def interval_midpoint(lower, upper):
    """Maximize the smaller logit margin to the two development constraints."""
    if 0 < lower <= upper < 1:
        margin = (math.log(lower / (1 - lower)) + math.log(upper / (1 - upper))) / 2
        middle = 1 / (1 + math.exp(-margin))
    else:
        middle = (lower + upper) / 2
    return max(lower, min(upper, round(middle, 6)))


def evaluate(baseline, candidate, baseline_cutoff):
    before = counts(baseline, baseline_cutoff)
    general_budget = before["general"]["fp"]
    # Marked improvement: keep general recall and its false-positive count,
    # retain aggregate recall, and remove at least 75% of context false alarms.
    context_budget = before["context"]["fp"] // 4
    cutoffs = [
        cutoff_for_budget(group[group.label == 0].score, budget)
        for group, budget in [
            (candidate[candidate.cohort == "general"], general_budget),
            (candidate[candidate.cohort == "context"], context_budget),
            (candidate[candidate.cohort == "public_context"], 0),
        ]
    ]
    cutoff = max(cutoffs) if all(t is not None for t in cutoffs) else None
    ceiling = recall_preserving_cutoff(candidate, before)
    minimum = cutoff
    if cutoff is not None and ceiling >= cutoff:
        cutoff = interval_midpoint(cutoff, ceiling)
    after = counts(candidate, cutoff)
    reasons = []
    if cutoff is None:
        reasons.append("saturated benign scores exceed a calibration budget")
    if after["general"]["fp"] > general_budget:
        reasons.append("general false positives increased")
    if after["general"]["tp"] < before["general"]["tp"]:
        reasons.append("general attack recall decreased")
    if sum(row["tp"] for row in after.values()) < sum(
        row["tp"] for row in before.values()
    ):
        reasons.append("total attack recall decreased")
    if after["context"]["fp"] > context_budget:
        reasons.append("context false positives did not fall by 75%")
    if after["public_context"]["fp"]:
        reasons.append("public development code produced a false alarm")
    return {
        "cutoff": cutoff,
        "minimum_cutoff_for_fp_budgets": minimum,
        "maximum_cutoff_preserving_recall": ceiling,
        "accepted_on_development": not reasons,
        "reasons": reasons,
        "baseline": before,
        "candidate": after,
        "budgets": {
            "general_fp": general_budget,
            "context_fp": context_budget,
            "public_context_fp": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", default="models/release")
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument(
        "--candidate", nargs=2, action="append", metavar=("NAME", "PATH")
    )
    parser.add_argument(
        "--exported-scores",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "MODEL", "SCORES"),
        help="reuse fp32 unwrapped validation scores exported by train_v2",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = [(name, path, None) for name, path in (args.candidate or [])]
    candidates += [tuple(row) for row in args.exported_scores]
    if not candidates:
        parser.error("supply --candidate or --exported-scores")
    data = pd.read_parquet(args.data)
    if set(data.cohort) != {"general", "context", "public_context"}:
        raise ValueError(
            "development needs general, context and public_context cohorts"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = Classifier.load(args.baseline, device=args.device)
    threshold = baseline.threshold
    identity = baseline.identity
    if args.baseline_cache:
        provenance = json.loads(
            (args.baseline_cache.parent / "report.json").read_text()
        )["baseline_model"]
        key = "artifact_sha256" if "artifact_sha256" in identity else "revision"
        if not identity.get(key) or provenance.get(key) != identity[key]:
            raise ValueError("baseline cache belongs to a different model")
        cached = pd.read_parquet(args.baseline_cache)
        size = len(cached)
        if not cached.text.reset_index(drop=True).equals(
            data.text.iloc[:size].reset_index(drop=True)
        ):
            raise ValueError("baseline cache population differs")
        if not np.array_equal(cached.label, data.label.iloc[:size]):
            raise ValueError("baseline cache labels differ")
        tail = scan_scores(
            data.iloc[size:], baseline, batch=16, segmenter=legacy_segments
        )
        scored_baseline = pd.concat([cached, tail], ignore_index=True)
        scored_baseline["cohort"] = data.cohort.to_numpy()
    else:
        scored_baseline = scan_scores(
            data, baseline, batch=16, segmenter=legacy_segments
        )
    scored_baseline.to_parquet(args.out / "release-development.parquet", index=False)
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report = {
        "baseline_model": identity,
        "baseline_cutoff": threshold,
        "selection_data": str(args.data),
        "test_used": False,
        "selection_protocol": "development-logit-margin-v4",
        "minimum_numeric_margin": 1e-5,
        "models": {},
    }
    for name, path, score_file in candidates:
        classifier = Classifier.load(path, device="cpu" if score_file else args.device)
        if score_file:
            scored = pd.read_parquet(score_file)
            if (
                not scored[["text", "label"]]
                .reset_index(drop=True)
                .equals(data[["text", "label"]].reset_index(drop=True))
            ):
                raise ValueError("exported score population or labels differ")
            scored["cohort"] = data.cohort.to_numpy()
        else:
            scored = scan_scores(data, classifier, batch=16)
        scored.to_parquet(args.out / (name + "-development.parquet"), index=False)
        result = evaluate(scored_baseline, scored, threshold)
        result["identity"] = classifier.identity
        result["path"] = path
        report["models"][name] = result
        print(json.dumps({"name": name, **result}), flush=True)
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    eligible = [
        (name, r)
        for name, r in report["models"].items()
        if r["accepted_on_development"]
    ]
    selected = max(
        eligible,
        key=lambda item: (
            -sum(row["fp"] for row in item[1]["candidate"].values()),
            sum(row["tp"] for row in item[1]["candidate"].values()),
            item[1]["candidate"]["context"]["tp"],
        ),
        default=(None, None),
    )[0]
    report["selected"] = selected
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_on_development": selected}), flush=True)


if __name__ == "__main__":
    main()
