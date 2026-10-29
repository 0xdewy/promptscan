#!/usr/bin/env python3
"""Compare complete CI scans using cutoffs selected on separate calibration files.

Input parquet columns: text, label, repo, path; optional pr_id identifies clean PRs.
Calibration/test repositories and normalized texts must be disjoint. Never tune
on the test results. A small corpus cannot certify a rare false-positive rate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promptscan.scan import classifier_segments, hidden_text  # noqa: E402
from scripts.build_dataset import norm  # noqa: E402


def scan_scores(data, classifier, batch=4, segmenter=classifier_segments):
    """Exact scanner decisions, batching files together without localisation.

    Timing is the batch average per file. Segments still come from the runtime
    scanner; combining inference requests changes neither the inputs nor max rule.
    """
    records = []
    for first in range(0, len(data), 64):
        chunk = data.iloc[first : first + 64]
        segments, owners, hidden = [], [], []
        start = time.perf_counter()
        for owner, row in enumerate(chunk.itertuples()):
            findings = hidden_text(row.text, row.path)
            hidden.append(bool(findings))
            views = segmenter(row.text, findings)
            segments.extend(text for text, _, _ in views)
            owners.extend([owner] * len(views))
        scores = np.asarray(classifier.score(segments, batch=batch), dtype=float)
        if len(scores) != len(segments) or not np.isfinite(scores).all():
            raise ValueError("scanner needs one finite score per segment")
        if not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("classifier scores must be in [0, 1]")
        maximum = np.zeros(len(chunk))
        np.maximum.at(maximum, owners, scores)
        seconds = (time.perf_counter() - start) / len(chunk)
        records.extend(
            {"score": score, "hidden": flag, "seconds": seconds}
            for score, flag in zip(maximum, hidden, strict=True)
        )
    return pd.concat(
        [
            data.reset_index(drop=True),
            pd.DataFrame(records, columns=["score", "hidden", "seconds"]),
        ],
        axis=1,
    )


def calibration_cutoff(data, target, fail_on_hidden=False):
    negatives = data[data.label == 0]
    allowed = int(len(negatives) * target)
    forced = int(negatives.hidden.sum()) if fail_on_hidden else 0
    if not len(negatives):
        raise ValueError("calibration needs benign files")
    if forced > allowed:
        return None, "hidden-Unicode findings alone exceed the calibration budget"
    eligible = negatives.loc[~negatives.hidden] if fail_on_hidden else negatives
    remaining = eligible.score.to_numpy()
    if not len(remaining):
        return None, "no classifier cutoff can satisfy the calibration budget"
    # Count the remaining allowed errors directly; avoid floating-point rounding
    # of (allowed-forced)/n when best_threshold converts it back to an integer.
    ordered = np.sort(remaining)[::-1]
    budget = allowed - forced
    if budget >= len(ordered):
        return 0.0, None
    cutoff = float(np.nextafter(ordered[budget], np.inf))
    if cutoff > 1:
        return None, "saturated benign scores leave no cutoff <= 1"
    return cutoff, None


def measure(data, cutoff, fail_on_hidden=False):
    predictions = (
        data.hidden.to_numpy(dtype=bool).copy()
        if fail_on_hidden
        else np.zeros(len(data), dtype=bool)
    )
    if cutoff is not None:
        predictions |= data.score.to_numpy() >= cutoff
    positive = data.label.to_numpy() == 1
    negatives = int((~positive).sum())
    positives = int(positive.sum())
    fp = int(predictions[~positive].sum())
    tp = int(predictions[positive].sum())
    prs = {}
    if "pr_id" in data:
        for i, row in enumerate(data.itertuples()):
            if not positive[i] and pd.notna(row.pr_id):
                prs[row.pr_id] = prs.get(row.pr_id, False) or bool(predictions[i])
    if "pr_ids" in data:
        for i, row in enumerate(data.itertuples()):
            if not positive[i]:
                for pr_id in json.loads(row.pr_ids):
                    prs[pr_id] = prs.get(pr_id, False) or bool(predictions[i])
    return {
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "classifier_fp": (
            int((data.loc[~positive, "score"] >= cutoff).sum())
            if cutoff is not None
            else 0
        ),
        "recall": tp / positives if positives else None,
        "recall_lower_95": (
            float(beta.ppf(0.05, tp, positives - tp + 1)) if tp else 0.0
        ),
        "fpr": fp / negatives if negatives else None,
        "fpr_upper_95": (
            float(beta.ppf(0.95, fp + 1, negatives - fp))
            if negatives and fp < negatives
            else 1.0
        ),
        "clean_prs": len(prs),
        "flagged_clean_prs": sum(prs.values()),
        "hidden_benign_files": int(data.loc[~positive, "hidden"].sum()),
    }


def check_cohorts(calibration, test):
    for data in (calibration, test):
        if data.empty or not data.label.isin([0, 1]).all():
            raise ValueError("cohorts need nonempty binary-labelled files")
        if data.text.isna().any() or data.path.isna().any() or data.repo.isna().any():
            raise ValueError("every file needs text, path and repository provenance")
    if set(calibration.repo.str.casefold()) & set(test.repo.str.casefold()):
        raise ValueError("calibration and test repositories overlap")
    if set(calibration.text.map(norm)) & set(test.text.map(norm)):
        raise ValueError("calibration and test texts overlap")


def select_model(models):
    """Select on calibration only; test results can reject, never rank candidates."""
    baseline = next(iter(models))
    feasible = [name for name, row in models.items() if row["calibration_feasible"]]
    selected = max(
        feasible,
        key=lambda name: (
            models[name]["calibration"]["tp"],
            -models[name]["parameters"],
            name == baseline,
        ),
        default=baseline,
    )
    reasons = []
    if selected == baseline:
        reasons.append("no candidate improved calibration recall at the same budget")
    before, after = models[baseline]["test"], models[selected]["test"]
    if after["tp"] < before["tp"]:
        reasons.append("held-out attack detections decreased")
    if after["fp"] > before["fp"]:
        reasons.append("held-out file false alarms increased")
    if after["flagged_clean_prs"] > before["flagged_clean_prs"]:
        reasons.append("held-out clean PR false alarms increased")
    if selected != baseline and all(
        before[key] == after[key] for key in ("tp", "fp", "flagged_clean_prs")
    ):
        reasons.append("no held-out CI improvement")
    return {
        "baseline": baseline,
        "selected_on_calibration": selected,
        "retained_model": baseline if reasons else selected,
        "needs_broader_validation": not reasons,
        "reasons": reasons,
    }


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from promptscan.classifier import Classifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "PATH", "WINDOW"),
    )
    parser.add_argument("--fpr-target", type=float, default=0.0001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.fpr_target < 1:
        parser.error("--fpr-target must be in [0, 1)")
    calibration = pd.read_parquet(args.calibration)
    test = pd.read_parquet(args.test)
    check_cohorts(calibration, test)
    if len({m[0] for m in args.model}) != len(args.model):
        parser.error("model names must be unique")
    args.out.mkdir(parents=True, exist_ok=False)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "target_fpr": args.fpr_target,
        "device": device,
        "models": {},
        "limitation": (
            "file-level binomial bounds assume independent, representative samples; "
            "public checkpoints may have seen these repositories"
        ),
    }
    for name, path, window in args.model:
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            use_safetensors=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        classifier = Classifier(
            model,
            AutoTokenizer.from_pretrained(path, trust_remote_code=False),
            0.5,
            int(window),
            device=device,
        )
        scored_cal = scan_scores(calibration, classifier)
        cutoff, error = calibration_cutoff(scored_cal, args.fpr_target)
        scored_test = scan_scores(test, classifier)
        scored = pd.concat(
            [
                scored_cal.assign(cohort="calibration"),
                scored_test.assign(cohort="test"),
            ],
            ignore_index=True,
        )
        scored["text_sha256"] = scored.text.map(
            lambda t: hashlib.sha256(t.encode()).hexdigest()
        )
        scored.to_parquet(args.out / f"{name}-scores.parquet", index=False)
        result = {
            "model": path,
            "parameters": sum(p.numel() for p in model.parameters()),
            "window": int(window),
            "cutoff": cutoff,
            "stored_cutoff": float(
                (getattr(model.config, "promptscan", None) or {}).get("threshold", 0.5)
            ),
            "calibration_feasible": error is None,
            "calibration_error": error,
            "calibration": measure(scored_cal, cutoff),
            "test": measure(scored_test, cutoff),
        }
        result["test_at_stored_cutoff"] = measure(scored_test, result["stored_cutoff"])
        report["models"][name] = result
        print(json.dumps({"name": name, **result}), flush=True)
        del classifier, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["selection"] = select_model(report["models"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
