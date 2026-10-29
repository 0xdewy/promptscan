#!/usr/bin/env python3
"""Veto a development-selected model on frozen holdout or regression data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptscan.classifier import Classifier  # noqa: E402
from scripts.context_baseline import legacy_segments  # noqa: E402
from scripts.eval_ci import measure, scan_scores  # noqa: E402


def summary(data, cutoff):
    result = measure(data, cutoff)
    result["by_source"] = {
        str(name): measure(group, cutoff) for name, group in data.groupby("source")
    }
    if "pair_id" in data:
        pairs = data[data.pair_id.notna()].copy()
        pairs["correct"] = (pairs.score >= cutoff) == pairs.label.astype(bool)
        grouped = pairs.groupby("pair_id").correct.all()
        result["pairs"] = len(grouped)
        result["correct_pairs"] = int(grouped.sum())
    return result


def veto(before, after, holdout):
    reasons = []
    if after["tp"] < before["tp"]:
        reasons.append("attack detections decreased")
    budget = before["fp"] // 2 if holdout else before["fp"]
    if after["fp"] > budget:
        reasons.append("false alarms exceeded the fixed comparison budget")
    for name, row in before["by_source"].items():
        if (name.startswith("public") or name == "toxicchat") and after["by_source"][
            name
        ]["fp"] > row["fp"]:
            reasons.append(f"{name} false alarms increased")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="required for the sealed holdout")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--raw-classifier", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        help="raw scores with an identity/checksum sidecar at the same .json path",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.holdout and args.raw_classifier:
        parser.error("sealed holdout requires the complete scanner")
    if args.baseline_cache and (not args.raw_classifier or args.holdout):
        parser.error("baseline cache is only for raw regression comparisons")
    selection = json.loads(args.selection.read_text())
    name = selection["selected"]
    if name is None or not selection["models"][name]["accepted_on_development"]:
        raise ValueError("no development-approved candidate")
    candidate = selection["models"][name]
    checksum = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.holdout:
        if not args.manifest:
            parser.error("--holdout requires --manifest")
        manifest = json.loads(args.manifest.read_text())
        if checksum != manifest["sealed_sha256"]:
            raise ValueError("sealed holdout changed")
    data = pd.read_parquet(args.data)
    data["path"] = data.path.fillna("evaluation.txt")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "candidate": name,
        "data_sha256": checksum,
        "holdout": args.holdout,
        "score_method": "raw_classifier" if args.raw_classifier else "complete_scanner",
        "selection": str(args.selection),
        "models": {},
    }
    for key, path, threshold in [
        (
            "release",
            selection.get("baseline_archive", selection["baseline_model"]["source"]),
            selection["baseline_cutoff"],
        ),
        ("candidate", candidate["path"], candidate["cutoff"]),
    ]:
        classifier = Classifier.load(
            path, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        expected = (
            selection["baseline_model"] if key == "release" else candidate["identity"]
        )
        field = "artifact_sha256" if "artifact_sha256" in expected else "revision"
        if not expected.get(field) or classifier.identity.get(field) != expected[field]:
            raise ValueError(f"{key} model changed after development selection")
        if key == "release" and args.baseline_cache:
            cached = pd.read_parquet(args.baseline_cache)
            provenance = json.loads(
                args.baseline_cache.with_suffix(".json").read_text()
            )
            if (
                provenance["model"][field] != expected[field]
                or provenance["score_method"] != "raw_classifier"
                or provenance["scores_sha256"]
                != hashlib.sha256(args.baseline_cache.read_bytes()).hexdigest()
                or not cached[["text", "label"]].equals(data[["text", "label"]])
                or list(cached["set"]) != list(data.source)
                or not cached.score.between(0, 1).all()
            ):
                raise ValueError("baseline raw cache identity or population differs")
            scored = data.assign(score=cached.score.to_numpy(), hidden=False)
        elif args.raw_classifier:
            scored = data.assign(
                score=classifier.score(data.text.tolist(), batch=args.batch),
                hidden=False,
            )
        else:
            scored = scan_scores(
                data,
                classifier,
                batch=args.batch,
                **({"segmenter": legacy_segments} if key == "release" else {}),
            )
        scored.to_parquet(args.out / (key + "-scores.parquet"), index=False)
        result = summary(scored, threshold)
        report["models"][key] = {
            "identity": classifier.identity,
            "cutoff": threshold,
            "metrics": result,
        }
        print(json.dumps({"model": key, **result}), flush=True)
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["reasons"] = veto(
        report["models"]["release"]["metrics"],
        report["models"]["candidate"]["metrics"],
        args.holdout,
    )
    report["accepted"] = not report["reasons"]
    report["limitation"] = (
        "Curated correlated cases; not a population-level safety estimate."
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"accepted": report["accepted"], "reasons": report["reasons"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
