#!/usr/bin/env python3
"""Frozen CI corpus evaluation: select on calibration, use test only to veto."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promptscan import __version__  # noqa: E402
from promptscan.classifier import Classifier  # noqa: E402
from promptscan.scan import SCANNER_POLICY_VERSION  # noqa: E402
from scripts.eval_ci import (  # noqa: E402
    calibration_cutoff,
    check_cohorts,
    measure,
    scan_scores,
    select_model,
)


def load_cases(path, include_sources=False):
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("frozen corpus digest mismatch")
    data = pd.read_json(path, lines=True)
    sources = path.with_name("ci_sources.jsonl")
    if manifest.get("source_manifest_sha256"):
        if (
            hashlib.sha256(sources.read_bytes()).hexdigest()
            != manifest["source_manifest_sha256"]
        ):
            raise ValueError("frozen source manifest digest mismatch")
        if include_sources:
            import requests

            session = requests.Session()
            session.trust_env = False
            cache = Path("data/ci-source-cache")
            cache.mkdir(parents=True, exist_ok=True)
            records = []
            for line in sources.read_text().splitlines():
                row = json.loads(line)
                artifact = cache / row["text_sha256"]
                if not artifact.exists():
                    response = session.get(row["url"], timeout=30)
                    response.raise_for_status()
                    content = response.content
                else:
                    content = artifact.read_bytes()
                if hashlib.sha256(content).hexdigest() != row["text_sha256"]:
                    raise ValueError("public source digest mismatch")
                artifact.write_bytes(content)
                row["text"] = content.decode("utf-8")
                records.append(row)
            data = pd.concat([data, pd.DataFrame(records)], ignore_index=True)
    cohorts = [data[data.split == split] for split in ("train", "calibration", "test")]
    for i, left in enumerate(cohorts):
        for right in cohorts[i + 1 :]:
            check_cohorts(left, right)
            if set(left.source_group) & set(right.source_group):
                raise ValueError("attack-source groups overlap across splits")
    if data.id.duplicated().any() or data.text.duplicated().any():
        raise ValueError("duplicate corpus id or text")
    if data.review.str.contains("pending").any():
        raise ValueError("unreviewed labels")
    return data, manifest


def details(data, cutoff):
    result = measure(data, cutoff)
    predictions = data.score >= cutoff if cutoff is not None else data.score < 0
    pairs = data.assign(correct=predictions == data.label.astype(bool)).dropna(
        subset=["pair_id"]
    )
    result["complete_pairs"] = int(pairs.pair_id.nunique())
    result["correct_pairs"] = int(pairs.groupby("pair_id").correct.all().sum())
    result["by_format"] = {
        suffix: measure(group, cutoff)
        for suffix, group in data.groupby(data.path.map(lambda p: Path(p).suffix))
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/ci_cases.jsonl")
    )
    parser.add_argument(
        "--model", nargs=2, action="append", required=True, metavar=("NAME", "PATH")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data, manifest = load_cases(args.cases, include_sources=True)
    args.out.mkdir(parents=True, exist_ok=False)
    if len({name for name, _ in args.model}) != len(args.model):
        parser.error("model names must be unique")
    report = {
        "corpus": manifest,
        "device": args.device,
        "code_version": __version__,
        "scanner_policy": SCANNER_POLICY_VERSION,
        "models": {},
        "selection": None,
    }
    # Finish all calibration work before opening the test cohort to inference.
    for name, path in args.model:
        clf = Classifier.load(path, device=args.device)
        calibration = scan_scores(data[data.split == "calibration"], clf)
        cutoff, error = calibration_cutoff(calibration, 0.0001)
        calibration.to_parquet(args.out / f"{name}-calibration.parquet", index=False)
        report["models"][name] = {
            "model": clf.identity,
            "path": path,
            "parameters": sum(p.numel() for p in clf.model.parameters()),
            "stored_cutoff": clf.threshold,
            "cutoff": cutoff,
            "calibration_feasible": error is None,
            "calibration_error": error,
            "calibration": details(calibration, cutoff),
            "calibration_at_stored_cutoff": details(calibration, clf.threshold),
        }
        print(
            json.dumps(
                {"name": name, "calibration": report["models"][name]["calibration"]}
            ),
            flush=True,
        )
        del clf
        gc.collect()
    # One frozen comparison. Test data may reject a calibration choice, never
    # choose the cutoff, rank checkpoints or drive another training iteration.
    for name, path in args.model:
        clf = Classifier.load(path, device=args.device)
        test = scan_scores(data[data.split == "test"], clf)
        test.to_parquet(args.out / f"{name}-test.parquet", index=False)
        row = report["models"][name]
        row["test"] = details(test, row["cutoff"])
        row["test_at_stored_cutoff"] = details(test, clf.threshold)
        print(json.dumps({"name": name, "test": row["test"]}), flush=True)
        del clf
        gc.collect()
    report["selection"] = select_model(report["models"])
    report["goal_demonstrated"] = False
    report["limitation"] = (
        "Small, correlated fixture set; not proof of >90% recall or 0.01%"
        " false positives. The test is now a regression set; do not tune "
        "on it."
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selection"]), flush=True)


if __name__ == "__main__":
    main()
