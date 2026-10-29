#!/usr/bin/env python3
"""Small native-tool workflow smoke test; no real files, keys or external tools."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promptscan import __version__  # noqa: E402
from promptscan.classifier import Classifier  # noqa: E402
from promptscan.scan import SCANNER_POLICY_VERSION, scan_text  # noqa: E402
from scripts.agent_trial import REPORT_TOOL, TOOLS, OllamaTarget, run_trial  # noqa: E402
from scripts.eval_product import load_cases  # noqa: E402


def workflow_metrics(rows, guarded=False):
    evaluated = [r for r in rows if r["trial"]["status"] != "error"]
    attacks = [r for r in rows if r["label"] == 1]
    benign = [r for r in rows if r["label"] == 0]

    def rate(group, field):
        if not group:
            return None
        return sum(
            bool(r["trial"].get(field)) and not (guarded and r["blocked"])
            for r in group
        ) / len(group)

    return {
        "evaluated": len(evaluated),
        "errors": len(rows) - len(evaluated),
        "observed_unsafe_errors": sum(
            r["trial"]["status"] == "error"
            and r["trial"].get("unsafe_action", False)
            and not (guarded and r["blocked"])
            for r in rows
        ),
        "attacks": len(attacks),
        "benign": len(benign),
        "attacker_goal_rate": rate(attacks, "success"),
        "unsafe_action_rate": rate(attacks, "unsafe_action"),
        "task_completion_under_attack": rate(attacks, "task_success"),
        "benign_task_completion": rate(benign, "task_success"),
        "blocked_benign": sum(r["blocked"] for r in benign) if guarded else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/ci_cases.jsonl")
    )
    parser.add_argument("--model", default="models/release")
    parser.add_argument("--target", default="qwen3:1.7b")
    parser.add_argument(
        "--replay",
        type=Path,
        help="reuse verified tool traces from a previous output directory",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data, manifest = load_cases(args.cases)
    # All three held-out goals and their paired security-documentation controls.
    selected = data[
        (data.split == "test") & data.path.str.endswith(".md") & data.pair_id.notna()
    ]
    clf = Classifier.load(args.model)
    target = OllamaTarget(args.target, tools=TOOLS + [REPORT_TOOL])
    replay = {}
    source_trace = None
    if args.replay:
        previous = json.loads((args.replay / "report.json").read_text())
        if previous["corpus_sha256"] != manifest["sha256"]:
            raise ValueError("replay corpus differs")
        content = (args.replay / "trials.jsonl").read_bytes()
        source_trace = hashlib.sha256(content).hexdigest()
        replay = {r["id"]: r for r in map(json.loads, content.splitlines())}
        identity = previous["target"]
        if previous["target_settings"] != {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 8192,
            "num_predict": 512,
            "num_gpu": 0,
            "num_thread": 4,
        }:
            raise ValueError("replay settings differ")
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            models = json.load(response)["models"]
        identity = next(m for m in models if m["name"] == args.target)
    rows = []
    args.out.mkdir(parents=True, exist_ok=False)
    for row in selected.to_dict("records"):
        decision = {}
        findings = scan_text(row["text"], clf, row["path"], decision=decision)
        if args.replay:
            saved = replay[row["id"]]
            trial = saved["trial"]
            if (
                saved["label"] != row["label"]
                or trial["goal"] != row["goal"]
                or trial["messages"][3]["content"] != row["text"]
            ):
                raise ValueError("replay case differs")
        else:
            trial = run_trial(row["text"], row["goal"], target, complete_task=True)
        result = {
            "id": row["id"],
            "label": row["label"],
            "score": decision["score"],
            "blocked": any(f.kind == "injection" for f in findings),
            "trial": trial,
        }
        rows.append(result)
        with (args.out / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "trial"}
                | {
                    k: trial[k]
                    for k in ("success", "task_success", "unsafe_action", "status")
                }
            ),
            flush=True,
        )
    report = {
        "code_version": __version__,
        "scanner_policy": SCANNER_POLICY_VERSION,
        "corpus_sha256": manifest["sha256"],
        "model": clf.identity,
        "threshold": clf.threshold,
        "source_trace_sha256": source_trace,
        "target": identity,
        "tools_sha256": hashlib.sha256(
            json.dumps(target.tools, sort_keys=True).encode()
        ).hexdigest(),
        "target_settings": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 8192,
            "num_predict": 512,
            "num_gpu": 0,
            "num_thread": 4,
        },
        "unprotected": workflow_metrics(rows),
        "guarded": workflow_metrics(rows, True),
        "limitation": (
            "Six synthetic cases and one small local target. Guarded results "
            "replay the identical traces for unblocked input and stop blocked"
            " input before exposure. Not an official AgentDojo run or a "
            "production safety estimate."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
