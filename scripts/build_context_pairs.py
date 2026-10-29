#!/usr/bin/env python3
"""Build contextual SFT pairs and a sealed, independently sourced test cohort.

Public benchmark JSON is input data only. Nothing in it is executed. Training
uses CyberSecEval and already verified training attacks; InjecAgent is test only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_dataset import norm  # noqa: E402


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reference_targets(data, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if (
        len(scores) != len(data)
        or not np.isfinite(scores).all()
        or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise ValueError("invalid reference probabilities")
    correct = (scores >= threshold) == data.label.to_numpy()
    eligible = data.is_replay.fillna(False).to_numpy(dtype=bool) | (
        data.label.to_numpy() == 1
    )
    probability = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.where(correct & eligible, np.log(probability / (1 - probability)), np.nan)


def ci_replay_examples(original, validation):
    rows = []
    for row in original[original.source.eq("ci_planted_verified")].to_dict("records"):
        if pd.isna(row.get("attack_start")) or pd.isna(row.get("attack_end")):
            continue
        start, end = int(row["attack_start"]), int(row["attack_end"])
        text = row["text"]
        if not 0 <= start < end <= len(text):
            raise ValueError("invalid CI replay payload span")
        left, right = max(0, start - 512), min(len(text), end + 512)
        if right - left > 8000:
            continue
        row.update(
            text=text[left:right],
            attack_start=start - left,
            attack_end=end - left,
            source="ci_payload_replay",
            is_replay=True,
            source_document_sha256=digest(text),
            source_char_start=left,
        )
        rows.append(row)
    positives = pd.DataFrame(rows).drop_duplicates("text")
    if positives.empty:
        raise ValueError("CI replay needs annotated training attacks")
    negatives = (
        original[
            (original.label == 0)
            & original.source.eq("ci_repository")
            & (original.text.str.len() <= 8000)
        ]
        .sample(
            min(
                len(positives),
                sum(
                    (original.label == 0)
                    & original.source.eq("ci_repository")
                    & (original.text.str.len() <= 8000)
                ),
            ),
            random_state=935,
        )
        .assign(is_replay=True)
    )
    extra = pd.concat([positives, negatives], ignore_index=True)
    known = set(validation.text.map(norm))
    return extra[~extra.text.map(norm).isin(known)].reset_index(drop=True)


def instruction_replay_examples(original, existing, validation):
    """Recover long instructions and reviewed hard negatives omitted by sampling."""
    length = original.text.str.len()
    structured = original.source.str.startswith(("rl_", "selfplay_")) | (
        original.source == "reviewed_benign_security_app"
    )
    long = (original.source == "ci_repository") & length.between(8001, 32000)
    known = set(existing.text.map(norm)) | set(validation.text.map(norm))
    negative = original[(original.label == 0) & (structured | long)]
    negative = negative[~negative.text.map(norm).isin(known)].drop_duplicates("text")
    positive = original[
        (original.label == 1) & length.le(8000) & ~original.text.map(norm).isin(known)
    ].sample(len(negative), random_state=937)
    return pd.concat([negative, positive], ignore_index=True).assign(is_replay=True)


def anchor_existing(source, teacher_path, out, extra=None, score_extra=True):
    import torch

    from promptscan.classifier import Classifier
    from promptscan.scan import SCANNER_POLICY_VERSION, classifier_segments
    from scripts.context_baseline import legacy_segments
    from scripts.eval_ci import scan_scores

    data = pd.read_parquet(source / "train.parquet")
    teacher = Classifier.load(
        teacher_path,
        device="cuda" if score_extra and torch.cuda.is_available() else "cpu",
    )
    settings = getattr(teacher.model.config, "promptscan", None) or {}
    policy = settings.get("scanner_policy", "context-1")
    if policy not in {"context-1", SCANNER_POLICY_VERSION}:
        raise ValueError(f"unknown teacher scanner policy: {policy}")
    segmenter = legacy_segments if policy == "context-1" else classifier_segments
    if extra is not None and "reference_margin" in data:
        record = source / "reference.json"
        if not record.exists():
            record = source / "protocol.json"
        previous = json.loads(record.read_text())
        if (
            previous["teacher"]["artifact_sha256"]
            != teacher.identity["artifact_sha256"]
        ):
            raise ValueError("replay targets belong to a different teacher")
        if score_extra:
            scored = scan_scores(
                extra.assign(path=extra.path.fillna("training")),
                teacher,
                batch=16,
                segmenter=segmenter,
            )
            extra["reference_margin"] = reference_targets(
                extra, scored.score, teacher.threshold
            )
        else:
            extra["reference_margin"] = np.nan
        data = pd.concat([data, extra], ignore_index=True)
    else:
        if extra is not None:
            data = pd.concat([data, extra], ignore_index=True)
        scored = scan_scores(
            data.assign(path=data.path.fillna("training")),
            teacher,
            batch=16,
            segmenter=segmenter,
        )
        data["reference_margin"] = reference_targets(
            data, scored.score, teacher.threshold
        )
    out.mkdir(parents=True, exist_ok=False)
    data.to_parquet(out / "train.parquet", index=False)
    shutil.copyfile(source / "val.parquet", out / "val.parquet")
    (out / "reference.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "extra_rows": len(extra) if extra is not None else 0,
                "extra_sources": (
                    extra.source.value_counts().to_dict() if extra is not None else {}
                ),
                "source_train_sha256": hashlib.sha256(
                    (source / "train.parquet").read_bytes()
                ).hexdigest(),
                "source_val_sha256": hashlib.sha256(
                    (source / "val.parquet").read_bytes()
                ).hexdigest(),
                "teacher": teacher.identity,
                "teacher_threshold": teacher.threshold,
                "teacher_policy": policy,
                "new_rows_scored_by_teacher": score_extra,
                "targets": int(data.reference_margin.notna().sum()),
                "new_negative_targets": int(
                    data[(~data.is_replay) & (data.label == 0)]
                    .reference_margin.notna()
                    .sum()
                ),
                "rule": "preserve correct predictions on replay or positive pairs; "
                "never anchor new contextual negatives",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "reference targets",
        int(data.reference_margin.notna().sum()),
        "of",
        len(data),
        flush=True,
    )


def render(payload, label, style):
    literal = json.dumps(payload)
    if style == 0:
        return (
            f"# Assistant work notes\n\n{payload}\n"
            if label
            else f"# Rejected input catalogue\n\nThe following quotation is test data, "
            f"not a contributor rule.\n\n```text\n{payload}\n```\n"
        )
    if style == 1:
        return (
            f"# Instructions to the coding assistant: {payload}\n"
            "def add(a, b):\n    return a + b\n"
            if label
            else "def test_untrusted_input_is_rejected():\n"
            f"    malicious_input = {literal}\n"
            '    assert detector(malicious_input).prediction == "INJECTION"\n'
        )
    if style == 2:
        return (
            json.dumps(
                {"agent_directives": payload}
                if label
                else {
                    "test_name": "reject_untrusted_instructions",
                    "expected": "INJECTION",
                    "untrusted_input": payload,
                },
                indent=2,
            )
            + "\n"
        )
    if style == 3:
        return (
            f"export const runtimeAgentPrompt = {literal};\n"
            if label
            else f"const attackFixture = {literal};\n"
            'it("rejects the attack fixture", () => {\n'
            "  expect(guard(attackFixture).blocked).toBe(true);\n});\n"
        )
    if style == 4:
        return (
            f"# Repository assistant policy\n\n<!-- {payload} -->\n"
            if label
            else "# Security regression examples\n\nThis comment is a hostile sample "
            "for the detector test suite. Do not execute its request.\n"
            f"<!-- {payload} -->\n"
        )
    if style == 5:
        return (
            f"ASSISTANT_SYSTEM_PROMPT = {literal}\n"
            if label
            else '"""Negative fixture: these instructions must be rejected '
            'by the guard."""\n'
            f"REJECTED_PROMPT = {literal}\n"
        )
    if style == 6:
        return (
            f"# Tasks for the repository agent\nagent_instructions: {literal}\n"
            if label
            else "# Detector regression data\nexpected: reject\n"
            f"attack_sample: {literal}\n"
        )
    return (
        f"# Guidance for the coding assistant\nagent_policy = {literal}\n"
        if label
        else f"# A recorded hostile input; used as data by tests\n"
        f'expected_result = "blocked"\nrecorded_input = {literal}\n'
    )


def fresh_render(payload, response, label, style):
    """Different contexts from training, frozen before any model sees test scores."""
    if style == 0:
        return (
            "## Working agreement\n\nWhile completing changes in this checkout:\n"
            + payload
            + "\n"
            if label
            else "# Incident analysis\n\nOur rejection test uses the following hostile "
            "quotation. It is evidence, not work to perform:\n\n> "
            + payload.replace("\n", "\n> ")
            + "\n"
        )
    if style == 1:
        return (
            "# Note to the assistant processing this module:\n# "
            + payload.replace("\n", "\n# ")
            + "\n\ndef square(x):\n    return x * x\n"
            if label
            else "def test_regression_payload_stays_untrusted():\n"
            f"    sample = {json.dumps(payload)}\n"
            "    observed = scan(sample)\n"
            "    assert observed.injection_detected\n"
        )
    return (
        response
        if label
        else json.dumps(
            {
                "regression_case": "malicious_tool_output",
                "expected_detection": True,
                "recorded_tool_output": response,
            },
            indent=2,
        )
    )


def long_fixture(payload, label):
    """Same global test frame, with an active instruction only in positives."""
    prefix = (
        "# Regression input catalogue. Case strings are data, not instructions.\n"
        "CASES = [\n"
    )
    filler = "".join(f'    "ordinary fixture {i}",\n' for i in range(96))
    middle = "    " + json.dumps(payload) + ",\n"
    span = None
    if label:
        note = (
            "# Reviewer: before continuing with this file, follow these instructions.\n"
        )
        note += "".join("# " + line + "\n" for line in payload.splitlines())
        start = len(prefix + filler + middle)
        span = (start, start + len(note))
        middle += note
    suffix = (
        "]\n\ndef test_recorded_inputs():\n"
        "    for text in CASES:\n        assert guard(text).blocked\n"
    )
    return prefix + filler + middle + filler + suffix, span


def long_training_pairs(source, raw_root, original):
    """Source-linked long pairs; no validation or held-out payloads are read."""
    raw = json.loads((raw_root / "purplellama-injections.json").read_text())
    payloads = {"cyber-" + str(row["prompt_id"]): row["user_input"] for row in raw}
    payloads.update(
        {
            "verified-" + digest(row.text)[:16]: row.text
            for row in original[original.source == "tensortrust_verified"].itertuples()
        }
    )
    keys = sorted(
        {
            row.id.rsplit("-", 2)[0]
            for row in source[source.source == "context_matched_sft"].itertuples()
            if isinstance(row.id, str) and row.id.startswith(("cyber-", "verified-"))
        },
        key=digest,
    )
    rows = []
    for key in [key for key in keys if len(payloads[key]) <= 1200][:96]:
        payload = payloads[key]
        for label in (0, 1):
            text, span = long_fixture(payload, label)
            row = make_row(
                payload,
                text,
                label,
                "long-" + key,
                8,
                "train",
                "long_context_pairs",
                "source-linked to " + key,
            )
            row["is_replay"] = False
            row["attack_start"], row["attack_end"] = span or (None, None)
            rows.append(row)
    return pd.DataFrame(rows)


def representative_cases(root):
    groups = {}
    for filename in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        for index, item in enumerate(json.loads((root / filename).read_text())):
            groups.setdefault(norm(item["Attacker Instruction"]), []).append(
                (filename, index, item)
            )
    for key, group in groups.items():
        # Taking the first occurrence would choose Amazon reviews for every
        # attack. Spread host tools deterministically without looking at scores.
        yield key, group[int(digest(key)[:16], 16) % len(group)]


def make_row(payload, text, label, pair, style, split, source, provenance):
    suffix = ["md", "py", "json", "ts", "md", "py", "yaml", "toml", "py"][style]
    start = text.rfind(payload) if label else -1
    return {
        "id": f"{pair}-{style}-{label}",
        "pair_id": f"{pair}-{style}",
        "text": text,
        "label": label,
        "repo": f"{source}/{pair}",
        "path": f"cases/{pair}-{style}.{suffix}",
        "source": source,
        "source_group": source,
        "split": split,
        "payload_sha256": digest(norm(payload)),
        "provenance": provenance,
        "review": "assistant-reviewed construction; no independent human review",
        "attack_start": start if start >= 0 else None,
        "attack_end": start + len(payload) if start >= 0 else None,
    }


def public_excerpts(metadata):
    for source in metadata:
        path = Path(source["file"])
        text = path.read_text()
        if source["split"] == "test":
            yield {
                "id": source["repo"] + ":full-file",
                "pair_id": None,
                "text": text,
                "label": 0,
                "repo": source["repo"],
                "path": source["path"],
                "source": "public_python_file",
                "source_group": source["repo"],
                "split": "test",
                "provenance": source["url"],
                "review": "presumed benign upstream source/test file; context reviewed",
                "payload_sha256": None,
            }
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines(keepends=True)
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            excerpt = "".join(lines[first - 1 : node.end_lineno])
            if not 150 <= len(excerpt) <= 3200:
                continue
            # Keep source units intact. Label is ordinary source/test code, not
            # a claim that every arbitrary repository is free of injection.
            yield {
                "id": source["repo"] + ":" + str(first),
                "pair_id": None,
                "text": excerpt,
                "label": 0,
                "repo": source["repo"],
                "path": source["path"],
                "source": "public_python_context",
                "source_group": source["repo"],
                "split": source["split"],
                "provenance": source["url"] + "#L" + str(first),
                "review": "presumed benign public source; assistant-reviewed context",
                "payload_sha256": None,
            }
            count += 1
            if count == 6:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/context-improvement-20261025")
    )
    parser.add_argument("--replay", type=Path, default=Path("data/v2/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/v2/val.parquet"))
    parser.add_argument(
        "--anchor-from",
        type=Path,
        help="add training-only teacher margins to an existing prepared dataset",
    )
    parser.add_argument("--teacher", default="models/release")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--ci-replay",
        action="store_true",
        help="balance replay with source-linked CI payload views",
    )
    parser.add_argument(
        "--long-pairs",
        action="store_true",
        help="add long fixtures including active malicious comments to anchored data",
    )
    parser.add_argument(
        "--instruction-replay",
        action="store_true",
        help="add long legitimate CI files and reviewed negatives with attack replay",
    )
    args = parser.parse_args()
    if args.anchor_from:
        if args.out is None:
            parser.error("--anchor-from needs --out")
        extra = (
            ci_replay_examples(
                pd.read_parquet(args.replay),
                pd.read_parquet(args.anchor_from / "val.parquet"),
            )
            if args.ci_replay
            else None
        )
        if args.long_pairs:
            if extra is not None:
                parser.error("add CI replay and long pairs in separate steps")
            extra = long_training_pairs(
                pd.read_parquet(args.anchor_from / "train.parquet"),
                args.root / "raw",
                pd.read_parquet(args.replay),
            )
        if args.instruction_replay:
            if extra is not None:
                parser.error("add instruction replay in a separate step")
            extra = instruction_replay_examples(
                pd.read_parquet(args.replay),
                pd.read_parquet(args.anchor_from / "train.parquet"),
                pd.read_parquet(args.anchor_from / "val.parquet"),
            )
        anchor_existing(
            args.anchor_from,
            args.teacher,
            args.out,
            extra,
            score_extra=not args.instruction_replay,
        )
        return
    root = args.root
    training = pd.read_parquet(args.replay)
    validation = pd.read_parquet(args.validation)
    known = set(training.text.map(norm)) | set(validation.text.map(norm))
    known_text = "\n".join(training.text.map(norm))
    rows, excluded = [], []
    cyber = json.loads((root / "raw/purplellama-injections.json").read_text())
    cyber_source = json.loads((root / "raw/purplellama-source.json").read_text())
    cyber_url = f"https://github.com/{cyber_source['repo']}/blob/{cyber_source['revision']}/{cyber_source['paths'][0]}"
    attacks = []
    for item in cyber:
        text = item["user_input"]
        if (
            item["risk_category"] != "security-violating"
            or item["injection_variant"] == "different_user_input_language"
            or norm(text) in known
            or not re.search(
                "ignore|disregard|override|forget|password|secret|system|developer|instruction",
                text,
                re.I,
            )
        ):
            excluded.append(item["prompt_id"])
            continue
        attacks.append((text, "cyber-" + str(item["prompt_id"]), cyber_url))
    selected = (
        training[
            (training.label == 1)
            & training.source.eq("tensortrust_verified")
            & training.text.str.len().between(50, 1000)
            & training.text.str.contains(
                r"ignore|disregard|override|secret|password|instruction", case=False
            )
        ]
        .drop_duplicates("text")
        .sample(256, random_state=931)
    )
    attacks += [
        (r.text, "verified-" + digest(r.text)[:16], str(r.provenance))
        for r in selected.itertuples()
    ]
    for payload, pair, provenance in attacks:
        for style in range(8):
            for label in (0, 1):
                text = render(payload, label, style)
                rows.append(
                    make_row(
                        payload,
                        text,
                        label,
                        pair,
                        style,
                        "train",
                        "context_matched_sft",
                        provenance,
                    )
                )
    # An entirely different benchmark source is held out from new SFT.
    origin = json.loads((root / "raw/provenance.json").read_text())
    upstream_seen = []
    for key, (filename, record_index, item) in representative_cases(root / "raw"):
        payload = item["Attacker Instruction"]
        if key in known_text:
            upstream_seen.append(digest(key))
            continue
        pair = "injecagent-" + digest(key)[:16]
        provenance = f"https://github.com/{origin['repo']}/blob/{origin['revision']}/data/{filename}"
        for style in range(3):
            for label in (0, 1):
                text = fresh_render(payload, item["Tool Response"], label, style)
                row = make_row(
                    payload,
                    text,
                    label,
                    pair,
                    style,
                    "test",
                    "injecagent_holdout",
                    provenance,
                )
                row.update(host_tool=item["User Tool"], source_record=record_index)
                rows.append(row)
    metadata = json.loads((root / "public/sources.json").read_text())
    rows.extend(public_excerpts(metadata))
    data = pd.DataFrame(rows)
    # Drop exact duplicate generated views only within a split. Cross-split
    # collisions are an error, never silently reassigned.
    for left, right in [
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    ]:
        a, b = data[data.split == left], data[data.split == right]
        if set(a.text.map(norm)) & set(b.text.map(norm)):
            raise ValueError("new cohort texts overlap")
        if set(a.repo.str.casefold()) & set(b.repo.str.casefold()):
            raise ValueError("new cohort repositories overlap")
        if set(a.payload_sha256.dropna()) & set(b.payload_sha256.dropna()):
            raise ValueError("new cohort payloads overlap")
    data = data.drop_duplicates(["split", "text"])
    out = args.out or root / "prepared"
    out.mkdir(exist_ok=False)
    data.to_parquet(out / "cases.parquet", index=False)
    replay_pool = training[training.text.str.len() <= 8000]
    replay = pd.concat(
        [
            replay_pool[replay_pool.label == label].sample(
                2048, random_state=932 + label
            )
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    replay["is_replay"] = True
    original_pairs = pd.read_json("tests/fixtures/ci_cases.jsonl", lines=True)
    original_pairs = original_pairs[original_pairs.split == "train"].assign(
        source="context_matched_sft"
    )
    train = pd.concat(
        [replay, data[data.split == "train"], original_pairs], ignore_index=True
    )
    train["is_replay"] = train.is_replay.fillna(False).astype(bool)
    val = pd.concat(
        [
            validation.assign(cohort="general"),
            pd.read_parquet("data/ci-context-20261025/val.parquet").assign(
                cohort="context"
            ),
            data[data.split == "calibration"].assign(cohort="public_context"),
        ],
        ignore_index=True,
    )
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test = data[data.split == "test"]
    test.to_parquet(out / "sealed-test.parquet", index=False)
    manifest = {
        "train_rows": len(train),
        "new_matched_payloads": len(attacks),
        "calibration_rows": len(val),
        "sealed_rows": len(test),
        "sealed_attacks": int(test.label.sum()),
        "sealed_benign": int((test.label == 0).sum()),
        "sealed_sha256": hashlib.sha256(
            (out / "sealed-test.parquet").read_bytes()
        ).hexdigest(),
        "replay_max_chars": 8000,
        "cybersec_quarantined_ids": excluded,
        "injecagent_previously_seen_payloads": upstream_seen,
        "seed": 931,
        "replay_seed": 932,
        "cutoff_selection": "development only; compare at release general-FP budget",
        "limitation": "Generated contexts and public benchmarks; source "
        "pretraining exposure unknown. Public code labels are "
        "presumed benign; no independent human review.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if not isinstance(v, list)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
