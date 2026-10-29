#!/usr/bin/env python3
"""Build reviewed hard negatives with injected siblings and balanced replay."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_context_pairs import make_row, render  # noqa: E402
from scripts.build_dataset import norm  # noqa: E402


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def insert_instruction(text, payload):
    """Use a middle paragraph outside fenced code; retain all original content."""
    offset, fence, choices = 0, None, []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else marker if fence is None else fence
        offset += len(line)
        if not fence and not line.strip():
            choices.append(offset)
    point = (
        min(choices, key=lambda n: abs(n - len(text) // 2)) if choices else len(text)
    )
    prefix, suffix = "\n<!--\n", "\n-->\n"
    note = prefix + payload + suffix
    start = point + len(prefix)
    return text[:point] + note + text[point:], (start, start + len(payload))


def active_prompt_pair(row, style):
    """Both labels use the same active field; only their content differs."""
    body = row["text"]
    text = render(body, 1, style)
    start = end = None
    if row["label"]:
        payload = body[int(row["attack_start"]) : int(row["attack_end"])]
        needle = payload if style in {0, 1, 4} else json.dumps(payload)[1:-1]
        start = text.rfind(needle)
        if start < 0 or not needle:
            raise ValueError("active field lost its malicious payload")
        end = start + len(needle)
    suffix = ["md", "py", "json", "ts", "md", "py", "yaml", "toml"][style]
    return row | {
        "text": text,
        "source": "active_prompt_pair",
        "reference_margin": np.nan,
        "id": f"{row['id']}-active-{style}",
        "pair_id": f"{row['pair_id']}-active-{style}",
        "path": f"active-pairs/{row['id']}-{style}.{suffix}",
        "attack_start": start,
        "attack_end": end,
    }


def save_dataset(out, train, validation, manifest):
    if set(train.text.map(norm)) & set(validation.text.map(norm)):
        raise ValueError("training overlaps validation")
    out.mkdir(parents=True, exist_ok=False)
    train.to_parquet(out / "train.parquet", index=False)
    validation.to_parquet(out / "val.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def build_active_fields(source, out, repeat=4):
    train = pd.read_parquet(source / "train.parquet")
    validation = pd.read_parquet(source / "val.parquet")
    pairs = train[train.source.eq("reviewed_hard_pair")].drop_duplicates(
        ["pair_id", "label"]
    )
    complete = pairs.groupby("pair_id").label.apply(
        lambda values: set(values) == {0, 1}
    )
    if pairs.empty or not complete.all():
        raise ValueError("active fields need complete reviewed pairs")
    active = pd.DataFrame(
        [
            active_prompt_pair(row, style)
            for row in pairs.to_dict("records")
            for style in range(8)
        ]
    )
    pool = train[train.is_replay].drop_duplicates("text")
    replay = pd.concat(
        [
            pool[pool.label == label].sample(1024, random_state=943 + label)
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    context = train[
        train.source.isin(["context_matched_sft", "long_context_pairs"])
    ].drop_duplicates(["text", "label"])
    result = pd.concat([replay, context] + [active] * repeat, ignore_index=True)
    save_dataset(
        out,
        result,
        validation,
        {
            "source": str(source),
            "rows": len(result),
            "new_active_pairs": len(active) // 2,
            "reviewed_source_files": len(pairs) // 2,
            "repeat": repeat,
            "replay": len(replay),
            "retained_context": len(context),
            "heldout_used": False,
            "rule": "identical active format for both labels; one sibling adds an "
            "existing malicious training payload; span excludes wrapper",
        },
    )


def build_goal_replay(source, original, out, repeat=2):
    base = pd.read_parquet(source / "train.parquet")
    validation = pd.read_parquet(source / "val.parquet")
    original = pd.read_parquet(original)
    payloads = []
    eligible = original[
        original.label.eq(1)
        & original.attack_start.notna()
        & original.attack_end.notna()
    ]
    for row in eligible.itertuples():
        payload = row.text[int(row.attack_start) : int(row.attack_end)]
        if 30 <= len(payload) <= 1600 and re.search(
            r"\.ssh/id_rsa|collector\.example\.com", payload, re.I
        ):
            payloads.append(payload)
    payloads = list(dict.fromkeys(payloads))
    if not payloads:
        raise ValueError("no original training payloads for the sandbox goals")
    rows = []
    for payload in payloads:
        key = "native-goal-" + digest(payload)[:16]
        for style in range(8):
            for label in (0, 1):
                text = render(payload, label, style)
                row = make_row(
                    payload,
                    text,
                    label,
                    key,
                    style,
                    "train",
                    "goal_context_pair",
                    "original training native-tool pilot; attempted injections, "
                    "not all successful",
                )
                if label:
                    needle = (
                        payload if style in {0, 1, 4} else json.dumps(payload)[1:-1]
                    )
                    start = text.rfind(needle)
                    if start < 0:
                        raise ValueError("goal payload span was lost")
                    row["attack_start"], row["attack_end"] = start, start + len(needle)
                row.update(reference_margin=np.nan, is_replay=False)
                rows.append(row)
    goals = pd.DataFrame(rows)
    active = base[base.source.eq("active_prompt_pair")].drop_duplicates(
        ["pair_id", "label"]
    )
    safe, bad = active[active.label == 0], active[active.label == 1]
    if active.empty:
        raise ValueError("goal replay needs harmless active-field controls")
    replay = base[base.is_replay].drop_duplicates("text")
    context = base[
        base.source.isin(["context_matched_sft", "long_context_pairs"])
    ].drop_duplicates(["text", "label"])
    authored = pd.read_json("tests/fixtures/ci_cases.jsonl", lines=True)
    authored = authored[authored.split == "train"].assign(
        source="contract_replay", is_replay=False, reference_margin=np.nan
    )
    train = pd.concat(
        [replay, context, bad, safe, safe] + [goals] * repeat + [authored] * 8,
        ignore_index=True,
    )
    save_dataset(
        out,
        train,
        validation,
        {
            "source": str(source),
            "rows": len(train),
            "native_goal_attempts": len(payloads),
            "new_goal_pairs": len(goals) // 2,
            "goal_repeat": repeat,
            "authored_train_repeat": 8,
            "replay": len(replay),
            "retained_context": len(context),
            "active_malicious": len(bad),
            "active_benign": len(safe) * 2,
            "heldout_used": False,
            "note": "Existing native pilot attempts only; no new native outcomes "
            "claimed. Failed attempts remain positive. Both explicit quotations and "
            "harmless active fields are negative.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path("data/v2/train.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--active-fields", action="store_true")
    mode.add_argument("--goal-replay", action="store_true")
    args = parser.parse_args()
    default_repeat = 4 if args.active_fields else 2 if args.goal_replay else 16
    repeat = args.repeat if args.repeat is not None else default_repeat
    if repeat < 1:
        parser.error("--repeat must be positive")
    if args.active_fields:
        if args.scores or args.review:
            parser.error("--active-fields uses existing reviewed pairs from --data")
        build_active_fields(args.data, args.out, repeat)
        return
    if args.goal_replay:
        if args.scores or args.review:
            parser.error("--goal-replay reuses existing original training payloads")
        build_goal_replay(args.data, args.original, args.out, repeat)
        return
    if not args.scores or not args.review:
        parser.error("need --scores and --review, or --active-fields")
    scored = pd.read_parquet(args.scores)
    reviews = [json.loads(line) for line in args.review.read_text().splitlines()]
    if len({r["text_sha256"] for r in reviews}) != len(reviews):
        raise ValueError("duplicate review entries")
    original = pd.read_parquet(args.original)
    allowed = set(original.text.map(norm))
    reviewed = {r["text_sha256"]: r for r in reviews}
    blocked = {r["text_sha256"] for r in reviews if r["label"] is None}
    blocked_text = set(scored[scored.text.map(digest).isin(blocked)].text.map(norm))
    hard = scored[scored.text.map(digest).isin(reviewed)].copy()
    hard["review_label"] = hard.text.map(lambda t: reviewed[digest(t)]["label"])
    hard = hard[
        hard.review_label.eq(0) & ~hard.text.map(norm).isin(blocked_text)
    ].drop_duplicates("text")
    hard = hard.loc[~hard.text.map(norm).duplicated()]
    if hard.empty or not hard.label.eq(0).all():
        raise ValueError("expected reviewed original training negatives")
    if not set(hard.text.map(norm)) <= allowed:
        raise ValueError("mining includes texts outside original training")
    train = pd.read_parquet(args.data / "train.parquet")
    val = pd.read_parquet(args.data / "val.parquet")
    train = train[
        ~train.text.map(norm).isin(blocked_text) & ~train.text.str.contains("\x00")
    ].copy()
    candidates = train[
        train.id.fillna("").str.startswith(("cyber-", "verified-"))
        & train.id.fillna("").str.endswith("-0-1")
    ]
    payloads = [
        row.text.removeprefix("# Assistant work notes\n\n").strip()
        for row in candidates.itertuples()
    ]
    payloads = sorted({p for p in payloads if 80 <= len(p) <= 1200}, key=digest)
    if not payloads:
        raise ValueError("need already-trained payloads for paired injections")
    pairs = []
    for row in hard.to_dict("records"):
        key = digest(row["text"])
        payload = payloads[int(key[:16], 16) % len(payloads)]
        for label in (0, 1):
            text, span = (
                insert_instruction(row["text"], payload)
                if label
                else (row["text"], (None, None))
            )
            pairs.append(
                row
                | {
                    "id": f"hard-{key[:16]}-{label}",
                    "pair_id": "hard-" + key[:16],
                    "text": text,
                    "label": label,
                    "source": "reviewed_hard_pair",
                    "attack_start": span[0],
                    "attack_end": span[1],
                    "reference_margin": np.nan,
                    "is_replay": False,
                    "review": (
                        "benign source with an existing attack inserted as an active "
                        "assistant instruction"
                        if label
                        else reviewed[key]["reason"]
                    ),
                    "payload_sha256": digest(payload),
                }
            )
    hard = pd.DataFrame(pairs).drop(
        columns=["score", "hidden", "seconds", "review_label"]
    )
    known = set(hard.text.map(norm))
    replay = train[train.is_replay & ~train.text.map(norm).isin(known)]
    replay = pd.concat(
        [
            replay[replay.label == label].sample(2048, random_state=938 + label)
            for label in (0, 1)
        ]
    )
    context = train[train.source.eq("context_matched_sft")]
    context = pd.concat(
        [
            context[context.label == label].sample(128, random_state=940 + label)
            for label in (0, 1)
        ]
    )
    authored = train[
        train.source.eq("context_matched_sft")
        & ~train.id.fillna("").str.startswith(("cyber-", "verified-"))
    ]
    result = pd.concat(
        [replay, context, authored, train[train.source.eq("long_context_pairs")]]
        + [hard] * repeat,
        ignore_index=True,
    )
    manifest = {
        "source": str(args.data),
        "rows": len(result),
        "unique_hard_pairs": len(hard) // 2,
        "repeat": repeat,
        "quarantined_reviews": len(blocked),
        "heldout_used": False,
        "review_sha256": digest(args.review.read_text()),
        "mining_precision": "bf16 for ranking only; validation remains fp32",
        "limitation": "Assistant-reviewed presumed-benign training examples; "
        "no independent human review. New positives use existing training payloads.",
    }
    save_dataset(args.out, result, val, manifest)


if __name__ == "__main__":
    main()
