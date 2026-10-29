#!/usr/bin/env python3
"""Combine complete CI counterfactuals with context pairs and original replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_dataset import norm  # noqa: E402
from scripts.refine_hard_examples import insert_instruction, save_dataset  # noqa: E402


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path("data/v2/train.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    original = pd.read_parquet(args.original)
    base = pd.read_parquet(root / "anchored/train.parquet")
    validation = pd.read_parquet(root / "goal-replay/val.parquet")
    reviews = [
        json.loads(line)
        for line in (root / "hard-negative-review.jsonl").read_text().splitlines()
    ]
    blocked = {r["text_sha256"] for r in reviews if r["label"] is None}
    blocked_repos = {
        r["repo"].casefold()
        for r in reviews
        if r["label"] is None and isinstance(r.get("repo"), str) and r["repo"] != "nan"
    }
    forbidden = set(validation.repo.dropna().str.casefold())
    for file in [Path("data/ci-iterations-yro7lzzv/test.parquet")]:
        data = pd.read_parquet(file)
        for column in ("repo", "host_repo"):
            forbidden.update(data[column].dropna().str.casefold())
    forbidden.update(
        row["repo"].casefold()
        for row in json.loads((root / "public/sources.json").read_text())
        if row["split"] != "train"
    )
    known_validation = set(validation.text.map(norm))
    excluded_repos = forbidden | blocked_repos

    def eligible(row):
        text = row["text"]
        identities = [row.get("repo"), row.get("host_repo")]
        return (
            text.strip()
            and "\x00" not in text
            and len(text.encode()) <= 1_000_000
            and digest(text) not in blocked
            and norm(text) not in known_validation
            and not any(
                isinstance(value, str) and value.casefold() in excluded_repos
                for value in identities
            )
        )

    base = base[[bool(eligible(row)) for row in base.to_dict("records")]]
    original = original[[bool(eligible(row)) for row in original.to_dict("records")]]
    references = {
        digest(row.text): row.reference_margin
        for row in base.itertuples()
        if pd.notna(row.reference_margin)
    }
    cropped = pd.read_parquet(root / "anchored-ci/train.parquet")
    crop_references = {
        row.source_document_sha256: row.reference_margin
        for row in cropped[cropped.source.eq("ci_payload_replay")].itertuples()
        if pd.notna(row.reference_margin)
    }
    rows = []
    for row in original[original.source.eq("ci_planted_verified")].to_dict("records"):
        start, end = int(row["attack_start"]), int(row["attack_end"])
        text = row["text"]
        if not 0 <= start < end <= len(text):
            raise ValueError("invalid original CI payload span")
        clean = text[:start] + text[end:]
        if not clean.strip() or norm(clean) in known_validation:
            continue
        key = digest(text)
        for label, content in [(1, text), (0, clean)]:
            rows.append(
                row
                | {
                    "id": f"full-{key[:16]}-{label}",
                    "pair_id": "full-" + key[:16],
                    "text": content,
                    "label": label,
                    "source": "ci_full_pair",
                    "attack_start": start if label else None,
                    "attack_end": end if label else None,
                    "is_replay": True,
                    "reference_margin": crop_references.get(key, np.nan)
                    if label
                    else references.get(digest(clean), np.nan),
                }
            )

    native = pd.read_parquet(root / "goal-directed-payloads.parquet").text.tolist()
    game = (
        original[
            original.source.eq("tensortrust_verified")
            & original.text.str.len().between(30, 1600)
            & original.text.str.contains(
                r"ignore|disregard|forget|override|system prompt", case=False
            )
            & ~original.text.str.contains(r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff]")
        ]
        .text.drop_duplicates()
        .tolist()
    )
    payloads = native + game
    clean_pool = original[original.source.eq("ci_repository")].copy()
    security = clean_pool[
        clean_pool.text.str.contains(
            r"AWS_SECRET_ACCESS_KEY|PRIVATE_KEY|private key|credentials|password|token",
            case=False,
        )
    ]
    security = security.sample(min(256, len(security)), random_state=945)
    ordinary = clean_pool[~clean_pool.index.isin(security.index)].sample(
        256, random_state=946
    )

    def pair(row, key, payload, source):
        text, span = insert_instruction(row["text"], payload)
        for label, content in [(0, row["text"]), (1, text)]:
            rows.append(
                row
                | {
                    "id": f"{key}-{label}",
                    "pair_id": key,
                    "text": content,
                    "label": label,
                    "source": source,
                    "is_replay": False,
                    "attack_start": span[0] if label else None,
                    "attack_end": span[1] if label else None,
                    "reference_margin": np.nan,
                }
            )

    for row in pd.concat([security, ordinary]).to_dict("records"):
        key = digest(row["text"])
        pair(
            row,
            "complete-" + key[:16],
            payloads[int(key[:16], 16) % len(payloads)],
            "ci_complete_pair",
        )
    long_sources = pd.read_parquet(root / "long-training-blobs-verified.parquet")
    long_sources = long_sources[long_sources.verified]
    for row in long_sources.to_dict("records"):
        record = {
            "text": row["content"],
            "repo": row["repo"],
            "path": row["file_path"],
            "provenance": row["verified_url"],
            "review": "presumed benign project guidance; assistant context review",
        }
        if not eligible(record):
            continue
        key = digest(record["text"])
        for variant in range(4):
            payload = payloads[(int(key[:16], 16) + variant * 131) % len(payloads)]
            for _ in range(4):
                pair(record, f"long-{key[:16]}-{variant}", payload, "long_ci_pair")
    additions = pd.DataFrame(rows)
    goal = pd.read_parquet(root / "goal-replay/train.parquet")
    goal = goal[goal.source.eq("goal_context_pair")].drop_duplicates("id")
    active = pd.read_parquet(root / "active-prompt-refinement/train.parquet")
    active = active[
        active.source.isin(["active_prompt_pair", "long_context_pairs"])
    ].drop_duplicates("id")
    hard = pd.read_parquet(root / "hard-refinement-v3/train.parquet")
    hard = hard[hard.source.eq("reviewed_hard_pair")].drop_duplicates("id")
    authored = pd.read_json("tests/fixtures/ci_cases.jsonl", lines=True)
    authored = authored[authored.split.eq("train")].assign(
        source="contract_replay", is_replay=False, reference_margin=np.nan
    )
    train = pd.concat(
        [base, additions, goal, active] + [hard] * 4 + [authored] * 7, ignore_index=True
    )
    train = train[[bool(eligible(row)) for row in train.to_dict("records")]]
    save_dataset(
        args.out,
        train,
        validation,
        {
            "original": str(args.original),
            "rows": len(train),
            "sources": train.source.value_counts().to_dict(),
            "maximum_characters": int(train.text.str.len().max()),
            "window_policy": "complete documents; microbatched forwards; "
            "no character sampling cap",
            "heldout_used": False,
            "reference_margin_limit": 6,
            "labels": "Original training, verified-source presumed-benign guides, "
            "and controlled counterfactuals; no independent human review",
        },
    )


if __name__ == "__main__":
    main()
