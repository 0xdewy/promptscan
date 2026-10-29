#!/usr/bin/env python3
"""Collect public-game secret leaks, independently of the access-grant flag.

Raw files contain synthetic TensorTrust game passwords, never local credentials.
Outputs retain attack text and hashed outcome evidence, not passwords/responses.
Use explicit_protected_scope to select the conservative paired-training cohort.
"""

import argparse
import bz2
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_dataset import norm  # noqa: E402

REVISION = "747a75e096761ebc01bd3970158827326b4add23"
BLOBS = {
    "raw_dump_attacks.jsonl.bz2": "1ed666b8aa7083a1698bdbd44158ca72d8315d04",
    "raw_dump_attacks_v2.jsonl.bz2": "3d8e42efce1b8ea9a7a36e3a78832b95e16b8c7b",
}
REQUEST = re.compile(
    r"\b(?:show|print|reveal|return|repeat|list|output|write|give|tell|echo|translate|"
    r"decode|display|describe|summari[sz]e)\b.{0,220}\b(?:password|secret|access.?code|"
    r"system.?prompt|initial.?instructions|hidden.?instructions|your.?instructions)\b",
    re.I | re.S,
)
NEGATED = re.compile(
    r"\b(?:do not|don't|never|avoid|cannot|must not|should not)\b", re.I
)
PROTECTED = re.compile(
    r"\b(?:your (?:secret |system |initial |hidden )?(?:prompt|password|instructions|"
    r"access code)|(?:original|initial|hidden|confidential) (?:system )?"
    r"(?:prompt|instructions|password)|actual access code)\b",
    re.I,
)
WORDS = re.compile(r"\b[a-z]+\b", re.I)
ENGLISH = set(
    "the a an your you my me of and in to from with is are this that please do "
    "not can will must what how output show repeat give tell return secret "
    "password instructions".split()
)


def verified_leak(row):
    """Require observed hidden-code disclosure and an affirmative extraction ask."""
    text, code, reply = (
        row.get(k) or "" for k in ("attacker_input", "access_code", "llm_output")
    )
    if not all(isinstance(v, str) for v in (text, code, reply)):
        return False
    if row.get("is_self_attack") or not 80 <= len(text) <= 4000:
        return False
    if not 6 <= len(code) <= 80 or len(set(code.casefold())) < 4:
        return False
    if code.casefold() not in reply.casefold() or code.casefold() in text.casefold():
        return False
    if re.sub(r"\W", "", code.casefold()) in re.sub(r"\W", "", text.casefold()):
        return False
    defense = (
        (row.get("opening_defense") or "") + " " + (row.get("closing_defense") or "")
    )
    if code.casefold() not in defense.casefold():
        return False
    if not any(
        not NEGATED.search(text[max(0, m.start() - 60) : m.start()])
        for m in REQUEST.finditer(text)
    ):
        return False
    tokens = WORDS.findall(text.lower())
    if len(tokens) < 8 or sum(w in ENGLISH for w in tokens) / len(tokens) < 0.15:
        return False
    if re.search("[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]", text):
        return False
    return not re.search(
        r"sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|BEGIN (?:OPENSSH|RSA) PRIVATE KEY", text
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--reserved-rendered", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    blocked, exclusion_hashes = set(), {}
    for p in args.exclude:
        exclusion_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        blocked.update(pd.read_parquet(p, columns=["text"]).text.map(norm))
    for p in args.raw_dir.glob("*_robustness_dataset.jsonl"):
        exclusion_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        blocked.update(norm(json.loads(line)["attack"]) for line in p.open())
    reserved, reserved_hashes = [], {}
    for p in args.reserved_rendered:
        reserved_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        reserved.extend(pd.read_parquet(p, columns=["text"]).text.map(norm))
    args.out.mkdir(parents=True, exist_ok=False)
    rows, sources, seen = [], [], set()
    for name, expected in BLOBS.items():
        p = args.raw_dir / name
        if not p.exists():
            continue
        digest = hashlib.sha1(f"blob {p.stat().st_size}\0".encode())
        sha = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                sha.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"raw source differs from pinned upstream blob: {p}")
        version = "v2" if "_v2" in name else "v1"
        sources.append(
            {
                "path": str(p),
                "blob_sha1": expected,
                "sha256": sha.hexdigest(),
                "url": f"https://raw.githubusercontent.com/HumanCompatibleAI/"
                f"tensor-trust-data/{REVISION}/raw-data/{version}/raw_dump_attacks.jsonl.bz2",
            }
        )
        with bz2.open(p, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if not verified_leak(row):
                    continue
                author = str(row["attacker_id_anonymized"])
                bucket = (
                    int(
                        hashlib.sha256(
                            ("ci-20261025:attacker:" + author).encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    % 100
                )
                text, code = row["attacker_input"], row["access_code"]
                key = norm(text)
                if bucket >= 65 or key in blocked or key in seen:
                    continue
                if any(key in held or held in key for held in reserved):
                    continue
                seen.add(key)
                rows.append(
                    {
                        "text": text,
                        "label": 1,
                        "source": "tensortrust_verified_extraction",
                        "repo": "tensortrust:" + author,
                        "group": "tensortrust:" + author,
                        "path": "AGENTS.md",
                        "split": "train",
                        "original_attack_id": row["attack_id"],
                        "provenance": "HumanCompatibleAI/tensor-trust-data@" + REVISION,
                        "success_evidence": (
                            "Published synthetic access code appears in defender "
                            "output, was hidden in defense, and was absent from "
                            "attacker input; not local agent success"
                        ),
                        "canary_sha256": hashlib.sha256(code.encode()).hexdigest(),
                        "canary_length": len(code),
                        "game_access_granted": bool(
                            row.get("output_is_access_granted")
                        ),
                        "explicit_protected_scope": bool(PROTECTED.search(text)),
                    }
                )
    if not sources:
        raise ValueError("no pinned public raw sources found")
    for p in args.reserved_rendered:
        if hashlib.sha256(p.read_bytes()).hexdigest() != reserved_hashes[str(p)]:
            raise ValueError("reserved inputs changed during the overlap check")
    data = pd.DataFrame(rows)
    data.to_parquet(args.out / "attacks.parquet", index=False)
    report = {
        "rows": len(data),
        "source_commit": REVISION,
        "sources": sources,
        "exclusion_sha256": exclusion_hashes,
        "reserved_used_for_overlap_only": reserved_hashes,
        "outcome": "synthetic secret disclosure; access-grant flag is irrelevant",
        "local_native_success_claimed": False,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
