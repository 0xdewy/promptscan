#!/usr/bin/env python3
"""Build data/v2/{train,val}.parquet for the v2 classifier.

Label 1 = the text tries to override, extract or hijack the instructions an LLM
is running under (direct or planted in a document). Label 0 = everything else,
including harmful-but-honest requests, role-play, security talk and documents
that merely contain imperative sentences. That is the Prompt Guard 2 / eval-suite
definition, and it is what keeps false positives down.

Only train/validation splits of public sets are used. Every row is deduplicated
against data/eval/*.parquet (see eval_suite.py) so the held-out numbers stay honest.
Each source is cached in data/v2/raw/<name>.parquet; delete one to refetch it.

  python scripts/build_dataset.py            # all sources
  python scripts/build_dataset.py --only wildchat,bipia
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "v2" / "raw"
OUT = ROOT / "data" / "v2"
EVAL = ROOT / "data" / "eval"
LOCAL = ROOT / "data" / "hf_datasets"
rng = random.Random(0)


def log(msg: str) -> None:
    print(f"[build] {msg}", file=sys.stderr, flush=True)


def hf(name, config=None, split="train", streaming=False):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=streaming)


def take(stream, n: int, fn):
    """Pull up to n rows from an iterable dataset through fn (returns text or None)."""
    out = []
    for row in stream:
        t = fn(row)
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


def df(texts, label, **cols) -> pd.DataFrame:
    d = pd.DataFrame({"text": list(texts), "label": label})
    for k, v in cols.items():
        d[k] = v
    return d


# ---------------------------------------------------------------- sources
# Each returns DataFrame[text, label] (+ optional group for pair-aware splitting).

def src_deepset():
    d = hf("deepset/prompt-injections")
    return df(d["text"], d["label"])


def src_jackhhao():
    d = hf("jackhhao/jailbreak-classification")
    return df(d["prompt"], [int(t == "jailbreak") for t in d["type"]])


def src_slabs():
    d = hf("S-Labs/prompt-injection-dataset")
    return df(d["text"], d["label"])


def src_agentic():
    parts = [hf("3nesdeniz/agentic-prompt-injection-5k", split=s) for s in ("train", "validation")]
    return pd.concat([df(d["text"], [int(x) for x in d["label"]]) for d in parts])


def src_boundary():
    parts = [hf("3nesdeniz/agentic-prompt-injection-boundary-pairs", split=s) for s in ("train", "validation")]
    return pd.concat([df(d["text"], [int(x) for x in d["label"]], group=d["pair_id"]) for d in parts])


def src_xtram():
    # Its positives are mostly harmful requests or "write a poem starting with ..." prompts, not
    # injections, so only the benign rows are used.
    d = hf("xTRam1/safe-guard-prompt-injection")
    return df([t for t, l in zip(d["text"], d["label"]) if l == 0], 0)


def src_gandalf():
    return df(hf("Lakera/gandalf_ignore_instructions")["text"], 1)


def src_spml():
    # Its "injection" label means "user strays from the system prompt's topic", not an
    # override attempt, so only the system prompts are used, as benign text.
    d = hf("reshabhs/SPML_Chatbot_Prompt_Injection")
    return df(sorted(set(d["System Prompt"]))[:3000], 0)


def src_llmail():
    d = hf("microsoft/llmail-inject-challenge", split="Phase1")
    texts = set()
    for subject, body, raw in zip(
        d["subject"], d["body"], d["objectives"], strict=True
    ):
        outcomes = json.loads(raw)
        if body and all(
            outcomes.get(key) is True
            for key in ("exfil.sent", "exfil.destination", "exfil.content")
        ):
            texts.add(f"Subject: {subject}\n\n{body}".strip())
    return df(rng.sample(sorted(texts), min(5000, len(texts))), 1)



def src_hackaprompt():
    d = pd.read_parquet(LOCAL / "imoxto_prompt_injection_hackaprompt_gpt35.parquet")
    d = d.dropna(subset=["text"]).drop_duplicates("text")
    d = d[d["labels"].astype(int) == 1] if "labels" in d.columns else d
    return df(d["text"].astype(str).sample(min(5000, len(d)), random_state=0), 1)


def src_orbench():
    d = hf("bench-llm/or-bench", "or-bench-80k")
    return df(rng.sample(list(d["prompt"]), 8000), 0)


def src_awesome():
    return df(hf("fka/awesome-chatgpt-prompts")["prompt"], 0)


def src_oasst2():
    d = hf("OpenAssistant/oasst2")
    t = [x for x, role, parent in zip(d["text"], d["role"], d["parent_id"]) if role == "prompter" and parent is None]
    return df(rng.sample(t, min(10000, len(t))), 0)


def src_dolly():
    d = hf("databricks/databricks-dolly-15k")
    return df([f"{i}\n\n{c}".strip() if c else i for i, c in zip(d["instruction"], d["context"])], 0)


def _local(name, n):
    d = pd.read_parquet(LOCAL / name)
    col = next(c for c in ("text", "prompt", "instruction", "human_prompt", "user_input") if c in d.columns)
    t = d[col].dropna().astype(str)
    return df(rng.sample(sorted(set(t)), min(n, t.nunique())), 0)


def src_alpaca():
    # instruction + input only; the "text" column also holds the template wrapper and the answer
    d = pd.read_parquet(LOCAL / "tatsu-lab_alpaca.parquet")
    t = sorted({f"{i}\n\n{x}".strip() for i, x in zip(d["instruction"], d["input"].fillna(""))})
    return df(rng.sample(t, 8000), 0)


def src_code():
    return _local("pvduy_70k_evol_code_prompts.parquet", 5000)


def src_chatgpt_prompts():
    return _local("MohamedRashad_ChatGPT-prompts.parquet", 3000)


def src_marketing():
    return _local("marketeam_marketing_user_prompts_unfiltered.parquet", 3000)


def src_fineweb():
    s = hf("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)
    return df(take(s, 6000, lambda r: r["text"][:2000]), 0)


def src_wikipedia():
    s = hf("wikimedia/wikipedia", "20231101.en", streaming=True).shuffle(seed=0, buffer_size=5000)
    return df(take(s, 4000, lambda r: r["text"][:2000]), 0)


def src_emails():
    d = hf("Yale-LILY/aeslc")
    t = [f"Subject: {s}\n\n{b}" for s, b in zip(d["subject_line"], d["email_body"]) if b and len(b) < 3000]
    return df(rng.sample(t, min(6000, len(t))), 0)


INJECTION_LINES = None


def src_planted():
    """BIPIA-style: benign documents with an injection line planted at a random spot, plus clean twins."""
    docs = pd.concat([pd.read_parquet(RAW / f"{n}.parquet") for n in ("emails", "wikipedia", "fineweb", "dolly")
                      if (RAW / f"{n}.parquet").exists()])
    docs = [t for t in docs[docs.label == 0]["text"] if 200 < len(t) < 1500]
    lines = set()
    for n in ("gandalf", "deepset", "xtram", "slabs", "agentic"):
        p = RAW / f"{n}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            lines |= {t.strip() for t in d[d.label == 1]["text"] if 20 < len(t.strip()) < 300 and "\n" not in t.strip()}
    lines = sorted(lines)
    rows = []
    for i, doc in enumerate(rng.sample(docs, min(5000, len(docs)))):
        inj = rng.choice(lines)
        paras = doc.split("\n")
        k = rng.randint(0, len(paras))
        wrap = rng.choice(["{}", "\n{}\n", "<!-- {} -->", "[{}]", "Note to AI assistant: {}", "{} "])
        inserted = wrap.format(inj)
        start = sum(len(p) + 1 for p in paras[:k]) + inserted.index(inj)
        paras.insert(k, inserted)
        rows.append(("\n".join(paras), 1, f"plant{i}", start, start + len(inj)))
        rows.append((doc, 0, f"plant{i}", None, None))
    return pd.DataFrame(
        rows, columns=["text", "label", "group", "attack_start", "attack_end"]
    )


# The label audit found ordinary questions labelled as attacks in neuralchemy,
# and explicit jailbreaks labelled benign in inthewild and non-toxic WildChat.
# Keep those corpora out of new builds; benchmark populations stay unchanged.
SOURCES = {
    "deepset": src_deepset, "jackhhao": src_jackhhao, "slabs": src_slabs,
    "agentic": src_agentic, "boundary": src_boundary, "xtram": src_xtram, "gandalf": src_gandalf,
    "spml": src_spml, "llmail": src_llmail,
    "hackaprompt": src_hackaprompt, "orbench": src_orbench, "awesome": src_awesome,
    "oasst2": src_oasst2, "dolly": src_dolly, "alpaca": src_alpaca,
    "code": src_code, "chatgpt_prompts": src_chatgpt_prompts, "marketing": src_marketing,
    "fineweb": src_fineweb, "wikipedia": src_wikipedia, "emails": src_emails,
    "planted": src_planted,   # last: built from the cached raw sets above
}


# ---------------------------------------------------------------- assembly

def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def trim_texts(data):
    """Keep character annotations aligned when trimming source whitespace."""
    raw = data.text.astype(str)
    leading = raw.str.len() - raw.str.lstrip().str.len()
    data["text"] = raw.str.strip()
    for field in ("attack_start", "attack_end"):
        if field in data:
            data[field] = (data[field] - leading).clip(0, data.text.str.len())
    return data


def output_columns(data):
    fields = ("text", "label", "source", "group", "repo", "path",
              "attack_start", "attack_end")
    return [c for c in fields if c in data]


def group_keys(data: pd.DataFrame) -> pd.Series:
    """Join families that share a normalized text before deduplicating rows."""
    parents = {}

    def root(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    keys = data.text.map(norm)
    groups = data.get("group", pd.Series(None, index=data.index, dtype=object))
    repos = data.get("repo", pd.Series(None, index=data.index, dtype=object))
    for key, group, repo in zip(keys, groups, repos, strict=True):
        links = []
        if pd.notna(group):
            links.append(("group", str(group)))
        if pd.notna(repo):
            links.append(("group", "repo:" + str(repo).casefold()))
        for link in links:
            left, right = root(("text", key)), root(link)
            parents[max(left, right)] = min(left, right)
    return pd.Series(
        [root(("text", key))[1] if ("text", key) in parents else key for key in keys],
        index=data.index,
    )


def in_validation(group: str, fraction: float) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def extend_splits(train, val, extra, fraction, eval_keys):
    """Add grouped rows without moving any existing training or validation text."""
    train, val, extra = (d.copy() for d in (train, val, extra))
    for data in (train, val, extra):
        data["key"] = data.text.map(norm)
    train_keys, val_keys = set(train.key), set(val.key)
    if train_keys & val_keys:
        raise ValueError("base training and validation contain overlapping texts")

    def repositories(data):
        return set(data.repo.dropna().str.casefold()) if "repo" in data else set()

    train_repos, val_repos = repositories(train), repositories(val)
    if train_repos & val_repos:
        raise ValueError("base training and validation repositories overlap")
    extra["group"] = group_keys(extra)
    keep_train, keep_val = [], []
    for _, family in extra.groupby("group", sort=False):
        keys = set(family.key)
        repos = repositories(family)
        seen_train = bool(keys & train_keys or repos & train_repos)
        seen_val = bool(keys & val_keys or repos & val_repos)
        if keys & eval_keys or (seen_train and seen_val):
            log(f"excluded {len(family)} new rows: family crosses a held-out boundary")
            continue
        target = (
            keep_val
            if seen_val
            or (not seen_train and in_validation(family.group.iloc[0], fraction))
            else keep_train
        )
        target.append(family)

    def append(base, families):
        added = pd.concat(families, ignore_index=True) if families else extra.iloc[:0]
        combined = pd.concat([base, added], ignore_index=True)
        linked = combined["group"].notna() | combined.key.isin(added.key)
        merged = group_keys(combined)
        combined.loc[linked, "group"] = merged[linked]
        return combined.drop_duplicates("key")

    return append(train, keep_train), append(val, keep_val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated source names to (re)fetch")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--max-pwned", type=int, default=2500)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--extra", nargs="*", type=Path, default=[])
    ap.add_argument(
        "--base-data", type=Path,
        help="extend these fixed train/val splits instead of rebuilding sources",
    )
    a = ap.parse_args()
    if not 0 <= a.val < 1:
        ap.error("--val must be in [0, 1)")
    a.out.mkdir(parents=True, exist_ok=True)
    eval_keys = set()
    for p in EVAL.glob("*.parquet"):
        eval_keys.update(pd.read_parquet(p)["text"].astype(str).map(norm))
    if a.base_data:
        extra = pd.concat([pd.read_parquet(p) for p in a.extra], ignore_index=True) \
            if a.extra else pd.DataFrame(columns=["text", "label", "source", "group"])
        train, val = extend_splits(
            pd.read_parquet(a.base_data / "train.parquet"),
            pd.read_parquet(a.base_data / "val.parquet"), extra, a.val, eval_keys,
        )
        train[output_columns(train)].to_parquet(a.out / "train.parquet", index=False)
        val[output_columns(val)].to_parquet(a.out / "val.parquet", index=False)
        log(f"train {len(train)}, val {len(val)}; existing split membership preserved")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    wanted = a.only.split(",") if a.only else list(SOURCES)

    frames = []
    for name in SOURCES:
        p = RAW / f"{name}.parquet"
        if name in wanted or not p.exists():
            try:
                d = SOURCES[name]().dropna(subset=["text"])
                d = trim_texts(d)
                d["label"] = d["label"].astype(int)
                d = d[(d.text.str.len() >= 3) & (d.text.str.len() <= 20000)]
                d.to_parquet(p, index=False)
                log(f"{name}: {len(d)} rows, {d.label.mean():.0%} injection")
            except Exception as e:  # keep going; the summary says what is missing
                log(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
                continue
        if p.exists():
            d = pd.read_parquet(p)
            d["source"] = name
            frames.append(d)
    for p in [*sorted(RAW.glob("selfplay_*.parquet")),
              *sorted(RAW.glob("rl_*.parquet")), *a.extra]:
        data = pd.read_parquet(p)
        frames.append(data[output_columns(data)])

    all_ = pd.concat(frames, ignore_index=True)
    all_["group"] = group_keys(all_)
    all_["key"] = all_["text"].map(norm)
    before = len(all_)
    all_ = all_.drop_duplicates("key")
    all_ = all_[all_.key.str.len() > 0]
    log(f"dedup within train: {before} -> {len(all_)}")

    hit = all_.key.isin(eval_keys)
    heldout = all_.group.isin(all_.loc[hit, "group"])
    log(f"dropped {int(heldout.sum())} rows in families touching the eval sets")
    all_ = all_[~heldout]
    # HackAPrompt-style "I have been PWNED" rows were 20% of positives; cap them so one game doesn't define the class
    pwned = all_.index[(all_.label == 1) & all_.text.str.contains("PWNED", case=False)]
    all_ = all_.drop(pwned[a.max_pwned:])

    # pair-aware split: rows sharing a group land in the same split
    is_val = all_["group"].map(lambda group: in_validation(group, a.val)).astype(bool)
    val, train = all_[is_val], all_[~is_val]
    train[output_columns(train)].sample(frac=1, random_state=0).to_parquet(
        a.out / "train.parquet", index=False,
    )
    val[output_columns(val)].to_parquet(a.out / "val.parquet", index=False)
    log(f"train {len(train)} ({train.label.mean():.1%} injection), val {len(val)}")
    summary = all_.groupby(["source", "label"]).size().unstack(fill_value=0)
    summary = summary.reindex(columns=[0, 1], fill_value=0)
    summary.columns = ["safe", "injection"]
    print(summary.sort_values("injection", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
