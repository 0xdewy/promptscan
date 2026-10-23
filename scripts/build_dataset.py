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


def src_neuralchemy():
    parts = [hf("neuralchemy/Prompt-injection-dataset", "full", s) for s in ("train", "validation")]
    return pd.concat([df(d["text"], [int(x) for x in d["label"]], group=d["group_id"]) for d in parts])


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


def src_inthewild():
    jb = hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
    reg = hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25")
    reg_texts = rng.sample(list(reg["prompt"]), min(4000, len(reg)))
    return pd.concat([df(jb["prompt"], 1), df(reg_texts, 0)])


def src_llmail():
    d = hf("microsoft/llmail-inject-challenge", split="Phase1")
    # keep only submissions that made the agent call send_email: proven injections, not near-misses
    texts = {f"Subject: {s}\n\n{b}".strip() for s, b, o in zip(d["subject"], d["body"], d["objectives"])
             if b and json.loads(o).get("exfil.sent")}
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


def src_wildchat():
    s = hf("allenai/WildChat-1M", streaming=True).shuffle(seed=0, buffer_size=20000)
    texts = take(s, 40000, lambda r: None if r.get("toxic") else (r["conversation"][0]["content"] or "")[:2000])
    return df(texts, 0)


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
        paras.insert(k, wrap.format(inj))
        rows.append(("\n".join(paras), 1, f"plant{i}"))
        rows.append((doc, 0, f"plant{i}"))
    return pd.DataFrame(rows, columns=["text", "label", "group"])


SOURCES = {
    "deepset": src_deepset, "jackhhao": src_jackhhao, "slabs": src_slabs, "neuralchemy": src_neuralchemy,
    "agentic": src_agentic, "boundary": src_boundary, "xtram": src_xtram, "gandalf": src_gandalf,
    "spml": src_spml, "inthewild": src_inthewild, "llmail": src_llmail,
    "hackaprompt": src_hackaprompt, "orbench": src_orbench, "awesome": src_awesome,
    "wildchat": src_wildchat, "oasst2": src_oasst2, "dolly": src_dolly, "alpaca": src_alpaca,
    "code": src_code, "chatgpt_prompts": src_chatgpt_prompts, "marketing": src_marketing,
    "fineweb": src_fineweb, "wikipedia": src_wikipedia, "emails": src_emails,
    "planted": src_planted,   # last: built from the cached raw sets above
}


# ---------------------------------------------------------------- assembly

def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated source names to (re)fetch")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--max-pwned", type=int, default=2500)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    wanted = a.only.split(",") if a.only else list(SOURCES)

    frames = []
    for name in SOURCES:
        p = RAW / f"{name}.parquet"
        if name in wanted or not p.exists():
            try:
                d = SOURCES[name]().dropna(subset=["text"])
                d["text"] = d["text"].astype(str).str.strip()
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
    for p in sorted(RAW.glob("selfplay_*.parquet")):   # written by selfplay.py
        frames.append(pd.read_parquet(p)[["text", "label", "source"]])

    all_ = pd.concat(frames, ignore_index=True)
    if "group" not in all_:
        all_["group"] = None
    all_["key"] = all_["text"].map(norm)
    before = len(all_)
    all_ = all_.drop_duplicates("key")
    all_ = all_[all_.key.str.len() > 0]
    log(f"dedup within train: {before} -> {len(all_)}")

    eval_keys = set()
    for p in EVAL.glob("*.parquet"):
        eval_keys |= set(pd.read_parquet(p)["text"].astype(str).map(norm))
    hit = all_.key.isin(eval_keys)
    log(f"dropped {int(hit.sum())} rows that appear in the eval sets")
    all_ = all_[~hit]
    # HackAPrompt-style "I have been PWNED" rows were 20% of positives; cap them so one game doesn't define the class
    pwned = all_.index[(all_.label == 1) & all_.text.str.contains("PWNED", case=False)]
    all_ = all_.drop(pwned[a.max_pwned:])

    # pair-aware split: rows sharing a group land in the same split
    gk = all_["group"].fillna(all_["key"]).astype(str)
    h = gk.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    val = all_[h < a.val]
    train = all_[h >= a.val]
    cols = ["text", "label", "source"]
    train[cols].sample(frac=1, random_state=0).to_parquet(OUT / "train.parquet", index=False)
    val[cols].to_parquet(OUT / "val.parquet", index=False)
    log(f"train {len(train)} ({train.label.mean():.1%} injection), val {len(val)}")
    summary = all_.groupby(["source", "label"]).size().unstack(fill_value=0)
    summary.columns = ["safe", "injection"][:len(summary.columns)]
    print(summary.sort_values("injection", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
