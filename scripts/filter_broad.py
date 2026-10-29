#!/usr/bin/env python3
"""Keep the broad regression set on-target: English text, no chatbot-prompt dataset.

promptscan is English-only and scans files agents read. ToxicChat is user prompts
to a chatbot (role-play, jailbreaks), not agent-read files, so it is dropped whole.
Labels are untouched; rows are only removed.

usage: uv run --with langdetect python scripts/filter_broad.py IN.parquet OUT.parquet
"""

import json
import sys

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0


def english(text):
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(not c.isascii() for c in letters) > 0.3 * len(letters):
        return False
    if len(text) < 80:  # too short for langdetect; it misreads short English
        return True
    try:
        top = detect_langs(text)[0]
    except LangDetectException:
        return True
    return top.lang == "en" or top.prob < 0.9


data = pd.read_parquet(sys.argv[1])
chat = data.source == "toxicchat"
keep = ~chat & data.text.map(english)
data[keep].to_parquet(sys.argv[2], index=False)
dropped = data[~keep].groupby(["source", "label"]).size().unstack(fill_value=0)
print(json.dumps({"rows": len(data), "kept": int(keep.sum()),
                  "dropped_by_source": dropped.to_dict("index")}, indent=1))
