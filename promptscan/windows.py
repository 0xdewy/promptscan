"""Shared tokenizer windows for inference and document-level training."""

from collections import defaultdict


def encode_windows(tokenizer, texts, max_length, context_tokens=0, offsets=False):
    kwargs = {
        "truncation": True,
        "max_length": max_length,
        "stride": max_length // 4,
        "return_overflowing_tokens": True,
        "return_offsets_mapping": offsets,
    }
    if not context_tokens:
        return tokenizer(texts, **kwargs)
    if not 0 <= context_tokens <= max_length // 4:
        raise ValueError("context_tokens must be between zero and max_length / 4")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    width = max_length - 2 * context_tokens
    if width - special <= kwargs["stride"]:
        raise ValueError(
            "context leaves insufficient room for overlapping body windows"
        )
    encoded = tokenizer(
        texts, **(kwargs | {"max_length": width, "return_special_tokens_mask": True})
    )
    groups = defaultdict(list)
    for index, owner in enumerate(encoded["overflow_to_sample_mapping"]):
        groups[owner].append(index)
    result = {"input_ids": [], "attention_mask": [], "overflow_to_sample_mapping": []}
    if offsets:
        result["offset_mapping"] = []

    def add(ids, owner, positions=None):
        result["input_ids"].append(ids)
        result["attention_mask"].append([1] * len(ids))
        result["overflow_to_sample_mapping"].append(owner)
        if offsets:
            result["offset_mapping"].append(positions)

    def bounds(index):
        content = [
            i
            for i, flag in enumerate(encoded["special_tokens_mask"][index])
            if not flag
        ]
        return (content[0], content[-1] + 1) if content else (0, 0)

    for owner, indices in groups.items():
        if len(indices) == 1:
            i = indices[0]
            add(
                encoded["input_ids"][i],
                owner,
                encoded["offset_mapping"][i] if offsets else None,
            )
            continue
        spans = [bounds(i) for i in indices]
        length = sum(right - left for left, right in spans) - kwargs["stride"] * (
            len(indices) - 1
        )
        if length + special <= max_length:
            # Inputs that fit in one ordinary window keep their original encoding.
            short = tokenizer([texts[owner]], **kwargs)
            add(
                short["input_ids"][0],
                owner,
                short["offset_mapping"][0] if offsets else None,
            )
            continue
        first, last = indices[0], indices[-1]
        a, b = spans[0]
        c, d = spans[-1]
        prefix = encoded["input_ids"][first][a:b][:context_tokens]
        suffix = encoded["input_ids"][last][c:d][-context_tokens:]
        if offsets:
            prefix_offsets = encoded["offset_mapping"][first][a:b][:context_tokens]
            suffix_offsets = encoded["offset_mapping"][last][c:d][-context_tokens:]
        for i, (left, right) in zip(indices, spans, strict=True):
            ids = encoded["input_ids"][i]
            framed = ids[:left] + prefix + ids[left:right] + suffix + ids[right:]
            positions = None
            if offsets:
                original = encoded["offset_mapping"][i]
                positions = (
                    original[:left]
                    + prefix_offsets
                    + original[left:right]
                    + suffix_offsets
                    + original[right:]
                )
            add(framed, owner, positions)
    return result
