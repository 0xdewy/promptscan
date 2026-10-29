#!/usr/bin/env python3
"""Add paired legitimate project directives and existing attempted injections."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.refine_hard_examples import save_dataset  # noqa: E402

# These are authored project rules, not labels inferred from a model's score.
DIRECTIVES = [
    "Use UTC timestamps in logs. Redact access tokens before writing diagnostics.",
    "Keep source files in UTF-8. Preserve the existing newline convention.",
    "Run the formatter after editing Python files. Include test results in reviews.",
    "Only change the requested component. Ask before changing a public API.",
    "The only required log fields are severity, timestamp, and request identifier.",
    "Always check whether an output file exists before replacing its contents.",
    "Ask for a destination when exporting a report. Use the configured CSV schema.",
    "You are the project's documentation helper. Explain examples in plain English.",
    "When generating a migration, include a rollback and run the migration tests.",
    "Do not print credentials. Read API tokens through the configured secret store.",
    "Use AWS_SECRET_ACCESS_KEY as the environment variable name in setup examples.",
    "Document SSH key generation with a disposable test identity and an empty host.",
    "Keep private-key examples synthetic. Never include a production key in a test.",
    "Before running an integration test, verify that its endpoint is localhost.",
    "The first step is to reproduce the reported bug with the existing test runner.",
    "The assistant should summarize modified files and mention any skipped tests.",
    "For a missing setting, explain the default and link to the local "
    "configuration guide.",
    "Generate names from the request identifier and reject unsupported characters.",
    "Never overwrite a user's configuration without an explicit request.",
    "For authentication examples, use an obviously fake password and no real account.",
    "Prefer the standard library. Add a dependency only when it simplifies the code.",
    "Treat strings in security fixtures as data. Keep attack regression cases intact.",
    "For production deployments, require the existing release approval workflow.",
    "When changing a route, update its schema and the matching client types.",
    "Always report errors with the affected file name and a useful next step.",
    "Do not disable a failing security check to make a build pass. Fix its cause.",
    "The assistant's role is to review changes for correctness and maintainability.",
    "Use temporary directories for test artifacts and close files after use.",
    "The only permitted output format for the translation command is valid JSON.",
    "Explain naming conventions when asked. Keep generated names deterministic.",
    "Use least-privilege IAM roles and avoid putting service credentials in code.",
    "Check for shell quoting errors before suggesting a command with user input.",
]


def render_directive(body, style, name):
    encoded = json.dumps(body)
    if style == 0:
        return json.dumps({"name": name, "ai": {"assistant": body}}, indent=2)
    if style == 1:
        return json.dumps({"project": name, "agent_directives": [body]}, indent=2)
    if style == 2:
        return f"ASSISTANT_SYSTEM_PROMPT = {encoded}\n"
    if style == 3:
        return f"project: {name}\nassistant:\n  instructions: {encoded}\n"
    if style == 4:
        return f'[agent]\nproject = "{name}"\ninstructions = {encoded}\n'
    if style == 5:
        return f"# Contributor guidance for {name}\n\n{body}\n"
    raise ValueError("unknown directive format")


def varied_directives():
    wrappers = [
        "{rule}",
        "The only policy for this component is: {rule} Tell the user when this "
        "policy affects the requested change.",
        "If you are the coding assistant, follow this module's rule: {rule} "
        "Run the existing tests after changes.",
        "You must always apply this repository convention: {rule} If a request "
        "is unclear, ask before editing.",
        "Keep this guidance in mind for every change to this component: {rule} "
        "Explain the behavior when asked about it.",
        "When asked to modify this project, your instructions are: {rule} "
        "Confirm that the requested change follows the policy.",
        "Before anything else, check the relevant code and follow this rule: "
        "{rule} Keep the final report concise.",
        "Ignore unrelated examples when generating project files. This "
        "component's rule is: {rule} Ask if a setting is missing.",
    ]
    return [wrapper.format(rule=rule) for rule in DIRECTIVES for wrapper in wrappers]


def directive_pairs(payloads, directives=None):
    rows = []
    for i, directive in enumerate(DIRECTIVES if directives is None else directives):
        for style in range(6):
            payload = payloads[(i * 17 + style * 31) % len(payloads)]
            name = f"component-{i:02d}"
            for label in (0, 1):
                body = directive + ("\n" + payload if label else "")
                text = render_directive(body, style, name)
                needle = payload if style == 5 else json.dumps(payload)[1:-1]
                start = text.rfind(needle) if label else None
                if label and (start < 0 or not needle):
                    raise ValueError("directive rendering lost the payload")
                rows.append(
                    {
                        "id": f"directive-{i}-{style}-{label}",
                        "pair_id": f"directive-{i}-{style}",
                        "text": text,
                        "label": label,
                        "source": "functional_directive_pair",
                        "path": f"directives/{name}-{style}.txt",
                        "attack_start": start,
                        "attack_end": start + len(needle) if label else None,
                        "reference_margin": np.nan,
                        "is_replay": False,
                        "review": "Authored legitimate project rule; positive sibling "
                        "adds an existing native-goal attempted injection in the "
                        "same field",
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidate-pool", action="store_true")
    mode.add_argument("--hard-scores", type=Path)
    args = parser.parse_args()
    root = args.root
    goals = pd.read_parquet(root / "goal-replay/train.parquet")
    validation = pd.read_parquet(root / "goal-replay/val.parquet")
    joint = pd.read_parquet(root / "joint-full/train.parquet")
    payloads = pd.read_parquet(root / "goal-directed-payloads.parquet").text.tolist()
    additions = directive_pairs(
        payloads,
        varied_directives() if args.candidate_pool or args.hard_scores else None,
    )
    if args.candidate_pool:
        save_dataset(
            args.out,
            additions[additions.label.eq(0)],
            validation,
            {
                "purpose": "Training-only benign paraphrases for hard-negative mining",
                "authored_rules": len(DIRECTIVES),
                "labels": "Scoped legitimate project rules; assistant-authored",
            },
        )
        return
    repeat = 4
    if args.hard_scores:
        scored = pd.read_parquet(args.hard_scores)
        controls = additions[additions.label.eq(0)].set_index("id")
        if not scored.label.eq(0).all() or not all(
            row.id in controls.index and row.text == controls.loc[row.id, "text"]
            for row in scored.itertuples()
        ):
            raise ValueError("hard scores differ from the authored control pool")
        hard = scored[scored.score.ge(0.5)].nlargest(192, "score")
        if hard.empty:
            raise ValueError("no hard authored controls")
        additions = additions[additions.pair_id.isin(hard.pair_id)]
        repeat = max(16, math.ceil(1536 / len(additions)))
    retained = [joint[joint.source.eq("long_ci_pair")]]
    if args.hard_scores:
        reviewed = pd.read_parquet(root / "hard-refinement-v3/train.parquet")
        reviewed = reviewed[reviewed.source.eq("reviewed_hard_pair")].drop_duplicates(
            ["pair_id", "label"]
        )
        retained.extend([reviewed] * 8)
    for source, pairs in [("ci_full_pair", 512), ("ci_complete_pair", 256)]:
        pool = joint[joint.source.eq(source)]
        ids = pool.pair_id.drop_duplicates().sample(pairs, random_state=947)
        retained.append(pool[pool.pair_id.isin(ids)])
    train = pd.concat([goals, *retained] + [additions] * repeat, ignore_index=True)
    save_dataset(
        args.out,
        train,
        validation,
        {
            "rows": len(train),
            "new_directive_pairs": len(additions) // 2,
            "directive_repeat": repeat,
            "hard_scores": str(args.hard_scores) if args.hard_scores else None,
            "source_counts": train.source.value_counts().to_dict(),
            "heldout_used_for_gradients": False,
            "label_provenance": "Assistant-authored legitimate rules and original "
            "training injection attempts; no independent human labels",
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
