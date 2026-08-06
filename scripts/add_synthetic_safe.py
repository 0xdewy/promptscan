#!/usr/bin/env python3
"""
Generate synthetic borderline-safe training examples to reduce false positives
on instruction-like text (e.g., skill documentation, README files, system prompts).
"""

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd


SYNTHETIC_TEMPLATES = [
    "You are now a {role}. {action}.",
    "System: you operate as a {role}. When asked to {action}, do so.",
    "TRIGGER on: {trigger}. SKIP on: {skip}.",
    "When the user says \"{trigger}\", respond with {action}.",
    "Load the {role} workflow. It will {action}.",
    "Activate {role} mode. From now on, {action}.",
    "You function as {role}. Your task is to {action}.",
    "The {role} agent spawns sub-agents who {action}.",
    "Use the {role} skill when the user asks to {action}.",
    "This skill handles {action}. Do not use for {skip}.",
    "Instructions for {role}: first {action}, then {action2}.",
    "As a {role}, you should always {action} before anything else.",
    "Run the {role} pipeline: gather info → analyze → {action}.",
    "You have been configured as {role}. Respond with {action} when prompted.",
    "This module provides {action}. It is triggered by users asking about {trigger}.",
]

ROLES = [
    "code reviewer", "security auditor", "data analyst", "documentation writer",
    "test engineer", "devops assistant", "refactoring specialist", "bug hunter",
    "performance optimizer", "API designer", "database expert", "frontend developer",
    "backend engineer", "ML researcher", "prompt engineer", "system architect",
    "quality assurance lead", "release manager", "mentor", "translator",
]

ACTIONS = [
    "find and fix bugs in the codebase", "generate a comprehensive report",
    "answer the user's questions accurately", "process the input data",
    "scan for security vulnerabilities", "refactor the legacy module",
    "write unit tests for new features", "optimize database queries",
    "deploy the application to production", "review the pull request",
    "explain the architecture to the team", "validate the configuration",
    "transform the data into the required format", "summarize the document",
    "classify the input into categories", "extract key information from the text",
    "evaluate the model's performance", "merge the feature branches",
    "create a new database migration", "set up the CI/CD pipeline",
]

TRIGGERS = [
    "user requests", "a specific keyword appears", "the input matches a pattern",
    "certain conditions are met", "the conversation reaches a specific topic",
    "an error occurs", "the user asks for help", "a file is modified",
    "the build fails", "new data arrives", "the system starts up",
    "an anomaly is detected", "a threshold is crossed",
]

SKIPS = [
    "unrelated topics", "pure informational queries", "single-file edits",
    "documentation-only requests", "questions without code context",
    "tasks outside the agent's scope", "requests requiring human judgment",
    "network operations without permission",
]


def parse_skill_md(filepath: Path) -> str:
    """Extract the description text from a SKILL.md YAML frontmatter."""
    content = filepath.read_text(encoding="utf-8")
    match = re.search(r"description:\s*>(.*?)(?:license:|metadata:|$)", content, re.DOTALL)
    if match:
        desc = match.group(1)
        desc = re.sub(r"\n\s+", " ", desc)
        return desc.strip()
    return ""


def collect_skill_descriptions(skills_dir: Path) -> list:
    """Collect all SKILL.md descriptions from a skills directory."""
    examples = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        desc = parse_skill_md(skill_md)
        if desc:
            examples.append({
                "text": desc,
                "is_injection": False,
                "source": f"skill:{skill_md.parent.name}",
            })
    return examples


def generate_synthetic_examples(num_examples: int = 100) -> list:
    """Generate synthetic borderline-safe instruction-like examples."""
    examples = []
    for _ in range(num_examples):
        template = random.choice(SYNTHETIC_TEMPLATES)
        role = random.choice(ROLES)
        action = random.choice(ACTIONS)
        action2 = random.choice(ACTIONS)
        trigger = random.choice(TRIGGERS)
        skip = random.choice(SKIPS)
        text = template.format(role=role, action=action, action2=action2,
                               trigger=trigger, skip=skip)
        examples.append({
            "text": text,
            "is_injection": False,
            "source": "synthetic:instruction-safe",
        })
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic borderline-safe training examples"
    )
    parser.add_argument(
        "--skills-dir",
        default="/home/user/code/skills/skills",
        help="Path to skills directory containing SKILL.md files",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=100,
        help="Number of synthetic examples to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_safe.parquet",
        help="Output parquet file (default: data/synthetic_safe.parquet)",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    if not skills_dir.exists():
        print(f"Skills directory not found: {skills_dir}")
        print("Skipping skill description extraction. Use --skills-dir to specify.")
        skill_examples = []
    else:
        skill_examples = collect_skill_descriptions(skills_dir)
        print(f"Collected {len(skill_examples)} skill descriptions")

    synthetic_examples = generate_synthetic_examples(args.num_synthetic)
    print(f"Generated {len(synthetic_examples)} synthetic examples")

    all_examples = skill_examples + synthetic_examples
    df = pd.DataFrame(all_examples)
    df["is_injection"] = df["is_injection"].astype(bool)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\nSaved {len(all_examples)} examples to {output_path}")
    print(f"  Skill descriptions: {len(skill_examples)}")
    print(f"  Synthetic: {len(synthetic_examples)}")


if __name__ == "__main__":
    main()
