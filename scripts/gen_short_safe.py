#!/usr/bin/env python3
"""Generate short safe text examples and insert them into merged.parquet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from promptscan.parquet_store import ParquetDataStore

SHORT_SAFE_TEXTS = [
    # Single common words
    "test", "hello", "hi", "hey", "yes", "no", "ok", "okay", "sure",
    "help", "thanks", "thank you", "sorry", "please", "bye", "goodbye",
    "welcome", "nice", "good", "bad", "great", "awesome", "cool",
    "maybe", "perhaps", "definitely", "absolutely", "exactly", "right",
    "wrong", "true", "false", "stop", "go", "start", "end", "done",
    "wait", "pause", "cancel", "confirm", "deny", "accept", "reject",
    "submit", "save", "delete", "update", "create", "read", "exit", "quit",
    # Short tech/testing terms that commonly get falsely flagged
    "foo", "bar", "baz", "qux", "foo bar", "test123", "hello world",
    "debug", "trace", "log", "info", "warn", "error", "ping", "pong",
    # Common short questions
    "what?", "how?", "why?", "when?", "where?", "who?", "which?",
    "what is this?", "how does this work?", "why not?", "what now?",
    # Numbers and identifiers
    "1234", "abc", "xyz", "123", "000", "test1", "test2", "abcd",
    # Greetings and common phrases
    "good morning", "good afternoon", "good evening", "good night",
    "how are you?", "I'm fine", "I'm good", "please help",
    "can you help", "need help", "got it", "understood", "noted",
    "makes sense", "not sure", "I don't know", "let me check",
    # Short natural requests
    "tell me more", "go on", "continue", "what next", "and then?",
    "really?", "seriously?", "are you sure?", "is that right?",
    "what do you mean?", "explain please", "more details",
    # Single-word task verbs (benign when used alone)
    "summarize", "translate", "explain", "describe", "list", "show",
    "find", "search", "calculate", "compute", "format", "convert", "sort",
    # Short conversational phrases
    "that's right", "of course", "no problem", "my bad", "my mistake",
    "never mind", "forget it", "just kidding", "I see", "I get it",
    "thanks a lot", "thank you so much", "you're welcome", "no worries",
    # Programming / benign identifiers
    "null", "undefined", "none", "empty", "void",
    "int", "str", "bool", "list", "dict", "set", "tuple", "float",
    # Short file/resource names (benign)
    "hello.txt", "test.py", "README", "index.html", "main.py",
    # Short instructional fragments used in normal conversation
    "I need help", "can you help me", "please answer", "answer this",
    "help me understand", "what is", "tell me about", "explain this",
    "I need you to help me", "please explain", "could you help",
    "I need you to answer this question",
    "hi how are you", "hi how are you?",
    "how are you doing",
    "hi how are you. i need you to answer this question for me",
    "please help me with this",
    "can you answer my question",
    "i have a question for you",
    "what can you do",
    "who are you",
    "what are you",
]


def main():
    parquet_path = "data/merged.parquet"
    store = ParquetDataStore(parquet_path)

    prompts = [
        {"text": text, "is_injection": False, "source": "short_safe_examples"}
        for text in SHORT_SAFE_TEXTS
    ]

    print(f"Inserting {len(prompts)} short safe examples...")
    added_ids, skipped = store.add_prompts_batch(prompts)
    print(f"  Added: {len(added_ids)}, Skipped (duplicates): {skipped}")

    stats = store.get_statistics()
    print(
        f"Database now: {stats['total']} total "
        f"({stats['injections']} inj / {stats['safe']} safe)"
    )


if __name__ == "__main__":
    main()
