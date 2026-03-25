#!/usr/bin/env python3
"""
Generate popular/safe prompts using DeepSeek API.
Adds them to the database automatically.
"""

import os
import time
import random
import sqlite3
from typing import List, Dict
import requests
import json


class SafePromptGenerator:
    def __init__(self, db_path: str = "prompts.db", api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            print(
                "Warning: No DeepSeek API key found. Set DEEPSEEK_API_KEY environment variable."
            )
            print("Will use predefined safe prompts instead.")

        self._init_database()

    def _init_database(self):
        """Ensure database table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            is_injection BOOLEAN NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API to generate safe prompts."""
        if not self.api_key:
            return self._get_fallback_response()

        url = "https://api.deepseek.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates popular, safe, and useful prompts for AI assistants. Generate prompts that people commonly ask AI assistants, covering various topics like education, creativity, productivity, and general knowledge.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Get a fallback safe prompt if API fails."""
        safe_prompts = [
            "What are the benefits of renewable energy sources compared to fossil fuels?",
            "Can you explain the basics of machine learning to a beginner with simple examples?",
            "What's the difference between artificial intelligence and machine learning?",
            "How can I improve my productivity while working from home?",
            "What are some effective study techniques for memorizing information?",
            "Can you suggest some healthy meal prep ideas for the week?",
            "What are the main causes of climate change and what can individuals do to help?",
            "How does the stock market work for someone who is just starting to invest?",
            "What are some good books to read for personal development?",
            "Can you explain how blockchain technology works in simple terms?",
            "What are the key differences between Python and JavaScript for web development?",
            "How can I start learning a new language effectively?",
            "What are some tips for improving public speaking skills?",
            "Can you explain the water cycle and its importance to the environment?",
            "What are the health benefits of regular exercise and meditation?",
            "How do I create a budget and stick to it?",
            "What are some creative writing prompts for beginners?",
            "Can you explain the theory of evolution in simple terms?",
            "What are the best practices for digital security and privacy?",
            "How does photosynthesis work in plants?",
        ]
        return random.choice(safe_prompts)

    def generate_safe_prompt(self) -> str:
        """Generate a prompt for the API to create safe prompts."""
        categories = [
            "education",
            "technology",
            "health",
            "finance",
            "productivity",
            "creativity",
            "science",
            "history",
            "personal development",
            "cooking",
        ]

        category = random.choice(categories)

        prompts = {
            "education": "Generate a popular educational question that students might ask an AI tutor.",
            "technology": "Create a common technology-related question that people ask AI assistants.",
            "health": "Generate a health and wellness question that is safe and appropriate for an AI to answer.",
            "finance": "Create a personal finance question that people commonly ask about budgeting or investing.",
            "productivity": "Generate a productivity tip or question that people ask to improve their work efficiency.",
            "creativity": "Create a creative writing or art-related prompt that inspires positive expression.",
            "science": "Generate a science question that explains a natural phenomenon or scientific concept.",
            "history": "Create a historical question that helps people understand important events or figures.",
            "personal development": "Generate a self-improvement question about habits, mindset, or skills development.",
            "cooking": "Create a cooking or recipe-related question that helps people prepare meals.",
        }

        return prompts[category]

    def add_to_database(self, text: str, is_injection: bool = False):
        """Add generated text to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO prompts (text, is_injection) VALUES (?, ?)",
            (text, is_injection),
        )

        conn.commit()
        conn.close()

        print(f"Added to database: {text[:100]}...")

    def generate_and_store(self, count: int = 5, delay: int = 2):
        """Generate multiple safe prompts and store them."""
        print(f"Generating {count} popular/safe prompts...")

        for i in range(count):
            print(f"\n--- Generating safe prompt {i + 1}/{count} ---")

            # Generate a prompt for the API
            prompt = self.generate_safe_prompt()
            print(f"Category prompt: {prompt}")

            # Call API or use fallback
            safe_prompt = self.call_deepseek_api(prompt)
            print(f"Generated: {safe_prompt[:150]}...")

            # Add to database
            self.add_to_database(safe_prompt, is_injection=False)

            # Delay to avoid rate limiting
            if i < count - 1:
                print(f"Waiting {delay} seconds before next generation...")
                time.sleep(delay)

        print(f"\n✅ Generated and stored {count} popular/safe prompts!")

    def get_statistics(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 1")
        injections = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 0")
        safe = cursor.fetchone()[0]

        conn.close()

        return {
            "total_injections": injections,
            "total_safe": safe,
            "total": injections + safe,
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate popular/safe prompts")
    parser.add_argument(
        "--count", type=int, default=5, help="Number of safe prompts to generate"
    )
    parser.add_argument(
        "--delay", type=int, default=2, help="Delay between generations (seconds)"
    )
    parser.add_argument(
        "--api-key", type=str, help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--top-20", action="store_true", help="Add the top 20 most popular prompts"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = SafePromptGenerator(api_key=args.api_key)

    # Show current stats
    stats = generator.get_statistics()
    print(f"Current database: {stats['total']} total prompts")
    print(f"  - Injections: {stats['total_injections']}")
    print(f"  - Safe: {stats['total_safe']}")

    # Generate and store new prompts
    generator.generate_and_store(count=args.count, delay=args.delay)

    # Show updated stats
    stats = generator.get_statistics()
    print(f"\nUpdated database: {stats['total']} total prompts")
    print(f"  - Injections: {stats['total_injections']}")
    print(f"  - Safe: {stats['total_safe']}")


if __name__ == "__main__":
    main()
