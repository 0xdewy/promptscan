#!/usr/bin/env python3
"""
Interactive review of unverified user submissions.
Presents each submission, asks for verification, and moves to main prompts.parquet.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import textwrap
import argparse

# Add project root to path to import promptscan modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from promptscan.feedback_store import ParquetFeedbackStore
    from promptscan.parquet_store import ParquetDataStore
except ImportError as e:
    print(f"Error importing promptscan modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class UnverifiedReviewer:
    """Interactive reviewer for unverified user submissions."""

    def __init__(
        self,
        unverified_path: str = "website/api/data/unverified_user_submissions.parquet",
        prompts_path: str = "data/prompts.parquet",
        progress_file: str = ".reviewed_ids.json",
    ):
        """
        Initialize the reviewer.

        Args:
            unverified_path: Path to unverified submissions parquet file
            prompts_path: Path to main prompts parquet file
            progress_file: File to track reviewed IDs
        """
        self.unverified_path = Path(unverified_path)
        self.prompts_path = Path(prompts_path)
        self.progress_file = Path(progress_file)

        # Initialize data stores
        print("Loading data stores...")
        self.feedback_store = ParquetFeedbackStore(str(self.unverified_path))
        self.prompts_store = ParquetDataStore(str(self.prompts_path))

        # Load progress
        self.reviewed_ids = self._load_reviewed_ids()
        self.stats = {
            "total_reviewed": 0,
            "added_as_injection": 0,
            "added_as_safe": 0,
            "skipped": 0,
            "agreements": 0,
            "disagreements": 0,
        }

    def _load_reviewed_ids(self) -> set:
        """Load previously reviewed entry IDs from progress file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("reviewed_ids", []))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load progress file: {e}")
                return set()
        return set()

    def _save_progress(self) -> None:
        """Save reviewed IDs to progress file."""
        try:
            progress_data = {
                "reviewed_ids": list(self.reviewed_ids),
                "last_updated": datetime.now().isoformat(),
                "stats": self.stats,
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save progress file: {e}")

    def _format_prompt_text(self, text: str, width: int = 80) -> str:
        """Format prompt text for display with wrapping."""
        # Clean up text for display
        clean_text = text.replace("\r", "").strip()

        # Truncate if very long
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "\n[...truncated...]"

        # Wrap text
        wrapped_lines = []
        for line in clean_text.split("\n"):
            if line.strip():
                wrapped = textwrap.wrap(line, width=width)
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append("")

        return "\n".join(wrapped_lines)

    def _display_prompt_info(self, entry: Dict[str, Any]) -> None:
        """Display prompt information with context."""
        print("\n" + "=" * 60)

        # Display prompt text
        print("📝 PROMPT:")
        formatted_text = self._format_prompt_text(entry["text"])
        for line in formatted_text.split("\n"):
            print(f"  {line}")

        print("\n🔍 ORIGINAL CONTEXT:")

        # Show prediction information
        predicted = entry.get("predicted_label", "UNKNOWN")
        user_label = entry.get("user_label", "UNKNOWN")
        confidence = entry.get("ensemble_confidence", 0.0)

        print(f"  • Model predicted: {predicted} ({confidence:.1%} confidence)")
        print(f"  • User labeled: {user_label}")

        # Show agreement/disagreement
        if predicted == user_label:
            print(f"  • Status: ✅ AGREEMENT")
        else:
            print(f"  • Status: ❌ DISAGREEMENT")

        # Show individual predictions if available
        individual_preds = entry.get("individual_predictions", [])
        if individual_preds and isinstance(individual_preds, list):
            print(f"  • Individual model predictions:")
            for i, pred in enumerate(individual_preds[:3]):  # Show up to 3
                model = pred.get("model", f"Model {i + 1}")
                pred_label = pred.get("prediction", "UNKNOWN")
                pred_conf = pred.get("confidence", 0.0)
                print(f"    - {model}: {pred_label} ({pred_conf:.1%})")

        # Show metadata
        source = entry.get("source", "unknown")
        model_type = entry.get("model_type", "unknown")
        timestamp = entry.get("timestamp", "")

        if timestamp:
            if isinstance(timestamp, str):
                time_str = timestamp
            else:
                try:
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = str(timestamp)
            print(f"  • Submitted: {time_str}")

        print(f"  • Source: {source}")
        print(f"  • Model type: {model_type}")

        print("=" * 60)

    def _get_user_decision(self) -> str:
        """Get user decision with validation."""
        while True:
            print("\n❓ DECISION:")
            print("  [y] Yes - This IS an injection")
            print("  [n] No - This is SAFE")
            print("  [s] Skip - Review later")
            print("  [q] Quit - Save and exit")

            choice = input("\nYour choice (y/n/s/q): ").strip().lower()

            if choice in ["y", "n", "s", "q"]:
                return choice
            else:
                print("Invalid choice. Please enter y, n, s, or q.")

    def _add_to_main_prompts(
        self, text: str, is_injection: bool, original_entry: Dict[str, Any]
    ) -> bool:
        """Add verified prompt to main prompts database."""
        try:
            # Prepare the prompt data with metadata
            prompt_data = {
                "text": text,
                "is_injection": is_injection,
                "source": "user_feedback_reviewed",
                "review_date": datetime.now().isoformat(),
                "original_prediction": original_entry.get("predicted_label", ""),
                "original_user_label": original_entry.get("user_label", ""),
                "original_confidence": float(
                    original_entry.get("ensemble_confidence", 0.0)
                ),
                "original_source": original_entry.get("source", ""),
                "original_timestamp": str(original_entry.get("timestamp", "")),
            }

            # Check if prompt already exists using the store's duplicate check
            if self.prompts_store._prompt_exists(text, is_injection):
                print("  ⚠️  Prompt already exists in database (duplicate)")
                return False

            # Get current data and add the new prompt
            df = self.prompts_store.export_to_dataframe()

            # Generate a UUID for the new prompt
            import uuid

            prompt_id = str(uuid.uuid4())
            prompt_data["id"] = prompt_id

            # Create new row
            new_row = pd.DataFrame([prompt_data])

            # Append to existing data
            df = pd.concat([df, new_row], ignore_index=True)

            # Save back to store
            self.prompts_store._data = df
            self.prompts_store._save_data()

            print(f"  ✅ Added to main database with ID: {prompt_id[:8]}...")
            return True

        except Exception as e:
            print(f"  ❌ Error adding to prompts: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _remove_from_unverified(self, entry_id: int) -> bool:
        """Remove entry from unverified submissions."""
        try:
            # Get current data
            df = self.feedback_store.export_to_dataframe()

            # Remove the entry
            df = df[df["id"] != entry_id]

            # Save back
            self.feedback_store.import_from_dataframe(df)
            return True
        except Exception as e:
            print(f"  ❌ Error removing from unverified: {e}")
            return False

    def _get_unreviewed_entries(self) -> List[Dict[str, Any]]:
        """Get all unreviewed entries from unverified submissions."""
        all_entries = self.feedback_store.get_all_feedback()

        # Filter out reviewed entries
        unreviewed = []
        for entry in all_entries:
            entry_id = entry.get("id")
            if entry_id and entry_id not in self.reviewed_ids:
                unreviewed.append(entry)

        # Sort by timestamp (oldest first) or by disagreement
        def sort_key(entry):
            # Prioritize disagreements
            predicted = entry.get("predicted_label", "")
            user_label = entry.get("user_label", "")
            is_disagreement = predicted != user_label

            # Get timestamp
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except:
                    ts = datetime.min
            elif hasattr(timestamp, "timestamp"):
                ts = timestamp
            else:
                ts = datetime.min

            return (not is_disagreement, ts)  # False (0) comes before True (1)

        unreviewed.sort(key=sort_key)
        return unreviewed

    def _display_stats(self, current: int, total: int) -> None:
        """Display review statistics."""
        print("\n" + "=" * 60)
        print("📊 REVIEW STATISTICS")
        print("=" * 60)
        print(f"  Progress: {current}/{total} ({current / total * 100:.1f}%)")
        print(f"  Added as injection: {self.stats['added_as_injection']}")
        print(f"  Added as safe: {self.stats['added_as_safe']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Agreements: {self.stats['agreements']}")
        print(f"  Disagreements: {self.stats['disagreements']}")

        if self.stats["total_reviewed"] > 0:
            agreement_rate = (
                self.stats["agreements"] / self.stats["total_reviewed"] * 100
            )
            print(f"  Agreement rate: {agreement_rate:.1f}%")
        print("=" * 60)

    def run_interactive_review(self) -> None:
        """Run the interactive review session."""
        # Get unreviewed entries
        unreviewed = self._get_unreviewed_entries()
        total_entries = len(unreviewed)

        if total_entries == 0:
            print("✅ No unreviewed entries found!")
            return

        print("\n" + "=" * 60)
        print("🔍 UNVERIFIED SUBMISSIONS REVIEW")
        print("=" * 60)
        print(f"Found {total_entries} unreviewed entries")
        print("=" * 60)

        # Review loop
        for i, entry in enumerate(unreviewed, 1):
            entry_id = entry.get("id", "unknown")

            print(f"\n📋 Entry {i}/{total_entries} (ID: {entry_id})")

            # Display prompt information
            self._display_prompt_info(entry)

            # Get user decision
            decision = self._get_user_decision()

            if decision == "q":
                print("\n💾 Saving progress and exiting...")
                self._save_progress()
                self._display_stats(i - 1, total_entries)
                return

            elif decision == "s":
                print("  ⏭️  Skipping this entry")
                self.stats["skipped"] += 1
                continue

            # Process decision
            is_injection = decision == "y"
            predicted = entry.get("predicted_label", "")
            user_label = entry.get("user_label", "")

            # Track agreement/disagreement
            if predicted == user_label:
                self.stats["agreements"] += 1
            else:
                self.stats["disagreements"] += 1

            # Add to main prompts
            print(
                f"  {'🔄' if is_injection else '✅'} Adding as {'INJECTION' if is_injection else 'SAFE'}..."
            )
            success = self._add_to_main_prompts(
                text=entry["text"], is_injection=is_injection, original_entry=entry
            )

            # Mark as reviewed regardless of success
            self.reviewed_ids.add(entry_id)
            self.stats["total_reviewed"] += 1

            if success:
                # Update stats for successful additions
                if is_injection:
                    self.stats["added_as_injection"] += 1
                else:
                    self.stats["added_as_safe"] += 1

                # Remove from unverified
                if self._remove_from_unverified(entry_id):
                    print(f"  ✅ Removed from unverified submissions")
            else:
                # Even if duplicate, we reviewed it
                print(f"  ⚠️  Not added (duplicate or error)")

            # Save progress periodically
            if i % 5 == 0:
                self._save_progress()
                print(f"  💾 Progress saved")

        # Final save and stats
        self._save_progress()
        print("\n" + "=" * 60)
        print("🎉 REVIEW COMPLETE!")
        print("=" * 60)
        self._display_stats(total_entries, total_entries)

        # Show final database stats
        prompts_stats = self.prompts_store.get_statistics()
        print(f"\n📈 MAIN DATABASE STATISTICS:")
        print(f"  Total prompts: {prompts_stats['total']}")
        print(
            f"  Injections: {prompts_stats['injections']} ({prompts_stats['injection_percentage']:.1f}%)"
        )
        print(
            f"  Safe prompts: {prompts_stats['safe']} ({prompts_stats['safe_percentage']:.1f}%)"
        )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Interactive review of unverified user submissions"
    )
    parser.add_argument(
        "--unverified",
        default="website/api/data/unverified_user_submissions.parquet",
        help="Path to unverified submissions parquet file",
    )
    parser.add_argument(
        "--prompts",
        default="data/prompts.parquet",
        help="Path to main prompts parquet file",
    )
    parser.add_argument(
        "--progress",
        default=".reviewed_ids.json",
        help="Path to progress tracking file",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous progress"
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.unverified).exists():
        print(f"Error: Unverified file not found: {args.unverified}")
        sys.exit(1)

    if not Path(args.prompts).exists():
        print(f"Warning: Prompts file not found: {args.prompts}")
        print("Creating new prompts file...")

    # Create reviewer and run
    reviewer = UnverifiedReviewer(
        unverified_path=args.unverified,
        prompts_path=args.prompts,
        progress_file=args.progress,
    )

    try:
        reviewer.run_interactive_review()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
        reviewer._save_progress()
        print("Progress saved. You can resume with --resume flag.")
    except Exception as e:
        print(f"\n❌ Error during review: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
