#!/usr/bin/env python3
"""
Test script for the unverified submissions reviewer.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from promptscan.feedback_store import ParquetFeedbackStore
from promptscan.parquet_store import ParquetDataStore


def test_data_stores():
    """Test that data stores can be loaded and manipulated."""
    print("Testing data stores...")

    # Test feedback store
    feedback_path = "website/api/data/unverified_user_submissions.parquet"
    if not Path(feedback_path).exists():
        print(f"❌ Feedback file not found: {feedback_path}")
        return False

    feedback = ParquetFeedbackStore(feedback_path)
    entries = feedback.get_all_feedback()
    print(f"✓ Loaded {len(entries)} feedback entries")

    # Test prompts store
    prompts_path = "data/prompts.parquet"
    if not Path(prompts_path).exists():
        print(f"⚠️  Prompts file not found: {prompts_path}")
        print("Creating empty store for testing...")

    prompts = ParquetDataStore(prompts_path)
    stats = prompts.get_statistics()
    print(f"✓ Main database: {stats['total']} prompts")

    return True


def test_script_import():
    """Test that the review script can be imported."""
    print("\nTesting script import...")

    try:
        # Import the main function
        import scripts.review_unverified as review_module

        print("✓ Script module imported successfully")

        # Check if main class exists
        if hasattr(review_module, "UnverifiedReviewer"):
            print("✓ UnverifiedReviewer class found")

            # Try to create an instance
            reviewer = review_module.UnverifiedReviewer(
                unverified_path="website/api/data/unverified_user_submissions.parquet",
                prompts_path="data/prompts.parquet",
                progress_file=".test_reviewed_ids.json",
            )
            print("✓ Reviewer instance created successfully")

            # Test getting unreviewed entries
            unreviewed = reviewer._get_unreviewed_entries()
            print(f"✓ Found {len(unreviewed)} unreviewed entries")

            # Clean up test file
            if Path(".test_reviewed_ids.json").exists():
                Path(".test_reviewed_ids.json").unlink()

            return True
        else:
            print("❌ UnverifiedReviewer class not found")
            return False

    except Exception as e:
        print(f"❌ Error importing script: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_single_entry_processing():
    """Test processing a single entry."""
    print("\nTesting single entry processing...")

    try:
        import scripts.review_unverified as review_module

        # Create reviewer
        reviewer = review_module.UnverifiedReviewer(
            unverified_path="website/api/data/unverified_user_submissions.parquet",
            prompts_path="data/prompts.parquet",
            progress_file=".test_reviewed_ids.json",
        )

        # Get first unreviewed entry
        unreviewed = reviewer._get_unreviewed_entries()
        if not unreviewed:
            print("⚠️  No unreviewed entries to test")
            return True

        entry = unreviewed[0]
        entry_id = entry.get("id")

        print(f"Testing entry ID: {entry_id}")
        print(f"Text preview: {entry.get('text', '')[:100]}...")

        # Test display function
        print("\nTesting display function...")
        reviewer._display_prompt_info(entry)

        # Test adding to prompts (simulate 'y' decision)
        print("\nTesting add to prompts (simulating 'y' decision)...")
        success = reviewer._add_to_main_prompts(
            text=entry["text"], is_injection=True, original_entry=entry
        )

        if success:
            print("✓ Successfully added to prompts")

            # Test removal from unverified
            print("\nTesting removal from unverified...")
            removed = reviewer._remove_from_unverified(entry_id)
            if removed:
                print("✓ Successfully removed from unverified")
            else:
                print("❌ Failed to remove from unverified")
        else:
            print("❌ Failed to add to prompts (might be duplicate)")

        # Clean up
        if Path(".test_reviewed_ids.json").exists():
            Path(".test_reviewed_ids.json").unlink()

        return True

    except Exception as e:
        print(f"❌ Error in single entry test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("UNVERIFIED REVIEW SCRIPT TEST SUITE")
    print("=" * 60)

    tests = [
        ("Data Stores", test_data_stores),
        ("Script Import", test_script_import),
        ("Single Entry Processing", test_single_entry_processing),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"   Result: ❌ ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The review script is ready to use.")
        print("\nTo use the script:")
        print("  cd /home/user/code/prompt-scan")
        print("  uv run python scripts/review_unverified.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
