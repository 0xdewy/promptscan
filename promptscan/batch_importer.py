#!/usr/bin/env python3
"""
Batch importer for prompt insertion with duplicate checking.
Supports local files, directories, and GitHub repositories.
"""

import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ImportStats:
    """Statistics for import operations."""

    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    added: int = 0
    duplicates: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Get import duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed - self.failed) / self.total_files * 100

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def finish(self):
        """Finish timing."""
        self.end_time = time.time()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Processed: {self.processed}/{self.total_files} files, "
            f"Added: {self.added} prompts, "
            f"Skipped: {self.skipped} files, "
            f"Duplicates: {self.duplicates} prompts, "
            f"Failed: {self.failed} files, "
            f"Duration: {self.duration:.2f}s"
        )


class BatchImporter:
    """Batch importer for prompt data with duplicate checking."""

    def __init__(self, store, verbose=False, github_token=None):
        """
        Initialize batch importer.

        Args:
            store: ParquetDataStore instance
            verbose: Show verbose output
            github_token: GitHub token for API access
        """
        self.store = store
        self.verbose = verbose
        self.github_token = github_token

    def import_from_files(
        self,
        files: List[Union[str, Path]],
        label: Optional[bool] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> ImportStats:
        """
        Import prompts from files.

        Args:
            files: List of file paths (strings or Path objects)
            label: Label for all prompts (True=injection, False=safe, None=interactive)
            dry_run: Preview without importing
            verbose: Show verbose output

        Returns:
            Import statistics
        """
        stats = ImportStats()
        stats.start()
        stats.total_files = len(files)

        print(f"📄 Importing {len(files)} files")
        print(
            f"   Label: {'injection' if label else 'safe' if label is not None else 'interactive'}"
        )
        print(f"   Dry run: {dry_run}")

        prompts = []

        for i, filepath in enumerate(files, 1):
            # Convert to Path object if it's a string
            filepath = Path(filepath) if isinstance(filepath, str) else filepath

            if verbose or i % max(1, len(files) // 20) == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] Processing: {filepath.name[:50]}...")

            try:
                # Check if file exists
                if not filepath.exists():
                    if verbose:
                        print(f"    ✗ File not found: {filepath}")
                    stats.skipped += 1
                    continue

                # Read file content
                content = filepath.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    if verbose:
                        print(f"    ⚠️  Empty file: {filepath.name}")
                    stats.skipped += 1
                    continue

                # Create prompt data
                prompt_data = {
                    "text": content,
                    "is_injection": label if label is not None else False,
                    "source": str(filepath),
                }

                prompts.append(prompt_data)
                stats.processed += 1

            except Exception as e:
                if verbose:
                    print(f"    ✗ Error processing {filepath.name}: {e}")
                stats.failed += 1
                continue

            # Batch insert every 100 prompts
            if len(prompts) >= 100:
                if not dry_run:
                    added_ids, duplicates = self.store.add_prompts_batch(prompts)
                    stats.added += len(added_ids)
                    stats.duplicates += duplicates
                else:
                    stats.added += len(prompts)
                prompts = []

        # Insert remaining prompts
        if prompts:
            if not dry_run:
                added_ids, duplicates = self.store.add_prompts_batch(prompts)
                stats.added += len(added_ids)
                stats.duplicates += duplicates
            else:
                stats.added += len(prompts)

        stats.finish()

        # Print summary
        self._print_import_summary(stats, dry_run)

        return stats

    def import_from_directory(
        self,
        directory: Union[str, Path],
        label: Optional[bool] = None,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        recursive: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> ImportStats:
        """
        Import prompts from directory.

        Args:
            directory: Directory path (string or Path)
            label: Label for all prompts
            extensions: File extensions to include (e.g., ['.txt', '.md', '.py', '.js'])
                         Default includes common text and source code files
            exclude: Path patterns to exclude
            max_size: Maximum file size in bytes (None for no limit)
            recursive: Whether to traverse subdirectories
            dry_run: Preview without importing
            verbose: Show verbose output

        Returns:
            Import statistics
        """
        # Convert to Path object if it's a string
        directory = Path(directory) if isinstance(directory, str) else directory

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Default extensions
        if extensions is None:
            extensions = [
                ".txt",
                ".md",
                ".py",
                ".json",
                ".csv",
                ".yaml",
                ".yml",
                ".xml",
                ".html",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".rs",
                ".go",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".cs",
                ".php",
                ".rb",
                ".pl",
                ".pm",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".ps1",
                ".bat",
                ".cmd",
                ".sql",
                ".r",
                ".m",
                ".swift",
                ".kt",
                ".kts",
                ".scala",
                ".clj",
                ".cljs",
                ".lua",
                ".erl",
                ".ex",
                ".exs",
                ".hs",
                ".ml",
                ".mli",
                ".fs",
                ".fsx",
                ".v",
                ".sv",
                ".vhd",
                ".vhdl",
                ".asm",
                ".s",
                ".ino",
                ".ino.cpp",
            ]

        print(f"📂 Found directory: {directory}")
        print(f"   Extensions: {', '.join(extensions)}")
        print(f"   Recursive: {recursive}")
        print(
            f"   Max file size: {'unlimited' if max_size is None else f'{max_size:,} bytes'}"
        )
        print(
            f"   Label: {'injection' if label else 'safe' if label is not None else 'interactive'}"
        )
        print(f"   Dry run: {dry_run}")

        # Find files
        files = []
        for ext in extensions:
            if recursive:
                files.extend(directory.rglob(f"*{ext}"))
            else:
                files.extend(directory.glob(f"*{ext}"))

        # Filter out excluded patterns
        if exclude:
            filtered_files = []
            for file in files:
                file_str = str(file)
                if not any(pattern in file_str for pattern in exclude):
                    filtered_files.append(file)
            files = filtered_files

        # Filter by file size if max_size is specified
        if max_size is not None:
            size_filtered_files = []
            for file in files:
                try:
                    if file.stat().st_size <= max_size:
                        size_filtered_files.append(file)
                    elif verbose:
                        print(
                            f"    ⚠️  Skipping large file: {file.name} ({file.stat().st_size:,} bytes)"
                        )
                except (OSError, IOError):
                    # Skip files we can't stat
                    if verbose:
                        print(f"    ⚠️  Cannot check size: {file.name}")
                    size_filtered_files.append(file)
            files = size_filtered_files

        print(f"📄 Found {len(files)} files to process")

        # Import files
        return self.import_from_files(files, label, dry_run, verbose or self.verbose)

    def import_from_github(
        self,
        github_url: str,
        label: Optional[bool] = None,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        max_depth: int = 10,
        dry_run: bool = False,
        github_token: Optional[str] = None,
        branch: str = "main",
    ) -> ImportStats:
        """
        Import prompts from GitHub repository.

        Note: This is a simplified stub. For full GitHub import functionality,
        use the original batch_importer.py or implement GitHub API integration.

        Args:
            github_url: GitHub repository URL
            label: Label for all prompts
            extensions: File extensions to include
            exclude: Path patterns to exclude
            max_size: Maximum file size in bytes
            max_depth: Maximum directory depth
            dry_run: Preview without importing
            github_token: GitHub personal access token
            branch: Git branch to use

        Returns:
            Import statistics
        """
        print(f"🌐 GitHub Import: {github_url}")
        print(f"🌿 Branch: {branch}")
        print(f"📏 Max depth: {max_depth}")

        # Show helpful message about limited functionality
        print("\n⚠️  NOTE: Simplified GitHub import has limited functionality.")
        print("   For full GitHub import features:")
        print("   1. Use the original batch_importer.py with GitHub API integration")
        print("   2. Or clone the repository locally and use --dir option")
        print("   3. Or implement proper GitHub API support")

        # Create empty stats to return
        stats = ImportStats()
        stats.start()

        print(f"\n📊 GitHub import would process repository: {github_url}")
        print("   (Actual import not implemented in simplified version)")

        stats.finish()
        self._print_import_summary(stats, dry_run)

        return stats

    def _print_import_summary(self, stats: ImportStats, dry_run: bool = False):
        """Print import summary statistics."""
        print(f"\n📊 IMPORT COMPLETE")
        print(f"   Total files: {stats.total_files}")
        print(f"   Processed: {stats.processed}")
        print(f"   Added: {stats.added}")
        print(f"   Skipped: {stats.skipped}")
        print(f"   Failed: {stats.failed}")
        print(f"   Duplicates: {stats.duplicates}")
        print(f"   Success rate: {stats.success_rate:.1f}%")
        print(f"   Duration: {stats.duration:.2f}s")

        # Show database stats if not dry run and something was added
        if not dry_run and stats.added > 0:
            all_prompts = self.store.get_all_prompts()
            injections = sum(1 for p in all_prompts if p.get("is_injection"))
            safe = len(all_prompts) - injections

            print(f"\n💾 Database now contains {len(all_prompts)} prompts:")
            print(
                f"   🔴 {injections} injection prompts ({injections / len(all_prompts) * 100:.1f}%)"
            )
            print(f"   🟢 {safe} safe prompts ({safe / len(all_prompts) * 100:.1f}%)")
