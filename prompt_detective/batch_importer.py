#!/usr/bin/env python3
"""
Batch importer for processing multiple files from various sources.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Generator
from collections import defaultdict

from .github_client import GitHubClient, GitHubRepoInfo
from .text_extractors import TextExtractor
from .parquet_store import ParquetDataStore


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
        """String representation of stats."""
        return (
            f"Import Statistics:\n"
            f"  Total files: {self.total_files}\n"
            f"  Processed: {self.processed}\n"
            f"  Added: {self.added}\n"
            f"  Skipped: {self.skipped}\n"
            f"  Failed: {self.failed}\n"
            f"  Duplicates: {self.duplicates}\n"
            f"  Success rate: {self.success_rate:.1f}%\n"
            f"  Duration: {self.duration:.2f}s"
        )


class BatchImporter:
    """Batch importer for processing files from various sources."""

    def __init__(
        self,
        store: ParquetDataStore,
        verbose: bool = False,
        github_token: Optional[str] = None,
    ):
        """
        Initialize batch importer.

        Args:
            store: Parquet data store for saving prompts
            verbose: Enable verbose output
            github_token: GitHub personal access token (optional)
        """
        self.store = store
        self.verbose = verbose

        # Initialize GitHub client with token if provided
        self.github_client = GitHubClient(token=github_token)
        self.text_extractor = TextExtractor()

        # Track seen content to avoid duplicates
        self.seen_content: Set[str] = set()

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
        Import prompts from a GitHub repository.

        Args:
            github_url: GitHub repository URL
            label: Label for all prompts (True for injection, False for safe)
            extensions: List of file extensions to include (None for all supported)
            exclude: List of path patterns to exclude
            max_size: Maximum file size in bytes
            max_depth: Maximum directory recursion depth
            dry_run: Preview without importing
            github_token: GitHub personal access token
            branch: Git branch to use

        Returns:
            Import statistics
        """
        stats = ImportStats()
        stats.start()

        try:
            # Parse GitHub URL
            repo_info = self.github_client.parse_github_url(github_url)
            if branch != "main":
                repo_info.branch = branch

            print(f"📦 Importing from GitHub: {repo_info.owner}/{repo_info.repo}")
            print(f"   Branch: {repo_info.branch}")
            print(f"   Path: {repo_info.path or '/'}")
            print(
                f"   Label: {'injection' if label else 'safe' if label is not None else 'interactive'}"
            )
            print(f"   Dry run: {dry_run}")
            print()

            # Walk repository
            files = list(
                self.github_client.walk_repository(
                    owner=repo_info.owner,
                    repo=repo_info.repo,
                    path=repo_info.path,
                    ref=repo_info.branch,
                    max_depth=max_depth,
                )
            )

            stats.total_files = len(files)
            print(f"Found {stats.total_files} files in repository")

            # Process files
            prompts = []
            for i, file in enumerate(files, 1):
                # Show progress
                if (
                    self.verbose
                    or i % max(1, stats.total_files // 20) == 0
                    or i == stats.total_files
                ):
                    progress = i / stats.total_files * 100
                    print(
                        f"  [{i}/{stats.total_files}] {progress:.1f}% - Processing: {file.name[:50]}..."
                    )

                result = self._process_github_file(
                    file=file,
                    repo_info=repo_info,
                    extensions=extensions,
                    exclude=exclude,
                    max_size=max_size,
                    label=label,
                    dry_run=dry_run,
                )

                if result:
                    file_stats, prompt_data = result
                    stats.processed += 1
                    stats.skipped += file_stats.get("skipped", 0)
                    stats.failed += file_stats.get("failed", 0)

                    if prompt_data:
                        prompts.append(prompt_data)

                # Batch insert
                if prompts and len(prompts) >= 100:
                    if not dry_run:
                        added = self.store.add_prompts_batch(prompts)
                        stats.added += len(added)
                    else:
                        stats.added += len(prompts)
                    prompts = []

            # Insert remaining prompts
            if prompts:
                if not dry_run:
                    added = self.store.add_prompts_batch(prompts)
                    stats.added += len(added)
                else:
                    stats.added += len(prompts)

            # Show completion message
            if stats.total_files > 0:
                success_rate = (
                    (stats.processed - stats.failed) / stats.total_files * 100
                )
                print(
                    f"\n✅ Processing complete: {stats.processed}/{stats.total_files} files ({success_rate:.1f}% success)"
                )

        except Exception as e:
            print(f"❌ Error importing from GitHub: {e}")
            stats.failed += 1

        stats.finish()
        return stats

    def _process_github_file(
        self,
        file: "GitHubFile",
        repo_info: GitHubRepoInfo,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        label: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Optional[Tuple[Dict, Optional[Dict]]]:
        """
        Process a single GitHub file.

        Returns:
            Tuple of (file_stats, prompt_data) or None if skipped
        """
        file_stats = {"skipped": 0, "failed": 0}

        # Check file extension
        ext = Path(file.name).suffix.lower()
        if extensions and ext not in extensions:
            if self.verbose:
                print(
                    f"  ⏭️  Skipping {file.path} (extension {ext} not in {extensions})"
                )
            file_stats["skipped"] = 1
            return (file_stats, None)

        # Check exclude patterns
        if exclude and any(excl in file.path for excl in exclude):
            if self.verbose:
                print(f"  ⏭️  Skipping {file.path} (excluded pattern)")
            file_stats["skipped"] = 1
            return (file_stats, None)

        # Check file size
        if max_size and file.size > max_size:
            if self.verbose:
                print(f"  ⏭️  Skipping {file.path} (size {file.size} > {max_size})")
            file_stats["skipped"] = 1
            return (file_stats, None)

        try:
            # Get file content
            content = self.github_client.get_file_content(
                owner=repo_info.owner,
                repo=repo_info.repo,
                path=file.path,
                ref=repo_info.branch,
            )

            # Extract text
            extracted = self.text_extractor.extract_text(file.path, content)
            if not extracted or len(extracted.strip()) < 10:
                if self.verbose:
                    print(f"  ⏭️  Skipping {file.path} (no extractable text)")
                file_stats["skipped"] = 1
                return (file_stats, None)

            # Check for duplicates
            content_hash = hash(extracted.strip())
            if content_hash in self.seen_content:
                if self.verbose:
                    print(f"  ⏭️  Skipping {file.path} (duplicate content)")
                file_stats["skipped"] = 1
                return (file_stats, None)

            self.seen_content.add(content_hash)

            # Get label if not provided
            if label is None:
                # Interactive mode - would need user input
                # For now, skip or use default
                print(
                    f"  ❓ File {file.path} needs label (interactive mode not supported in batch)"
                )
                file_stats["skipped"] = 1
                return (file_stats, None)

            # Create prompt data
            prompt_data = {
                "text": extracted.strip(),
                "is_injection": label,
                "source": f"github:{repo_info.owner}/{repo_info.repo}/{file.path}",
            }

            if self.verbose:
                print(f"  ✅ Processed {file.path} ({len(extracted)} chars)")

            return (file_stats, prompt_data)

        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to process {file.path}: {e}")
            file_stats["failed"] = 1
            return (file_stats, None)

    def import_from_directory(
        self,
        directory: str,
        label: Optional[bool] = None,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        recursive: bool = True,
        dry_run: bool = False,
    ) -> ImportStats:
        """
        Import prompts from a local directory.

        Args:
            directory: Local directory path
            label: Label for all prompts
            extensions: List of file extensions to include
            exclude: List of path patterns to exclude
            max_size: Maximum file size in bytes
            recursive: Whether to search recursively
            dry_run: Preview without importing

        Returns:
            Import statistics
        """
        stats = ImportStats()
        stats.start()

        directory = Path(directory).resolve()
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            stats.failed += 1
            stats.finish()
            return stats

        print(f"📂 Importing from directory: {directory}")
        print(
            f"   Label: {'injection' if label else 'safe' if label is not None else 'interactive'}"
        )
        print(f"   Recursive: {recursive}")
        print(f"   Dry run: {dry_run}")
        print()

        # Find files
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.glob("*"))

        # Filter to files only
        files = [f for f in files if f.is_file()]
        stats.total_files = len(files)

        print(f"Found {stats.total_files} files in directory")

        # Process files
        prompts = []
        for i, file_path in enumerate(files, 1):
            if self.verbose and i % 10 == 0:
                print(f"  Processing file {i}/{stats.total_files}...")

            result = self._process_local_file(
                file_path=file_path,
                directory=directory,
                extensions=extensions,
                exclude=exclude,
                max_size=max_size,
                label=label,
                dry_run=dry_run,
            )

            if result:
                file_stats, prompt_data = result
                stats.processed += 1
                stats.skipped += file_stats.get("skipped", 0)
                stats.failed += file_stats.get("failed", 0)

                if prompt_data:
                    prompts.append(prompt_data)

            # Batch insert
            if prompts and len(prompts) >= 100:
                if not dry_run:
                    added = self.store.add_prompts_batch(prompts)
                    stats.added += len(added)
                else:
                    stats.added += len(prompts)
                prompts = []

        # Insert remaining prompts
        if prompts:
            if not dry_run:
                added = self.store.add_prompts_batch(prompts)
                stats.added += len(added)
            else:
                stats.added += len(prompts)

        stats.finish()
        return stats

    def _process_local_file(
        self,
        file_path: Path,
        directory: Path,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        label: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Optional[Tuple[Dict, Optional[Dict]]]:
        """
        Process a single local file.
        """
        file_stats = {"skipped": 0, "failed": 0}

        # Get relative path for display
        try:
            rel_path = file_path.relative_to(directory)
        except ValueError:
            rel_path = file_path

        # Check file extension
        ext = file_path.suffix.lower()
        if extensions and ext not in extensions:
            if self.verbose:
                print(f"  ⏭️  Skipping {rel_path} (extension {ext} not in {extensions})")
            file_stats["skipped"] = 1
            return (file_stats, None)

        # Check exclude patterns
        if exclude and any(excl in str(rel_path) for excl in exclude):
            if self.verbose:
                print(f"  ⏭️  Skipping {rel_path} (excluded pattern)")
            file_stats["skipped"] = 1
            return (file_stats, None)

        # Check file size
        if max_size:
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size:
                    if self.verbose:
                        print(
                            f"  ⏭️  Skipping {rel_path} (size {file_size} > {max_size})"
                        )
                    file_stats["skipped"] = 1
                    return (file_stats, None)
            except OSError:
                pass

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract text
            extracted = self.text_extractor.extract_text(str(file_path), content)
            if not extracted or len(extracted.strip()) < 10:
                if self.verbose:
                    print(f"  ⏭️  Skipping {rel_path} (no extractable text)")
                file_stats["skipped"] = 1
                return (file_stats, None)

            # Check for duplicates
            content_hash = hash(extracted.strip())
            if content_hash in self.seen_content:
                if self.verbose:
                    print(f"  ⏭️  Skipping {rel_path} (duplicate content)")
                file_stats["skipped"] = 1
                return (file_stats, None)

            self.seen_content.add(content_hash)

            # Get label if not provided
            if label is None:
                # Interactive mode - would need user input
                print(
                    f"  ❓ File {rel_path} needs label (interactive mode not supported in batch)"
                )
                file_stats["skipped"] = 1
                return (file_stats, None)

            # Create prompt data
            prompt_data = {
                "text": extracted.strip(),
                "is_injection": label,
                "source": f"file:{rel_path}",
            }

            if self.verbose:
                print(f"  ✅ Processed {rel_path} ({len(extracted)} chars)")

            return (file_stats, prompt_data)

        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to process {rel_path}: {e}")
            file_stats["failed"] = 1
            return (file_stats, None)

    def import_from_files(
        self, files: List[str], label: Optional[bool] = None, dry_run: bool = False
    ) -> ImportStats:
        """
        Import prompts from specific files.

        Args:
            files: List of file paths
            label: Label for all prompts
            dry_run: Preview without importing

        Returns:
            Import statistics
        """
        stats = ImportStats()
        stats.start()

        print(f"📄 Importing {len(files)} files")
        print(
            f"   Label: {'injection' if label else 'safe' if label is not None else 'interactive'}"
        )
        print(f"   Dry run: {dry_run}")
        print()

        stats.total_files = len(files)

        # Process files
        prompts = []
        for i, file_path_str in enumerate(files, 1):
            file_path = Path(file_path_str).resolve()

            if self.verbose and i % 10 == 0:
                print(f"  Processing file {i}/{stats.total_files}...")

            if not file_path.exists():
                print(f"  ❌ File not found: {file_path}")
                stats.failed += 1
                continue

            result = self._process_local_file(
                file_path=file_path,
                directory=file_path.parent,
                extensions=None,  # No extension filtering for specific files
                exclude=None,
                max_size=None,
                label=label,
                dry_run=dry_run,
            )

            if result:
                file_stats, prompt_data = result
                stats.processed += 1
                stats.skipped += file_stats.get("skipped", 0)
                stats.failed += file_stats.get("failed", 0)

                if prompt_data:
                    prompts.append(prompt_data)

            # Batch insert
            if prompts and len(prompts) >= 100:
                if not dry_run:
                    added = self.store.add_prompts_batch(prompts)
                    stats.added += len(added)
                else:
                    stats.added += len(prompts)
                prompts = []

        # Insert remaining prompts
        if prompts:
            if not dry_run:
                added = self.store.add_prompts_batch(prompts)
                stats.added += len(added)
            else:
                stats.added += len(prompts)

        stats.finish()
        return stats


def test_batch_importer():
    """Test function for batch importer."""
    from .parquet_store import ParquetDataStore

    # Create a temporary store
    store = ParquetDataStore("/tmp/test_prompts.parquet")
    importer = BatchImporter(store, verbose=True)

    # Test with a small public repo
    stats = importer.import_from_github(
        github_url="https://github.com/openai/openai-python",
        label=False,  # safe
        extensions=[".md", ".txt"],
        max_size=100000,  # 100KB
        max_depth=2,
        dry_run=True,  # Don't actually import
    )

    print(stats)


if __name__ == "__main__":
    test_batch_importer()
