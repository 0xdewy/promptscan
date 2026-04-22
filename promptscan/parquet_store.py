"""
Parquet-based data store for prompt injection detection.
Replaces the SQLite-based PromptDatabase with Parquet format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class ParquetDataStore:
    """Parquet-based data store for prompt injection data."""

    def __init__(self, parquet_path: str = "data/merged.parquet"):
        """
        Initialize the parquet data store.

        Args:
            parquet_path: Path to the parquet file
        """
        self.parquet_path = Path(parquet_path)
        self._data = None
        self._text_index = None
        self._load_data()

    def _load_data(self) -> None:
        """Load data from parquet file into memory."""
        if not self.parquet_path.exists():
            self._data = pd.DataFrame(
                {
                    "id": pd.Series([], dtype="int64"),
                    "text": pd.Series([], dtype="string"),
                    "is_injection": pd.Series([], dtype="bool"),
                }
            )
            self._save_data()
        else:
            self._data = pd.read_parquet(self.parquet_path)
            if "id" not in self._data.columns:
                self._data = self._data.reset_index(drop=True)
                self._data["id"] = self._data.index + 1
                self._save_data()
        self._build_text_index()

    def _build_text_index(self) -> None:
        """Build a set index of normalized text + is_injection for fast lookup."""
        self._text_index = set()
        if len(self._data) > 0:
            for _, row in self._data.iterrows():
                normalized = (row["text"].strip().lower(), row["is_injection"])
                self._text_index.add(normalized)

    def _invalidate_index(self) -> None:
        """Mark text index as needing rebuild."""
        self._text_index = None

    def _save_data(self) -> None:
        """Save data to parquet file."""
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self._data.to_parquet(self.parquet_path, index=False)

    def _get_next_id(self) -> str:
        """Get the next available ID for a new prompt."""
        import uuid

        return str(uuid.uuid4())

    def _prompt_exists(self, text: str, is_injection: bool) -> bool:
        """
        Check if a prompt with the same text and label already exists.

        Args:
            text: The prompt text
            is_injection: Whether it's an injection prompt

        Returns:
            True if prompt exists, False otherwise
        """
        if self._text_index is None:
            self._build_text_index()

        normalized = (text.strip().lower(), is_injection)
        return normalized in self._text_index

    def add_prompt(self, text: str, is_injection: bool) -> Optional[str]:
        """
        Add a single prompt to the store.

        Args:
            text: The prompt text
            is_injection: Whether it's an injection prompt

        Returns:
            The ID of the added prompt, or None if duplicate
        """
        # Check if prompt already exists
        if self._prompt_exists(text, is_injection):
            return None

        prompt_id = self._get_next_id()
        new_row = pd.DataFrame(
            {"id": [prompt_id], "text": [text], "is_injection": [is_injection]}
        )

        self._data = pd.concat([self._data, new_row], ignore_index=True)
        self._save_data()
        self._invalidate_index()

        return prompt_id

    def add_prompts_batch(self, prompts: List[Dict[str, Any]]) -> Tuple[List[str], int]:
        """
        Add multiple prompts in batch.

        Args:
            prompts: List of prompt dictionaries with 'text' and 'is_injection' keys
                     Optional keys: 'source', 'category', 'variation_type'

        Returns:
            List of IDs for the added prompts
        """
        if not prompts:
            return [], 0

        import uuid

        new_data = []
        added_ids = []
        skipped_count = 0

        for prompt in prompts:
            text = prompt["text"]
            is_injection = prompt["is_injection"]

            # Check if prompt already exists
            if self._prompt_exists(text, is_injection):
                skipped_count += 1
                continue

            # Generate unique ID for each prompt
            prompt_id = str(uuid.uuid4())

            # Create base record
            record = {
                "id": prompt_id,
                "text": text,
                "is_injection": is_injection,
            }

            # Add optional fields if present
            optional_fields = ["source", "category", "variation_type"]
            for field in optional_fields:
                if field in prompt:
                    record[field] = prompt[field]

            new_data.append(record)
            added_ids.append(prompt_id)

        if new_data:
            new_df = pd.DataFrame(new_data)
            self._data = pd.concat([self._data, new_df], ignore_index=True)
            self._save_data()
            self._invalidate_index()

        # Return tuple with added IDs and skipped count
        return added_ids, skipped_count

    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all prompts from the store.

        Returns:
            List of prompt dictionaries
        """
        return self._data.to_dict("records")

    def get_prompt_by_id(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a prompt by its ID.

        Args:
            prompt_id: The prompt ID

        Returns:
            Prompt dictionary or None if not found
        """
        result = self._data[self._data["id"] == prompt_id]
        if len(result) == 0:
            return None
        return result.iloc[0].to_dict()

    def update_prompt(self, prompt_id: int, text: str, is_injection: bool) -> bool:
        """
        Update a prompt.

        Args:
            prompt_id: The prompt ID
            text: New text
            is_injection: New injection status

        Returns:
            True if successful, False if prompt not found
        """
        if prompt_id not in self._data["id"].values:
            return False

        # Update the row
        mask = self._data["id"] == prompt_id
        self._data.loc[mask, "text"] = text
        self._data.loc[mask, "is_injection"] = is_injection

        self._save_data()
        return True

    def delete_prompt(self, prompt_id: int) -> bool:
        """
        Delete a prompt by ID.

        Args:
            prompt_id: The prompt ID

        Returns:
            True if successful, False if prompt not found
        """
        if prompt_id not in self._data["id"].values:
            return False

        # Remove the row
        self._data = self._data[self._data["id"] != prompt_id]
        self._save_data()
        return True

    def search_prompts(self, query: str) -> List[Dict[str, Any]]:
        """
        Search prompts by text content.

        Args:
            query: Search query string

        Returns:
            List of matching prompt dictionaries
        """
        if not query:
            return self.get_all_prompts()

        # Case-insensitive search
        mask = self._data["text"].str.contains(query, case=False, na=False)
        results = self._data[mask]
        return results.to_dict("records")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the data.

        Returns:
            Dictionary with statistics
        """
        total = len(self._data)
        if total == 0:
            return {
                "total": 0,
                "injections": 0,
                "safe": 0,
                "injection_percentage": 0.0,
                "safe_percentage": 0.0,
            }

        injections = int(self._data["is_injection"].sum())
        safe = total - injections

        return {
            "total": total,
            "injections": injections,
            "safe": safe,
            "injection_percentage": (injections / total * 100),
            "safe_percentage": (safe / total * 100),
        }

    def clear_data(self) -> None:
        """Clear all data from the store."""
        self._data = pd.DataFrame(
            {
                "id": pd.Series([], dtype="int64"),
                "text": pd.Series([], dtype="string"),
                "is_injection": pd.Series([], dtype="bool"),
            }
        )
        self._save_data()
        self._invalidate_index()

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export data as pandas DataFrame.

        Returns:
            DataFrame with all prompts
        """
        return self._data.copy()

    def import_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Import data from pandas DataFrame.

        Args:
            df: DataFrame with 'text' and 'is_injection' columns
        """
        # Validate required columns
        required_columns = {"text", "is_injection"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Generate IDs if not present
        if "id" not in df.columns:
            import uuid

            df = df.copy()
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Ensure correct dtypes
        df = df.astype({"text": "string", "is_injection": "bool"})

        # Append to existing data
        self._data = pd.concat([self._data, df], ignore_index=True)
        self._save_data()

    def get_training_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_samples: int = 0,
        max_samples_per_source: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get train/validation/test splits with stratification.

        Uses stratified splitting to maintain consistent class distribution
        across train/val/test sets, which is critical for imbalanced datasets.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            max_samples: Maximum number of samples to use (0 = use all)
            max_samples_per_source: Cap samples per source to reduce source dominance (0 = no cap)

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        from sklearn.model_selection import train_test_split

        if len(self._data) == 0:
            return {
                "train": pd.DataFrame(),
                "val": pd.DataFrame(),
                "test": pd.DataFrame(),
            }

        df = self._data

        # Per-source cap: prevents any single source from dominating training
        if max_samples_per_source > 0 and "source" in df.columns:
            before = len(df)
            capped = []
            for src in df["source"].unique():
                src_df = df[df["source"] == src]
                if len(src_df) > max_samples_per_source:
                    src_df = src_df.sample(max_samples_per_source, random_state=42)
                capped.append(src_df)
            df = pd.concat(capped, ignore_index=True)
            print(f"  Per-source cap ({max_samples_per_source:,}): {before:,} → {len(df):,} rows")
            for src in df["source"].unique():
                grp = df[df["source"] == src]
                inj = grp["is_injection"].sum()
                print(f"    {src}: {len(grp):,} ({inj} inj / {len(grp)-inj} safe)")

        if max_samples > 0 and max_samples < len(self._data):
            # Use stratified sampling when limiting samples
            df, _ = train_test_split(
                self._data,
                train_size=max_samples,
                stratify=self._data["is_injection"],
                random_state=42,
            )
            df = df.reset_index(drop=True)

        # Stratified split to maintain class balance across splits
        # First split: train vs (val + test)
        test_val_ratio = 1 - train_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=test_val_ratio,
            stratify=df["is_injection"],
            random_state=42,
        )

        # Second split: val vs test (from the temp set)
        # val_ratio is relative to total, so we need to compute relative to temp
        # val_ratio / test_val_ratio gives us the proportion of temp that should be val
        val_proportion = val_ratio / test_val_ratio if test_val_ratio > 0 else 0.5
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_proportion,
            stratify=temp_df["is_injection"],
            random_state=42,
        )

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }
