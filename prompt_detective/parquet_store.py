"""
Parquet-based data store for prompt injection detection.
Replaces the SQLite-based PromptDatabase with Parquet format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ParquetDataStore:
    """Parquet-based data store for prompt injection data."""

    def __init__(self, parquet_path: str = "data/prompts.parquet"):
        """
        Initialize the parquet data store.

        Args:
            parquet_path: Path to the parquet file
        """
        self.parquet_path = Path(parquet_path)
        self._data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load data from parquet file into memory."""
        if not self.parquet_path.exists():
            # Create empty DataFrame with correct schema
            self._data = pd.DataFrame(
                {
                    "id": pd.Series([], dtype="int64"),
                    "text": pd.Series([], dtype="string"),
                    "is_injection": pd.Series([], dtype="bool"),
                }
            )
            # Save empty DataFrame to create the file
            self._save_data()
        else:
            self._data = pd.read_parquet(self.parquet_path)
            # Ensure the DataFrame has an 'id' column
            if "id" not in self._data.columns:
                # Add an 'id' column with sequential numbers
                self._data = self._data.reset_index(drop=True)
                self._data["id"] = self._data.index + 1
                # Save with the new ID column
                self._save_data()

    def _save_data(self) -> None:
        """Save data to parquet file."""
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self._data.to_parquet(self.parquet_path, index=False)

    def _get_next_id(self) -> int:
        """Get the next available ID."""
        if len(self._data) == 0 or "id" not in self._data.columns:
            return 1
        return int(self._data["id"].max() + 1)

    def add_prompt(self, text: str, is_injection: bool) -> int:
        """
        Add a single prompt to the store.

        Args:
            text: The prompt text
            is_injection: Whether it's an injection prompt

        Returns:
            The ID of the added prompt
        """
        prompt_id = self._get_next_id()
        new_row = pd.DataFrame(
            {"id": [prompt_id], "text": [text], "is_injection": [is_injection]}
        )

        self._data = pd.concat([self._data, new_row], ignore_index=True)
        self._save_data()

        return prompt_id

    def add_prompts_batch(self, prompts: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple prompts in batch.

        Args:
            prompts: List of prompt dictionaries with 'text' and 'is_injection' keys
                     Optional keys: 'source', 'category', 'variation_type'

        Returns:
            List of IDs for the added prompts
        """
        if not prompts:
            return []

        start_id = self._get_next_id()
        new_data = []

        for i, prompt in enumerate(prompts):
            # Create base record
            record = {
                "id": start_id + i,
                "text": prompt["text"],
                "is_injection": prompt["is_injection"],
            }

            # Add optional fields if present
            optional_fields = ["source", "category", "variation_type"]
            for field in optional_fields:
                if field in prompt:
                    record[field] = prompt[field]

            new_data.append(record)

        new_df = pd.DataFrame(new_data)
        self._data = pd.concat([self._data, new_df], ignore_index=True)
        self._save_data()

        return list(range(start_id, start_id + len(prompts)))

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
            start_id = self._get_next_id()
            df = df.copy()
            df["id"] = range(start_id, start_id + len(df))

        # Ensure correct dtypes
        df = df.astype({"id": "int64", "text": "string", "is_injection": "bool"})

        # Append to existing data
        self._data = pd.concat([self._data, df], ignore_index=True)
        self._save_data()

    def get_training_splits(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Get train/validation/test splits.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if len(self._data) == 0:
            return {
                "train": pd.DataFrame(),
                "val": pd.DataFrame(),
                "test": pd.DataFrame(),
            }

        # Shuffle the data
        df_shuffled = self._data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split sizes
        n_total = len(df_shuffled)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        # Split the data
        train_df = df_shuffled.iloc[:n_train]
        val_df = df_shuffled.iloc[n_train : n_train + n_val]
        test_df = df_shuffled.iloc[n_train + n_val :]

        return {"train": train_df, "val": val_df, "test": test_df}
