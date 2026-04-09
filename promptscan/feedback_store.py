"""
Parquet-based feedback store for user submissions on model predictions.
Stores unverified user feedback to improve model accuracy.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import pandas as pd


class ParquetFeedbackStore:
    """Parquet-based store for user feedback on model predictions."""

    def __init__(self, parquet_path: str = "data/unverified_user_submissions.parquet"):
        """
        Initialize the feedback data store.

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
                    "predicted_label": pd.Series([], dtype="string"),
                    "user_label": pd.Series([], dtype="string"),
                    "ensemble_confidence": pd.Series([], dtype="float64"),
                    "individual_predictions": pd.Series(
                        [], dtype="string"
                    ),  # JSON string
                    "model_type": pd.Series([], dtype="string"),
                    "voting_strategy": pd.Series([], dtype="string"),
                    "timestamp": pd.Series([], dtype="datetime64[ns]"),
                    "source": pd.Series([], dtype="string"),
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

    def _feedback_exists(
        self, text: str, predicted_label: str, user_label: str
    ) -> bool:
        """
        Check if feedback with the same text and labels already exists.

        Args:
            text: The prompt text
            predicted_label: What the model predicted
            user_label: What the user labeled

        Returns:
            True if feedback exists, False otherwise
        """
        if len(self._data) == 0:
            return False

        # Normalize text for comparison
        normalized_text = text.strip().lower()

        # Check for exact match on text and labels
        mask = (
            (self._data["text"].str.strip().str.lower() == normalized_text)
            & (self._data["predicted_label"] == predicted_label)
            & (self._data["user_label"] == user_label)
        )

        return mask.any()

    def add_feedback(
        self,
        text: str,
        predicted_label: str,
        user_label: str,
        ensemble_confidence: float,
        individual_predictions: List[Dict[str, Any]],
        model_type: str = "ensemble",
        voting_strategy: str = "majority",
        source: str = "web_interface",
    ) -> int:
        """
        Add user feedback to the store.

        Args:
            text: The prompt text that was analyzed
            predicted_label: What the model predicted ("SAFE" or "INJECTION")
            user_label: What the user labeled ("SAFE" or "INJECTION")
            ensemble_confidence: Model's confidence score (0.0 to 1.0)
            individual_predictions: List of individual model predictions
            model_type: Type of model used ("ensemble", "cnn", "lstm", "transformer")
            voting_strategy: Voting strategy used ("majority", "weighted", "confidence", "soft")
            source: Source of the feedback ("web_interface", "api", "cli")

        Returns:
            The ID of the added feedback entry
        """
        # Check if feedback already exists
        if self._feedback_exists(text, predicted_label, user_label):
            # Find existing entry and return its ID
            normalized_text = text.strip().lower()
            mask = (
                (self._data["text"].str.strip().str.lower() == normalized_text)
                & (self._data["predicted_label"] == predicted_label)
                & (self._data["user_label"] == user_label)
            )
            existing = self._data[mask]
            if len(existing) > 0:
                return int(existing.iloc[0]["id"])

        feedback_id = self._get_next_id()

        # Convert individual_predictions to JSON string
        individual_predictions_json = json.dumps(individual_predictions)

        new_row = pd.DataFrame(
            {
                "id": [feedback_id],
                "text": [text],
                "predicted_label": [predicted_label],
                "user_label": [user_label],
                "ensemble_confidence": [ensemble_confidence],
                "individual_predictions": [individual_predictions_json],
                "model_type": [model_type],
                "voting_strategy": [voting_strategy],
                "timestamp": [datetime.now()],
                "source": [source],
            }
        )

        self._data = pd.concat([self._data, new_row], ignore_index=True)
        self._save_data()

        return feedback_id

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all feedback entries from the store.

        Returns:
            List of feedback dictionaries
        """
        if len(self._data) == 0:
            return []

        records = self._data.to_dict("records")

        # Parse JSON strings back to objects
        for record in records:
            if "individual_predictions" in record and isinstance(
                record["individual_predictions"], str
            ):
                try:
                    record["individual_predictions"] = json.loads(
                        record["individual_predictions"]
                    )
                except json.JSONDecodeError:
                    record["individual_predictions"] = []

        return records

    def get_feedback_by_id(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """
        Get feedback by its ID.

        Args:
            feedback_id: The feedback ID

        Returns:
            Feedback dictionary or None if not found
        """
        result = self._data[self._data["id"] == feedback_id]
        if len(result) == 0:
            return None

        record = result.iloc[0].to_dict()

        # Parse JSON string back to object
        if "individual_predictions" in record and isinstance(
            record["individual_predictions"], str
        ):
            try:
                record["individual_predictions"] = json.loads(
                    record["individual_predictions"]
                )
            except json.JSONDecodeError:
                record["individual_predictions"] = []

        return record

    def search_feedback(
        self,
        text_query: Optional[str] = None,
        predicted_label: Optional[str] = None,
        user_label: Optional[str] = None,
        model_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search feedback entries with filters.

        Args:
            text_query: Search query for text content
            predicted_label: Filter by predicted label
            user_label: Filter by user label
            model_type: Filter by model type
            source: Filter by source

        Returns:
            List of matching feedback dictionaries
        """
        if len(self._data) == 0:
            return []

        mask = pd.Series([True] * len(self._data), index=self._data.index)

        if text_query:
            mask = mask & self._data["text"].str.contains(
                text_query, case=False, na=False
            )

        if predicted_label:
            mask = mask & (self._data["predicted_label"] == predicted_label)

        if user_label:
            mask = mask & (self._data["user_label"] == user_label)

        if model_type:
            mask = mask & (self._data["model_type"] == model_type)

        if source:
            mask = mask & (self._data["source"] == source)

        results = self._data[mask]
        records = results.to_dict("records")

        # Parse JSON strings back to objects
        for record in records:
            if "individual_predictions" in record and isinstance(
                record["individual_predictions"], str
            ):
                try:
                    record["individual_predictions"] = json.loads(
                        record["individual_predictions"]
                    )
                except json.JSONDecodeError:
                    record["individual_predictions"] = []

        return records

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback data.

        Returns:
            Dictionary with statistics
        """
        if len(self._data) == 0:
            return {
                "total": 0,
                "by_predicted_label": {},
                "by_user_label": {},
                "by_model_type": {},
                "by_source": {},
                "agreement_rate": 0.0,
            }

        total = len(self._data)

        # Count by predicted label
        predicted_counts = self._data["predicted_label"].value_counts().to_dict()

        # Count by user label
        user_counts = self._data["user_label"].value_counts().to_dict()

        # Count by model type
        model_counts = self._data["model_type"].value_counts().to_dict()

        # Count by source
        source_counts = self._data["source"].value_counts().to_dict()

        # Calculate agreement rate (where predicted_label == user_label)
        agreement_mask = self._data["predicted_label"] == self._data["user_label"]
        agreement_count = agreement_mask.sum()
        agreement_rate = agreement_count / total if total > 0 else 0.0

        return {
            "total": total,
            "by_predicted_label": predicted_counts,
            "by_user_label": user_counts,
            "by_model_type": model_counts,
            "by_source": source_counts,
            "agreement_rate": float(agreement_rate),
            "agreement_percentage": float(agreement_rate * 100),
        }

    def clear_data(self) -> None:
        """Clear all data from the store."""
        self._data = pd.DataFrame(
            {
                "id": pd.Series([], dtype="int64"),
                "text": pd.Series([], dtype="string"),
                "predicted_label": pd.Series([], dtype="string"),
                "user_label": pd.Series([], dtype="string"),
                "ensemble_confidence": pd.Series([], dtype="float64"),
                "individual_predictions": pd.Series([], dtype="string"),
                "model_type": pd.Series([], dtype="string"),
                "voting_strategy": pd.Series([], dtype="string"),
                "timestamp": pd.Series([], dtype="datetime64[ns]"),
                "source": pd.Series([], dtype="string"),
            }
        )
        self._save_data()

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export data as pandas DataFrame.

        Returns:
            DataFrame with all feedback entries
        """
        return self._data.copy()

    def import_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Import data from pandas DataFrame.

        Args:
            df: DataFrame with required columns
        """
        # Validate required columns
        required_columns = {
            "text",
            "predicted_label",
            "user_label",
            "ensemble_confidence",
        }
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Generate IDs if not present
        if "id" not in df.columns:
            start_id = self._get_next_id()
            df = df.copy()
            df["id"] = range(start_id, start_id + len(df))

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = datetime.now()

        # Ensure source column exists
        if "source" not in df.columns:
            df = df.copy()
            df["source"] = "imported"

        # Ensure individual_predictions is JSON string
        if "individual_predictions" in df.columns:
            # Convert list/dict to JSON string if needed
            def convert_to_json(x):
                if isinstance(x, (list, dict)):
                    return json.dumps(x)
                return x

            df["individual_predictions"] = df["individual_predictions"].apply(
                convert_to_json
            )

        # Append to existing data
        self._data = pd.concat([self._data, df], ignore_index=True)
        self._save_data()


if __name__ == "__main__":
    # Test the feedback store
    store = ParquetFeedbackStore()

    # Test adding feedback
    test_predictions = [
        {"model": "CNN", "prediction": "SAFE", "confidence": 0.85},
        {"model": "LSTM", "prediction": "INJECTION", "confidence": 0.72},
        {"model": "Transformer", "prediction": "SAFE", "confidence": 0.91},
    ]

    feedback_id = store.add_feedback(
        text="Test prompt for feedback system",
        predicted_label="SAFE",
        user_label="INJECTION",  # User disagrees
        ensemble_confidence=0.82,
        individual_predictions=test_predictions,
        model_type="ensemble",
        voting_strategy="majority",
        source="test",
    )

    print(f"Added feedback with ID: {feedback_id}")

    # Get statistics
    stats = store.get_statistics()
    print(f"Statistics: {stats}")

    # Get all feedback
    all_feedback = store.get_all_feedback()
    print(f"Total feedback entries: {len(all_feedback)}")
