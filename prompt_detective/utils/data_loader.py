"""
Data loading utilities for prompt injection detection.
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Simple dataset for prompt injection data."""

    def __init__(self, data: List[Dict], processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        # Processor.encode() returns a dict with "input_ids" key
        encoded = self.processor.encode(text)

        # Extract the tensor from the dict and squeeze batch dimension
        input_ids = encoded["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "label": torch.tensor(label, dtype=torch.long),
        }
