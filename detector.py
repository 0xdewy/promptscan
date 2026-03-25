#!/usr/bin/env python3
"""
Simple Prompt Injection Detector
A clean, minimal neural network for detecting prompt injections.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SimpleTextProcessor:
    """Minimal text processor for prompt injection detection."""

    def __init__(self, max_length=100):
        self.max_length = max_length
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2

    def build_vocab(self, texts: List[str], min_freq=2):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = self.next_id
                self.inverse_vocab[self.next_id] = word
                self.next_id += 1

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = self._tokenize(text)
        token_ids = []

        for word in words:
            token_ids.append(self.vocab.get(word, 1))  # 1 = <UNK>

        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        return token_ids


class SimpleInjectionDetector(nn.Module):
    """Simple CNN model for prompt injection detection."""

    def __init__(self, vocab_size, embedding_dim=64, num_filters=50, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Simple CNN with multiple filter sizes
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(0.3)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embeddings = embeddings.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # Apply convolutions
        conv3_out = torch.relu(self.conv3(embeddings))
        conv4_out = torch.relu(self.conv4(embeddings))
        conv5_out = torch.relu(self.conv5(embeddings))

        # Max pooling
        pooled3 = torch.max(conv3_out, dim=2)[0]
        pooled4 = torch.max(conv4_out, dim=2)[0]
        pooled5 = torch.max(conv5_out, dim=2)[0]

        # Concatenate
        concatenated = torch.cat([pooled3, pooled4, pooled5], dim=1)
        concatenated = self.dropout(concatenated)

        # Classify
        logits = self.fc(concatenated)
        return logits


class PromptDataset(Dataset):
    """Simple dataset for prompt injection data."""

    def __init__(self, data: List[Dict], processor: SimpleTextProcessor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        token_ids = self.processor.encode(text)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SimpleTrainer:
    """Minimal trainer for the model."""

    def __init__(self, model, train_loader, val_loader, processor, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            total_loss += loss.item() * input_ids.size(0)
            total_correct += correct
            total_samples += input_ids.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()

                total_loss += loss.item() * input_ids.size(0)
                total_correct += correct
                total_samples += input_ids.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def train(self, epochs=20):
        """Train the model."""
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "vocab": self.processor.vocab,  # Save full vocabulary
                        "max_length": self.processor.max_length,
                        "val_acc": val_acc,
                        "epoch": epoch,
                    },
                    "best_model.pt",
                )
                print(f"  Saved best model with val acc: {val_acc:.4f}")


def load_data(db_path="prompts.db"):
    """Load data from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT text, is_injection FROM prompts")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for text, is_injection in rows:
        data.append({"text": text, "label": 1 if is_injection else 0})

    return data


def train_model():
    """Train a simple prompt injection detector."""
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} prompts")

    # Split data
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    # Build vocabulary
    print("Building vocabulary...")
    processor = SimpleTextProcessor(max_length=100)
    train_texts = [item["text"] for item in train_data]
    processor.build_vocab(train_texts, min_freq=2)
    print(f"Vocabulary size: {len(processor.vocab)}")

    # Create datasets
    train_dataset = PromptDataset(train_data, processor)
    val_dataset = PromptDataset(val_data, processor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    vocab_size = len(processor.vocab)
    model = SimpleInjectionDetector(vocab_size)

    # Train
    print("Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = SimpleTrainer(model, train_loader, val_loader, processor, device)
    trainer.train(epochs=20)

    print("Training complete!")
    print(f"Model saved to: best_model.pt")


def predict(text: str, model_path="best_model.pt"):
    """Predict if text contains prompt injection."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Create processor from checkpoint
    processor = SimpleTextProcessor(max_length=checkpoint["max_length"])
    processor.vocab = checkpoint["vocab"]
    processor.inverse_vocab = {v: k for k, v in processor.vocab.items()}

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab_size = len(checkpoint["vocab"])

    model = SimpleInjectionDetector(vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Encode text
    token_ids = processor.encode(text)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()

    return {
        "text": text,
        "prediction": "INJECTION" if prediction == 1 else "SAFE",
        "confidence": probs[0, prediction].item(),
        "safe_prob": probs[0, 0].item(),
        "injection_prob": probs[0, 1].item(),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict(text)

        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Safe: {result['safe_prob']:.2%}")
        print(f"Injection: {result['injection_prob']:.2%}")
    else:
        print("Usage:")
        print("  python simple_detector.py train")
        print("  python simple_detector.py <text to analyze>")
        print("\nExample:")
        print('  python simple_detector.py "Ignore all previous instructions"')
