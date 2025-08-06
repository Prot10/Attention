"""Data utilities for attention research."""

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

from ..shared.config import Config

logger = logging.getLogger(__name__)


class WikiTextDataset(Dataset):
    """WikiText dataset wrapper for PyTorch."""

    def __init__(self, texts: list[str], tokenizer: Tokenizer, max_length: int = 1024) -> None:
        """Initialize dataset.

        Args:
            texts: List of text strings
            tokenizer: Trained tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item."""
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids

        # Truncate or pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
        else:
            # Pad with padding token (assumed to be 0)
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        # Create attention mask
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),  # For language modeling
        }


def create_tokenizer(config: Config) -> Tokenizer:
    """Create and train a custom tokenizer.

    Args:
        config: Configuration object

    Returns:
        Trained tokenizer
    """
    tokenizer_config = config.get_dataset_config()
    vocab_size = config.get("tokenizer.vocab_size", 32000)
    special_tokens = config.get("tokenizer.special_tokens", ["<pad>", "<unk>", "<s>", "</s>"])
    save_path = Path(config.get("tokenizer.save_path", "./data/tokenizer"))

    # Check if tokenizer already exists
    tokenizer_file = save_path / "tokenizer.json"
    if tokenizer_file.exists():
        logger.info(f"Loading existing tokenizer from {tokenizer_file}")
        return Tokenizer.from_file(str(tokenizer_file))

    logger.info("Training new tokenizer...")

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFD()
    tokenizer.normalizer = StripAccents()
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()

    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Load dataset for training
    dataset = load_dataset(
        tokenizer_config.name,
        tokenizer_config.subset,
        split="train",
        cache_dir=tokenizer_config.cache_dir,
    )

    # Get text iterator
    def get_training_corpus():
        for item in dataset:
            yield item["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Configure post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )

    # Save tokenizer
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_file))
    logger.info(f"Tokenizer saved to {tokenizer_file}")

    return tokenizer


def load_wikitext_data(config: Config) -> tuple[DatasetDict, Tokenizer]:
    """Load WikiText dataset and tokenizer.

    Args:
        config: Configuration object

    Returns:
        Tuple of (dataset, tokenizer)
    """
    dataset_config = config.get_dataset_config()

    # Load dataset
    logger.info(f"Loading dataset: {dataset_config.name}")
    dataset = load_dataset(
        dataset_config.name,
        dataset_config.subset,
        cache_dir=dataset_config.cache_dir,
    )

    # Create/load tokenizer
    tokenizer = create_tokenizer(config)

    return dataset, tokenizer


def create_dataloaders(
    dataset: DatasetDict,
    tokenizer: Tokenizer,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Trained tokenizer
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    training_config = config.get_training_config()
    max_length = config.get("model.max_sequence_length", 1024)
    batch_size = training_config.batch_size

    # Create datasets
    train_dataset = WikiTextDataset(dataset["train"]["text"], tokenizer, max_length)
    val_dataset = WikiTextDataset(dataset["validation"]["text"], tokenizer, max_length)
    test_dataset = WikiTextDataset(dataset["test"]["text"], tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
