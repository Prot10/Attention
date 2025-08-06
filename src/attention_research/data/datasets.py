"""Dataset utilities for attention research."""

import logging
from typing import TypeAlias

import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from attention_research.shared.config import Config

from .tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

# Type alias for cleaner type hints
BatchType: TypeAlias = dict[str, torch.Tensor]


class WikiTextDataset(Dataset[BatchType]):
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

    def __getitem__(self, idx: int) -> BatchType:
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


def load_wikitext_data(config: Config) -> tuple[DatasetDict, Tokenizer]:
    """Load WikiText dataset and tokenizer.

    Args:
        config: Configuration object

    Returns
    -------
        Tuple of (dataset, tokenizer)
    """
    dataset_config = config.get_dataset_config()

    # Load dataset
    logger.info("Loading dataset: %s", dataset_config.name)
    dataset = load_dataset(
        dataset_config.name,
        dataset_config.subset,
        cache_dir=dataset_config.cache_dir,
    )

    # Create/load tokenizer
    tokenizer = load_tokenizer(config)

    return dataset, tokenizer


def create_dataloaders(
    dataset: DatasetDict,
    tokenizer: Tokenizer,
    config: Config,
) -> tuple[DataLoader[BatchType], DataLoader[BatchType], DataLoader[BatchType]]:
    """Create PyTorch DataLoaders.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Trained tokenizer
        config: Configuration object

    Returns
    -------
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
