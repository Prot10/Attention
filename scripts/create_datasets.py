#!/usr/bin/env python3
"""Create and prepare datasets for attention research."""

import logging
import sys
from pathlib import Path

from attention_research.data import create_dataloaders, load_wikitext_data
from attention_research.shared.config import Config, setup_logging


def create_datasets(config_path: str | None = None) -> bool:
    """Create datasets for attention research.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file, by default None

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = Config(Path(config_path)) if config_path else Config()

        # Setup logging
        setup_logging(config)

        logger.info("Creating datasets for attention research...")
        logger.info("Framework: %s", config.get_framework())
        logger.info("Attention type: %s", config.get_attention_type())

        # Load data and tokenizer
        logger.info("Loading WikiText dataset and tokenizer...")
        dataset, tokenizer = load_wikitext_data(config)

        logger.info("Dataset loaded successfully!")
        logger.info("Tokenizer vocabulary size: %d", tokenizer.get_vocab_size())

        # Print dataset statistics
        for split_name, split_data in dataset.items():
            logger.info("%s split: %d examples", split_name, len(split_data))

        # Create dataloaders for PyTorch
        if config.get_framework() == "torch":
            logger.info("Creating PyTorch DataLoaders...")
            train_loader, val_loader, test_loader = create_dataloaders(dataset, tokenizer, config)

            logger.info("DataLoaders created successfully!")
            logger.info("Train batches: %d", len(train_loader))
            logger.info("Validation batches: %d", len(val_loader))
            logger.info("Test batches: %d", len(test_loader))

            # Test a batch
            sample_batch = next(iter(train_loader))
            logger.info("Sample batch shape:")
            for key, value in sample_batch.items():
                logger.info("  %s: %s", key, value.shape)

        # For JAX, we'll use the raw dataset
        elif config.get_framework() == "jax":
            logger.info("JAX framework selected - using raw dataset")
            logger.info("Dataset will be processed during training")

        else:
            logger.error("Unknown framework: %s", config.get_framework())
            return False

        logger.info("Dataset creation completed successfully!")

    except Exception:
        logger.exception("Failed to create datasets")
        return False
    else:
        return True


def test_datasets(config_path: str | None = None) -> bool:
    """Test the created datasets.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file, by default None

    Returns
    -------
    bool
        True if tests pass, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = Config(Path(config_path)) if config_path else Config()

        logger.info("Testing datasets...")

        # Load data and tokenizer
        dataset, tokenizer = load_wikitext_data(config)

        # Test tokenizer
        test_text = "Hello, this is a test sentence for the tokenizer."
        encoding = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(encoding.ids)

        logger.info("Tokenizer test:")
        logger.info("  Original: %s", test_text)
        logger.info("  Encoded: %s", encoding.ids[:10])  # First 10 tokens
        logger.info("  Decoded: %s", decoded_text)

        # Test special tokens
        vocab = tokenizer.get_vocab()
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

        logger.info("Special tokens:")
        for token in special_tokens:
            if token in vocab:
                logger.info("  %s: %d", token, vocab[token])
            else:
                logger.warning("  %s: NOT FOUND", token)

        # Test dataset samples
        logger.info("Dataset samples:")
        for split_name, split_data in dataset.items():
            if len(split_data) > 0:
                sample = split_data[0]["text"]
                logger.info("  %s sample: %s...", split_name, sample[:100])

        # Test dataloader if PyTorch
        if config.get_framework() == "torch":
            train_loader, _, _ = create_dataloaders(dataset, tokenizer, config)

            # Test a batch
            sample_batch = next(iter(train_loader))

            logger.info("DataLoader test:")
            logger.info("  Batch size: %d", sample_batch["input_ids"].size(0))
            logger.info("  Sequence length: %d", sample_batch["input_ids"].size(1))
            logger.info(
                "  Vocab range: %d - %d", sample_batch["input_ids"].min().item(), sample_batch["input_ids"].max().item()
            )

            # Check for proper masking
            attention_mask = sample_batch["attention_mask"]
            padding_ratio = (attention_mask == 0).float().mean().item()
            logger.info("  Padding ratio: %.2f%%", padding_ratio * 100)

        logger.info("Dataset tests completed successfully!")

    except Exception:
        logger.exception("Dataset tests failed")
        return False
    else:
        return True


def main() -> int:
    """Run dataset creation and testing.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        logger.info("Using config file: %s", config_path)
    else:
        logger.info("Using default config")

    # Create datasets
    if not create_datasets(config_path):
        logger.error("Dataset creation failed")
        return 1

    # Test datasets
    if not test_datasets(config_path):
        logger.error("Dataset testing failed")
        return 1

    logger.info("All dataset operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
