"""Data loading and preprocessing utilities."""

import logging

from datasets import load_dataset

from attention_research.shared.config import Config

logger = logging.getLogger(__name__)


def download_and_verify_data(config: Config) -> None:
    """Download and verify dataset.

    Args:
        config: Configuration object
    """
    dataset_config = config.get_dataset_config()

    logger.info("Downloading dataset: %s", dataset_config.name)

    # Download dataset
    dataset = load_dataset(
        dataset_config.name,
        dataset_config.subset,
        cache_dir=dataset_config.cache_dir,
    )

    # Verify splits exist
    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in dataset:
            msg = f"Required split '{split}' not found in dataset"
            raise ValueError(msg)

    logger.info("Dataset verification completed successfully")
    logger.info("Train examples: %d", len(dataset["train"]))
    logger.info("Validation examples: %d", len(dataset["validation"]))
    logger.info("Test examples: %d", len(dataset["test"]))


def get_dataset_info(config: Config) -> dict[str, int]:
    """Get basic information about the dataset.

    Args:
        config: Configuration object

    Returns
    -------
        Dictionary with dataset statistics
    """
    dataset_config = config.get_dataset_config()

    dataset = load_dataset(
        dataset_config.name,
        dataset_config.subset,
        cache_dir=dataset_config.cache_dir,
    )

    return {
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "test_size": len(dataset["test"]),
        "total_size": len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"]),
    }
