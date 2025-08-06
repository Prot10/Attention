#!/usr/bin/env python3
"""Download and prepare WikiText dataset for attention research."""

import logging
import sys
from pathlib import Path

from datasets import load_dataset


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def download_wikitext_data(data_dir: Path = Path("./data")) -> bool:
    """Download WikiText dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to save the data, by default Path("./data")

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Create data directory
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading WikiText-103-raw-v1 dataset...")

        # Download the dataset
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            cache_dir=str(cache_dir),
        )

        logger.info("Dataset downloaded successfully!")
        logger.info("Dataset splits: %s", list(dataset.keys()))

        # Print dataset statistics
        for split_name, split_data in dataset.items():
            logger.info("%s split: %d examples", split_name, len(split_data))

            # Sample a few examples to show the data format
            if len(split_data) > 0:
                sample_text = split_data[0]["text"]
                logger.info("Sample from %s: %s...", split_name, sample_text[:100])

        # Save dataset info
        info_file = data_dir / "dataset_info.txt"
        with info_file.open("w") as f:
            f.write("WikiText-103-raw-v1 Dataset Information\n")
            f.write("=" * 40 + "\n\n")
            f.write("Dataset: Salesforce/wikitext\n")
            f.write("Subset: wikitext-103-raw-v1\n")
            f.write(f"Cache directory: {cache_dir}\n\n")

            for split_name, split_data in dataset.items():
                f.write(f"{split_name.capitalize()} split: {len(split_data)} examples\n")

        logger.info("Dataset information saved to %s", info_file)

    except Exception:
        logger.exception("Failed to download dataset")
        return False
    else:
        return True


def verify_data(data_dir: Path = Path("./data")) -> bool:
    """Verify that the dataset is properly downloaded.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data, by default Path("./data")

    Returns
    -------
    bool
        True if data is valid, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        cache_dir = data_dir / "cache"

        if not cache_dir.exists():
            logger.error("Cache directory does not exist: %s", cache_dir)
            return False

        # Try to load the dataset
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            cache_dir=str(cache_dir),
        )

        # Check all required splits exist
        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset:
                logger.error("Missing required split: %s", split)
                return False

            if len(dataset[split]) == 0:
                logger.error("Empty split: %s", split)
                return False

        logger.info("Data verification successful!")

    except Exception:
        logger.exception("Data verification failed")
        return False
    else:
        return True


def main() -> int:
    """Download and verify data.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")

    logger.info("Using data directory: %s", data_dir.resolve())

    # Check if data already exists
    cache_dir = data_dir / "cache"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        logger.info("Data directory already exists. Verifying...")
        if verify_data(data_dir):
            logger.info("Data is valid. Skipping download.")
            return 0
        logger.warning("Existing data is invalid. Re-downloading...")

    # Download the data
    if download_wikitext_data(data_dir):
        # Verify the downloaded data
        if verify_data(data_dir):
            logger.info("Data download and verification completed successfully!")
            return 0
        logger.error("Data verification failed after download.")
        return 1
    logger.error("Data download failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
