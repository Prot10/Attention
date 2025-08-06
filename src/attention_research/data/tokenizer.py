"""Tokenizer utilities for attention research."""

import logging
from collections.abc import Generator
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from attention_research.shared.config import Config

logger = logging.getLogger(__name__)


def create_tokenizer(config: Config) -> Tokenizer:
    """Create and train a custom tokenizer.

    Args:
        config: Configuration object

    Returns
    -------
        Trained tokenizer
    """
    tokenizer_config = config.get_dataset_config()
    vocab_size = config.get("tokenizer.vocab_size", 32000)
    special_tokens = config.get("tokenizer.special_tokens", ["<pad>", "<unk>", "<s>", "</s>"])
    save_path = Path(config.get("tokenizer.save_path", "./data/tokenizer"))

    # Check if tokenizer already exists
    tokenizer_file = save_path / "tokenizer.json"
    if tokenizer_file.exists():
        print(f"Loading existing tokenizer from {tokenizer_file}")
        return Tokenizer.from_file(str(tokenizer_file))

    print("Training new tokenizer...")

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
    def get_training_corpus() -> Generator[str, None, None]:
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
    print(f"Tokenizer saved to {tokenizer_file}")

    return tokenizer


def load_tokenizer(config: Config) -> Tokenizer:
    """Load an existing tokenizer or create a new one if it doesn't exist.

    Args:
        config: Configuration object

    Returns
    -------
        Loaded tokenizer
    """
    save_path = Path(config.get("tokenizer.save_path", "./data/tokenizer"))
    tokenizer_file = save_path / "tokenizer.json"

    if tokenizer_file.exists():
        print(f"Loading existing tokenizer from {tokenizer_file}")
        return Tokenizer.from_file(str(tokenizer_file))
    print("Tokenizer not found, creating new one...")
    return create_tokenizer(config)
