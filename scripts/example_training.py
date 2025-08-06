#!/usr/bin/env python3
"""Example training script for attention research."""

import logging
import sys
from pathlib import Path

from attention_research.shared.config import Config, setup_logging


def main() -> int:
    """Run example training."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        logger.info("Using config file: %s", config_path)
    else:
        logger.info("Using default config")

    try:
        # Load configuration
        config = Config(Path(config_path) if config_path else None)
        setup_logging(config)

        logger.info("=" * 60)
        logger.info("🔬 ATTENTION RESEARCH FRAMEWORK")
        logger.info("=" * 60)
        logger.info("Framework: %s", config.get_framework())
        logger.info("Attention type: %s", config.get_attention_type())
        logger.info("Model configuration:")
        model_config = config.get_model_config()
        for key, value in model_config.items():
            logger.info("  %s: %s", key, value)

        logger.info("\nTraining configuration:")
        training_config = config.get_training_config()
        for key, value in training_config.items():
            logger.info("  %s: %s", key, value)

        logger.info("\nDataset configuration:")
        dataset_config = config.get_dataset_config()
        for key, value in dataset_config.items():
            logger.info("  %s: %s", key, value)

        logger.info("\n✅ Configuration loaded successfully!")
        logger.info("🚀 Ready to train attention mechanisms!")

        # Example of how to switch frameworks and attention types
        logger.info("\n%s", "=" * 60)
        logger.info("🔄 FRAMEWORK SWITCHING DEMO")
        logger.info("=" * 60)

        # Test switching to JAX
        config.set_framework("jax")
        config.set_attention_type("multi_head")
        logger.info("Switched to JAX with multi-head attention")
        logger.info("Framework: %s", config.get_framework())
        logger.info("Attention type: %s", config.get_attention_type())

        # Test switching to PyTorch
        config.set_framework("torch")
        config.set_attention_type("vanilla")
        logger.info("Switched to PyTorch with vanilla attention")
        logger.info("Framework: %s", config.get_framework())
        logger.info("Attention type: %s", config.get_attention_type())

        logger.info("\n✅ Framework switching works correctly!")

        # Show available attention types
        logger.info("\n%s", "=" * 60)
        logger.info("📋 AVAILABLE ATTENTION MECHANISMS")
        logger.info("=" * 60)

        attention_types = ["vanilla", "multi_head", "flash", "linear", "performer", "longformer"]
        frameworks = ["torch", "jax"]

        logger.info("Supported attention mechanisms:")
        for attention_type in attention_types:
            logger.info("  • %s", attention_type)

        logger.info("\nSupported frameworks:")
        for framework in frameworks:
            logger.info("  • %s", framework)

        logger.info("\n🎯 To start training:")
        logger.info("1. Run: python scripts/download_data.py")
        logger.info("2. Run: python scripts/create_datasets.py")
        logger.info("3. Implement your training loop using the trainers")
        logger.info("4. Use profiling tools for performance analysis")

        logger.info("\n🔧 To customize:")
        logger.info("1. Edit config/base.yaml")
        logger.info("2. Implement additional attention mechanisms")
        logger.info("3. Add your own evaluation metrics")

    except Exception:
        logger.exception("Example script failed")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
