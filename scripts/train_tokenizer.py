"""Train the tokenizer for the attention research project.

This script trains a custom BPE tokenizer on the WikiText dataset as specified
in the configuration. The tokenizer will be saved and can be reused for
training attention models.
"""

import sys
from pathlib import Path

from attention_research.data.tokenizer import create_tokenizer
from attention_research.shared.config import Config

# Add the project root to the Python path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/
sys.path.insert(0, str(project_root))


def main():
    """Train the tokenizer."""
    print("🚀 Starting tokenizer training...")

    # Load configuration
    config_path = project_root / "config" / "base.yaml"
    config = Config(config_path)

    print(f"📖 Dataset: {config.get('dataset.name')} ({config.get('dataset.subset')})")
    print(f"🔤 Vocabulary size: {config.get('tokenizer.vocab_size'):,}")
    print(f"✨ Special tokens: {config.get('tokenizer.special_tokens')}")
    print(f"💾 Save path: {config.get('tokenizer.save_path')}")

    # Train the tokenizer
    try:
        tokenizer = create_tokenizer(config)
        print("✅ Tokenizer training completed successfully!")

        # Test the tokenizer with a sample text
        sample_text = "Hello world! This is a test of the attention mechanism tokenizer."
        encoding = tokenizer.encode(sample_text)

        print("\n🧪 Testing tokenizer:")
        print(f"   Input: {sample_text}")
        print(f"   Tokens: {encoding.tokens}")
        print(f"   IDs: {encoding.ids}")
        print(f"   Token count: {len(encoding.tokens)}")

        # Test decoding
        decoded = tokenizer.decode(encoding.ids)
        print(f"   Decoded: {decoded}")

        # Show vocabulary info
        vocab_size = tokenizer.get_vocab_size()
        print("\n📊 Tokenizer statistics:")
        print(f"   Vocabulary size: {vocab_size:,}")
        print(f"   Special tokens: {[tokenizer.decode([i]) for i in range(min(10, vocab_size))]}")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"❌ Error during tokenizer training: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
