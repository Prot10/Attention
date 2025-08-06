"""Tests for tokenizer utilities."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tokenizers import Tokenizer

from attention_research.data.tokenizer import create_tokenizer, load_tokenizer
from attention_research.shared.config import Config


class TestTokenizerUtilities:
    """Test suite for tokenizer utilities."""

    def test_load_existing_tokenizer(self):
        """Test loading an existing tokenizer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock config
            config = Mock(spec=Config)
            config.get.return_value = temp_dir

            # Create a dummy tokenizer file
            tokenizer_path = Path(temp_dir) / "tokenizer.json"
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a simple tokenizer and save it
            from tokenizers import Tokenizer
            from tokenizers.models import BPE

            simple_tokenizer = Tokenizer(BPE())
            simple_tokenizer.save(str(tokenizer_path))

            # Test loading
            loaded_tokenizer = load_tokenizer(config)
            assert isinstance(loaded_tokenizer, Tokenizer)

    @patch("attention_research.data.tokenizer.load_dataset")
    def test_create_new_tokenizer(self, mock_load_dataset):
        """Test creating a new tokenizer when none exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock dataset
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(
                return_value=iter(
                    [
                        {"text": "hello world"},
                        {"text": "test sentence"},
                    ]
                )
            )
            mock_load_dataset.return_value = mock_dataset

            # Create a mock config
            config = Mock(spec=Config)
            config.get_dataset_config.return_value = Mock(
                name="test_dataset", subset="test_subset", cache_dir=tempfile.gettempdir()
            )
            config.get.side_effect = lambda key, default=None: {
                "tokenizer.vocab_size": 1000,
                "tokenizer.special_tokens": ["<pad>", "<unk>"],
                "tokenizer.save_path": temp_dir,
            }.get(key, default)

            # Test creating tokenizer
            tokenizer = create_tokenizer(config)
            assert isinstance(tokenizer, Tokenizer)

            # Check that tokenizer file was saved
            tokenizer_file = Path(temp_dir) / "tokenizer.json"
            assert tokenizer_file.exists()

    def test_load_tokenizer_creates_if_missing(self):
        """Test that load_tokenizer creates a new tokenizer if none exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock(spec=Config)
            config.get.return_value = temp_dir

            with patch("attention_research.data.tokenizer.create_tokenizer") as mock_create:
                mock_tokenizer = Mock(spec=Tokenizer)
                mock_create.return_value = mock_tokenizer

                result = load_tokenizer(config)

                mock_create.assert_called_once_with(config)
                assert result == mock_tokenizer
