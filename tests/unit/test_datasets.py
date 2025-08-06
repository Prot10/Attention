"""Tests for dataset utilities."""

import tempfile
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from attention_research.data.datasets import WikiTextDataset, create_dataloaders, load_wikitext_data
from attention_research.shared.config import Config


class TestWikiTextDataset:
    """Test suite for WikiTextDataset."""

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=[1, 2, 3, 4, 5])

        texts = ["hello world", "test sentence"]
        dataset = WikiTextDataset(texts, mock_tokenizer, max_length=10)

        assert len(dataset) == 2
        assert dataset.max_length == 10

    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=[1, 2, 3])

        texts = ["hello world"]
        dataset = WikiTextDataset(texts, mock_tokenizer, max_length=5)

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert len(item["input_ids"]) == 5  # Should be padded/truncated to max_length

    def test_dataset_padding(self):
        """Test that sequences are properly padded."""
        # Mock tokenizer with short sequence
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=[1, 2])

        texts = ["short"]
        dataset = WikiTextDataset(texts, mock_tokenizer, max_length=5)

        item = dataset[0]

        # Should be padded with zeros
        expected_ids = [1, 2, 0, 0, 0]
        expected_mask = [1, 1, 0, 0, 0]

        assert item["input_ids"].tolist() == expected_ids
        assert item["attention_mask"].tolist() == expected_mask

    def test_dataset_truncation(self):
        """Test that sequences are properly truncated."""
        # Mock tokenizer with long sequence
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=[1, 2, 3, 4, 5, 6, 7, 8])

        texts = ["very long sentence that should be truncated"]
        dataset = WikiTextDataset(texts, mock_tokenizer, max_length=5)

        item = dataset[0]

        # Should be truncated
        expected_ids = [1, 2, 3, 4, 5]
        expected_mask = [1, 1, 1, 1, 1]

        assert item["input_ids"].tolist() == expected_ids
        assert item["attention_mask"].tolist() == expected_mask


class TestDataLoaders:
    """Test suite for data loader creation."""

    @patch("attention_research.data.datasets.load_tokenizer")
    @patch("attention_research.data.datasets.load_dataset")
    def test_load_wikitext_data(self, mock_load_dataset, mock_load_tokenizer):
        """Test loading wikitext data."""
        # Mock dataset
        mock_dataset = {"train": {"text": ["text1"]}}
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock config
        config = Mock(spec=Config)
        dataset_config = Mock()
        dataset_config.name = "test_dataset"
        dataset_config.subset = "test_subset"
        dataset_config.cache_dir = tempfile.gettempdir()
        config.get_dataset_config.return_value = dataset_config

        dataset, tokenizer = load_wikitext_data(config)

        assert dataset == mock_dataset
        assert tokenizer == mock_tokenizer

    def test_create_dataloaders(self):
        """Test creating data loaders."""
        # Mock dataset
        mock_dataset = {
            "train": {"text": ["train text 1", "train text 2"]},
            "validation": {"text": ["val text 1"]},
            "test": {"text": ["test text 1"]},
        }

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=[1, 2, 3])

        # Mock config
        config = Mock(spec=Config)
        training_config = Mock()
        training_config.batch_size = 2
        config.get_training_config.return_value = training_config
        config.get.return_value = 512  # max_sequence_length

        train_loader, val_loader, test_loader = create_dataloaders(mock_dataset, mock_tokenizer, config)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2
