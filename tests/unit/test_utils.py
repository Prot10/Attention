"""Tests for data utilities."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from attention_research.data.utils import download_and_verify_data, get_dataset_info
from attention_research.shared.config import Config


class TestDataUtils:
    """Test suite for data utility functions."""

    @patch("attention_research.data.utils.load_dataset")
    @patch("attention_research.data.utils.logger")
    def test_download_and_verify_data_success(self, mock_logger, mock_load_dataset):
        """Test successful data download and verification."""
        # Mock dataset with all required splits
        mock_dataset = {"train": ["data1", "data2"], "validation": ["val1"], "test": ["test1"]}
        mock_load_dataset.return_value = mock_dataset

        # Mock config
        config = Mock(spec=Config)
        dataset_config = Mock()
        dataset_config.name = "test_dataset"
        dataset_config.subset = "test_subset"
        dataset_config.cache_dir = tempfile.gettempdir()
        config.get_dataset_config.return_value = dataset_config

        # Should not raise any exception
        download_and_verify_data(config)

        # Verify dataset was loaded with correct parameters
        mock_load_dataset.assert_called_once_with("test_dataset", "test_subset", cache_dir=tempfile.gettempdir())

    @patch("attention_research.data.utils.load_dataset")
    def test_download_and_verify_data_missing_split(self, mock_load_dataset):
        """Test data verification with missing required split."""
        # Mock dataset missing test split
        mock_dataset = {
            "train": ["data1", "data2"],
            "validation": ["val1"],
            # Missing "test" split
        }
        mock_load_dataset.return_value = mock_dataset

        # Mock config
        config = Mock(spec=Config)
        dataset_config = Mock()
        dataset_config.name = "test_dataset"
        dataset_config.subset = "test_subset"
        dataset_config.cache_dir = tempfile.gettempdir()
        config.get_dataset_config.return_value = dataset_config

        # Should raise ValueError for missing split
        with pytest.raises(ValueError, match="Required split 'test' not found"):
            download_and_verify_data(config)

    @patch("attention_research.data.utils.load_dataset")
    def test_get_dataset_info(self, mock_load_dataset):
        """Test getting dataset information."""
        # Mock dataset
        mock_dataset = {"train": ["data1", "data2", "data3"], "validation": ["val1", "val2"], "test": ["test1"]}
        mock_load_dataset.return_value = mock_dataset

        # Mock config
        config = Mock(spec=Config)
        dataset_config = Mock()
        dataset_config.name = "test_dataset"
        dataset_config.subset = "test_subset"
        dataset_config.cache_dir = tempfile.gettempdir()
        config.get_dataset_config.return_value = dataset_config

        result = get_dataset_info(config)

        expected = {"train_size": 3, "validation_size": 2, "test_size": 1, "total_size": 6}

        assert result == expected

        # Verify dataset was loaded with correct parameters
        mock_load_dataset.assert_called_once_with("test_dataset", "test_subset", cache_dir=tempfile.gettempdir())
