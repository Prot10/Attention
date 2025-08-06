"""Comprehensive tests for PyTorch attention mechanisms."""

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from attention_research.models.torch.attention import (
    MultiHeadAttention,
    VanillaAttention,
)


class TestVanillaAttention:
    """Test suite for VanillaAttention implementation."""

    @pytest.fixture
    def vanilla_attention(self):
        """Create a VanillaAttention instance for testing."""
        return VanillaAttention(hidden_size=64)

    def test_initialization(self, vanilla_attention):
        """Test VanillaAttention initialization."""
        assert vanilla_attention.hidden_size == 64
        assert vanilla_attention.scale == 1.0 / (8**0.5)  # head_dim = 64/8 = 8
        assert isinstance(vanilla_attention.query_proj, nn.Linear)
        assert isinstance(vanilla_attention.key_proj, nn.Linear)
        assert isinstance(vanilla_attention.value_proj, nn.Linear)
        assert isinstance(vanilla_attention.output_proj, nn.Linear)

    def test_forward_pass_shape(self, vanilla_attention):
        """Test VanillaAttention forward pass output shape."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)

        output, attention_weights = vanilla_attention(query, key, value)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights(self, vanilla_attention):
        """Test attention weights calculation."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)

        # Set to eval mode to disable dropout for precise testing
        vanilla_attention.eval()
        with torch.no_grad():
            _, attention_weights = vanilla_attention(query, key, value)

            # Check that attention weights sum to 1 along the last dimension
            weight_sums = attention_weights.sum(dim=-1)
            expected_sums = torch.ones_like(weight_sums)
            assert_close(weight_sums, expected_sums, atol=1e-6, rtol=1e-6)

    def test_masked_attention(self, vanilla_attention):
        """Test masked attention mechanism."""
        batch_size, seq_len, hidden_size = 2, 5, 64
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)

        # Create a mask that masks out the last two positions
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = False  # Mask last two positions

        output, _ = vanilla_attention(query, key, value, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, hidden_size)
        # Output should not be NaN or Inf
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, vanilla_attention):
        """Test that gradients flow properly through attention."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        query = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        key = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        value = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

        output, _ = vanilla_attention(query, key, value)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert query.grad is not None
        assert not torch.isnan(query.grad).any()

        # Check that parameter gradients exist
        for param in vanilla_attention.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention implementation."""

    @pytest.fixture
    def multi_head_attention(self):
        """Create a MultiHeadAttention instance for testing."""
        return MultiHeadAttention(hidden_size=64, num_heads=8)

    def test_initialization(self, multi_head_attention):
        """Test MultiHeadAttention initialization."""
        assert multi_head_attention.hidden_size == 64
        assert multi_head_attention.num_heads == 8
        assert multi_head_attention.head_dim == 8  # 64 / 8
        assert multi_head_attention.scale == 1.0 / (8**0.5)
        assert len(multi_head_attention.query_projections) == 8
        assert len(multi_head_attention.key_projections) == 8
        assert len(multi_head_attention.value_projections) == 8
        assert isinstance(multi_head_attention.output_proj, nn.Linear)

    def test_head_dimension_mismatch(self):
        """Test that initialization fails when hidden_size is not divisible by num_heads."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            MultiHeadAttention(hidden_size=65, num_heads=8)

    def test_forward_pass_shape(self, multi_head_attention):
        """Test MultiHeadAttention forward pass output shape."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)

        output, attention_weights = multi_head_attention(query, key, value)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_gradient_flow(self, multi_head_attention):
        """Test that gradients flow properly through multi-head attention."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        query = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        key = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        value = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

        output, _ = multi_head_attention(query, key, value)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert query.grad is not None
        assert not torch.isnan(query.grad).any()

        # Check that parameter gradients exist
        for param in multi_head_attention.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        attention = VanillaAttention(hidden_size=32)

        # Test with different sequence lengths
        for seq_len in [1, 5, 50]:
            query = torch.randn(1, seq_len, 32)
            key = torch.randn(1, seq_len, 32)
            value = torch.randn(1, seq_len, 32)
            output, _ = attention(query, key, value)
            assert output.shape == (1, seq_len, 32)

    def test_large_sequence_length(self):
        """Test attention with a large sequence length."""
        attention = VanillaAttention(hidden_size=32)

        # Test with a reasonably large sequence
        query = torch.randn(1, 100, 32)
        key = torch.randn(1, 100, 32)
        value = torch.randn(1, 100, 32)
        output, _ = attention(query, key, value)
        assert output.shape == (1, 100, 32)
        assert torch.isfinite(output).all()

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        attention = VanillaAttention(hidden_size=32)

        # Test with very small values
        query_small = torch.randn(1, 10, 32) * 1e-6
        key_small = torch.randn(1, 10, 32) * 1e-6
        value_small = torch.randn(1, 10, 32) * 1e-6
        output_small, _ = attention(query_small, key_small, value_small)
        assert torch.isfinite(output_small).all()

        # Test with larger values (but not extreme to avoid overflow)
        query_large = torch.randn(1, 10, 32) * 10
        key_large = torch.randn(1, 10, 32) * 10
        value_large = torch.randn(1, 10, 32) * 10
        output_large, _ = attention(query_large, key_large, value_large)
        assert torch.isfinite(output_large).all()
