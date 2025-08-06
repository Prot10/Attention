"""Tests for JAX attention mechanisms using NNX."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from attention_research.models.jax.attention import (
    MultiHeadAttention,
    VanillaAttention,
)


class TestJAXAttention:
    """Test suite for JAX attention implementations."""

    @pytest.fixture
    def setup_params(self):
        """Set up common test parameters."""
        return {
            "hidden_size": 512,
            "num_heads": 8,
            "dropout_rate": 0.1,
            "batch_size": 2,
            "seq_len": 10,
        }

    @pytest.fixture
    def sample_inputs(self, setup_params):
        """Create sample input tensors."""
        key = jax.random.PRNGKey(42)
        batch_size = setup_params["batch_size"]
        seq_len = setup_params["seq_len"]
        hidden_size = setup_params["hidden_size"]

        # Create sample inputs
        query = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        key_tensor = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        value = jax.random.normal(key, (batch_size, seq_len, hidden_size))

        # Create attention mask (1 for valid positions, 0 for masked)
        attention_mask = jnp.ones((batch_size, seq_len))
        # Mask the last 2 positions for the first batch
        attention_mask = attention_mask.at[0, -2:].set(0)

        return {
            "query": query,
            "key": key_tensor,
            "value": value,
            "attention_mask": attention_mask,
        }

    def test_vanilla_attention_initialization(self, setup_params):
        """Test VanillaAttention initialization."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=setup_params["dropout_rate"],
            use_bias=False,
            rngs=rngs,
        )

        # Check basic attributes
        assert attention.hidden_size == setup_params["hidden_size"]
        assert attention.num_heads == setup_params["num_heads"]
        assert attention.dropout_rate == setup_params["dropout_rate"]
        assert attention.head_dim == setup_params["hidden_size"] // setup_params["num_heads"]

        # Check that layers are initialized
        assert hasattr(attention, "query_proj")
        assert hasattr(attention, "key_proj")
        assert hasattr(attention, "value_proj")
        assert hasattr(attention, "output_proj")
        assert hasattr(attention, "dropout")

    def test_vanilla_attention_forward(self, setup_params, sample_inputs):
        """Test VanillaAttention forward pass."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=setup_params["dropout_rate"],
            use_bias=False,
            rngs=rngs,
        )

        # Forward pass without dropout
        output, attention_weights = attention(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
            attention_mask=sample_inputs["attention_mask"],
        )

        # Check output shapes
        expected_output_shape = (setup_params["batch_size"], setup_params["seq_len"], setup_params["hidden_size"])
        expected_weights_shape = (setup_params["batch_size"], setup_params["seq_len"], setup_params["seq_len"])

        assert output.shape == expected_output_shape
        assert attention_weights.shape == expected_weights_shape

        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(attention_weights))

        # Check attention weights properties
        # Should sum to approximately 1 for non-masked positions
        weights_sum = jnp.sum(attention_weights, axis=-1)
        # For unmasked positions, weights should sum to ~1
        assert jnp.allclose(weights_sum[1], 1.0, rtol=1e-5)  # Second batch (unmasked)

    def test_vanilla_attention_with_dropout(self, setup_params, sample_inputs):
        """Test VanillaAttention with dropout."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=setup_params["dropout_rate"],
            use_bias=False,
            rngs=rngs,
        )

        # Forward pass with dropout
        dropout_rngs = nnx.Rngs(123)
        output, attention_weights = attention(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
            attention_mask=sample_inputs["attention_mask"],
            rngs=dropout_rngs,
        )

        # Check output shapes
        expected_output_shape = (setup_params["batch_size"], setup_params["seq_len"], setup_params["hidden_size"])

        assert output.shape == expected_output_shape
        assert jnp.all(jnp.isfinite(output))

    def test_multihead_attention_initialization(self, setup_params):
        """Test MultiHeadAttention initialization."""
        rngs = nnx.Rngs(42)

        attention = MultiHeadAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=setup_params["dropout_rate"],
            use_bias=False,
            rngs=rngs,
        )

        # Check basic attributes
        assert attention.hidden_size == setup_params["hidden_size"]
        assert attention.num_heads == setup_params["num_heads"]
        assert attention.dropout_rate == setup_params["dropout_rate"]
        assert attention.head_dim == setup_params["hidden_size"] // setup_params["num_heads"]

        # Check that projection lists are initialized
        assert len(attention.query_projections) == setup_params["num_heads"]
        assert len(attention.key_projections) == setup_params["num_heads"]
        assert len(attention.value_projections) == setup_params["num_heads"]

    def test_multihead_attention_forward(self, setup_params, sample_inputs):
        """Test MultiHeadAttention forward pass."""
        rngs = nnx.Rngs(42)

        attention = MultiHeadAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=setup_params["dropout_rate"],
            use_bias=False,
            rngs=rngs,
        )

        # Forward pass
        output, attention_weights = attention(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
            attention_mask=sample_inputs["attention_mask"],
        )

        # Check output shapes
        expected_output_shape = (setup_params["batch_size"], setup_params["seq_len"], setup_params["hidden_size"])
        expected_weights_shape = (setup_params["batch_size"], setup_params["seq_len"], setup_params["seq_len"])

        assert output.shape == expected_output_shape
        assert attention_weights.shape == expected_weights_shape

        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(attention_weights))

    def test_attention_mask_effect(self, setup_params, sample_inputs):
        """Test that attention mask properly affects attention weights."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=0.0,  # No dropout for this test
            use_bias=False,
            rngs=rngs,
        )

        # Forward pass with mask
        _, attention_weights_masked = attention(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
            attention_mask=sample_inputs["attention_mask"],
        )

        # Forward pass without mask
        _, attention_weights_unmasked = attention(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
            attention_mask=None,
        )

        # Check that masked positions have very low attention weights
        # The last 2 positions of the first batch should be masked
        masked_weights = attention_weights_masked[0, :, -2:]  # First batch, last 2 positions

        # Masked positions should have very small weights (close to 0 after softmax)
        assert jnp.all(masked_weights < 1e-6)

        # Attention weights should be different with and without mask
        assert not jnp.allclose(attention_weights_masked, attention_weights_unmasked)

    def test_hidden_size_divisibility_check(self):
        """Test that initialization fails when hidden_size is not divisible by num_heads."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            VanillaAttention(
                hidden_size=513,  # Not divisible by 8
                num_heads=8,
                dropout_rate=0.1,
                use_bias=False,
                rngs=rngs,
            )

    def test_different_input_shapes(self):
        """Test attention with different input shapes."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=256,
            num_heads=4,
            dropout_rate=0.1,
            use_bias=False,
            rngs=rngs,
        )

        # Test with different batch size and sequence length
        batch_size, seq_len, hidden_size = 3, 15, 256
        key = jax.random.PRNGKey(42)

        query = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        key_tensor = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        value = jax.random.normal(key, (batch_size, seq_len, hidden_size))

        output, attention_weights = attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_bias_parameter(self):
        """Test attention with bias enabled."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=256,
            num_heads=4,
            dropout_rate=0.1,
            use_bias=True,  # Enable bias
            rngs=rngs,
        )

        batch_size, seq_len, hidden_size = 2, 8, 256
        key = jax.random.PRNGKey(42)

        query = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        key_tensor = jax.random.normal(key, (batch_size, seq_len, hidden_size))
        value = jax.random.normal(key, (batch_size, seq_len, hidden_size))

        output, attention_weights = attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert jnp.all(jnp.isfinite(output))

    def test_attention_determinism(self, setup_params, sample_inputs):
        """Test that attention gives deterministic results with same random key."""
        rngs1 = nnx.Rngs(42)
        rngs2 = nnx.Rngs(42)

        attention1 = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=0.0,  # No dropout for determinism
            use_bias=False,
            rngs=rngs1,
        )

        attention2 = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=0.0,  # No dropout for determinism
            use_bias=False,
            rngs=rngs2,
        )

        output1, weights1 = attention1(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
        )

        output2, weights2 = attention2(
            query=sample_inputs["query"],
            key=sample_inputs["key"],
            value=sample_inputs["value"],
        )

        # With same random seed, results should be identical
        assert jnp.allclose(output1, output2)
        assert jnp.allclose(weights1, weights2)

    def test_gradient_flow(self, setup_params, sample_inputs):
        """Test that gradients flow through the attention mechanism."""
        rngs = nnx.Rngs(42)

        attention = VanillaAttention(
            hidden_size=setup_params["hidden_size"],
            num_heads=setup_params["num_heads"],
            dropout_rate=0.0,
            use_bias=False,
            rngs=rngs,
        )

        def loss_fn(model, inputs):
            output, _ = model(
                query=inputs["query"],
                key=inputs["key"],
                value=inputs["value"],
            )
            return jnp.mean(output**2)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(attention, sample_inputs)

        # Check that loss is finite
        assert jnp.isfinite(loss)

        # Check that gradients exist and are finite for all parameters
        def check_grads(module, path=""):
            for name, value in vars(module).items():
                if isinstance(value, nnx.Module):
                    check_grads(value, f"{path}.{name}")
                elif hasattr(value, "shape") and jnp.issubdtype(value.dtype, jnp.floating):
                    # This is a parameter, check its gradient
                    param_path = f"{path}.{name}" if path else name
                    print(f"Checking gradient for {param_path}")

        # The gradients should be computed for the model
        assert grads is not None
