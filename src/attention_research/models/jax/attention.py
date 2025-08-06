"""Base attention mechanisms for JAX/NNX.

This module implements the attention mechanisms described in "Attention Is All You Need"
(Vaswani et al., 2017) using JAX and Flax NNX - the seminal paper that introduced
the Transformer architecture.

The core innovation of the paper is the scaled dot-product attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
- Q (Query): What we're looking for - represents the current position's information
- K (Key): What we're looking at - represents all positions' information for matching
- V (Value): What we extract - represents the actual content to be aggregated
- d_k: Dimension of the key vectors (for scaling to prevent saturation in softmax)

Multi-Head Attention extends this by:
1. Linearly projecting Q, K, V into h different representation subspaces
2. Applying attention in parallel across all heads
3. Concatenating the results and projecting back to the original dimension

This allows the model to attend to information from different representation
subspaces at different positions simultaneously, which is crucial for understanding
complex relationships in sequences.

Key benefits:
- Parallelizable (unlike RNNs)
- Long-range dependencies (unlike CNNs with limited receptive fields)
- Interpretable attention weights
- No sequential computation bottleneck
- Functional programming paradigm with JAX for efficient computation

JAX/NNX Specific Features:
- Immutable functional transformations
- JIT compilation for performance
- Automatic differentiation (grad, vmap, etc.)
- Explicit handling of random number generators (RNGs)
- Pure functions with explicit state management

References
----------
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
In Advances in neural information processing systems (pp. 5998-6008).
"""

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx


class BaseAttention(nnx.Module, ABC):
    """Abstract base class for attention mechanisms in JAX/NNX.

    This class provides the common interface and utilities shared by all
    attention implementations in the functional programming paradigm of JAX.
    It handles the mathematical foundations like scaling factors and masking
    operations while maintaining immutability and explicit state management.

    JAX/NNX Design Patterns:
    - Explicit RNG handling for reproducibility
    - Functional transformations with pure functions
    - Immutable state with explicit updates
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize base attention parameters with JAX/NNX conventions.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size (d_model in the paper)
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout_rate : float, optional
            Dropout probability for attention weights, by default 0.1
        use_bias : bool, optional
            Whether to use bias in linear projections, by default False
            (Transformers typically don't use bias in attention layers)
        rngs : nnx.Rngs
            Random number generators for parameter initialization and dropout
            JAX requires explicit RNG management for reproducibility
        """
        # VALIDATION: Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            msg = f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        # CORE PARAMETERS: Store attention configuration
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.head_dim = hidden_size // num_heads  # d_k = d_model / num_heads

        # SCALING FACTOR: √d_k from the paper - prevents softmax saturation
        # When d_k is large, the dot products grow large in magnitude, pushing
        # the softmax function into regions where it has extremely small gradients
        self.scale = 1.0 / math.sqrt(self.head_dim)

    @abstractmethod
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Abstract method for forward pass in JAX functional style.

        Parameters
        ----------
        query : jnp.ndarray
            Query tensor of shape (batch_size, seq_len, hidden_size)
        key : jnp.ndarray
            Key tensor of shape (batch_size, seq_len, hidden_size)
        value : jnp.ndarray
            Value tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask : jnp.ndarray | None, default=None
            Attention mask of shape (batch_size, seq_len) where 1 means attend, 0 means mask
        rngs : nnx.Rngs | None, default=None
            Random number generators for stochastic operations (dropout)
            JAX requires explicit RNG threading for reproducible randomness

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Tuple of (output, attention_weights)
            - output: shape (batch_size, seq_len, hidden_size)
            - attention_weights: shape (batch_size, seq_len, seq_len)
        """

    def _apply_mask(
        self,
        attention_scores: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply attention mask to prevent attention to certain positions.

        This is crucial for:
        1. Padding tokens: Don't attend to padding positions
        2. Causal masking: In decoder, don't attend to future positions
        3. Custom masking: Any application-specific attention restrictions

        JAX Implementation Notes:
        - Uses jnp.where for conditional assignment (functional style)
        - Explicit broadcasting with jnp.broadcast_to
        - Immutable operations - returns new array instead of in-place modification

        Parameters
        ----------
        attention_scores : jnp.ndarray
            Raw attention scores of shape (batch_size, num_heads, seq_len, seq_len)
        attention_mask : jnp.ndarray | None, default=None
            Mask tensor of shape (batch_size, seq_len) where 1 means "attend", 0 means "mask"

        Returns
        -------
        jnp.ndarray
            Masked attention scores with -1e9 where attention is not allowed
        """
        if attention_mask is not None:
            # EXPAND MASK: Transform from (batch_size, seq_len) to broadcast-compatible shape
            # We need (batch_size, 1, 1, seq_len) to broadcast with attention_scores
            mask = jnp.expand_dims(attention_mask, axis=(1, 2))  # (batch_size, 1, 1, seq_len)

            # BROADCAST: Expand mask to match attention_scores shape
            mask = jnp.broadcast_to(mask, attention_scores.shape)

            # APPLY MASK: Use functional jnp.where instead of in-place operations
            # Set masked positions to large negative value (-1e9)
            # After softmax, these become effectively 0
            attention_scores = jnp.where(mask == 0, -1e9, attention_scores)

        return attention_scores


class VanillaAttention(BaseAttention):
    """Vanilla scaled dot-product attention using JAX/NNX functional paradigm.

    This implements the standard multi-head attention as described in
    "Attention Is All You Need" using JAX's functional programming approach.
    It projects all heads at once and uses efficient JAX operations.

    JAX-Specific Implementation Features:
    - Immutable operations with jnp.* functions
    - Explicit shape transformations with reshape and transpose
    - Functional matrix operations with jnp.matmul
    - Pure functions without hidden state mutations
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize vanilla attention with JAX/NNX modules.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout_rate : float, optional
            Dropout probability, by default 0.1
        use_bias : bool, optional
            Whether to use bias in linear layers, by default False
        rngs : nnx.Rngs
            Random number generators for initialization
        """
        super().__init__(hidden_size, num_heads, dropout_rate, use_bias=use_bias, rngs=rngs)

        # STEP 1: Initialize linear projection layers using NNX modules
        # These project the input to Query, Key, and Value representations
        # Note: use_bias=False is common in transformer implementations
        self.query_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.key_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.value_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Final output projection (W^O in the paper)
        self.output_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )

        # REGULARIZATION: Dropout layer using explicit RNG management
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of vanilla attention using JAX functional operations.

        This implements the complete attention mechanism with detailed step-by-step
        comments to understand the mathematical operations in the JAX context.

        Parameters
        ----------
        query : jnp.ndarray
            Query tensor of shape (batch_size, seq_len, hidden_size)
        key : jnp.ndarray
            Key tensor of shape (batch_size, seq_len, hidden_size)
        value : jnp.ndarray
            Value tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask : jnp.ndarray | None, default=None
            Attention mask of shape (batch_size, seq_len)
        rngs : nnx.Rngs | None, default=None
            Random number generators for dropout

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Tuple of (output, attention_weights)
        """
        # STEP 1: Get input dimensions for shape tracking
        batch_size, seq_len, _ = query.shape

        # STEP 2: Project inputs to Query, Key, Value spaces using NNX modules
        # Each projection: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        q = self.query_proj(query)  # What we're looking for
        k = self.key_proj(key)  # What we're comparing against
        v = self.value_proj(value)  # What we want to extract

        # STEP 3: Reshape for multi-head attention using JAX operations
        # Split hidden_size into (num_heads, head_dim) for parallel processing
        # JAX reshape is immutable - creates new array with different shape

        # Transform: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # STEP 4: Transpose to put heads in dimension 1 for efficient computation
        # JAX transpose is also immutable and functional
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # STEP 5: Compute attention scores using JAX matrix multiplication
        # This computes Q @ K^T for all heads in parallel
        # jnp.matmul handles batched matrix multiplication efficiently
        # Shape: (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        #     -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scale

        # STEP 6: Apply mask if provided using functional JAX operations
        # This prevents attention to padding tokens or future positions
        attention_scores = self._apply_mask(attention_scores, attention_mask)

        # STEP 7: Apply softmax to get attention weights using JAX activation
        # This normalizes attention scores so they sum to 1 across the key dimension
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # STEP 8: Apply dropout for regularization if RNGs provided
        # JAX requires explicit RNG threading for reproducible randomness
        if rngs is not None:
            attention_weights = self.dropout(attention_weights, rngs=rngs)

        # STEP 9: Apply attention to values using JAX matrix multiplication
        # This computes the weighted sum of values based on attention weights
        # Shape: (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        #     -> (batch_size, num_heads, seq_len, head_dim)
        context = jnp.matmul(attention_weights, v)

        # STEP 10: Reshape back to original format using JAX operations
        # Combine heads back into hidden dimension
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))
        # -> (batch_size, seq_len, hidden_size)
        context = context.reshape(batch_size, seq_len, self.hidden_size)

        # STEP 11: Final linear projection using NNX module
        # This is the W^O matrix from the paper, combining information from all heads
        output = self.output_proj(context)

        # STEP 12: Return output and attention weights (averaged across heads for visualization)
        # jnp.mean is functional and creates new array
        return output, jnp.mean(attention_weights, axis=1)  # Average attention weights across heads


class MultiHeadAttention(BaseAttention):
    """Multi-head attention with separate head processing in JAX/NNX.

    This is an alternative implementation that processes each attention head
    separately using explicit loops and functional operations. This is less
    efficient than the vectorized approach but more explicit about the
    multi-head structure and educational for understanding the mechanism.

    JAX Implementation Benefits:
    - Explicit functional operations for each head
    - Clear separation of head-specific computations
    - Educational value for understanding multi-head structure
    - Immutable operations throughout the computation
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize multi-head attention with separate projections per head.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout_rate : float, optional
            Dropout probability, by default 0.1
        use_bias : bool, optional
            Whether to use bias in linear projections, by default False
        rngs : nnx.Rngs
            Random number generators for initialization
        """
        super().__init__(hidden_size, num_heads, dropout_rate, use_bias=use_bias, rngs=rngs)

        # STEP 1: Create separate projection layers for each head using list comprehension
        # This makes the multi-head structure very explicit and educational
        # Each head gets its own learned linear transformation parameters
        self.query_projections = [
            nnx.Linear(
                hidden_size,
                self.head_dim,  # Project to smaller dimension per head
                use_bias=use_bias,
                rngs=rngs,
            )
            for _ in range(num_heads)
        ]
        self.key_projections = [
            nnx.Linear(
                hidden_size,
                self.head_dim,
                use_bias=use_bias,
                rngs=rngs,
            )
            for _ in range(num_heads)
        ]
        self.value_projections = [
            nnx.Linear(
                hidden_size,
                self.head_dim,
                use_bias=use_bias,
                rngs=rngs,
            )
            for _ in range(num_heads)
        ]

        # Final output projection to combine all heads
        self.output_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )

        # REGULARIZATION: Dropout with explicit RNG management
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of multi-head attention with explicit head processing.

        This implementation processes each head separately to make the multi-head
        mechanism more explicit and educational, using JAX functional operations.

        Parameters
        ----------
        query : jnp.ndarray
            Query tensor of shape (batch_size, seq_len, hidden_size)
        key : jnp.ndarray
            Key tensor of shape (batch_size, seq_len, hidden_size)
        value : jnp.ndarray
            Value tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask : jnp.ndarray | None, default=None
            Attention mask of shape (batch_size, seq_len)
        rngs : nnx.Rngs | None, default=None
            Random number generators for dropout

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Tuple of (output, attention_weights)
        """
        # STEP 1: Get input dimensions for shape tracking
        batch_size, seq_len, _ = query.shape

        # STEP 2: Initialize lists to store outputs from each head
        # This approach makes the multi-head structure very explicit
        head_outputs = []
        head_attention_weights = []

        # STEP 3: Process each attention head separately using explicit iteration
        # This is less efficient but more educational than vectorized operations
        for i in range(self.num_heads):
            # PROJECT: Transform inputs for this specific head using dedicated projections
            # Each head gets its own linear transformation with learned parameters
            # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, head_dim)
            q_i = self.query_projections[i](query)  # What this head is looking for
            k_i = self.key_projections[i](key)  # What this head compares against
            v_i = self.value_projections[i](value)  # What this head extracts

            # ATTENTION SCORES: Compute Q @ K^T using JAX matrix multiplication
            # Shape: (batch_size, seq_len, head_dim) @ (batch_size, head_dim, seq_len)
            #     -> (batch_size, seq_len, seq_len)
            attention_scores = jnp.matmul(q_i, jnp.transpose(k_i, (0, 2, 1))) * self.scale

            # MASKING: Apply mask if provided using functional JAX operations
            # Note: Different masking logic for 2D vs 4D attention scores
            if attention_mask is not None:
                # Expand mask from (batch_size, seq_len) to (batch_size, seq_len, 1)
                mask = jnp.expand_dims(attention_mask, axis=-1)
                # Broadcast to match attention_scores shape (batch_size, seq_len, seq_len)
                mask = jnp.broadcast_to(mask, attention_scores.shape)
                # Apply mask using functional jnp.where
                attention_scores = jnp.where(mask == 0, -1e9, attention_scores)

            # SOFTMAX: Normalize attention scores using JAX activation function
            # This ensures attention weights sum to 1 across the key dimension
            attention_weights_i = jax.nn.softmax(attention_scores, axis=-1)

            # DROPOUT: Apply regularization if RNGs provided
            # JAX requires explicit RNG threading for reproducible randomness
            if rngs is not None:
                attention_weights_i = self.dropout(attention_weights_i, rngs=rngs)

            # WEIGHTED SUM: Apply attention to values using JAX matrix multiplication
            # Shape: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, head_dim)
            #     -> (batch_size, seq_len, head_dim)
            context_i = jnp.matmul(attention_weights_i, v_i)

            # COLLECT: Store this head's output and attention weights
            # JAX lists are immutable - append creates new list
            head_outputs.append(context_i)
            head_attention_weights.append(attention_weights_i)

        # STEP 4: Concatenate all head outputs using JAX functional operation
        # Combine: list of (batch_size, seq_len, head_dim) -> (batch_size, seq_len, hidden_size)
        # jnp.concatenate is functional and creates new array
        context = jnp.concatenate(head_outputs, axis=-1)

        # STEP 5: Final output projection using NNX module
        # Combine information from all heads with learned linear transformation
        output = self.output_proj(context)

        # STEP 6: Combine attention weights from all heads for visualization
        # Stack and average: list -> (batch_size, num_heads, seq_len, seq_len) -> (batch_size, seq_len, seq_len)
        attention_weights = jnp.stack(head_attention_weights, axis=1)
        attention_weights = jnp.mean(attention_weights, axis=1)

        return output, attention_weights
