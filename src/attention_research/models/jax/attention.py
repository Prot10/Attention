"""Base attention mechanisms for JAX/NNX."""

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx


class BaseAttention(nnx.Module, ABC):
    """Base class for all attention mechanisms in JAX/NNX."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize base attention module.

        Parameters
        ----------
        hidden_size : int
            Hidden size dimension
        num_heads : int, default=8
            Number of attention heads
        dropout_rate : float, default=0.1
            Dropout rate
        use_bias : bool, default=False
            Whether to use bias in linear layers
        rngs : nnx.Rngs
            Random number generators
        """
        if hidden_size % num_heads != 0:
            msg = f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.head_dim = hidden_size // num_heads
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
        """Forward pass of attention mechanism.

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

    def _apply_mask(
        self,
        attention_scores: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply attention mask to scores.

        Parameters
        ----------
        attention_scores : jnp.ndarray
            Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
        attention_mask : jnp.ndarray | None, default=None
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        jnp.ndarray
            Masked attention scores
        """
        if attention_mask is not None:
            # Convert mask to proper shape
            mask = jnp.expand_dims(attention_mask, axis=(1, 2))  # (batch_size, 1, 1, seq_len)
            mask = jnp.broadcast_to(mask, attention_scores.shape)

            # Apply mask (set masked positions to large negative value)
            attention_scores = jnp.where(mask == 0, -1e9, attention_scores)

        return attention_scores


class VanillaAttention(BaseAttention):
    """Vanilla scaled dot-product attention in JAX/NNX."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize vanilla attention.

        Parameters
        ----------
        hidden_size : int
            Hidden size dimension
        num_heads : int, default=8
            Number of attention heads
        dropout_rate : float, default=0.1
            Dropout rate
        use_bias : bool, default=False
            Whether to use bias in linear layers
        rngs : nnx.Rngs
            Random number generators
        """
        super().__init__(hidden_size, num_heads, dropout_rate, use_bias=use_bias, rngs=rngs)

        # Initialize layers
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
        self.output_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
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
        """Forward pass of vanilla attention.

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
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.query_proj(query)  # (batch_size, seq_len, hidden_size)
        k = self.key_proj(key)
        v = self.value_proj(value)

        # Reshape to multi-head format
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply mask if provided
        attention_scores = self._apply_mask(attention_scores, attention_mask)

        # Apply softmax
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        if rngs is not None:
            attention_weights = self.dropout(attention_weights, rngs=rngs)

        # Apply attention to values
        context = jnp.matmul(attention_weights, v)

        # Reshape back to original format
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.output_proj(context)

        return output, jnp.mean(attention_weights, axis=1)  # Average attention weights across heads


class MultiHeadAttention(BaseAttention):
    """Multi-head attention with separate head processing in JAX/NNX."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize multi-head attention.

        Parameters
        ----------
        hidden_size : int
            Hidden size dimension
        num_heads : int, default=8
            Number of attention heads
        dropout_rate : float, default=0.1
            Dropout rate
        use_bias : bool, default=False
            Whether to use bias in linear layers
        rngs : nnx.Rngs
            Random number generators
        """
        super().__init__(hidden_size, num_heads, dropout_rate, use_bias=use_bias, rngs=rngs)

        # Separate projections for each head
        self.query_projections = [
            nnx.Linear(
                hidden_size,
                self.head_dim,
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

        self.output_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
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
        """Forward pass of multi-head attention.

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
        batch_size, seq_len, _ = query.shape

        head_outputs = []
        head_attention_weights = []

        for i in range(self.num_heads):
            # Project for this head
            q_i = self.query_projections[i](query)  # (batch_size, seq_len, head_dim)
            k_i = self.key_projections[i](key)
            v_i = self.value_projections[i](value)

            # Compute attention scores
            attention_scores = jnp.matmul(q_i, jnp.transpose(k_i, (0, 2, 1))) * self.scale

            # Apply mask if provided
            if attention_mask is not None:
                mask = jnp.expand_dims(attention_mask, axis=-1)  # (batch_size, seq_len, 1)
                mask = jnp.broadcast_to(mask, attention_scores.shape)  # (batch_size, seq_len, seq_len)
                attention_scores = jnp.where(mask == 0, -1e9, attention_scores)

            # Apply softmax
            attention_weights_i = jax.nn.softmax(attention_scores, axis=-1)
            if rngs is not None:
                attention_weights_i = self.dropout(attention_weights_i, rngs=rngs)

            # Apply attention to values
            context_i = jnp.matmul(attention_weights_i, v_i)

            head_outputs.append(context_i)
            head_attention_weights.append(attention_weights_i)

        # Concatenate head outputs
        context = jnp.concatenate(head_outputs, axis=-1)  # (batch_size, seq_len, hidden_size)

        # Final projection
        output = self.output_proj(context)

        # Average attention weights across heads
        attention_weights = jnp.stack(head_attention_weights, axis=1)
        attention_weights = jnp.mean(attention_weights, axis=1)

        return output, attention_weights
