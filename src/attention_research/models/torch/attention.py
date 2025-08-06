"""Base attention mechanisms for PyTorch.

This module implements the attention mechanisms described in "Attention Is All You Need"
(Vaswani et al., 2017) - the seminal paper that introduced the Transformer architecture.

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

This implementation uses torch.einsum for efficient tensor operations and includes
educational comments to help understand each step of the computation.

References
----------
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
In Advances in neural information processing systems (pp. 5998-6008).
"""

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.

    This class provides the common interface and utilities shared by all
    attention implementations. It handles the mathematical foundations
    like scaling factors and masking operations.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Initialize base attention parameters.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size (d_model in the paper)
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout : float, optional
            Dropout probability for attention weights, by default 0.1
        """
        super().__init__()

        # VALIDATION: Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            msg = f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        # CORE PARAMETERS: Store attention configuration
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # d_k = d_model / num_heads

        # SCALING FACTOR: √d_k from the paper - prevents softmax saturation
        # When d_k is large, the dot products grow large in magnitude, pushing
        # the softmax function into regions where it has extremely small gradients
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # REGULARIZATION: Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)

    def _apply_mask(self, attention_scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Apply attention mask to prevent attention to certain positions.

        This is crucial for:
        1. Padding tokens: Don't attend to padding positions
        2. Causal masking: In decoder, don't attend to future positions
        3. Custom masking: Any application-specific attention restrictions

        Parameters
        ----------
        attention_scores : torch.Tensor
            Raw attention scores of shape (..., seq_len, seq_len)
        attention_mask : torch.Tensor | None
            Mask tensor where 0 means "don't attend" and 1 means "attend"

        Returns
        -------
        torch.Tensor
            Masked attention scores with -inf where attention is not allowed
        """
        if attention_mask is None:
            return attention_scores

        # EXPAND MASK: Ensure mask has the right shape for broadcasting
        # attention_mask is typically (batch_size, seq_len)
        # We need it to be (batch_size, 1, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)

        if attention_mask.dim() == 2:  # (batch_size, seq_len)
            # Expand to (batch_size, 1, 1, seq_len) for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        elif attention_mask.dim() == 3:  # (batch_size, seq_len, seq_len)
            # Expand to (batch_size, 1, seq_len, seq_len) for broadcasting
            attention_mask = attention_mask.unsqueeze(1)

        # APPLY MASK: Set masked positions to large negative value
        # After softmax, these become effectively 0
        return attention_scores.masked_fill(attention_mask == 0, -1e9)

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Abstract method for forward pass.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor
        key : torch.Tensor
            Key tensor
        value : torch.Tensor
            Value tensor
        attention_mask : torch.Tensor | None, default=None
            Attention mask
        **kwargs
            Additional keyword arguments

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (output, attention_weights)
        """


class VanillaAttention(BaseAttention):
    """Vanilla scaled dot-product attention using efficient tensor operations.

    This implements the standard multi-head attention as described in
    "Attention Is All You Need". It projects all heads at once and uses
    efficient matrix operations.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Initialize vanilla attention.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout : float, optional
            Dropout probability, by default 0.1
        """
        super().__init__(hidden_size, num_heads, dropout)

        # STEP 1: Initialize linear projection layers
        # These project the input to Query, Key, and Value representations
        # Note: bias=False is common in transformer implementations
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Final output projection (W^O in the paper)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of vanilla attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_size)
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_size)
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask : torch.Tensor | None, default=None
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (output, attention_weights)
        """
        # STEP 1: Get input dimensions
        batch_size, seq_len, _ = query.shape

        # STEP 2: Project inputs to Query, Key, Value spaces
        # Each projection: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        q = self.query_proj(query)  # What we're looking for
        k = self.key_proj(key)  # What we're comparing against
        v = self.value_proj(value)  # What we want to extract

        # STEP 3: Reshape for multi-head attention
        # Split hidden_size into (num_heads, head_dim) and move heads to dimension 1
        # This allows parallel processing of all attention heads

        # Using einops-style reshape: 'batch seq (heads dim) -> batch heads seq dim'
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape after reshape: (batch_size, num_heads, seq_len, head_dim)

        # STEP 4: Compute attention scores using einsum
        # This computes Q @ K^T for all heads in parallel
        # 'bhid,bhjd->bhij': batch, heads, seq_i, dim @ batch, heads, seq_j, dim -> batch, heads, seq_i, seq_j
        attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # STEP 5: Apply mask if provided
        # This prevents attention to padding tokens or future positions
        attention_scores = self._apply_mask(attention_scores, attention_mask)

        # STEP 6: Apply softmax to get attention weights
        # This normalizes attention scores so they sum to 1 across the key dimension
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # STEP 7: Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)

        # STEP 8: Apply attention to values using einsum
        # This computes the weighted sum of values based on attention weights
        # 'bhij,bhjd->bhid': batch, heads, seq_i, seq_j @ batch, heads, seq_j, dim -> batch, heads, seq_i, dim
        context = torch.einsum("bhij,bhjd->bhid", attention_weights, v)

        # STEP 9: Reshape back to original format
        # Combine heads back into hidden dimension: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # STEP 10: Final linear projection
        # This is the W^O matrix from the paper, combining information from all heads
        output = self.output_proj(context)

        # STEP 11: Return output and attention weights (averaged across heads for visualization)
        return output, attention_weights.mean(dim=1)  # Average attention weights across heads


class MultiHeadAttention(BaseAttention):
    """Multi-head attention with separate head processing.

    This is an alternative implementation that processes each attention head
    separately. This is less efficient but more explicit about the multi-head
    structure and can be educational.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Initialize multi-head attention with separate projections per head.

        Parameters
        ----------
        hidden_size : int
            Hidden dimension size
        num_heads : int, optional
            Number of attention heads, by default 8
        dropout : float, optional
            Dropout probability, by default 0.1
        """
        super().__init__(hidden_size, num_heads, dropout)

        # STEP 1: Create separate projection layers for each head
        # This makes the multi-head structure very explicit
        self.query_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)]
        )

        # Final output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention with explicit head processing.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_size)
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_size)
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask : torch.Tensor | None, default=None
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (output, attention_weights)
        """
        # STEP 1: Get input dimensions
        batch_size, seq_len, _ = query.shape

        # STEP 2: Initialize lists to store outputs from each head
        head_outputs = []
        head_attention_weights = []

        # STEP 3: Process each attention head separately
        for i in range(self.num_heads):
            # PROJECT: Transform inputs for this specific head
            # Each head gets its own linear transformation
            q_i = self.query_projections[i](query)  # (batch_size, seq_len, head_dim)
            k_i = self.key_projections[i](key)  # (batch_size, seq_len, head_dim)
            v_i = self.value_projections[i](value)  # (batch_size, seq_len, head_dim)

            # ATTENTION SCORES: Compute Q @ K^T using einsum
            # 'bid,bjd->bij': batch, seq_i, dim @ batch, seq_j, dim -> batch, seq_i, seq_j
            attention_scores = torch.einsum("bid,bjd->bij", q_i, k_i) * self.scale

            # MASKING: Apply mask if provided
            attention_scores = self._apply_mask(attention_scores, attention_mask)

            # SOFTMAX: Normalize attention scores
            attention_weights_i = torch.nn.functional.softmax(attention_scores, dim=-1)

            # DROPOUT: Apply regularization
            attention_weights_i = self.dropout(attention_weights_i)

            # WEIGHTED SUM: Apply attention to values using einsum
            # 'bij,bjd->bid': batch, seq_i, seq_j @ batch, seq_j, dim -> batch, seq_i, dim
            context_i = torch.einsum("bij,bjd->bid", attention_weights_i, v_i)

            # COLLECT: Store this head's output and attention weights
            head_outputs.append(context_i)
            head_attention_weights.append(attention_weights_i)

        # STEP 4: Concatenate all head outputs
        # Stack: list of (batch, seq, head_dim) -> (batch, seq, num_heads * head_dim)
        context = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, hidden_size)

        # STEP 5: Final output projection
        # Combine information from all heads with learned linear transformation
        output = self.output_proj(context)

        # STEP 6: Combine attention weights from all heads for visualization
        # Stack and average: (num_heads, batch, seq, seq) -> (batch, seq, seq)
        attention_weights = torch.stack(head_attention_weights, dim=1).mean(dim=1)

        return output, attention_weights
