"""Credibility-aware multi-head attention.

Implements the attention mechanism from Section 2.2–2.3 of
Richman, Scognamiglio & Wüthrich (arXiv:2409.16653).

The CLS token (last position in the token sequence) plays a special role:
- c_prior: CLS after the FNN blocks WITHOUT attention (portfolio mean)
- c_trans: CLS after the full Transformer (has seen all covariates)

At training time, Z ~ Bernoulli(alpha) selects which to use. At inference,
always use c_trans. Alpha controls the credibility regularisation.

The attention weight P = a_{T+1, T+1} (CLS self-attention) has the
Bühlmann-Straub interpretation: P = prior weight, (1-P) = individual weight.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCredibilityAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    Standard multi-head attention (Vaswani et al. 2017) applied to the
    token sequence including the CLS token. The attention matrix is
    returned for explainability (the CLS self-attention weight P).

    Args:
        token_dim: Input/output token dimension (2b in paper).
        n_heads: Number of attention heads M. Default 1 (base CT).
        dropout: Attention dropout rate.
    """

    def __init__(self, token_dim: int, n_heads: int = 1, dropout: float = 0.01) -> None:
        super().__init__()
        assert token_dim % n_heads == 0, (
            f"token_dim ({token_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.token_dim = token_dim
        self.n_heads = n_heads
        self.head_dim = token_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        # Projection matrices for K, Q, V (eq. 2.2)
        # Paper uses separate projections per head; we use the standard
        # concatenated-then-split implementation
        self.W_Q = nn.Linear(token_dim, token_dim, bias=True)
        self.W_K = nn.Linear(token_dim, token_dim, bias=True)
        self.W_V = nn.Linear(token_dim, token_dim, bias=True)
        self.W_O = nn.Linear(token_dim, token_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

        # NormFormer per-head scaling (Section 4, training stability)
        # Learned scale coefficient per head applied after softmax
        self.head_scale = nn.Parameter(torch.ones(n_heads))

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Self-attention over the token sequence.

        Args:
            x: (batch, T+1, token_dim) token sequence including CLS.
            return_attn: If True, also return attention weight matrix.

        Returns:
            output: (batch, T+1, token_dim)
            attn_weights: (batch, n_heads, T+1, T+1) if return_attn else None.
        """
        batch, seq_len, _ = x.shape

        # Project → Q, K, V: (batch, seq, token_dim)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head: (batch, n_heads, seq, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled dot-product attention: A ∈ R^{T+1 × T+1} (eq. 2.2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (..., seq, seq)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_heads, seq, seq)

        # NormFormer per-head scaling (applied after softmax for stability)
        scale = self.head_scale.view(1, self.n_heads, 1, 1)
        attn_weights_scaled = attn_weights * scale

        if self.training and self.dropout > 0:
            attn_weights_scaled = self.attn_dropout(attn_weights_scaled)

        # H = AV: (batch, n_heads, seq, head_dim)
        H = torch.matmul(attn_weights_scaled, V)

        # Merge heads: (batch, seq, token_dim)
        H = H.transpose(1, 2).contiguous().view(batch, seq_len, self.token_dim)
        output = self.W_O(H)

        return output, (attn_weights if return_attn else None)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU-gated feed-forward network.

    Replaces the standard 2-layer FNN in the deep CT (Section 4 of paper):
        z^GLU(x) = z^FNN_linear(x) ⊙ z^FNN_SiLU(x)

    SiLU (Sigmoid Linear Unit) = x * sigmoid(x) = swish activation.
    The gating provides expressive nonlinearity without stacking layers.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension. Defaults to 4*dim.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.01) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.linear_gate = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_value = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_out = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) → (..., dim)."""
        gate = F.silu(self.linear_gate(x))   # SiLU(Wx)
        value = self.linear_value(x)          # Vx
        h = gate * value                       # element-wise product
        return self.linear_out(self.dropout(h))


class StandardFeedForward(nn.Module):
    """Standard 2-layer FNN with tanh activation (base CT).

    The time-distributed FNN from Section 2.2 applied after attention.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension. Defaults to 4*dim.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.01) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) → (..., dim)."""
        h = torch.tanh(self.linear1(x))
        return self.linear2(self.dropout(h))


class CredibilityMechanism(nn.Module):
    """Bernoulli credibility switch between c_prior and c_trans.

    Implements the credibility mechanism from Section 2.3 (eq. 2.3):
        Z ~ Bernoulli(alpha)
        c_cred = Z * c_trans + (1-Z) * c_prior

    At training time, Z is sampled randomly. At eval time, Z=1 always
    (i.e., always use c_trans = the individual risk estimate).

    Args:
        alpha: Probability of using c_trans during training. Paper uses 0.90.
            In alpha% of gradient steps, model trains on individual info;
            in (1-alpha)% it trains to encode the portfolio mean.
    """

    def __init__(self, alpha: float = 0.90) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        c_trans: torch.Tensor,
        c_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Select between individual and portfolio estimate.

        Args:
            c_trans: (batch, token_dim) CLS token after full attention.
            c_prior: (batch, token_dim) CLS token through FNN only (no attention).

        Returns:
            c_cred: (batch, token_dim) credibility-weighted CLS token.
        """
        if self.training:
            # Sample one Z per batch element
            Z = torch.bernoulli(
                torch.full((c_trans.shape[0], 1), self.alpha, device=c_trans.device)
            )  # (batch, 1)
            return Z * c_trans + (1 - Z) * c_prior
        else:
            # At inference: always use c_trans (individual estimate)
            return c_trans
