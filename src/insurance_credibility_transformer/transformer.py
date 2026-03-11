"""Credibility Transformer layers and main model.

Implements the Transformer architecture from Section 2 of
Richman, Scognamiglio & Wüthrich (arXiv:2409.16653).

The forward pass structure:
1. Tokenize features → x+_{1:T+1} ∈ R^{(T+1)×2b}
2. L Transformer layers (attention + skip + FNN + skip + LayerNorm x2)
3. Extract CLS token as c_trans; build c_prior from FNN-only path
4. Credibility mechanism: c_cred = Z*c_trans + (1-Z)*c_prior
5. Decoder FNN → exposure-adjusted prediction
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import (
    CredibilityMechanism,
    MultiHeadCredibilityAttention,
    StandardFeedForward,
    SwiGLUFeedForward,
)
from .tokenizer import FeatureTokenizer


class CredibilityTransformerLayer(nn.Module):
    """Single Transformer layer with skip connections and LayerNorm.

    Implements the standard Pre-LN Transformer block adapted for CT:
        h = LayerNorm(x + Dropout(Attention(x)))
        output = LayerNorm(h + Dropout(FNN(h)))

    NormFormer variant (used by default for stability): per-head scaling
    is applied inside the attention module.

    Args:
        token_dim: Token dimension (2b).
        n_heads: Number of attention heads M.
        use_swiglu: If True, use SwiGLU gating instead of tanh FNN.
        dropout: Dropout rate applied after attention and FNN.
    """

    def __init__(
        self,
        token_dim: int,
        n_heads: int = 1,
        use_swiglu: bool = False,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadCredibilityAttention(token_dim, n_heads, dropout)
        if use_swiglu:
            self.ffn: nn.Module = SwiGLUFeedForward(token_dim, dropout=dropout)
        else:
            self.ffn = StandardFeedForward(token_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: (batch, T+1, token_dim)
            return_attn: Return attention weights for explainability.

        Returns:
            output: (batch, T+1, token_dim)
            attn_weights: (batch, n_heads, T+1, T+1) or None
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(x, return_attn=return_attn)
        x = self.norm1(x + self.dropout(attn_out))

        # FNN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class CredibilityTransformer(nn.Module):
    """The Credibility Transformer (CT) model.

    Implements the full model from arXiv:2409.16653. Produces an
    exposure-adjusted frequency (or severity) prediction via:
        mu_CT(x) = exp(decoder(c_cred(x)))

    The base CT with default settings has ~1,746 parameters and trains
    comfortably on CPU. The deep CT (n_heads=2, n_layers=3, use_ple=True,
    use_swiglu=True) has ~320K parameters and benefits from GPU.

    Args:
        cat_cardinalities: Number of levels per categorical feature.
            E.g. [6, 2, 11, 22] for 4 categorical features.
        n_num_features: Number of continuous features.
        embed_dim: Base embedding dimension b. Features are tokenized to
            b, then concatenated with b-dim positional encoding → 2b.
        n_heads: Attention heads M. Paper base=1, deep=2.
        n_layers: Transformer layers L. Paper base=1, deep=3.
        decoder_hidden: Hidden units in the decoder FNN.
        alpha: Credibility parameter. alpha=0.90 means 90% of gradient
            steps train on individual info, 10% on portfolio mean.
        use_ple: Use PiecewiseLinearEncoding for continuous features.
        n_ple_bins: PLE bins per continuous feature.
        use_swiglu: Use SwiGLU gating in FNN blocks.
        dropout: Dropout rate throughout.
        link: Output link function. 'log' for frequency, 'identity' for
            severity, 'softplus' for non-negative severity.
    """

    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num_features: int,
        embed_dim: int = 5,
        n_heads: int = 1,
        n_layers: int = 1,
        decoder_hidden: int = 16,
        alpha: float = 0.90,
        use_ple: bool = False,
        n_ple_bins: int = 16,
        use_swiglu: bool = False,
        dropout: float = 0.01,
        link: str = "log",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.alpha = alpha
        self.link = link

        token_dim = 2 * embed_dim

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities=cat_cardinalities,
            n_num_features=n_num_features,
            embed_dim=embed_dim,
            use_ple=use_ple,
            n_ple_bins=n_ple_bins,
        )

        # L Transformer layers
        self.layers = nn.ModuleList([
            CredibilityTransformerLayer(token_dim, n_heads, use_swiglu, dropout)
            for _ in range(n_layers)
        ])

        # Credibility mechanism (Bernoulli switch)
        self.credibility = CredibilityMechanism(alpha)

        # c_prior path: same FNN as the Transformer layers but applied to
        # CLS token directly (no attention). We use a separate FNN here
        # that mirrors the last layer's FNN for the prior path.
        # Paper: c_prior is CLS after FNN blocks WITHOUT attention.
        if use_swiglu:
            self.prior_ffn: nn.Module = SwiGLUFeedForward(token_dim, dropout=dropout)
        else:
            self.prior_ffn = StandardFeedForward(token_dim, dropout=dropout)
        self.prior_norm1 = nn.LayerNorm(token_dim)
        self.prior_norm2 = nn.LayerNorm(token_dim)
        self.prior_dropout = nn.Dropout(dropout)

        # Decoder: 2-layer FNN, R^{2b} → R
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, decoder_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )

    def encode(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run the encoder and return the credibility-weighted CLS token.

        This is useful for ICL (which needs c_cred as input) and for
        the AttentionExplainer.

        Args:
            x_cat: (batch, n_cat) integer categorical features.
            x_num: (batch, n_num) float continuous features.
            return_attn: Return per-layer attention weight matrices.

        Returns:
            c_cred: (batch, token_dim) credibility CLS token.
            attn_weights_list: List of attention tensors per layer, or None.
        """
        # Tokenize
        tokens = self.tokenizer(x_cat, x_num)  # (batch, T+1, 2b)

        # Extract initial CLS token for the prior path
        cls_init = tokens[:, -1, :]  # (batch, 2b)

        # c_prior: CLS token through FNN only (no attention)
        # Matches the FNN structure from one Transformer layer
        c_prior = self.prior_norm1(cls_init + self.prior_dropout(
            self.prior_ffn(cls_init)
        ))
        c_prior = self.prior_norm2(c_prior)

        # Full Transformer forward
        attn_weights_list = [] if return_attn else None
        x = tokens
        for layer in self.layers:
            x, attn_w = layer(x, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_weights_list.append(attn_w)

        # c_trans: CLS token after full attention (last position)
        c_trans = x[:, -1, :]  # (batch, 2b)

        # Credibility mechanism
        c_cred = self.credibility(c_trans, c_prior)  # (batch, 2b)

        return c_cred, attn_weights_list

    def forward(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        exposure: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Full forward pass to produce predictions.

        Args:
            x_cat: (batch, n_cat) integer categorical features.
            x_num: (batch, n_num) float continuous features.
            exposure: (batch,) or (batch, 1) exposure weights v_i.
                If None, assumes exposure=1.
            return_attn: Return per-layer attention weight matrices.

        Returns:
            mu: (batch,) predicted mean (claim frequency or severity).
            attn_weights_list: Per-layer attention weights or None.
        """
        c_cred, attn_weights_list = self.encode(x_cat, x_num, return_attn)

        # Decoder: R^{2b} → R
        log_mu = self.decoder(c_cred).squeeze(-1)  # (batch,)

        # Apply link function
        if self.link == "log":
            # Clamp log_mu to prevent overflow (exp(88) ~ float32 max)
            mu = torch.exp(log_mu.clamp(min=-10.0, max=10.0))
        elif self.link == "softplus":
            mu = nn.functional.softplus(log_mu)
        else:  # identity
            mu = log_mu

        # Multiply by exposure if provided
        if exposure is not None:
            if exposure.dim() > 1:
                exposure = exposure.squeeze(-1)
            mu = mu * exposure

        return mu, attn_weights_list

    def predict(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        exposure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method: predict in eval mode.

        Returns mu (batch,) without gradient tracking.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.forward(x_cat, x_num, exposure, return_attn=False)
        return mu

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
