"""Feature tokenization for the Credibility Transformer.

Implements the input tokenization described in Section 2.1 of
Richman, Scognamiglio & Wüthrich (arXiv:2409.16653).

Categorical features → entity embeddings: e_EE_t : {a1,...,a_nt} → R^b
Continuous features  → 2-layer FNN:        z_t^(2:1) : R → R^b
All tokens to dimension b, then positional encodings concatenated → 2b.
CLS token appended: augmented tensor x+_{1:T+1} ∈ R^{(T+1)×2b}.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityEmbedding(nn.Module):
    """Categorical feature → dense embedding of dimension embed_dim.

    Standard entity embedding (Guo & Berkhahn 2016). One embedding
    table per categorical feature.

    Args:
        cardinality: Number of distinct levels for this feature.
        embed_dim: Output embedding dimension (b in paper).
    """

    def __init__(self, cardinality: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cardinality, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) integer indices → (..., embed_dim)."""
        return self.embedding(x)


class NumericalEmbedding(nn.Module):
    """Continuous feature → embedding via 2-layer FNN.

    Implements z_t^(2:1): R → R^b from equation (2.1) in paper.
    Layer 1: linear (no activation), Layer 2: tanh.

    Args:
        embed_dim: Output dimension (b in paper).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1, embed_dim, bias=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) scalar feature → (..., embed_dim)."""
        x = x.unsqueeze(-1)  # (..., 1)
        x = self.linear1(x)  # (..., embed_dim), linear activation
        x = torch.tanh(self.linear2(x))  # (..., embed_dim)
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Differentiable Piecewise Linear Encoding for continuous features.

    Implements the PLE extension from Section 4.1 of arXiv:2409.16653,
    following the learnable-boundaries variant: bin widths are parameterised
    via log(delta_k) so that delta_k > 0 always. After PLE the encoding
    passes through a tanh FNN to produce a token of dimension embed_dim.

    The encoding for value x with bins [t_0, t_1, ..., t_K] is:
        v_k = clamp((x - t_{k-1}) / (t_k - t_{k-1}), 0, 1)  for k=1..K
    giving a K-dimensional vector that is linear within each bin.

    Args:
        embed_dim: Output dimension (b in paper).
        n_bins: Number of PLE bins (K). Default 16.
        eps: Minimum bin width to prevent collapse.
    """

    def __init__(self, embed_dim: int, n_bins: int = 16, eps: float = 1e-3) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.eps = eps

        # Learnable log bin widths: delta_k = softplus(log_delta_k) + eps
        self.log_delta = nn.Parameter(torch.zeros(n_bins))

        # FNN after PLE: n_bins → embed_dim with tanh
        self.linear1 = nn.Linear(n_bins, embed_dim, bias=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=True)

        # Base position (left edge of first bin, learned)
        self.t0 = nn.Parameter(torch.tensor(0.0))

    @property
    def bin_widths(self) -> torch.Tensor:
        """Positive bin widths, min eps."""
        return F.softplus(self.log_delta) + self.eps

    @property
    def bin_boundaries(self) -> torch.Tensor:
        """Cumulative sum of bin widths from t0. Shape: (n_bins+1,)."""
        widths = self.bin_widths
        boundaries = torch.cat([self.t0.unsqueeze(0), self.t0 + widths.cumsum(0)])
        return boundaries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) scalar → (..., embed_dim)."""
        boundaries = self.bin_boundaries  # (n_bins+1,)
        left = boundaries[:-1]  # (n_bins,)
        right = boundaries[1:]  # (n_bins,)
        widths = right - left  # (n_bins,)

        # Broadcast: x (...,) → (..., n_bins)
        x_exp = x.unsqueeze(-1)  # (..., 1)
        v = (x_exp - left) / widths  # (..., n_bins)
        v = torch.clamp(v, 0.0, 1.0)  # PLE encoding

        # Pass through FNN with tanh
        h = torch.tanh(self.linear1(v))
        h = torch.tanh(self.linear2(h))
        return h


class FeatureTokenizer(nn.Module):
    """Tokenize mixed categorical/continuous features → token sequence.

    Produces the augmented input tensor x+_{1:T+1} ∈ R^{(T+1)×2b} with:
    - T tokens for the T input features (b-dim embedding + b-dim positional = 2b)
    - 1 CLS token (learned, dimension 2b)

    Args:
        cat_cardinalities: List of cardinalities for each categorical feature.
            Empty list if no categoricals.
        n_num_features: Number of continuous features. 0 if none.
        embed_dim: Token dimension b before positional encoding.
        use_ple: If True, use PiecewiseLinearEncoding for continuous features
            instead of plain NumericalEmbedding. For the deep CT (Section 4).
        n_ple_bins: Bins for PLE (used when use_ple=True).
    """

    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num_features: int,
        embed_dim: int,
        use_ple: bool = False,
        n_ple_bins: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num_features
        self.n_tokens = self.n_cat + self.n_num  # T
        assert self.n_tokens > 0, "Must have at least one feature"

        # Entity embeddings for categorical features
        self.cat_embeddings = nn.ModuleList(
            [EntityEmbedding(c, embed_dim) for c in cat_cardinalities]
        )

        # Numerical embeddings
        if n_num_features > 0:
            if use_ple:
                self.num_embeddings: nn.ModuleList = nn.ModuleList(
                    [PiecewiseLinearEncoding(embed_dim, n_ple_bins) for _ in range(n_num_features)]
                )
            else:
                self.num_embeddings = nn.ModuleList(
                    [NumericalEmbedding(embed_dim) for _ in range(n_num_features)]
                )
        else:
            self.num_embeddings = nn.ModuleList()

        # Positional encodings: one per feature position, dimension b
        # Concatenated to feature embedding → final token dimension 2b
        self.pos_encodings = nn.Embedding(self.n_tokens, embed_dim)
        nn.init.normal_(self.pos_encodings.weight, std=0.02)

        # CLS token: learned parameter, dimension 2b
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 2 * embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    @property
    def token_dim(self) -> int:
        """Output token dimension (2b)."""
        return 2 * self.embed_dim

    def forward(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Tokenize features and append CLS token.

        Args:
            x_cat: (batch, n_cat) integer-encoded categorical features.
                None if no categorical features.
            x_num: (batch, n_num) float32 continuous features.
                None if no numerical features.

        Returns:
            tokens: (batch, T+1, 2*embed_dim) where T = n_cat + n_num.
                Last position is the CLS token.
        """
        batch_size = (x_cat if x_cat is not None else x_num).shape[0]
        embeddings = []

        # Categorical embeddings
        if x_cat is not None and self.n_cat > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                embeddings.append(emb_layer(x_cat[:, i]))  # (batch, b)

        # Numerical embeddings
        if x_num is not None and self.n_num > 0:
            for i, emb_layer in enumerate(self.num_embeddings):
                embeddings.append(emb_layer(x_num[:, i]))  # (batch, b)

        # Stack: (batch, T, b)
        x = torch.stack(embeddings, dim=1)

        # Positional encodings: (T, b) → broadcast to (batch, T, b)
        positions = torch.arange(self.n_tokens, device=x.device)
        pos_enc = self.pos_encodings(positions)  # (T, b)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, T, b)

        # Concatenate feature embedding + positional encoding → (batch, T, 2b)
        x = torch.cat([x, pos_enc], dim=-1)

        # Append CLS token: (batch, T+1, 2b)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, 2b)
        x = torch.cat([x, cls], dim=1)

        return x
