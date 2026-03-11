"""Decoder modules for the Credibility Transformer.

The decoder is a 2-layer FNN: R^{2b} → R.
For frequency: output activation is exp (log link).
For severity: output activation is identity or softplus.

These are thin wrappers kept separate from the main CredibilityTransformer
so that the ICL extension can freeze the decoder during phase 2 training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoder(nn.Module):
    """Decode the credibility CLS token to claim frequency.

    Implements z^{2:1}: R^{2b} → R with exp output (log link).
    Final prediction is mu_CT(x) = exp(z^{2:1}(c_cred(x))).

    Args:
        token_dim: Input dimension (2b).
        hidden_dim: Hidden layer size. Paper uses 16 for base CT.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, c_cred: torch.Tensor) -> torch.Tensor:
        """c_cred: (batch, token_dim) → (batch,) predicted frequency."""
        log_mu = self.net(c_cred).squeeze(-1)
        return torch.exp(log_mu)


class SeverityDecoder(nn.Module):
    """Decode the credibility CLS token to claim severity.

    For severity modelling (average loss amount given a claim).
    Uses softplus output to ensure positivity.

    Args:
        token_dim: Input dimension (2b).
        hidden_dim: Hidden layer size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, c_cred: torch.Tensor) -> torch.Tensor:
        """c_cred: (batch, token_dim) → (batch,) predicted severity."""
        raw = self.net(c_cred).squeeze(-1)
        return F.softplus(raw)
