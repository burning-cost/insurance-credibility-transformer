"""Attention-based explainability for the Credibility Transformer.

The key explainability insight from arXiv:2409.16653 Section 2.3:

    P = a_{T+1, T+1} = attention weight the CLS token assigns to itself

Interpretation:
    - P = prior weight (how much the model relies on the portfolio mean)
    - (1-P) = individual credibility weight (how much individual features matter)

This is the Bühlmann-Straub credibility formula in attention form:
    v_trans = P * v_{T+1} + (1-P) * v_covariate

High P → model is uncertain about this risk, falls back to portfolio mean.
Low P → model is confident in individual covariate signal.

The per-feature attention weights a_{T+1, t} / (1-P) give the relative
importance of each feature in determining the individual premium.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .transformer import CredibilityTransformer


class AttentionExplainer:
    """Extract credibility weights and feature importances from a trained CT.

    Args:
        model: A trained CredibilityTransformer.
        device: Device for inference.
    """

    def __init__(
        self,
        model: CredibilityTransformer,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        self.model.eval()

    def cls_attention(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        layer: int = -1,
        head: int = 0,
    ) -> np.ndarray:
        """Compute per-policy CLS self-attention weight P.

        P = a_{T+1, T+1}: the weight the CLS token assigns to itself.
        High P → relies on portfolio prior. Low P → individual credibility.

        Args:
            x_cat: (n, n_cat) integer categorical features.
            x_num: (n, n_num) float continuous features.
            layer: Which Transformer layer to extract attention from.
                -1 = last layer (default).
            head: Which attention head. 0 = first head.

        Returns:
            P: (n,) array of CLS self-attention weights (prior weights).
        """
        with torch.no_grad():
            if x_cat is not None:
                x_cat = x_cat.to(self.device)
            if x_num is not None:
                x_num = x_num.to(self.device)
            _, attn_list = self.model.encode(x_cat, x_num, return_attn=True)

        if attn_list is None or len(attn_list) == 0:
            raise RuntimeError("No attention weights returned. Ensure return_attn=True in encode().")

        # attn_list[layer]: (batch, n_heads, T+1, T+1)
        attn = attn_list[layer]  # (batch, n_heads, T+1, T+1)
        cls_idx = attn.shape[-1] - 1  # CLS is last position

        # CLS self-attention: row CLS, column CLS
        P = attn[:, head, cls_idx, cls_idx]  # (batch,)
        return P.cpu().numpy()

    def feature_attention(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        feature_names: Optional[List[str]] = None,
        layer: int = -1,
        head: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Per-feature attention weights from the CLS token.

        Returns a_{T+1, t} for each feature t. These are the raw attention
        weights the CLS token assigns to each feature. Normalising by
        (1-P) gives the individual credibility decomposition.

        Args:
            x_cat: (n, n_cat) integer categorical features.
            x_num: (n, n_num) float continuous features.
            feature_names: Optional list of feature names for the dict keys.
                If None, uses 'cat_0', 'cat_1', ..., 'num_0', 'num_1', ...
            layer: Transformer layer.
            head: Attention head.

        Returns:
            Dict mapping feature_name → (n,) attention weight array.
        """
        with torch.no_grad():
            if x_cat is not None:
                x_cat = x_cat.to(self.device)
            if x_num is not None:
                x_num = x_num.to(self.device)
            _, attn_list = self.model.encode(x_cat, x_num, return_attn=True)

        attn = attn_list[layer]  # (batch, n_heads, T+1, T+1)
        cls_idx = attn.shape[-1] - 1

        # CLS → all features: (batch, T)
        cls_to_features = attn[:, head, cls_idx, :cls_idx]  # (batch, T)

        n_cat = self.model.tokenizer.n_cat
        n_num = self.model.tokenizer.n_num

        if feature_names is None:
            feature_names = (
                [f"cat_{i}" for i in range(n_cat)] +
                [f"num_{i}" for i in range(n_num)]
            )

        result = {}
        for i, name in enumerate(feature_names):
            result[name] = cls_to_features[:, i].cpu().numpy()

        return result

    def individual_credibility(
        self,
        x_cat: Optional[torch.Tensor],
        x_num: Optional[torch.Tensor],
        layer: int = -1,
        head: int = 0,
    ) -> np.ndarray:
        """Individual credibility weight (1-P) per policy.

        (1-P) = 1 - a_{T+1, T+1} is the weight on individual covariate
        signal, analogous to the Bühlmann-Straub individual weight.

        Returns:
            z: (n,) individual credibility weights in [0, 1].
        """
        P = self.cls_attention(x_cat, x_num, layer=layer, head=head)
        return 1.0 - P
