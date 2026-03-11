"""Tests for ICL module: OutcomeTokenDecorator, ICLTransformerLayer, masking."""

import numpy as np
import pytest
import torch

from insurance_credibility_transformer import (
    CredibilityTransformer,
    ICLCredibilityTransformer,
)
from insurance_credibility_transformer.icl import (
    ICLTransformerLayer,
    OutcomeTokenDecorator,
)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def make_base_ct():
    return CredibilityTransformer(
        cat_cardinalities=[4, 2],
        n_num_features=2,
        embed_dim=4,
        n_heads=1,
        n_layers=1,
        dropout=0.0,
    )


class TestOutcomeTokenDecorator:
    def test_context_decorated_target_unchanged(self):
        token_dim = 8
        dec = OutcomeTokenDecorator(token_dim, kappa_init=1.0)
        dec.eval()

        batch = 6
        c_cred = torch.randn(batch, token_dim)
        y = torch.rand(batch) * 0.5
        exposure = torch.ones(batch)
        # First 4 are context, last 2 are target
        mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)

        c_decor = dec(c_cred, y, exposure, mask)

        # Target tokens should equal c_cred (no decoration)
        assert torch.allclose(c_decor[4:], c_cred[4:])
        # Context tokens should differ (decoration added)
        assert not torch.allclose(c_decor[:4], c_cred[:4])

    def test_kappa_is_positive(self):
        dec = OutcomeTokenDecorator(8)
        assert float(dec.kappa) > 0

    def test_high_exposure_high_weight(self):
        """Higher exposure → more weight on observed outcome (closer to full decoration)."""
        dec = OutcomeTokenDecorator(8, kappa_init=1.0)
        dec.eval()

        token_dim = 8
        c_cred = torch.zeros(2, token_dim)
        y = torch.ones(2)
        # One low exposure, one high
        exposure = torch.tensor([0.01, 100.0])
        mask = torch.ones(2, dtype=torch.bool)

        c_decor = dec(c_cred, y, exposure, mask)
        # High exposure row should have larger decoration norm
        norm_low = c_decor[0].norm()
        norm_high = c_decor[1].norm()
        assert norm_high > norm_low

    def test_output_shape(self):
        dec = OutcomeTokenDecorator(12)
        c_cred = torch.randn(10, 12)
        y = torch.rand(10)
        exposure = torch.ones(10)
        mask = torch.ones(10, dtype=torch.bool)
        out = dec(c_cred, y, exposure, mask)
        assert out.shape == (10, 12)


class TestICLTransformerLayer:
    def test_output_shape(self):
        layer = ICLTransformerLayer(token_dim=8, n_heads=1, dropout=0.0)
        c_decor = torch.randn(1, 10, 8)  # (batch=1, n=10, dim=8)
        is_target = torch.zeros(1, 10, dtype=torch.bool)
        is_target[:, -3:] = True  # last 3 are targets

        out = layer(c_decor, is_target)
        assert out.shape == (1, 10, 8)

    def test_causal_mask_prevents_target_target_attention(self):
        """Two target rows should not attend to each other.

        We verify this by checking the mask directly, not by checking
        that outputs are identical (they differ because of self-attention).
        """
        layer = ICLTransformerLayer(token_dim=8, n_heads=1, dropout=0.0)

        n = 5
        is_target = torch.tensor([[False, False, True, True, True]])  # (1, 5)
        mask = layer._build_causal_mask(is_target, device=torch.device("cpu"))
        # mask: (1, 5, 5) — target rows (2,3,4) should not attend to each other
        # i=2, j=3 → both target, different → -inf
        assert mask[0, 2, 3] == float("-inf")
        assert mask[0, 3, 2] == float("-inf")
        # i=2, j=2 → same index → 0 (self-attention allowed)
        assert mask[0, 2, 2] == 0.0
        # i=0 (context) → no mask
        assert mask[0, 0, 2] == 0.0
        assert mask[0, 0, 3] == 0.0

    def test_context_not_masked(self):
        """Context instances can attend to everyone."""
        layer = ICLTransformerLayer(token_dim=8, n_heads=1, dropout=0.0)
        is_target = torch.tensor([[True, False, True]])  # (1, 3)
        mask = layer._build_causal_mask(is_target, device=torch.device("cpu"))
        # Context (index 1) → no -inf anywhere in its row
        assert (mask[0, 1, :] == 0.0).all()

    def test_linearized_mode(self):
        layer = ICLTransformerLayer(token_dim=8, n_heads=1, dropout=0.0, linearized=True)
        c_decor = torch.randn(1, 6, 8)
        c_base = torch.randn(1, 6, 8)
        is_target = torch.zeros(1, 6, dtype=torch.bool)
        is_target[:, -2:] = True
        out = layer(c_decor, is_target, c_cred_base=c_base)
        assert out.shape == (1, 6, 8)


class TestICLCredibilityTransformer:
    def test_forward_output_shape(self):
        base_ct = make_base_ct()
        base_ct.eval()
        icl_ct = ICLCredibilityTransformer(base_ct, icl_layers=1)
        icl_ct.eval()

        n_ctx, n_tgt = 10, 5

        def rand_inputs(n):
            x_cat = torch.stack([torch.randint(0, 4, (n,)), torch.randint(0, 2, (n,))], dim=1)
            x_num = torch.randn(n, 2)
            return x_cat, x_num

        x_cat_c, x_num_c = rand_inputs(n_ctx)
        x_cat_t, x_num_t = rand_inputs(n_tgt)
        y_ctx = torch.rand(n_ctx) * 0.3
        exp_ctx = torch.ones(n_ctx)
        exp_tgt = torch.ones(n_tgt)

        with torch.no_grad():
            mu = icl_ct(x_cat_c, x_num_c, y_ctx, exp_ctx, x_cat_t, x_num_t, exp_tgt)

        assert mu.shape == (n_tgt,)
        assert (mu > 0).all()

    def test_decoder_shares_weights_with_base(self):
        """ICL decoder is the same object as base_ct.decoder."""
        base_ct = make_base_ct()
        icl_ct = ICLCredibilityTransformer(base_ct)
        # Check same underlying parameters
        for p1, p2 in zip(base_ct.decoder.parameters(), icl_ct.decoder.parameters()):
            assert p1.data_ptr() == p2.data_ptr()
