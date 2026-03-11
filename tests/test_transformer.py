"""Tests for transformer.py: CredibilityTransformer end-to-end."""

import pytest
import torch
import numpy as np

from insurance_credibility_transformer.transformer import (
    CredibilityTransformer,
    CredibilityTransformerLayer,
)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def make_small_ct(**kwargs) -> CredibilityTransformer:
    defaults = dict(
        cat_cardinalities=[6, 2, 11],
        n_num_features=2,
        embed_dim=5,
        n_heads=1,
        n_layers=1,
        alpha=0.90,
        dropout=0.0,
        link="log",
    )
    defaults.update(kwargs)
    return CredibilityTransformer(**defaults)


def make_batch(n=32, n_cat=3, n_num=2):
    x_cat = torch.stack([
        torch.randint(0, 6, (n,)),
        torch.randint(0, 2, (n,)),
        torch.randint(0, 11, (n,)),
    ], dim=1)
    x_num = torch.randn(n, n_num)
    exposure = torch.ones(n) * 0.5
    return x_cat, x_num, exposure


class TestCredibilityTransformerLayer:
    def test_output_shape(self):
        layer = CredibilityTransformerLayer(token_dim=10, n_heads=1, dropout=0.0)
        x = torch.randn(8, 6, 10)
        out, attn_w = layer(x, return_attn=False)
        assert out.shape == (8, 6, 10)
        assert attn_w is None

    def test_returns_attn_when_requested(self):
        layer = CredibilityTransformerLayer(token_dim=10, n_heads=1, dropout=0.0)
        x = torch.randn(4, 6, 10)
        _, attn_w = layer(x, return_attn=True)
        assert attn_w is not None
        assert attn_w.shape == (4, 1, 6, 6)

    def test_swiglu_layer(self):
        layer = CredibilityTransformerLayer(token_dim=10, n_heads=1, use_swiglu=True, dropout=0.0)
        x = torch.randn(4, 5, 10)
        out, _ = layer(x)
        assert out.shape == (4, 5, 10)


class TestCredibilityTransformer:
    def test_parameter_count_base(self):
        ct = make_small_ct()
        n_params = ct.count_parameters()
        # Base CT with tiny settings should be in hundreds, not millions
        assert n_params < 50_000
        assert n_params > 0

    def test_forward_output_shape(self):
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, exposure = make_batch(32)
        mu, attn_list = ct(x_cat, x_num, exposure)
        assert mu.shape == (32,)
        assert attn_list is None

    def test_forward_with_attn(self):
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, exposure = make_batch(16)
        mu, attn_list = ct(x_cat, x_num, exposure, return_attn=True)
        assert attn_list is not None
        assert len(attn_list) == 1  # 1 layer

    def test_predictions_positive(self):
        ct = make_small_ct(link="log")
        ct.eval()
        x_cat, x_num, exposure = make_batch(100)
        mu, _ = ct(x_cat, x_num, exposure)
        assert (mu > 0).all()

    def test_no_exposure_still_works(self):
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, _ = make_batch(8)
        mu, _ = ct(x_cat, x_num, exposure=None)
        assert mu.shape == (8,)

    def test_exposure_scaling(self):
        """Doubling exposure should double predictions (log link)."""
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, _ = make_batch(10)
        exp1 = torch.ones(10)
        exp2 = torch.ones(10) * 2.0
        mu1, _ = ct(x_cat, x_num, exp1)
        mu2, _ = ct(x_cat, x_num, exp2)
        ratio = mu2 / mu1
        assert torch.allclose(ratio, torch.ones(10) * 2.0, atol=1e-5)

    def test_encode_returns_correct_shape(self):
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, _ = make_batch(8)
        c_cred, _ = ct.encode(x_cat, x_num)
        assert c_cred.shape == (8, 2 * ct.embed_dim)

    def test_multilayer_ct(self):
        ct = make_small_ct(n_layers=3)
        ct.eval()
        x_cat, x_num, exposure = make_batch(16)
        mu, attn_list = ct(x_cat, x_num, exposure, return_attn=True)
        assert mu.shape == (16,)
        assert len(attn_list) == 3

    def test_multihead_ct(self):
        ct = make_small_ct(n_heads=2, embed_dim=6)  # token_dim=12, divisible by 2
        ct.eval()
        x_cat, x_num, exposure = make_batch(8)
        mu, _ = ct(x_cat, x_num, exposure)
        assert mu.shape == (8,)

    def test_deep_ct_with_ple_swiglu(self):
        ct = make_small_ct(use_ple=True, use_swiglu=True, n_heads=2, n_layers=3, embed_dim=6)
        ct.eval()
        x_cat, x_num, exposure = make_batch(8)
        mu, _ = ct(x_cat, x_num, exposure)
        assert mu.shape == (8,)

    def test_zero_exposure_predictions(self):
        """Zero exposure should give zero prediction (rate × 0)."""
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, _ = make_batch(4)
        exp_zero = torch.zeros(4)
        mu, _ = ct(x_cat, x_num, exp_zero)
        assert (mu == 0).all()

    def test_cat_only_no_num(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[5, 3],
            n_num_features=0,
            embed_dim=4,
        )
        ct.eval()
        x_cat = torch.randint(0, 3, (8, 2))
        x_cat[:, 0] = x_cat[:, 0] % 5
        mu, _ = ct(x_cat, None, torch.ones(8))
        assert mu.shape == (8,)

    def test_num_only_no_cat(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[],
            n_num_features=4,
            embed_dim=4,
        )
        ct.eval()
        x_num = torch.randn(8, 4)
        mu, _ = ct(None, x_num, torch.ones(8))
        assert mu.shape == (8,)

    def test_credibility_diverges_prior_from_trans(self):
        """Verify c_prior and c_trans differ after any initialisation."""
        ct = make_small_ct()
        ct.eval()
        x_cat, x_num, _ = make_batch(16)

        # Extract c_trans (eval mode: always c_trans)
        c_cred_eval, _ = ct.encode(x_cat, x_num)

        # Force training mode + alpha=0 to get c_prior
        ct.train()
        ct.credibility.alpha = 0.0
        c_cred_prior, _ = ct.encode(x_cat, x_num)

        # They should differ (c_prior vs c_trans)
        assert not torch.allclose(c_cred_eval, c_cred_prior, atol=1e-6)

    def test_predict_no_grad(self):
        ct = make_small_ct()
        x_cat, x_num, exposure = make_batch(8)
        # predict() should work without needing requires_grad
        pred = ct.predict(x_cat, x_num, exposure)
        assert pred.shape == (8,)
