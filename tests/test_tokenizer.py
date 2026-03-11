"""Tests for tokenizer.py: EntityEmbedding, NumericalEmbedding, PLE, FeatureTokenizer."""

import pytest
import torch

from insurance_credibility_transformer.tokenizer import (
    EntityEmbedding,
    FeatureTokenizer,
    NumericalEmbedding,
    PiecewiseLinearEncoding,
)

# Deterministic tests
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


class TestEntityEmbedding:
    def test_output_shape(self):
        emb = EntityEmbedding(cardinality=6, embed_dim=5)
        x = torch.randint(0, 6, (32,))
        out = emb(x)
        assert out.shape == (32, 5)

    def test_single_level(self):
        """Edge case: cardinality=1 (all policies same level)."""
        emb = EntityEmbedding(cardinality=1, embed_dim=8)
        x = torch.zeros(10, dtype=torch.long)
        out = emb(x)
        assert out.shape == (10, 8)

    def test_output_not_all_zero(self):
        emb = EntityEmbedding(cardinality=4, embed_dim=5)
        x = torch.randint(0, 4, (20,))
        out = emb(x)
        assert out.abs().sum() > 0


class TestNumericalEmbedding:
    def test_output_shape(self):
        emb = NumericalEmbedding(embed_dim=5)
        x = torch.randn(32)
        out = emb(x)
        assert out.shape == (32, 5)

    def test_bounded_by_tanh(self):
        emb = NumericalEmbedding(embed_dim=8)
        x = torch.randn(100)
        out = emb(x)
        # tanh output bounded in (-1, 1) at last layer
        assert out.abs().max() < 1.0 + 1e-6


class TestPiecewiseLinearEncoding:
    def test_output_shape(self):
        ple = PiecewiseLinearEncoding(embed_dim=5, n_bins=8)
        x = torch.randn(32)
        out = ple(x)
        assert out.shape == (32, 5)

    def test_bin_widths_positive(self):
        ple = PiecewiseLinearEncoding(embed_dim=5, n_bins=8)
        widths = ple.bin_widths
        assert (widths > 0).all()

    def test_boundaries_monotone(self):
        ple = PiecewiseLinearEncoding(embed_dim=5, n_bins=8)
        bounds = ple.bin_boundaries
        diffs = bounds[1:] - bounds[:-1]
        assert (diffs > 0).all()

    def test_gradients_flow(self):
        ple = PiecewiseLinearEncoding(embed_dim=5, n_bins=8)
        x = torch.randn(10, requires_grad=False)
        out = ple(x)
        loss = out.sum()
        loss.backward()
        assert ple.log_delta.grad is not None


class TestFeatureTokenizer:
    def test_mixed_features_shape(self):
        tok = FeatureTokenizer(cat_cardinalities=[6, 2, 11], n_num_features=2, embed_dim=5)
        x_cat = torch.randint(0, 6, (32, 3))
        x_cat[:, 1] = x_cat[:, 1] % 2
        x_cat[:, 2] = x_cat[:, 2] % 11
        x_num = torch.randn(32, 2)
        out = tok(x_cat, x_num)
        # T = 3 cat + 2 num = 5, +1 CLS = 6, token_dim = 2*5 = 10
        assert out.shape == (32, 6, 10)

    def test_cat_only(self):
        tok = FeatureTokenizer(cat_cardinalities=[4, 3], n_num_features=0, embed_dim=4)
        x_cat = torch.randint(0, 3, (10, 2))
        x_cat[:, 0] = x_cat[:, 0] % 4
        out = tok(x_cat, None)
        assert out.shape == (10, 3, 8)  # T=2, +CLS=3, dim=2*4=8

    def test_num_only(self):
        tok = FeatureTokenizer(cat_cardinalities=[], n_num_features=3, embed_dim=4)
        x_num = torch.randn(10, 3)
        out = tok(None, x_num)
        assert out.shape == (10, 4, 8)  # T=3, +CLS=4, dim=8

    def test_cls_appended_last(self):
        """CLS token is the last in sequence."""
        tok = FeatureTokenizer(cat_cardinalities=[5], n_num_features=1, embed_dim=4)
        x_cat = torch.randint(0, 5, (8, 1))
        x_num = torch.randn(8, 1)
        out1 = tok(x_cat, x_num)
        out2 = tok(x_cat, x_num)
        # CLS token is deterministic (learned param, no randomness in eval)
        assert torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_token_dim_property(self):
        tok = FeatureTokenizer(cat_cardinalities=[3], n_num_features=2, embed_dim=7)
        assert tok.token_dim == 14

    def test_ple_mode(self):
        tok = FeatureTokenizer(cat_cardinalities=[3], n_num_features=2, embed_dim=5, use_ple=True, n_ple_bins=8)
        x_cat = torch.randint(0, 3, (16, 1))
        x_num = torch.randn(16, 2)
        out = tok(x_cat, x_num)
        assert out.shape == (16, 4, 10)  # T=3, +CLS=4, dim=10
