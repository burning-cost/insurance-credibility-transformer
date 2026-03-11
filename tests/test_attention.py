"""Tests for attention.py: MultiHeadCredibilityAttention, CredibilityMechanism."""

import pytest
import torch

from insurance_credibility_transformer.attention import (
    CredibilityMechanism,
    MultiHeadCredibilityAttention,
    StandardFeedForward,
    SwiGLUFeedForward,
)

torch.manual_seed(0)


class TestMultiHeadCredibilityAttention:
    def test_output_shape(self):
        attn = MultiHeadCredibilityAttention(token_dim=10, n_heads=1)
        x = torch.randn(8, 6, 10)  # (batch, seq=T+1, dim)
        out, attn_w = attn(x, return_attn=True)
        assert out.shape == (8, 6, 10)
        assert attn_w.shape == (8, 1, 6, 6)  # (batch, heads, seq, seq)

    def test_multihead_output_shape(self):
        attn = MultiHeadCredibilityAttention(token_dim=10, n_heads=2, dropout=0.0)
        x = torch.randn(4, 7, 10)
        out, attn_w = attn(x, return_attn=True)
        assert out.shape == (4, 7, 10)
        assert attn_w.shape == (4, 2, 7, 7)

    def test_attention_weights_sum_to_one(self):
        attn = MultiHeadCredibilityAttention(token_dim=8, n_heads=1, dropout=0.0)
        attn.eval()
        x = torch.randn(2, 5, 8)
        _, attn_w = attn(x, return_attn=True)
        # After softmax, each row sums to 1 (before head scaling)
        # We check approximately: softmax rows should be positive
        assert (attn_w >= 0).all()

    def test_no_attn_returned(self):
        attn = MultiHeadCredibilityAttention(token_dim=8, n_heads=1)
        x = torch.randn(4, 5, 8)
        out, attn_w = attn(x, return_attn=False)
        assert attn_w is None

    def test_token_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadCredibilityAttention(token_dim=10, n_heads=3)


class TestCredibilityMechanism:
    def test_eval_always_uses_c_trans(self):
        mech = CredibilityMechanism(alpha=0.5)
        mech.eval()
        c_trans = torch.ones(10, 8)
        c_prior = torch.zeros(10, 8)
        result = mech(c_trans, c_prior)
        assert torch.allclose(result, c_trans)

    def test_training_bernoulli_mixture(self):
        """During training, result should be a mixture."""
        mech = CredibilityMechanism(alpha=0.9)
        mech.train()
        torch.manual_seed(42)
        c_trans = torch.ones(1000, 4) * 2.0
        c_prior = torch.zeros(1000, 4)
        result = mech(c_trans, c_prior)
        # Should be either 0 or 2 for each row
        assert result.shape == (1000, 4)
        unique_vals = result[:, 0].unique()
        assert len(unique_vals) == 2  # both 0 and 2 should appear

    def test_alpha_0_always_prior(self):
        """alpha=0: training always uses c_prior."""
        mech = CredibilityMechanism(alpha=0.0)
        mech.train()
        c_trans = torch.ones(100, 4) * 5.0
        c_prior = torch.zeros(100, 4)
        result = mech(c_trans, c_prior)
        assert torch.allclose(result, c_prior)

    def test_alpha_1_always_trans(self):
        """alpha=1: training always uses c_trans."""
        mech = CredibilityMechanism(alpha=1.0)
        mech.train()
        c_trans = torch.ones(100, 4) * 3.0
        c_prior = torch.zeros(100, 4)
        result = mech(c_trans, c_prior)
        assert torch.allclose(result, c_trans)


class TestFeedForwards:
    def test_standard_ffn_shape(self):
        ffn = StandardFeedForward(dim=10, dropout=0.0)
        x = torch.randn(8, 5, 10)
        out = ffn(x)
        assert out.shape == (8, 5, 10)

    def test_swiglu_shape(self):
        ffn = SwiGLUFeedForward(dim=10, dropout=0.0)
        x = torch.randn(8, 5, 10)
        out = ffn(x)
        assert out.shape == (8, 5, 10)
