"""Integration tests: small synthetic data, end-to-end training.

Uses 50 policies, 3 categorical + 2 numerical features.
Verifies training reduces loss and model trains without errors.
"""

import numpy as np
import pytest
import torch

from insurance_credibility_transformer import (
    AttentionExplainer,
    CredibilityTransformer,
    CredibilityTransformerTrainer,
)

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


def make_synthetic_data(n=50, seed=42):
    rng = np.random.default_rng(seed)
    x_cat = np.column_stack([
        rng.integers(0, 6, n),
        rng.integers(0, 2, n),
        rng.integers(0, 4, n),
    ])
    x_num = rng.standard_normal((n, 2)).astype(np.float32)
    exposure = rng.uniform(0.5, 2.0, n).astype(np.float32)
    # Simple GLM-like DGP: lambda ~ exp(0.1*cat0 - 0.05*num0) * 0.1
    log_lambda = 0.1 * x_cat[:, 0] - 0.05 * x_num[:, 0] - 2.0
    rate = np.exp(log_lambda)
    y = rng.poisson(rate * exposure).astype(np.float32)
    return x_cat, x_num, y, exposure


class TestEndToEndTraining:
    def test_training_reduces_loss(self):
        """Training for a few epochs should reduce validation loss."""
        x_cat, x_num, y, exposure = make_synthetic_data(n=100)

        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=5,
            n_heads=1,
            n_layers=1,
            alpha=0.9,
            dropout=0.0,
        )

        trainer = CredibilityTransformerTrainer(
            model=ct,
            loss="poisson",
            lr=1e-2,
            batch_size=32,
            val_split=0.2,
            early_stopping_patience=100,  # don't stop early
            max_epochs=20,
            n_ensemble=1,
            random_seed=42,
            verbose=0,
        )
        trainer.fit(x_cat, x_num, y, exposure)

        history = trainer.train_history[0]
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        assert final_loss < initial_loss, (
            f"Training did not reduce loss: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_predict_returns_correct_shape(self):
        x_cat, x_num, y, exposure = make_synthetic_data(n=60)
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=4,
        )
        trainer = CredibilityTransformerTrainer(
            model=ct, max_epochs=5, n_ensemble=1, verbose=0, batch_size=30
        )
        trainer.fit(x_cat[:50], x_num[:50], y[:50], exposure[:50])
        preds = trainer.predict(x_cat[50:], x_num[50:], exposure[50:])
        assert preds.shape == (10,)
        assert (preds > 0).all()

    def test_ensemble_averages_predictions(self):
        x_cat, x_num, y, exposure = make_synthetic_data(n=60)
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=4,
        )
        trainer = CredibilityTransformerTrainer(
            model=ct, max_epochs=3, n_ensemble=3, verbose=0, batch_size=30
        )
        trainer.fit(x_cat[:50], x_num[:50], y[:50], exposure[:50])
        assert len(trainer._ensemble_states) == 3
        preds = trainer.predict(x_cat[50:], x_num[50:], exposure[50:])
        assert preds.shape == (10,)

    def test_explain_cls_attention(self):
        x_cat, x_num, y, exposure = make_synthetic_data(n=60)
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=5,
        )
        trainer = CredibilityTransformerTrainer(
            model=ct, max_epochs=3, n_ensemble=1, verbose=0
        )
        trainer.fit(x_cat[:50], x_num[:50], y[:50], exposure[:50])

        explainer = AttentionExplainer(ct)
        x_cat_t = torch.tensor(x_cat[:20], dtype=torch.long)
        x_num_t = torch.tensor(x_num[:20], dtype=torch.float32)
        P = explainer.cls_attention(x_cat_t, x_num_t)
        assert P.shape == (20,)
        # P is an attention weight, should be in [0, 1]
        assert (P >= 0).all() and (P <= 1).all()

    def test_explain_individual_credibility(self):
        x_cat, x_num, y, exposure = make_synthetic_data(n=60)
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=5,
        )
        trainer = CredibilityTransformerTrainer(
            model=ct, max_epochs=3, n_ensemble=1, verbose=0
        )
        trainer.fit(x_cat[:50], x_num[:50], y[:50], exposure[:50])

        explainer = AttentionExplainer(ct)
        x_cat_t = torch.tensor(x_cat[:20], dtype=torch.long)
        x_num_t = torch.tensor(x_num[:20], dtype=torch.float32)
        z = explainer.individual_credibility(x_cat_t, x_num_t)
        assert z.shape == (20,)
        assert (z >= 0).all() and (z <= 1).all()

    def test_explain_feature_attention(self):
        x_cat, x_num, y, exposure = make_synthetic_data(n=60)
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=2,
            embed_dim=5,
        )
        trainer = CredibilityTransformerTrainer(
            model=ct, max_epochs=3, n_ensemble=1, verbose=0
        )
        trainer.fit(x_cat[:50], x_num[:50], y[:50], exposure[:50])

        explainer = AttentionExplainer(ct)
        x_cat_t = torch.tensor(x_cat[:20], dtype=torch.long)
        x_num_t = torch.tensor(x_num[:20], dtype=torch.float32)
        feat_attn = explainer.feature_attention(x_cat_t, x_num_t)
        assert len(feat_attn) == 5  # 3 cat + 2 num
        for name, weights in feat_attn.items():
            assert weights.shape == (20,)


class TestEdgeCases:
    def test_single_category_level(self):
        """Feature with cardinality=1 (all policies same level)."""
        ct = CredibilityTransformer(
            cat_cardinalities=[1, 3],  # first feature has single level
            n_num_features=1,
            embed_dim=4,
        )
        ct.eval()
        x_cat = torch.zeros(10, 2, dtype=torch.long)
        x_cat[:, 1] = torch.randint(0, 3, (10,))
        x_num = torch.randn(10, 1)
        mu, _ = ct(x_cat, x_num, torch.ones(10))
        assert mu.shape == (10,)
        assert not torch.isnan(mu).any()

    def test_no_num_features(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[6, 2, 4],
            n_num_features=0,
            embed_dim=5,
        )
        ct.eval()
        x_cat = torch.stack([
            torch.randint(0, 6, (8,)),
            torch.randint(0, 2, (8,)),
            torch.randint(0, 4, (8,)),
        ], dim=1)
        mu, _ = ct(x_cat, None, torch.ones(8))
        assert mu.shape == (8,)

    def test_batch_size_one(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[3, 2],
            n_num_features=2,
            embed_dim=4,
        )
        ct.eval()
        x_cat = torch.tensor([[1, 0]])
        x_num = torch.randn(1, 2)
        mu, _ = ct(x_cat, x_num, torch.ones(1))
        assert mu.shape == (1,)
        assert not torch.isnan(mu).any()

    def test_severity_decoder(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[3],
            n_num_features=2,
            embed_dim=4,
            link="softplus",
        )
        ct.eval()
        x_cat = torch.randint(0, 3, (10, 1))
        x_num = torch.randn(10, 2)
        mu, _ = ct(x_cat, x_num, torch.ones(10))
        assert (mu > 0).all()

    def test_identity_link(self):
        ct = CredibilityTransformer(
            cat_cardinalities=[3],
            n_num_features=2,
            embed_dim=4,
            link="identity",
        )
        ct.eval()
        x_cat = torch.randint(0, 3, (8, 1))
        x_num = torch.randn(8, 2)
        mu, _ = ct(x_cat, x_num)
        # Identity link: output can be any value, just check shape/not nan
        assert mu.shape == (8,)
        assert not torch.isnan(mu).any()
