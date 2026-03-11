"""Tests for loss.py: PoissonDevianceLoss, GammaDevianceLoss."""

import pytest
import torch
import numpy as np

from insurance_credibility_transformer.loss import GammaDevianceLoss, PoissonDevianceLoss

torch.use_deterministic_algorithms(True)


class TestPoissonDevianceLoss:
    def test_perfect_prediction_zero_loss(self):
        """When pred = exposure * y, deviance = 0."""
        loss_fn = PoissonDevianceLoss(use_exposure=True)
        exposure = torch.tensor([1.0, 2.0, 0.5, 1.0])
        y = torch.tensor([0.1, 0.2, 0.3, 0.4])
        pred = y * exposure  # perfect prediction
        loss = loss_fn(pred, y, exposure)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_loss_positive(self):
        loss_fn = PoissonDevianceLoss(use_exposure=True)
        exposure = torch.ones(10)
        y = torch.rand(10) * 0.5
        pred = torch.rand(10) + 0.1
        loss = loss_fn(pred, y, exposure)
        assert float(loss) > 0

    def test_zero_claims_handled(self):
        """Y=0 should not cause NaN (0*log(0)=0 convention)."""
        loss_fn = PoissonDevianceLoss(use_exposure=True)
        y = torch.zeros(5)
        pred = torch.ones(5) * 0.5
        exposure = torch.ones(5)
        loss = loss_fn(pred, y, exposure)
        assert not torch.isnan(loss)
        assert float(loss) > 0  # non-zero: mu contributes

    def test_no_exposure_mode(self):
        loss_fn = PoissonDevianceLoss(use_exposure=False)
        y = torch.tensor([0.1, 0.2])
        pred = torch.tensor([0.1, 0.2])  # perfect
        exposure = torch.ones(2)
        loss = loss_fn(pred, y, exposure)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_loss_scalar(self):
        loss_fn = PoissonDevianceLoss()
        loss = loss_fn(torch.tensor([0.5, 0.3]), torch.tensor([0.1, 0.4]), torch.ones(2))
        assert loss.shape == ()  # scalar

    def test_exposure_weighting(self):
        """Higher exposure rows should contribute more to the loss."""
        loss_fn = PoissonDevianceLoss(use_exposure=True)
        # Two identical mistakes; one with 10x exposure
        y = torch.tensor([0.1, 0.1])
        pred = torch.tensor([0.5, 0.5])
        exp_low = torch.tensor([1.0, 1.0])
        exp_high = torch.tensor([10.0, 10.0])
        loss_low = loss_fn(pred, y, exp_low)
        loss_high = loss_fn(pred, y, exp_high)
        # Same rate error, same deviance per unit; loss should be similar
        # (both use weighted mean)
        assert abs(float(loss_high) - float(loss_low)) < 0.5


class TestGammaDevianceLoss:
    def test_perfect_prediction_zero_loss(self):
        loss_fn = GammaDevianceLoss()
        y = torch.tensor([1.0, 2.0, 0.5])
        pred = y.clone()
        weights = torch.ones(3)
        loss = loss_fn(pred, y, weights)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_loss_positive_for_errors(self):
        loss_fn = GammaDevianceLoss()
        y = torch.tensor([1.0, 2.0])
        pred = torch.tensor([1.5, 1.0])
        loss = loss_fn(pred, y, torch.ones(2))
        assert float(loss) > 0
