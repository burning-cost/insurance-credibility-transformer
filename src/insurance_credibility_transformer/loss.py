"""Loss functions for the Credibility Transformer.

Implements the exposure-weighted Poisson deviance loss from Section 2.4
of Richman, Scognamiglio & Wüthrich (arXiv:2409.16653):

    L = (2/n) * sum_i v_i * [mu_i - Y_i - Y_i * log(mu_i / Y_i)]

where:
    v_i  = exposure (policy years)
    Y_i  = observed claim count
    mu_i = predicted mean (before exposure multiplication)

Note: the model outputs mu_i * v_i (exposure-adjusted prediction), so
the loss receives the raw rate mu_i = pred_i / v_i, then:
    deviance = 2 * (mu_i - Y_i + Y_i * log(Y_i / mu_i))

Poisson deviance is a strictly consistent scoring rule for the mean.
Zero claims (Y_i = 0) are handled via the convention 0*log(0) = 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoissonDevianceLoss(nn.Module):
    """Exposure-weighted Poisson deviance loss.

    The model should be called with exposure, returning mu * exposure.
    This loss un-applies exposure to get the rate, then computes deviance
    against the raw claim count Y.

    Alternatively, pass the raw rate (mu without exposure) and set
    use_exposure=False to compute deviance directly.

    Args:
        use_exposure: If True (default), divides predictions by exposure
            to recover the rate before computing deviance.
            Set False if the model already outputs the raw rate.
        eps: Small constant to prevent log(0) for zero predictions.
    """

    def __init__(self, use_exposure: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.use_exposure = use_exposure
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        exposure: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean exposure-weighted Poisson deviance.

        Args:
            pred: (batch,) model output. If use_exposure=True, this is
                mu * exposure; otherwise it is the raw rate mu.
            target: (batch,) observed claim counts Y_i (can be float).
            exposure: (batch,) exposure weights v_i.

        Returns:
            Scalar mean Poisson deviance.
        """
        if self.use_exposure:
            mu = pred / (exposure + self.eps)  # raw rate
        else:
            mu = pred

        mu = mu.clamp(min=self.eps)  # prevent log(0)

        # Deviance contribution: 2 * [mu - Y - Y * log(mu/Y)]
        # For Y=0: contribution is 2 * mu (since 0*log(0)=0)
        log_ratio = torch.where(
            target > 0,
            target * torch.log(target / mu),
            torch.zeros_like(target),
        )
        deviance = 2.0 * (mu - target + log_ratio)  # (batch,)

        # Exposure-weighted mean
        return (exposure * deviance).sum() / (exposure.sum() + self.eps)


class GammaDevianceLoss(nn.Module):
    """Exposure-weighted Gamma deviance loss for severity modelling.

    Gamma deviance: 2 * sum_i w_i * [Y_i/mu_i - 1 - log(Y_i/mu_i)]

    Suitable for modelling average loss amounts (severity), where
    the Gamma distribution assumption is standard.

    Args:
        eps: Small constant to prevent division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean exposure-weighted Gamma deviance.

        Args:
            pred: (batch,) predicted severity (positive).
            target: (batch,) observed severity Y_i (positive).
            weights: (batch,) observation weights (e.g., claim counts).

        Returns:
            Scalar mean Gamma deviance.
        """
        mu = pred.clamp(min=self.eps)
        y = target.clamp(min=self.eps)

        deviance = 2.0 * (y / mu - 1.0 - torch.log(y / mu))  # (batch,)
        return (weights * deviance).sum() / (weights.sum() + self.eps)
