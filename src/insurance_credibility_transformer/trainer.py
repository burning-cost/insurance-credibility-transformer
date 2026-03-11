"""Training infrastructure for the Credibility Transformer.

Implements the training protocol from Section 2.4 of arXiv:2409.16653
and the 3-phase ICL training from Section 2.4 of arXiv:2509.08122.

Phase 1 (base CT):
    AdamW, lr=1e-3, wd=1e-2, beta2=0.95, batch=1024
    Early stopping patience=20, val_split=0.15
    Multiple independent runs averaged (n_ensemble)

Phase 2 (ICL, frozen decoder):
    AdamW, lr=3e-4, wd=1e-2, beta2=0.95
    50 epochs, early stopping patience 10

Phase 3 (full fine-tune):
    AdamW, lr=3e-5
    20 epochs, early stopping patience 10
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .datasets import InsuranceDataset, collate_insurance
from .loss import GammaDevianceLoss, PoissonDevianceLoss


class EarlyStopping:
    """Monitor validation loss and signal when to stop.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
        mode: 'min' (lower is better) or 'max'.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.best_state: Optional[dict] = None

    def step(self, value: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            value: Current validation metric.
            model: Model to snapshot if this is the best epoch.

        Returns:
            True if training should stop.
        """
        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best checkpoint into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class CredibilityTransformerTrainer:
    """Train a CredibilityTransformer (phase 1 of the 3-phase protocol).

    Supports ensemble averaging over n_ensemble independent runs.

    Args:
        model: A CredibilityTransformer instance to train.
        loss: Loss function name: 'poisson' or 'gamma'. Default 'poisson'.
        lr: Learning rate. Paper default 1e-3.
        weight_decay: AdamW weight decay. Paper default 1e-2.
        beta2: Adam beta2. Paper default 0.95.
        batch_size: Mini-batch size. Paper default 1024.
        val_split: Fraction of data to use for validation (early stopping).
        early_stopping_patience: Epochs without improvement before stopping.
        max_epochs: Maximum training epochs. Paper uses 100.
        n_ensemble: Number of independent training runs to average.
            Set 1 for single run. Paper uses 20 for final results.
        device: Torch device. Auto-detects cuda if available.
        random_seed: Base random seed. Each ensemble run adds its index.
        verbose: Print progress every N epochs. 0 for silent.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: str = "poisson",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        beta2: float = 0.95,
        batch_size: int = 1024,
        val_split: float = 0.15,
        early_stopping_patience: int = 20,
        max_epochs: int = 100,
        n_ensemble: int = 1,
        device: Optional[str] = None,
        random_seed: int = 42,
        verbose: int = 10,
    ) -> None:
        self.model = model
        self.loss_name = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta2 = beta2
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.n_ensemble = n_ensemble
        self.random_seed = random_seed
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Ensemble storage: list of model state dicts
        self._ensemble_states: List[dict] = []
        self.train_history: List[Dict[str, List[float]]] = []

    def _make_loss(self) -> nn.Module:
        if self.loss_name == "poisson":
            return PoissonDevianceLoss(use_exposure=True)
        elif self.loss_name == "gamma":
            return GammaDevianceLoss()
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}. Choose 'poisson' or 'gamma'.")

    def _train_single_run(
        self,
        dataset: InsuranceDataset,
        seed: int,
    ) -> Tuple[dict, Dict[str, List[float]]]:
        """Train one independent run. Returns best state dict and history."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Re-initialise model weights for each ensemble run
        # We do this by re-applying weight init to all parameters
        model = copy.deepcopy(self.model)
        model = model.to(self.device)
        self._reinit_weights(model)

        # Train/val split
        n_val = max(1, int(len(dataset) * self.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_insurance,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 4,
            shuffle=False,
            collate_fn=collate_insurance,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, self.beta2),
        )
        criterion = self._make_loss()
        stopper = EarlyStopping(patience=self.patience, mode="min")

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            # Training
            model.train()
            train_losses = []
            for x_cat, x_num, y, exposure in train_loader:
                x_cat = x_cat.to(self.device) if x_cat is not None else None
                x_num = x_num.to(self.device) if x_num is not None else None
                y = y.to(self.device)
                exposure = exposure.to(self.device)

                optimizer.zero_grad()
                pred, _ = model(x_cat, x_num, exposure)
                loss = criterion(pred, y, exposure)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_cat, x_num, y, exposure in val_loader:
                    x_cat = x_cat.to(self.device) if x_cat is not None else None
                    x_num = x_num.to(self.device) if x_num is not None else None
                    y = y.to(self.device)
                    exposure = exposure.to(self.device)
                    pred, _ = model(x_cat, x_num, exposure)
                    loss = criterion(pred, y, exposure)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"  Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")

            if stopper.step(val_loss, model):
                if self.verbose > 0:
                    print(f"  Early stopping at epoch {epoch}")
                break

        stopper.restore_best(model)
        return model.state_dict(), history

    @staticmethod
    def _reinit_weights(model: nn.Module) -> None:
        """Re-initialise all module weights in-place."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def fit(
        self,
        x_cat: Optional[np.ndarray],
        x_num: Optional[np.ndarray],
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> "CredibilityTransformerTrainer":
        """Train the model (n_ensemble independent runs).

        Args:
            x_cat: (n, n_cat) integer-encoded categorical features.
            x_num: (n, n_num) float32 continuous features.
            y: (n,) claim counts.
            exposure: (n,) exposure in calendar years.

        Returns:
            self (for chaining)
        """
        dataset = InsuranceDataset(x_cat, x_num, y, exposure)
        self._ensemble_states = []
        self.train_history = []

        for run in range(self.n_ensemble):
            seed = self.random_seed + run
            if self.verbose > 0 and self.n_ensemble > 1:
                print(f"Ensemble run {run + 1}/{self.n_ensemble} (seed={seed})")
            state, history = self._train_single_run(dataset, seed)
            self._ensemble_states.append(state)
            self.train_history.append(history)

        # Load the first run's weights into self.model for single-run usage
        self.model.load_state_dict(self._ensemble_states[0])
        self.model.eval()
        return self

    def predict(
        self,
        x_cat: Optional[np.ndarray],
        x_num: Optional[np.ndarray],
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict, averaging across ensemble runs.

        Args:
            x_cat: (n, n_cat) integer features.
            x_num: (n, n_num) float features.
            exposure: (n,) exposure.

        Returns:
            predictions: (n,) numpy array of averaged predictions.
        """
        dataset = InsuranceDataset(x_cat, x_num, np.zeros(len(x_cat if x_cat is not None else x_num)), exposure)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            collate_fn=collate_insurance,
        )

        all_preds = []

        for state in self._ensemble_states:
            self.model.load_state_dict(state)
            self.model.eval()
            self.model.to(self.device)

            run_preds = []
            with torch.no_grad():
                for x_cat_b, x_num_b, _, exp_b in loader:
                    x_cat_b = x_cat_b.to(self.device) if x_cat_b is not None else None
                    x_num_b = x_num_b.to(self.device) if x_num_b is not None else None
                    exp_b = exp_b.to(self.device)
                    pred, _ = self.model(x_cat_b, x_num_b, exp_b)
                    run_preds.append(pred.cpu().numpy())

            all_preds.append(np.concatenate(run_preds))

        # Average across ensemble runs
        return np.mean(all_preds, axis=0)
