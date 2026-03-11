"""PyTorch Dataset for insurance tabular data.

InsuranceDataset wraps the separate categorical/numerical/target/exposure
arrays into a PyTorch Dataset for use with DataLoader. Handles:
- None for x_cat or x_num (single modality)
- Float32 casting
- Exposure weighting

Data format expected:
    X_cat:    (n, n_cat)   integer-encoded categorical features
    X_num:    (n, n_num)   float32 continuous features
    y:        (n,)         claim counts (or severity amounts)
    exposure: (n,)         policy exposure (calendar years)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class InsuranceDataset(Dataset):
    """PyTorch Dataset for insurance frequency/severity modelling.

    Args:
        x_cat: (n, n_cat) integer array of categorical features.
            Pass None if there are no categorical features.
        x_num: (n, n_num) float array of continuous features.
            Pass None if there are no numerical features.
        y: (n,) array of claim counts or severity values.
        exposure: (n,) policy exposure in calendar years.
            Defaults to ones if None.
    """

    def __init__(
        self,
        x_cat: Optional[np.ndarray],
        x_num: Optional[np.ndarray],
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> None:
        n = len(y)
        self.x_cat: Optional[torch.Tensor] = None
        self.x_num: Optional[torch.Tensor] = None

        if x_cat is not None:
            self.x_cat = torch.tensor(np.asarray(x_cat), dtype=torch.long)
            assert self.x_cat.shape[0] == n

        if x_num is not None:
            self.x_num = torch.tensor(np.asarray(x_num, dtype=np.float32), dtype=torch.float32)
            assert self.x_num.shape[0] == n

        self.y = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)

        if exposure is None:
            self.exposure = torch.ones(n, dtype=torch.float32)
        else:
            self.exposure = torch.tensor(np.asarray(exposure, dtype=np.float32), dtype=torch.float32)
            assert len(self.exposure) == n

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        x_cat_i = self.x_cat[idx] if self.x_cat is not None else None
        x_num_i = self.x_num[idx] if self.x_num is not None else None
        return x_cat_i, x_num_i, self.y[idx], self.exposure[idx]


def collate_insurance(
    batch: list,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Custom collate function for InsuranceDataset.

    Handles None in x_cat or x_num (no features of that type).
    """
    x_cat_list, x_num_list, y_list, exp_list = zip(*batch)

    x_cat = torch.stack(x_cat_list) if x_cat_list[0] is not None else None
    x_num = torch.stack(x_num_list) if x_num_list[0] is not None else None
    y = torch.stack(y_list)
    exposure = torch.stack(exp_list)

    return x_cat, x_num, y, exposure
