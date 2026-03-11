"""Tests for datasets.py: InsuranceDataset, collate_insurance."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from insurance_credibility_transformer.datasets import InsuranceDataset, collate_insurance


class TestInsuranceDataset:
    def test_basic_creation(self):
        x_cat = np.random.randint(0, 5, (100, 3))
        x_num = np.random.randn(100, 2).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        exposure = np.ones(100, dtype=np.float32)
        ds = InsuranceDataset(x_cat, x_num, y, exposure)
        assert len(ds) == 100

    def test_getitem_returns_correct_types(self):
        x_cat = np.array([[0, 1], [2, 0]])
        x_num = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([0.1, 0.2])
        ds = InsuranceDataset(x_cat, x_num, y)
        xc, xn, yi, exp = ds[0]
        assert xc.dtype == torch.int64
        assert xn.dtype == torch.float32
        assert yi.dtype == torch.float32
        assert exp.dtype == torch.float32

    def test_none_exposure_defaults_to_ones(self):
        x_num = np.random.randn(10, 2).astype(np.float32)
        y = np.zeros(10)
        ds = InsuranceDataset(None, x_num, y, exposure=None)
        _, _, _, exp = ds[0]
        assert float(exp) == 1.0

    def test_no_cat_features(self):
        x_num = np.random.randn(20, 3).astype(np.float32)
        y = np.random.rand(20)
        ds = InsuranceDataset(None, x_num, y)
        xc, xn, yi, exp = ds[0]
        assert xc is None
        assert xn.shape == (3,)

    def test_dataloader_collates(self):
        x_cat = np.random.randint(0, 4, (50, 2))
        x_num = np.random.randn(50, 3).astype(np.float32)
        y = np.random.rand(50)
        ds = InsuranceDataset(x_cat, x_num, y)
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_insurance)
        batch = next(iter(loader))
        xc, xn, yi, exp = batch
        assert xc.shape == (16, 2)
        assert xn.shape == (16, 3)
        assert yi.shape == (16,)
