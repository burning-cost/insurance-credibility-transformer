"""pytest configuration for insurance-credibility-transformer tests."""

import numpy as np
import torch
import pytest


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
