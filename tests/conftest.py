"""pytest configuration for insurance-credibility-transformer tests."""

import torch
import pytest


def pytest_configure(config):
    """Set deterministic algorithms for all tests."""
    torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    yield
