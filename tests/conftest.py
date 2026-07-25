"""pytest fixtures for splatreg's test suite.

Deterministic seeding before every test (the gsplat/Theseus autouse-seed discipline)
so failures reproduce, plus a CUDA-first device fixture with an explicit
``SPLATREG_TEST_DEVICE`` override.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed torch (CPU + CUDA) before every test."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


@pytest.fixture
def device():
    """Test device — CUDA when available, otherwise CPU, unless explicitly overridden."""
    default = "cuda" if torch.cuda.is_available() else "cpu"
    want = os.environ.get("SPLATREG_TEST_DEVICE", default)
    if want.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return want
