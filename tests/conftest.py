"""Shared pytest setup: put the project root on sys.path."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control import build_ur3e  # noqa: E402


@pytest.fixture
def robot():
    return build_ur3e()


@pytest.fixture
def rng():
    return np.random.default_rng(1234)