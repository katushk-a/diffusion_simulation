"""Reproducibility helpers."""

import random


def set_global_seed(seed: int) -> None:
    """Set random seed for Python's random module."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
