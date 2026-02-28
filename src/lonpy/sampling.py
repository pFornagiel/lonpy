"""
Basin-Hopping sampling for Local Optima Network construction.

This module provides a Rust-accelerated implementation of Basin-Hopping
sampling. The core loop (perturbation, acceptance, hashing, trace
construction) runs in Rust via PyO3, while scipy.optimize.minimize and
user-supplied objective functions are called back into Python.

The public API is identical to the original pure-Python implementation.
"""

from lonpy._lonpy_rust import (  # noqa: F401
    BasinHoppingSampler,
    BasinHoppingSamplerConfig,
    compute_lon,
)

__all__ = [
    "BasinHoppingSampler",
    "BasinHoppingSamplerConfig",
    "compute_lon",
]
