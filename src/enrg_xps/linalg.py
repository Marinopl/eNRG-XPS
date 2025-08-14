# core/linalg.py
from __future__ import annotations
import numpy as np
from numpy import linalg as la

# Simple explicit cache keyed ONLY by `key`
_EIGH_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

def eigh_cached(key: tuple, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if key in _EIGH_CACHE:
        return _EIGH_CACHE[key]

    # Optional safety (uncomment if you want):
    # if not np.allclose(H, H.T.conj(), atol=1e-12):
    #     raise ValueError("Matrix is not Hermitian/symmetric within tolerance.")

    evals, evecs = la.eigh(H)       # evecs: columns = eigenvectors
    res = (evals, evecs.T)          # store as ROWS = eigenvectors (convenient downstream)
    _EIGH_CACHE[key] = res
    return res

def clear_eigh_cache() -> None:
    """Clear the internal eigendecomposition cache."""
    _EIGH_CACHE.clear()
