
from __future__ import annotations
import numpy as np
from numpy import linalg as la

_EIGH_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

def eigh_cached(key: tuple, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if key in _EIGH_CACHE:
        return _EIGH_CACHE[key]

    # if not np.allclose(H, H.T.conj(), atol=1e-12):

    evals, evecs = la.eigh(H)       # evecs: columns = eigenvectors
    res = (evals, evecs.T)          # store as ROWS = eigenvectors
    _EIGH_CACHE[key] = res
    return res

def clear_eigh_cache() -> None:
    """Clear the internal eigendecomposition cache."""
    _EIGH_CACHE.clear()
