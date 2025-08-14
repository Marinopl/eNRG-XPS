"""
Impurity Hamiltonian construction and diagonalization for the XPS-eNRG model.

Original structure (no zeta offset here):
- site 0: impurity level with on-site energy d
- site 1: conduction site with local scattering k
- hopping (0,1) = v: hybridization between impurity and conduction site
- hopping (1,2) = T / sqrt(w)
- hopping (i, i+1) = t_n(i-2, w) for i >= 2
"""

from __future__ import annotations
import numpy as np

from .params import T, lam
from .utils import ensure_odd
from .discretization import t_n, normaliz
from .linalg import eigh_cached


def dig_ham_imp(n: int, k: float, d: float, v: float, w: float):
    """
    Builds and diagonalizes the (n+1) x (n+1) impurity Hamiltonian.

    Parameters
    ----------
    n : int
        Must be odd. Matrix dimension is (n+1).
    k : float
        Scattering potential on conduction site (index 1).
    d : float
        Impurity on-site energy at site 0.
    v : float
        Hybridization between impurity (0) and conduction site (1).
    w : float
        Discretization scale factor (w = Λ^θ).
    """
    ensure_odd(n)
    dim = n + 1

    ham__ = np.zeros((dim, dim), dtype=float)
    # on-site terms
    ham__[0, 0] = d
    ham__[1, 1] = k
    # impurity–conduction coupling
    ham__[0, 1] = ham__[1, 0] = v
    # first conduction link
    ham__[1, 2] = ham__[2, 1] = T / np.sqrt(w)
    # Wilson chain tail
    for i in range(2, dim - 1):
        ham__[i, i + 1] = ham__[i + 1, i] = t_n(i - 2, w)

    key = ("dig_ham_imp", n, float(k), float(d), float(v), float(w), T, float(lam))
    eval_, evec_T = eigh_cached(key, ham__)
    evec_ = evec_T.T  # columns = eigenvectors (to match spectra/projection)
    eval_norm = eval_ / normaliz(n, w)
    return eval_, evec_, eval_norm
