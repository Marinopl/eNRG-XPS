# core/hamiltonian.py

"""
Hamiltonian construction and diagonalization for the XPS-eNRG model.

Notes
-----
- 'n' must be odd; we enforce this via ensure_odd (raises ValueError if violated).
- eval_ are eigenvalues in ascending order (NumPy eigh convention).
- evec_ is the eigenvector matrix (each column is an eigenvector).
- eval_norm are eigenvalues normalized by normaliz(n, w).
"""

from __future__ import annotations
import numpy as np

from .params import T, lam, zeta
from .utils import ensure_odd
from .discretization import t_n, normaliz
from .linalg import eigh_cached  # <- use the central cached eigensolver


def dig_ham(n: int, k: float, w: float):
    """
    Builds and diagonalizes the (n+1)x(n+1) tridiagonal Hamiltonian.

    Parameters
    ----------
    n : int
        Must be odd. Hamiltonian dimension is (n+1) x (n+1).
    k : float
        Local scattering potential at site 0 (or see offset branch below).
    w : float
        Discretization scale factor (w = Λ^θ).

    Returns
    -------
    If zeta == 0:
        (eval_, evec_, eval_norm)
    Else:
        (eval_, evec_, eval_norm, ham__)
    """
    ensure_odd(n)
    Z = zeta
    dim = n + 1

    if Z == 0:
        # Standard Wilson chain:
        # - on-site potential k at site 0
        # - first link (0-1): T / sqrt(w)
        # - subsequent links: t_n(i-1, w), i = 1..dim-2
        ham__ = np.zeros((dim, dim), dtype=float)
        ham__[0, 0] = k
        ham__[0, 1] = ham__[1, 0] = T / np.sqrt(w)
        for i in range(1, dim - 1):
            ham__[i, i + 1] = ham__[i + 1, i] = t_n(i - 1, w)

        key = ("dig_ham", n, float(k), float(w), Z, T, float(lam))
        eval_, evec_T = eigh_cached(key, ham__)   # evec_T: rows = eigenvectors
        evec_ = evec_T.T                          # convert back to columns (as documented)
        eval_norm = eval_ / normaliz(n, w)
        return eval_, evec_, eval_norm

    else:
        # Offset chain:
        # - on-site potential k at site 0
        # - first Z links have uniform hopping T
        # - from site Z onward, hoppings follow t_n with index shift (1+Z)
        ham__ = np.zeros((dim, dim), dtype=float)
        ham__[0, 0] = k
        for i in range(1, Z + 1):
            ham__[i - 1, i] = ham__[i, i - 1] = T
        for i in range(Z, dim - 1):
            ham__[i, i + 1] = ham__[i + 1, i] = t_n(i - (1 + Z), w)

        key = ("dig_ham_offset", n, float(k), float(w), Z, T, float(lam))
        eval_, evec_T = eigh_cached(key, ham__)
        evec_ = evec_T.T
        eval_norm = eval_ / normaliz(n, w)
        return eval_, evec_, eval_norm, ham__


def delta(n: int, k: float, w: float) -> float:
    """
    Phase shift (delta / pi) associated with scattering potential k.

    Uses a level a few states ABOVE the Fermi level:
        nfermi3 = int((n + 1) / 2 + 3)
        delta/pi = ln(E_norm[nfermi3] / E0_norm[nfermi3]) / ln(lam^2)
    """
    ensure_odd(n)

    ene0_norm = dig_ham(n, 0.0, w)[2]
    ene_norm  = dig_ham(n, k,   w)[2]

    nfermi3: int = int((n + 1) / 2 + 3)
    num = np.log(ene_norm[nfermi3] / ene0_norm[nfermi3])
    den = 2.0 * np.log(lam)
    return float(num / den)
