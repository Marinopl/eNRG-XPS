"""
This module centralizes all discretization utilities
- int_w() -> np.ndarray
- int_ws() -> np.ndarray
- int_epsilon() -> (np.ndarray, np.ndarray)
- find_E() -> list[int]
"""

from __future__ import annotations
import numpy as np
from .params import lam, T, zeta


def int_w(num: int = 200) -> np.ndarray:
    """
    Returns a logarithmic mesh for w

    """
    theta = np.linspace(-1.0, 1.0, int(num))
    return np.power(lam, theta)

def int_epsilon():
    eps_long = 10.0 ** np.linspace(1.0, -5.0, 101)

    denom = 2.0 * np.log(lam) 

    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("LAMBDA must satisfy LAMBDA > 1 and yield finite log.")

    i = int(np.log(1e5 / denom))
    i = max(i, 3)

    eps_short = 10.0 ** np.linspace(1.0, -5.0, i)
    return eps_short, eps_long

def find_E() -> list[int]:
    """
    Computes a list of odd chain lengths n from eps_short
    """
    eps_short, _ = int_epsilon()
    ns: list[int] = []
    logL = np.log(lam)

    for eps in eps_short:
        a = int(-np.log(0.1 * float(eps)) / logL)
        if a % 2 == 1:
            ns.append(a)
    return ns

def t_n(n: int, w: float) -> float:
    """
    Wilson-chain hopping amplitude for site index n:
        t_n = (T / w) * Λ^(-n - 1/2)
    """
    return T / float(w) * np.power(lam, -n - 0.5)


def normaliz(n: int, w: float) -> float:
    """
    Energy normalization factor used to adimensionalize eigenvalues
    """
    return t_n(n - 2, w)