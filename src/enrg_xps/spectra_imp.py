"""
Spectral calculations for the impurity XPS-eNRG model.
"""

from __future__ import annotations
import numpy as np

from .hamiltonian_imp import dig_ham_imp
from .discretization import normaliz, int_w, int_epsilon
from .projection import overlap_matrix, det_abs_sq


def xps_proj_imp(n: int, k: float, d: float, v: float, w: float,
                 hole: int, head: int):
    """
    Determinant of the projection of the final impurity state onto the initial one.

    Parameters
    ----------
    n : int
        Odd chain length (Hamiltonian dimension is (n+1) x (n+1)).
    k : float
        Scattering potential for the final Hamiltonian (on site 1).
    d : float
        Impurity on-site energy at site 0 (same for initial/final).
    v : float
        Hybridization between impurity and conduction site (same for initial/final).
    w : float
        Discretization scale (w = Λ^θ).
    hole : int
        Hole index below the Fermi level.
    head : int
        Particle index above the Fermi level.
    """
    # Initial (k=0) and final (k != 0) impurity eigenpairs
    eval_i, evec_i, _ = dig_ham_imp(n, 0.0, d, v, w)
    eval_f, evec_f, _ = dig_ham_imp(n, k,   d, v, w)

    nfermi = (n + 1) // 2

    # Overlap with eigenvectors in COLUMNS (⟨final|init⟩ = Fᴴ I), hole-row replaced by head-row
    M = overlap_matrix(evec_f, evec_i, nfermi, hole, head)

    ener_excit = float(eval_f[head] - eval_f[hole])
    rate = det_abs_sq(M)
    
    return ener_excit, rate


def spectrum_imp(ni: int, nf: int, k: float, d: float, v: float, w: float, head: int):
    """
    Compute impurity XPS primary rates for a logarithmic sequence of energies at fixed head.

    Parameters
    ----------
    ni, nf : int
        Odd range [ni, nf) stepped by 2, as in the original code.
    k : float
        Scattering potential (final Hamiltonian).
    d : float
        Impurity on-site energy at site 0.
    v : float
        Hybridization between impurity and conduction site.
    w : float
        Discretization scale.
    head : int
        Particle index above Fermi level.
    """
    n_erg = int((nf - ni) / 2)
    erg_imp_ = np.zeros(n_erg, dtype=float)
    rate_imp_ = np.zeros_like(erg_imp_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = (n + 1) // 2
        n_hole = nfermi - head
        n_excit = nfermi + head - 1
        ener, rate = xps_proj_imp(n, k, d, v, w, n_hole, n_excit)
        erg_imp_[count] = ener
        rate_imp_[count] = rate
        count += 1

    return erg_imp_, rate_imp_

def spectrum_imp_continuous(
    ni: int,
    nf: int,
    k: float,
    d: float,
    v: float,
    head: int,
    w_grid: np.ndarray | None = None,
    eps_edges: np.ndarray | None = None,
    use_jacobian: bool = True,
    *,
    pick: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous spectrum for the impurity case.

    Logic:
      - choose a w-grid (default: int_w(), θ∈[-1,1])
      - choose ε values (default: int_epsilon()[1])
      - for each odd n in [ni, nf), compute E(w) = E_excit(w) and rate R(w)
      - detect crossings of E(w) with each ε between consecutive w points
      - deposit one rate per crossing (no Jacobian), either:
          * 'left'  : use R at the left endpoint (matches the original 'inside')
          * 'interp': linearly interpolate R between the endpoints at the crossing

    Returns
    -------
    (eps_vals, A_eps)
      eps_vals : ε grid used (same order as provided/default)
      A_eps    : accumulated rates per ε
    """
    # 1) grids
    if w_grid is None:
        w_grid = int_w()
    if eps_edges is None:
        eps_vals = int_epsilon()[1]
    else:
        eps_vals = np.asarray(eps_edges, dtype=float)

    A_eps = np.zeros_like(eps_vals, dtype=float)

    # 2) sweep n
    for n in range(ni, nf, 2):
        nfermi = (n + 1) // 2
        hole = nfermi - head
        excit = nfermi + head - 1
        if hole < 0 or excit >= (n + 1):
            continue

        # 3) E(w) and R(w) along the w-grid
        M = len(w_grid)
        E = np.empty(M, dtype=float)
        R = np.empty(M, dtype=float)
        for p, w in enumerate(w_grid):
            eval_i, evec_i, _ = dig_ham_imp(n, 0.0, d, v, w)
            eval_f, evec_f, _ = dig_ham_imp(n, k,   d, v, w)
            # Overlap matrix ⟨final|initial⟩ with hole-row replaced by head-row
            M_ov = overlap_matrix(evec_f, evec_i, nfermi, hole, excit)
            R[p] = det_abs_sq(M_ov)
            E[p] = float(eval_f[excit] - eval_f[hole])

        # 4) detect crossings with ε
        E0, E1 = E[:-1], E[1:]
        R0, R1 = R[:-1], R[1:]
    
        S0 = E0[:, None] - eps_vals[None, :]
        S1 = E1[:, None] - eps_vals[None, :]
        
        cross = ((S0 <= 0.0) & (S1 >= 0.0)) | ((S0 >= 0.0) & (S1 <= 0.0))

        if not cross.any():
            continue

        seg_idx, eps_idx = np.nonzero(cross)

        if pick == "left":
            A_eps[eps_idx] += R0[seg_idx]

        elif pick == "interp":
            denom = (E1[seg_idx] - E0[seg_idx])
            
            t = (eps_vals[eps_idx] - E0[seg_idx]) / (denom + 1e-30)
            t = np.clip(t, 0.0, 1.0)
            R_star = (1.0 - t) * R0[seg_idx] + t * R1[seg_idx]
            A_eps[eps_idx] += R_star

        else:
            raise ValueError("pick must be 'left' or 'interp'")

    return eps_vals, A_eps