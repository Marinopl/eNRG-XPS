"""
Spectral calculations for the base XPS-NRG model (no impurity).
"""

from __future__ import annotations
import numpy as np

from .hamiltonian import dig_ham
from .discretization import normaliz, int_w, int_epsilon
from .projection import overlap_matrix, det_abs_sq
from .params import lam


def xps_proj(n: int, k: float, w: float, hole: int, head: int):
    """
    Determinant of the projection of the final state onto the initial state.

    Parameters
    ----------
    n : int
        Odd chain length (Hamiltonian dimension is (n+1) x (n+1)).
    k : float
        Scattering potential for the final Hamiltonian.
    w : float
        Discretization scale (w = Λ^θ).
    hole : int
        Hole index below the Fermi level.
    head : int
        Particle index above the Fermi level.
    """
    # Initial (k = 0) and final (k != 0) eigenstates and eigenenergies
    eval_i, evec_i, _ = dig_ham(n, 0.0, w)
    eval_f, evec_f, _ = dig_ham(n, k,   w)

    nfermi = (n + 1) // 2
    M = overlap_matrix(evec_f, evec_i, nfermi, hole, head)

    ener_excit = float(eval_f[head] - eval_f[hole])
    rate = det_abs_sq(M)
    ener_norm = ener_excit / normaliz(n, w)

    return ener_excit, rate, ener_norm


def spectrum(ni: int, nf: int, k: float, w: float, head: int):
    """
    Compute XPS primary rates for a logarithmic sequence of energies at fixed head.

    Parameters
    ----------
    ni, nf : int
        Odd range [ni, nf) stepped by 2, as in the original code.
    k : float
        Scattering potential.
    w : float
        Discretization scale.
    head : int
        Particle index above Fermi level.

    Returns
    -------
    (np.ndarray, np.ndarray)
        erg_  : excitation energies for each n
        rate_ : corresponding |det(overlap)|^2
    """
    if ni % 2 == 0:
        raise ValueError("ni must be odd (dig_ham requires odd n).")
    n_erg = int((nf - ni) / 2)
    erg_ = np.zeros(n_erg, dtype=float)
    rate_ = np.zeros_like(erg_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = (n + 1) // 2
        n_hole = nfermi - head
        n_excit = nfermi + head - 1
        erg_[count], rate_[count], _ = xps_proj(n, k, w, n_hole, n_excit)
        count += 1

    return erg_, rate_


def spectrum_sec(ni: int, nf: int, k: float, w: float, head: int):
    if ni % 2 == 0:
        raise ValueError("ni must be odd (dig_ham requires odd n).")

    # Matches the original allocation: (#n points) * (head-1)^2
    total = int(((nf - ni) / 2) * np.power(head - 1, 2))
    erg_sec_ = np.zeros(total, dtype=float)
    rate_sec_ = np.zeros_like(erg_sec_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = (n + 1) // 2
        n_hole = nfermi - head
        n_excit = nfermi + head - 1

        # Loop over all secondary (j,u) between hole and head
        for j in range(n_hole + 1, nfermi):          # holes above the main hole
            for u in range(nfermi, n_excit):         # particles below the main particle
                ener, rate, _ = xps_proj(n, k, w, j, u)
                erg_sec_[count] = ener
                rate_sec_[count] = rate
                count += 1

    return erg_sec_, rate_sec_

def convolution(ni: int, nf: int, k: float, w: float, head: int):
    """
    eNRG XPS rate with a box-function convolution.

      - builds windows using geometric means of consecutive primary energies
      - accumulates secondary rates whose energies fall inside each window
      - adds the primary rate at i+1
      - normalizes by log(lam^2)
    """
    erg_, rate_ = spectrum(ni, nf, k, w, head)
    erg_sec_, rate_sec_ = spectrum_sec(ni, nf, k, w, head)

    # Need at least 3 primary points to define (N-2) windows
    if erg_.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Window bounds: for i = 0..N-3
    # Upper U_i = sqrt(E[i]   * E[i+1])
    # Lower L_i = sqrt(E[i+1] * E[i+2])
    U = np.sqrt(erg_[:-2] * erg_[1:-1])
    L = np.sqrt(erg_[1:-1] * erg_[2:])

    # Broadcast all secondary energies against all windows:
    # mask[j, i] = True  iff  L[i] < erg_sec_[j] < U[i]
    if erg_sec_.size:
        Esec = erg_sec_[:, None]
        mask = (Esec > L[None, :]) & (Esec < U[None, :])

        has_bin = mask.any(axis=1)

        idx = np.argmax(mask, axis=1)[has_bin]
        vals = rate_sec_[has_bin]

        sec_sum = np.zeros_like(U)
        np.add.at(sec_sum, idx, vals)
    else:
        sec_sum = np.zeros_like(U)

    # Add primary rate at i+1 and normalize by log(lam^2)
    rate_conv_ = (sec_sum + rate_[1:-1]) / (2.0 * np.log(lam))
    erg_conv_  = erg_[1:-1]

    return erg_conv_, rate_conv_

def spectrum_continuous(
    ni: int,
    nf: int,
    k: float,
    head: int,
    w_grid: np.ndarray | None = None,
    eps_edges: np.ndarray | None = None,
    use_jacobian: bool = True,  # mantido por compatibilidade; NÃO usado aqui
    *,
    pick: str = "left",         # "left" (igual ao inside original) ou "interp" (interpolado em w)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inside-like continuous spectrum for the base (no-impurity) model.

    Logic
    -----
    - Choose a w-grid (default: int_w(), θ ∈ [-1, 1]).
    - Choose ε values (default: int_epsilon()[1]).
    - For each odd n in [ni, nf), compute along w:
        E(w) = E_final[excit](w) - E_final[hole](w),  R(w) = |det(overlap)|^2
      where 'final' uses 'k', 'initial' uses 'k=0'.
    - Detect crossings of E(w) with each ε between consecutive w points.
    - Deposit one rate per crossing (no Jacobian):
        * pick="left"  : R at the left endpoint (matches inside-like behavior)
        * pick="interp": R linearly interpolated in w at the crossing.

    Returns
    -------
    eps_vals : np.ndarray
        ε grid (same order as provided/default).
    A_eps    : np.ndarray
        Accumulated rates per ε (same shape as eps_vals).
    """
    # 1) grids
    if w_grid is None:
        w_grid = int_w()            # suave e amplo: θ ∈ [-1, 1]
    if eps_edges is None:
        eps_vals = int_epsilon()[1] # no seu código original é a malha “longa” de ε
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

        # 3) compute E(w) and R(w) along the w-grid
        M = len(w_grid)
        E = np.empty(M, dtype=float)
        R = np.empty(M, dtype=float)
        for p, w in enumerate(w_grid):
            eval_i, evec_i, _ = dig_ham(n, 0.0, w)
            eval_f, evec_f, _ = dig_ham(n, k,   w)
            # overlap: eigenvectors are columns (NumPy convention)
            M_ov = overlap_matrix(evec_f, evec_i, nfermi, hole, excit)
            R[p] = det_abs_sq(M_ov)
            E[p] = float(eval_f[excit] - eval_f[hole])

        # 4) crossing detection against all eps (vectorized)
        E0, E1 = E[:-1], E[1:]
        R0, R1 = R[:-1], R[1:]

        # broadcasting: (segments, eps_len)
        S0 = E0[:, None] - eps_vals[None, :]
        S1 = E1[:, None] - eps_vals[None, :]
        # treat exact hits as crossings too
        cross = ((S0 <= 0.0) & (S1 >= 0.0)) | ((S0 >= 0.0) & (S1 <= 0.0))

        if not cross.any():
            continue

        seg_idx, eps_idx = np.nonzero(cross)

        if pick == "left":
            # deposit left-endpoint rate (matches inside-like behavior)
            A_eps[eps_idx] += R0[seg_idx]

        elif pick == "interp":
            # interpolate in w at the crossing:
            denom = (E1[seg_idx] - E0[seg_idx])
            t = (eps_vals[eps_idx] - E0[seg_idx]) / (denom + 1e-30)
            t = np.clip(t, 0.0, 1.0)
            R_star = (1.0 - t) * R0[seg_idx] + t * R1[seg_idx]
            A_eps[eps_idx] += R_star

        else:
            raise ValueError("pick must be 'left' or 'interp'")

    return eps_vals, A_eps