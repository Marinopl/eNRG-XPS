# core/projection.py

import numpy as np

def overlap_matrix(evec_final: np.ndarray,
                   evec_init:  np.ndarray,
                   nfermi: int,
                   hole_idx: int,
                   head_idx: int) -> np.ndarray:
    
    F = evec_final[:, :nfermi]           # (dim, nfermi)
    I = evec_init[:,  :nfermi]           # (dim, nfermi)

    # Base overlap: <final|init> = Fᴴ I
    M = F.conj().T @ I                   # (nfermi, nfermi)

    # Replace hole row with head projection row
    M[hole_idx, :] = evec_final[:, head_idx].conj().T @ I
    return M


def det_abs_sq(M: np.ndarray, method: str = "auto") -> float:
    if method in ("auto", "slogdet"):
        sign, logabs = np.linalg.slogdet(M)
        if sign != 0 and np.isfinite(logabs):
            return float(np.exp(2.0 * logabs))
        if method == "slogdet":
            s = np.linalg.svd(M, compute_uv=False)
            return float(np.exp(2.0 * np.sum(np.log(s + 0.0))))
    # robust SVD path
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.exp(2.0 * np.sum(np.log(s + 0.0))))
