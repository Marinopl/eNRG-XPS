"""
API for the eNRG XPS package.
"""

from . import params  # keep as a module to allow runtime mutation

# Discretization
from .discretization import int_w, int_epsilon, find_E, t_n, normaliz

# Base Hamiltonian & spectra
from .hamiltonian import dig_ham, delta
from .spectra import xps_proj, spectrum, spectrum_sec, convolution, spectrum_continuous

# Impurity Hamiltonian & spectra
from .hamiltonian_imp import dig_ham_imp
from .spectra_imp import xps_proj_imp, spectrum_imp, spectrum_imp_continuous

# Helpers (optional)
from .linalg import eigh_cached, clear_eigh_cache
from .projection import overlap_matrix, det_abs_sq

__all__ = [
    "params",
    "int_w", "int_epsilon", "find_E", "t_n", "normaliz",
    "dig_ham", "delta",
    "xps_proj", "spectrum", "spectrum_sec", "convolution", "spectrum_continuous",
    "dig_ham_imp", "xps_proj_imp", "spectrum_imp", "spectrum_imp_continuous",
    "eigh_cached", "clear_eigh_cache", "overlap_matrix", "det_abs_sq",
]

__version__ = "0.1.0"
