# eNRG XPS

Utilities to compute **XPS-like spectra** on a **Wilson chain** using an eNRG-style logarithmic discretization.

- **Base (no impurity)** and **impurity** Hamiltonians
- **Primary**, **box-convolved**, and **continuous** spectra
- CLI “testbench” for quick runs
---


## What each module does

- **`params.py`** – Global model parameters
- **`utils.py`** – Small utilities
- **`discretization.py`** – Logarithmic discretization parameters and grids for the Wilson-chain hopping.
- **`projection.py`** – Builds and computes the **overlap matrix** between initial (null local scattering)/final eigenstates (non-null local scattering).
- **`hamiltonian.py`** – **Base** tridiagonal Hamiltonian.
- **`spectra.py`** – Base spectra: for one particle at a time (primary), more than one excitations is permited (secondary), convolved rate around primary excitations (convolution box), continuous spectrum for a crescent chain and a fixed head.
- **`hamiltonian_imp.py`** – **Impurity** Hamiltonian (local level `d` hybridized by `v`).
- **`spectra_imp.py`** – Impurity spectra.
- **`testbench_base.py`** / **`test_spectra_imp.py`** – Simple **CLI** helpers to compute arrays (NPZ) and optionally make a quick plot. They are **not** unit tests; they’re bench scripts for exploration.
- **`xps_ic_original_ipynb`"" - Original code made to obtain the title of Bachelor in Theoretical Physics at the University of São Paulo.

---

## Theory References 
 - https://doi.org/10.1590/1806-9126-RBEF-2025-0103
 - https://doi.org/10.48550/arXiv.2502.11317

### Installation

It’s recommended to use a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .


