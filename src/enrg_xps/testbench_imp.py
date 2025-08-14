# src/enrg_xps/test_spectra_imp.py
from __future__ import annotations
import argparse
import numpy as np

from enrg_xps.spectra_imp import spectrum_imp, spectrum_imp_continuous
from enrg_xps.linalg import clear_eigh_cache
from enrg_xps.discretization import int_w


def compute_impurity_spectra(
    ni: int,
    nf: int,
    k: float,
    d: float,
    v: float,
    w: float,
    head: int,
    use_jacobian: bool = True,
) -> dict[str, np.ndarray]:
    """
    Return impurity spectra as plain NumPy arrays (no plotting).
    Keys mirror the base testbench for consistency.
    """
    clear_eigh_cache()

    # 1) Primary (no convolution variant here for impurity)
    erg_primary, rate_primary = spectrum_imp(ni, nf, k, d, v, w, head)

    # 2) Continuous (iso-energy crossings)
    eps_centers, A_eps = spectrum_imp_continuous(ni, nf, k, d, v, head, w_grid=None)

    return {
        "erg_primary":  erg_primary,
        "rate_primary": rate_primary,
        "eps_centers":  eps_centers,
        "A_eps":        A_eps,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compute impurity spectra and dump arrays (no plotting).")
    p.add_argument("--ni", type=int, default=7, help="Odd start n (inclusive).")
    p.add_argument("--nf", type=int, default=13, help="Odd stop n (exclusive, step=2).")
    p.add_argument("--k",  type=float, default=0.5, help="Scattering potential (site 1).")
    p.add_argument("--d",  type=float, default=0.1, help="Impurity on-site energy (site 0).")
    p.add_argument("--v",  type=float, default=0.2, help="Hybridization between impurity and site 1.")
    p.add_argument("--w",  type=float, default=1.0, help="Discretization scale w=Λ^θ (for primary).")
    p.add_argument("--head", type=int, default=2, help="Particle index above Fermi level.")
    p.add_argument("--no-jacobian", action="store_true",
                   help="Disable 1/|dE/dw| weighting in the continuous spectrum.")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Path to .npz to save arrays (e.g., out/imp_spectra.npz).")
    # optional plots (your style)
    p.add_argument("--plot-primary", action="store_true",
                   help="Plot (erg_primary, rate_primary).")
    p.add_argument("--plot-continuous", action="store_true",
                   help="Plot (eps_centers, A_eps).")
    p.add_argument("--plot-primary-continuous", action="store_true",
                   help="Overlay: primary vs continuous in one figure.")
    args = p.parse_args(argv)

    data = compute_impurity_spectra(
        ni=args.ni, nf=args.nf, k=args.k, d=args.d, v=args.v, w=args.w, head=args.head,
        use_jacobian=not args.no_jacobian
    )

    if args.output:
        np.savez(args.output, **data)
        print(f"[saved] {args.output} -> keys: {list(data.keys())}")
    else:
        for kname, arr in data.items():
            print(f"{kname:>15} | shape={arr.shape} | min={arr.min():.4e} | max={arr.max():.4e} | sum={arr.sum():.4e}")

    # ---------- optional plotting (minimal, you control the style) ----------
    # Dica: para log-log, garanta x>0 e y>0; abaixo eu normalizo por energia (y/x).
    if args.plot_primary:
        import matplotlib.pyplot as plt
        x = data["erg_primary"]
        y_raw = data["rate_primary"]
        m = x > 0
        x = x[m]; y = y_raw[m] / x    # troque para y_raw[m] se não quiser dividir por E
        plt.figure()
        plt.plot(x, y, "o")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Excitation energy (E)")
        plt.ylabel("Rate |det|^2 / E")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.title(f"Impurity primary (ni={args.ni}, nf={args.nf}, k={args.k}, d={args.d}, v={args.v}, w={args.w}, head={args.head})")
        plt.show()

    if args.plot_continuous:
        import matplotlib.pyplot as plt
        x = data["eps_centers"]
        y_raw = data["A_eps"]
        m = x > 0
        x = x[m]; y = y_raw[m] / x
        plt.figure()
        plt.plot(x, y, "o")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("ε (bin centers)")
        plt.ylabel("A(ε) / ε")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.title(f"Impurity continuous (ni={args.ni}, nf={args.nf}, k={args.k}, d={args.d}, v={args.v}, head={args.head})")
        plt.show()

    if args.plot_primary_continuous:
        import matplotlib.pyplot as plt
        x1, y1_raw = data["erg_primary"], data["rate_primary"]
        x2, y2_raw = data["eps_centers"], data["A_eps"]
        m1 = x1 > 0; m2 = x2 > 0
        x1, y1 = x1[m1], y1_raw[m1] / x1[m1]
        x2, y2 = x2[m2], y2_raw[m2] / x2[m2]
        fig, ax = plt.subplots()
        ax.plot(x1, y1, "o-", label="Primary")
        ax.plot(x2, y2, "^--", label="Continuous")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Energy (E)"); ax.set_ylabel("Rate / E")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_title(f"Impurity: Primary vs Continuous (ni={args.ni}, nf={args.nf}, k={args.k}, d={args.d}, v={args.v})")
        plt.show()


if __name__ == "__main__":
    main()
