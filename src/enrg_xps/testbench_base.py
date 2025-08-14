from __future__ import annotations
import argparse
import numpy as np

from enrg_xps.spectra import spectrum, convolution, spectrum_continuous
from enrg_xps.linalg import clear_eigh_cache


def compute_base_spectra(
    ni: int,
    nf: int,
    k: float,
    w: float,
    head: int,
    use_jacobian: bool = True,
) -> dict[str, np.ndarray]:
    clear_eigh_cache()

    # 1) Primary
    erg_primary, rate_primary = spectrum(ni, nf, k, w, head)

    # 2) Convolved
    erg_convolved, rate_convolved = convolution(ni, nf, k, w, head)

    # 3) Continuous
    eps_centers, A_eps = spectrum_continuous(
        ni, nf, k, head, w_grid=None, eps_edges=None, use_jacobian=use_jacobian
    )

    return {
        "erg_primary": erg_primary,
        "rate_primary": rate_primary,
        "erg_convolved": erg_convolved,
        "rate_convolved": rate_convolved,
        "eps_centers": eps_centers,
        "A_eps": A_eps,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compute base spectra and dump arrays (no plotting).")
    p.add_argument("--ni", type=int, default=7)
    p.add_argument("--nf", type=int, default=13)
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--head", type=int, default=2)
    p.add_argument("--no-jacobian", action="store_true")
    p.add_argument("-o", "--output", type=str, default=None)
    # plots
    p.add_argument("--plot-primary", action="store_true",
                   help="Plot (erg_primary, rate_primary).")
    p.add_argument("--plot-convolved", action="store_true",
                   help="Plot (erg_convolved, rate_convolved).")
    p.add_argument("--plot-continuous", action="store_true",
                   help="Plot (eps_centers, A_eps).")
    args = p.parse_args(argv)

    data = compute_base_spectra(
        ni=args.ni, nf=args.nf, k=args.k, w=args.w, head=args.head,
        use_jacobian=not args.no_jacobian
    )

    if args.output:
        np.savez(args.output, **data)
        print(f"[saved] {args.output} -> keys: {list(data.keys())}")
    else:
        for kname, arr in data.items():
            print(f"{kname:>15} | shape={arr.shape} | min={arr.min():.4e} | max={arr.max():.4e} | sum={arr.sum():.4e}")

    # -------- optional plots (your style) --------
    # Dica: log-log precisa de x > 0 e y > 0
    if args.plot_primary:
        import matplotlib.pyplot as plt
        x = data["erg_primary"]
        y_raw = data["rate_primary"]
        mask = x > 0
        x = x[mask]
        y = (y_raw[mask] / x)  # use y_raw[mask] se não quiser normalizar por energia
        plt.figure()
        plt.plot(x, y, marker="o", linestyle="-")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Excitation energy (E)")
        plt.ylabel("Rate / E")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.title(f"Primary spectrum (ni={args.ni}, nf={args.nf}, k={args.k}, w={args.w}, head={args.head})")
        plt.show()

    if args.plot_convolved:
        import matplotlib.pyplot as plt
        x = data["erg_convolved"]
        y_raw = data["rate_convolved"]
        if x.size:
            mask = x > 0
            x = x[mask]
            y = (y_raw[mask] / x)  # ou y_raw[mask]
            plt.figure()
            plt.plot(x, y, marker="o", linestyle="-")
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("Energy (E)")
            plt.ylabel("Convolved rate / E")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.title(f"Convolved spectrum (ni={args.ni}, nf={args.nf}, k={args.k}, w={args.w}, head={args.head})")
            plt.show()
        else:
            print("[info] convolved spectrum empty (need at least 3 primary points).")

    if args.plot_continuous:
        import matplotlib.pyplot as plt
        x = data["eps_centers"]
        y_raw = data["A_eps"]
        mask = x > 0
        x = x[mask]
        y = (y_raw[mask] / x)  # ou y_raw[mask]
        plt.figure()
        plt.plot(x, y, marker="o", linestyle="-")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("ε (bin centers)")
        plt.ylabel("A(ε) / ε")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.title(f"Continuous spectrum (ni={args.ni}, nf={args.nf}, k={args.k}, head={args.head})")
        plt.show()


if __name__ == "__main__":
    main()