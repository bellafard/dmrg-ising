#!/usr/bin/env python3
"""Plot the DMRG ground-state energy per site versus Ising coupling J.

Reads ``data/energy_vs_coupling.dat`` (produced by ``ising_infinite_sweep.py``)
and writes ``figures/energy_vs_coupling.png``.

Run:  python3 plot_energy.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "energy_vs_coupling.dat")
FIG_PATH = os.path.join("figures", "energy_vs_coupling.png")


def main():
    J, E = np.loadtxt(DATA_PATH, unpack=True)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(J, E, "o-", ms=4, lw=1.5, color="#2b6cb0")
    ax.set_xlabel(r"Ising coupling  $J$  (transverse field $h = 1$)")
    ax.set_ylabel(r"Ground-state energy per site  $E/L$")
    ax.set_title("Infinite-system DMRG: transverse-field Ising chain")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print("Wrote", FIG_PATH)


if __name__ == "__main__":
    main()
