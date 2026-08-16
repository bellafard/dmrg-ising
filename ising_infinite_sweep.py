#!/usr/bin/env python3
"""Sweep the Ising coupling J and record the ground-state energy per site.

Runs the infinite-system DMRG engine (see ``ising_infinite.py``) at a fixed
transverse field ``h`` for a range of Ising couplings ``J``, and writes the
resulting bulk energy per site to ``data/energy_vs_coupling.dat``. This traces
out how the ground-state energy evolves as the chain is driven across the
transverse-field Ising quantum phase transition.

Run:  python3 ising_infinite_sweep.py
"""

import os
import sys
import io
import contextlib
import numpy as np

import ising_infinite as dmrg  # reuse the DMRG engine and model definition

# Sweep settings
L = 100            # chain length grown by the infinite-system algorithm
M = 20             # states kept per block (bond dimension)
H_FIELD = 1.0      # fixed transverse field
J_VALUES = np.round(np.arange(0.1, 7.0, 0.1), 1)  # Ising couplings to scan

OUT_PATH = os.path.join("data", "energy_vs_coupling.dat")


def main():
    os.makedirs("data", exist_ok=True)
    dmrg.h = H_FIELD

    with open(OUT_PATH, "w") as f:
        f.write("# J\tE_per_site\t(L=%d, m=%d, h=%g)\n" % (L, M, H_FIELD))
        for j in J_VALUES:
            dmrg.J = float(j)  # the engine's H2() reads dmrg.J / dmrg.h
            # Silence the engine's per-step diagnostics during the sweep.
            with contextlib.redirect_stdout(io.StringIO()):
                energy_per_site = dmrg.infinite_system_algorithm(L=L, m=M)
            f.write("%g\t%.6f\n" % (j, energy_per_site))
            print("J = %g  ->  E/L = %.6f" % (j, energy_per_site))

    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
    main()
