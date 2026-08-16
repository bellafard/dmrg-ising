#!/usr/bin/env python3
"""Finite-system DMRG for the 1D transverse-field Ising model.

The finite-system algorithm improves on the infinite-system result by sweeping
back and forth across a chain of fixed length ``L``. During each sweep one
block grows at the expense of the other (using previously stored blocks as the
environment), which variationally refines the ground state well beyond the
single infinite-system pass. Increasing the number of kept states ``m`` on
successive sweeps systematically lowers the truncation error.

The model (spin-1/2 transverse-field Ising chain, ``H = J S^z S^z + h S^x``)
and the DMRG engine are imported from ``ising_infinite.py``.

Attribution: the DMRG engine is adapted from the pedagogical "simple-dmrg" code
by James R. Garrison and Ryan V. Mishmash (MIT license,
https://github.com/simple-dmrg/simple-dmrg).

Run:  python3 ising_finite.py
"""

import numpy as np

from ising_infinite import initial_block, single_dmrg_step


def graphic(sys_block, env_block, sys_label="l"):
    """ASCII picture of the current DMRG step: '=' system sites, '**' the two
    active sites, '-' environment sites.
    """
    assert sys_label in ("l", "r")
    picture = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        picture = picture[::-1]  # system on the right, environment on the left
    return picture


def finite_system_algorithm(L, m_warmup, m_sweep_list):
    """Run finite-system DMRG on a chain of length `L`.

    `m_warmup` states are kept during the infinite-system warm-up that builds
    the chain to full length; `m_sweep_list` gives the number of states to keep
    on each subsequent finite-system sweep.
    """
    assert L % 2 == 0  # require an even chain length

    # In-memory stand-in for on-disk storage of blocks, keyed by (side, length).
    block_disk = {}

    # Warm-up: build the chain to length L with the infinite-system algorithm,
    # saving each block as both a left ("l") and right ("r") block.
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block
    while 2 * block.length < L:
        print(graphic(block, block))
        block, energy = single_dmrg_step(block, block, m=m_warmup)
        print("E/L =", energy / (block.length * 2))
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    # Finite-system sweeps. The system block grows while the environment shrinks;
    # at the chain end the roles reverse, and a full round trip is one sweep.
    sys_label, env_label = "l", "r"
    sys_block = block
    del block

    for m in m_sweep_list:
        while True:
            env_block = block_disk[env_label, L - sys_block.length - 2]
            if env_block.length == 1:
                # Reached the end of the chain: turn around.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            print(graphic(sys_block, env_block, sys_label))
            sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)
            print("E/L =", energy / L)

            block_disk[sys_label, sys_block.length] = sys_block

            # A full sweep is complete when the left block reaches the midpoint.
            if sys_label == "l" and 2 * sys_block.length == L:
                break

    return energy / L


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
    finite_system_algorithm(L=20, m_warmup=10, m_sweep_list=[10, 20, 30, 40, 40])
