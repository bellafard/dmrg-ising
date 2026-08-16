#!/usr/bin/env python3
"""Infinite-system DMRG for the 1D transverse-field Ising model.

The density-matrix renormalization group (DMRG) finds the ground state of a
one-dimensional quantum lattice model by growing the chain one site at a time
and, at each step, keeping only the ``m`` most important states of the reduced
density matrix. This keeps the effective Hilbert space small while capturing the
dominant entanglement, giving near-exact ground-state energies at a tiny
fraction of the cost of exact diagonalization.

Model
-----
Spin-1/2 transverse-field Ising chain,

    H = J * sum_<i,j> S^z_i S^z_j  +  h * sum_i S^x_i,

with Ising coupling ``J`` and transverse field ``h``. The model has a
quantum phase transition between an ordered (ferromagnetic) and a disordered
(paramagnetic) phase as the ratio h/J is varied.

This is the "infinite-system" algorithm: the chain is grown symmetrically
(system block + two sites + a mirror-image environment block) until the target
length is reached, giving the bulk energy per site.

Attribution
-----------
The DMRG engine is adapted from the pedagogical "simple-dmrg" code by
James R. Garrison and Ryan V. Mishmash (MIT license,
https://github.com/simple-dmrg/simple-dmrg). The model-specific parts (the
transverse-field Ising Hamiltonian and the parameter studies) are my own.
"""

import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
from collections import namedtuple

# A "Block" is a renormalized chunk of the chain: its length in sites, the
# size of its (truncated) basis, and a dictionary of operators expressed in
# that basis. An "EnlargedBlock" is a Block that has grown by one bare site.
Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

# ---------------------------------------------------------------------------
# Model definition: spin-1/2 transverse-field Ising chain
# ---------------------------------------------------------------------------
model_d = 2  # single-site Hilbert space dimension

Sz1 = np.array([[0.5, 0.0], [0.0, -0.5]], dtype="d")  # single-site S^z
Sx1 = np.array([[0.0, 0.5], [0.5, 0.0]], dtype="d")   # single-site S^x
H1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype="d")    # single-site term (zero here)

# Coupling constants. J is the Ising (S^z S^z) coupling; h is the transverse
# field. These are module-level so a parameter sweep can vary them.
J = 1.0
h = 1.0


def is_valid_block(block):
    """Sanity check: every operator must be square in the block's basis."""
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True


# The enlarged block is validated the same way.
is_valid_enlarged_block = is_valid_block


def H2(Sz1, Sx1, Id1, Sz2, Sx2, Id2):
    """Two-site bond term of the transverse-field Ising Hamiltonian.

    Combines an Ising S^z S^z coupling with a transverse field S^x on each of
    the two sites, expressed as Kronecker products across the two Hilbert
    spaces being joined.
    """
    return J * kron(Sz1, Sz2) + h * kron(Sx1, Id2) + h * kron(Id1, Sx2)


# The initial block is a single site. "conn_*" operators live on the block edge
# and are needed to attach the next bond as the chain grows.
initial_block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sz": Sz1,
    "conn_Sx": Sx1,
})


def enlarge_block(block):
    """Grow a block by one bare site, returning the EnlargedBlock."""
    mblock = block.basis_size
    o = block.operator_dict

    # The new basis is the Kronecker product of the block basis and the
    # single-site basis. scipy's `kron` scales blocks of the second array by
    # the first; we follow that convention everywhere.
    enlarged_operator_dict = {
        "H": (kron(o["H"], identity(model_d))
              + kron(identity(mblock), H1)
              + H2(o["conn_Sz"], o["conn_Sx"], identity(mblock), Sz1, Sx1, identity(model_d))),
        "conn_Sz": kron(identity(mblock), Sz1),
        "conn_Sx": kron(identity(mblock), Sx1),
    }

    return EnlargedBlock(length=block.length + 1,
                         basis_size=block.basis_size * model_d,
                         operator_dict=enlarged_operator_dict)


def rotate_and_truncate(operator, transformation_matrix):
    """Project an operator into the new, truncated basis."""
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))


def single_dmrg_step(sys, env, m):
    """One DMRG step: grow `sys` and `env` by a site, solve the superblock
    ground state, then renormalize keeping the `m` most important states.
    Returns the new system block and the ground-state energy.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys)
    if sys is env:  # reflection symmetry: reuse the result
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    # Build the full superblock Hamiltonian (system + environment + the bond
    # connecting them).
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    superblock_hamiltonian = (
        kron(sys_enl_op["H"], identity(m_env_enl))
        + kron(identity(m_sys_enl), env_enl_op["H"])
        + H2(sys_enl_op["conn_Sz"], sys_enl_op["conn_Sx"], identity(m_sys_enl),
             env_enl_op["conn_Sz"], env_enl_op["conn_Sx"], identity(m_env_enl))
    )

    # Ground state via Lanczos ("SA" = smallest algebraic eigenvalue).
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

    # Reduced density matrix of the system, obtained by tracing out the
    # environment. The environment index runs fastest in the Kronecker
    # product, so psi0 reshapes row-major into (sys, env).
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())

    # Diagonalize rho and keep the eigenvectors with the largest weight.
    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = sorted(
        zip(evals, evecs.transpose()), reverse=True, key=lambda x: x[0]
    )

    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype="d", order="F")
    for i, (eval_, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum(x[0] for x in possible_eigenstates[:my_m])
    print("truncation error", truncation_error)

    # Renormalize every operator into the truncated basis.
    new_operator_dict = {name: rotate_and_truncate(op, transformation_matrix)
                         for name, op in sys_enl.operator_dict.items()}

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    return newblock, energy


def infinite_system_algorithm(L, m):
    """Grow a symmetric chain to length `L`, keeping `m` states per block."""
    block = initial_block
    while 2 * block.length < L:
        print("L =", block.length * 2 + 2)
        block, energy = single_dmrg_step(block, block, m=m)
        print("E/L =", energy / (block.length * 2))
    return energy / (block.length * 2)


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
    infinite_system_algorithm(L=100, m=20)
