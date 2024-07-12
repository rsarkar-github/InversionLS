import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])

    if model_mode == 0:
        filepath1 = "InversionLS/Data/sigsbee-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-sigsbee-source.npz"
        filepath3_ = "InversionLS/Data/p03-sigsbee-"
    elif model_mode == 1:
        filepath1 = "InversionLS/Data/marmousi-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-marmousi-source.npz"
        filepath3_ = "InversionLS/Data/p03-marmousi-"
    elif model_mode == 2:
        filepath1 = "InversionLS/Data/seiscope-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-seiscope-source.npz"
        filepath3_ = "InversionLS/Data/p03-seiscope-"
    else:
        raise ValueError("model mode = ", model_mode, " is not supported. Must be 0, 1, or 2.")


    if freq_mode == 0:
        freq = 5.0   # in Hz
    elif freq_mode == 1:
        freq = 7.5   # in Hz
    elif freq_mode == 2:
        freq = 10.0  # in Hz
    elif freq_mode == 3:
        freq = 15.0  # in Hz
    else:
        raise ValueError("freq mode = ", freq_mode, " is not supported. Must be 0, 1, 2, or 3.")


    # ----------------------------------------------
    # Load vel
    # ----------------------------------------------
    with np.load(filepath1) as data:
        vel = data["arr_0"]

    # ----------------------------------------------
    # Set parameters
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 1.0 / (n_ - 1)
    dz = dx
    freq_ = freq * 5.25
    omega_ = 2 * np.pi * freq_
    precision_ = np.complex64

    # ----------------------------------------------
    # Set receiver locations
    # ----------------------------------------------
    rec_locs_ = [[10, i] for i in range(n_)]

    # ----------------------------------------------
    # Initialize helmholtz matrix
    # ----------------------------------------------

    pml_cells = int((np.max(vel) / freq_) / dx)

    n_helmholtz_ = n_ + 2 * pml_cells
    nz_helmholtz_ = nz_ + 2 * pml_cells
    vel_helmholtz = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=np.float32)
    vel_helmholtz[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_] += vel

    vel_helmholtz[:, 0:pml_cells] += np.reshape(vel_helmholtz[:, pml_cells], newshape=(nz_helmholtz_, 1))
    vel_helmholtz[:, pml_cells + n_:] += np.reshape(vel_helmholtz[:, pml_cells + n_ - 1], newshape=(nz_helmholtz_, 1))
    vel_helmholtz[0:pml_cells, :] += vel_helmholtz[pml_cells, :]
    vel_helmholtz[pml_cells + nz_:, :] += vel_helmholtz[pml_cells + nz_ - 1, :]

    mat = create_helmholtz2d_matrix(
        a1=dz * nz_helmholtz_,
        a2=dx * n_helmholtz_,
        pad1=pml_cells,
        pad2=pml_cells,
        omega=omega_,
        precision=precision_,
        vel=vel_helmholtz,
        pml_damping=50.0,
        adj=False,
        warnings=True
    )
    mat_lu = splu(mat)
    print("Factorization done...\n")

    # ----------------------------------------------
    # Load source
    # ----------------------------------------------
    with np.load(filepath2) as data:
        sou_ = data["arr_0"]

    sou_ = sou_.astype(precision_)
    sou_helmholtz_ = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=precision_)
    sou_helmholtz_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_] += sou_

    rhs_norm = np.linalg.norm(sou_helmholtz_)
    rhs_ = sou_helmholtz_ / rhs_norm

    # ----------------------------------------------
    # Solve for true solution, sample at receivers
    # ----------------------------------------------
    sol_ = mat_lu.solve(np.reshape(rhs_, newshape=(nz_helmholtz_ * n_helmholtz_, 1)))
    sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
    sol_ = sol_ * rhs_norm
    sol_ = sol_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_]

    sol_recs_ = np.zeros(shape=(len(rec_locs_), 1), dtype=precision_)
    for i in range(len(rec_locs_)):
        sol_recs_[i, 0] = sol_[rec_locs_[i][0], rec_locs_[i][1]]

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------
    np.savez(filepath3_ + "true-sol-rec-" + "-" + "{:4.2f}".format(freq) + ".npz", sol_recs_)
