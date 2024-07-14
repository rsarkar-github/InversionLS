import sys
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 5:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])
    solver_mode = int(sys.argv[3])
    mu_mode = int(sys.argv[4])

    if model_mode == 0:
        filepath = "InversionLS/Data/sigsbee-new-vz-2d.npz"
        filepath1 = "InversionLS/Data/sigsbee-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-sigsbee-source.npz"
        filepath3_ = "InversionLS/Data/p03-sigsbee-"
        filepath4_ = "InversionLS/Data/p05-sigsbee-"
        filepath5_ = "InversionLS/Fig/p05-sigsbee-"

    elif model_mode == 1:
        filepath = "InversionLS/Data/marmousi-new-vz-2d.npz"
        filepath1 = "InversionLS/Data/marmousi-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-marmousi-source.npz"
        filepath3_ = "InversionLS/Data/p03-marmousi-"
        filepath4_ = "InversionLS/Data/p05-marmousi-"
        filepath5_ = "InversionLS/Fig/p05-marmousi-"

    elif model_mode == 2:
        filepath = "InversionLS/Data/seiscope-new-vz-2d.npz"
        filepath1 = "InversionLS/Data/seiscope-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-seiscope-source.npz"
        filepath3_ = "InversionLS/Data/p03-seiscope-"
        filepath4_ = "InversionLS/Data/p05-seiscope-"
        filepath5_ = "InversionLS/Fig/p05-seiscope-"

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


    if solver_mode == 1:
        solver_name = "lsqr"
    elif solver_mode == 2:
        solver_name = "lsmr"
    else:
        raise ValueError("solver mode = ", solver_mode, " is not supported. Must be 1 or 2.")

    if mu_mode == 0:
        mu_ = 1000000.0
    elif mu_mode == 1:
        mu_ = 5000000.0
    elif mu_mode == 2:
        mu_ = 10000000.0
    else:
        raise ValueError("mu mode = ", mu_mode, " is not supported. Must be 0, 1 or 2.")

    # ----------------------------------------------
    # Load vel
    # ----------------------------------------------

    with np.load(filepath) as data:
        vel = data["arr_0"]

    with np.load(filepath1) as data:
        vel1 = data["arr_0"]

    psi_ = (1.0 / vel) ** 2.0 - (1.0 / vel1) ** 2.0
    psi_ *= 0.5

    vel = ((1.0 / vel) ** 2.0 - psi_) ** (-0.5)

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

    matH = create_helmholtz2d_matrix(
        a1=dz * nz_helmholtz_,
        a2=dx * n_helmholtz_,
        pad1=pml_cells,
        pad2=pml_cells,
        omega=omega_,
        precision=precision_,
        vel=vel_helmholtz,
        pml_damping=50.0,
        adj=True,
        warnings=True
    )

    # ----------------------------------------------
    # Set receiver locations
    # Load true data
    # ----------------------------------------------
    rec_locs_ = [[10, i] for i in range(n_)]
    num_recs_ = len(rec_locs_)
    rec_locs_ = np.asarray(rec_locs_) + pml_cells

    with np.load(filepath3_ + "true-sol-rec-" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        rec_data_ = data["arr_0"]

    rec_data_ = rec_data_.astype(precision_)

    # ----------------------------------------------
    # Load source
    # ----------------------------------------------
    with np.load(filepath2) as data:
        sou_ = data["arr_0"]

    sou_ = sou_.astype(precision_)
    sou_helmholtz_ = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=precision_)
    sou_helmholtz_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_] += sou_

    # ----------------------------------------------
    # Initialize linear operator objects
    # ----------------------------------------------

    def func_matvec(v):
        u = mat.dot(v)
        v = np.reshape(v, newshape=(nz_helmholtz_, n_helmholtz_))
        u1 = mu_ * np.reshape(v[rec_locs_[:, 0], rec_locs_[:, 1]], newshape=(num_recs_,))

        out = np.zeros(shape=(nz_helmholtz_ * n_helmholtz_ + num_recs_,), dtype=precision_)
        out[0:nz_helmholtz_ * n_helmholtz_] = u
        out[nz_helmholtz_ * n_helmholtz_:] = u1

        return out

    def func_matvec_adj(v):

        u = matH.dot(v[0:nz_helmholtz_ * n_helmholtz_])
        v1 = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=precision_)
        v1[rec_locs_[:, 0], rec_locs_[:, 1]] = mu_ * v[nz_helmholtz_ * n_helmholtz_:]
        v1 = np.reshape(v1, newshape=(nz_helmholtz_ * n_helmholtz_,))

        return u + v1


    linop_helm = LinearOperator(
        shape=(nz_helmholtz_ * n_helmholtz_ + num_recs_, nz_helmholtz_ * n_helmholtz_),
        matvec=func_matvec,
        rmatvec=func_matvec_adj,
        dtype=precision_
    )

    # ----------------------------------------------
    # Run solver iterations
    # ----------------------------------------------
    sou_helmholtz_ = np.reshape(sou_helmholtz_, newshape=(nz_helmholtz_ * n_helmholtz_,))
    rec_data_ = np.reshape(rec_data_, newshape=(num_recs_,))
    rhs1_ = np.zeros(shape=(nz_helmholtz_ * n_helmholtz_ + num_recs_,), dtype=precision_)
    rhs1_[0:nz_helmholtz_ * n_helmholtz_] = sou_helmholtz_
    rhs1_[nz_helmholtz_ * n_helmholtz_:] = mu_ * rec_data_
    rhs_ = rhs1_

    if solver_name == "lsqr":

        print("----------------------------------------------")
        print("Solver: LSQR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsqr(
            linop_helm,
            np.reshape(rhs_, newshape=(nz_helmholtz_ * n_helmholtz_ + num_recs_, 1)),
            atol=tol_,
            btol=0,
            show=True,
            iter_lim=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    if solver_name == "lsmr":

        print("----------------------------------------------")
        print("Solver: LSMR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsmr(
            linop_helm,
            np.reshape(rhs_, newshape=(nz_helmholtz_ * n_helmholtz_ + num_recs_, 1)),
            atol=tol_,
            btol=0,
            show=True,
            maxiter=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
    sol_ = sol_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_]
    plt.imshow(np.real(sol_), cmap="Greys", vmin=-1e-5, vmax=1e-5)
    plt.show()

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------
    savefig_fname = (filepath5_ + "sol-" + solver_name +
                     "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_/1e6) + ".pdf")
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    np.savez(filepath4_ + "sol-" + solver_name +
             "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_/1e6) + ".npz", sol_)

    file_data = {}
    file_data["niter"] = total_iter
    file_data["tsolve"] = "{:4.2f}".format(tsolve)

    with open(
            filepath4_ + "stats-" + solver_name +
            "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_/1e6) + ".json", "w"
    ) as file:
        json.dump(file_data, file, indent=4)
