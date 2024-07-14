import sys
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


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
        filepath4_ = "InversionLS/Data/p04-sigsbee-"
        filepath5_ = "InversionLS/Fig/p04-sigsbee-"

    elif model_mode == 1:
        filepath = "InversionLS/Data/marmousi-new-vz-2d.npz"
        filepath1 = "InversionLS/Data/marmousi-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-marmousi-source.npz"
        filepath3_ = "InversionLS/Data/p03-marmousi-"
        filepath4_ = "InversionLS/Data/p04-marmousi-"
        filepath5_ = "InversionLS/Fig/p04-marmousi-"

    elif model_mode == 2:
        filepath = "InversionLS/Data/seiscope-new-vz-2d.npz"
        filepath1 = "InversionLS/Data/seiscope-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-seiscope-source.npz"
        filepath3_ = "InversionLS/Data/p03-seiscope-"
        filepath4_ = "InversionLS/Data/p04-seiscope-"
        filepath5_ = "InversionLS/Fig/p04-seiscope-"

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
        mu_ = 1.0
    elif mu_mode == 1:
        mu_ = 5.0
    elif mu_mode == 2:
        mu_ = 10.0
    else:
        raise ValueError("mu mode = ", mu_mode, " is not supported. Must be 0, 1 or 2.")

    # ----------------------------------------------
    # Load vz and calculate psi
    # Change psi to 0.5*psi
    # ----------------------------------------------
    with np.load(filepath) as data:
        vel = data["arr_0"]
    vel_trace = vel[:, 0]
    n1_vel_trace = vel_trace.shape[0]
    vel_trace = np.reshape(vel_trace, newshape=(n1_vel_trace, 1)).astype(np.float32)

    with np.load(filepath1) as data:
        vel1 = data["arr_0"]

    psi_ = (1.0 / vel) ** 2.0 - (1.0 / vel1) ** 2.0
    psi_ *= 0.5

    # ----------------------------------------------
    # Set parameters
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    a_ = 0.
    b_ = a_ + (1.0 / (n_ - 1)) * (nz_ - 1)
    freq_ = freq * 5.25
    omega_ = 2 * np.pi * freq_
    m_ = 4
    sigma_ = 0.0015
    precision_ = np.complex64
    green_func_dir_ = "InversionLS/Data/p01-green-func-" + str(model_mode) + "-" + str(freq_mode)
    num_threads_ = 4
    vz_ = np.zeros(shape=(nz_, 1), dtype=np.float32) + vel_trace

    psi_ = psi_.astype(precision_)

    # ----------------------------------------------
    # Set receiver locations
    # Load true data
    # ----------------------------------------------
    rec_locs_ = [[10, i] for i in range(n_)]
    num_recs_ = len(rec_locs_)
    rec_locs_ = np.asarray(rec_locs_)

    with np.load(filepath3_ + "true-sol-rec-" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        rec_data_ = data["arr_0"]

    rec_data_ = rec_data_.astype(precision_)

    # ----------------------------------------------
    # Load source
    # ----------------------------------------------
    with np.load(filepath2) as data:
        sou_ = data["arr_0"]

    sou_ = sou_.astype(precision_)

    # ----------------------------------------------
    # Initialize operator
    # ----------------------------------------------
    op = TruncatedKernelGeneralVz2d(
        n=n_,
        nz=nz_,
        a=a_,
        b=b_,
        k=omega_,
        vz=vz_,
        m=m_,
        sigma=sigma_,
        precision=precision_,
        green_func_dir=green_func_dir_,
        num_threads=num_threads_,
        verbose=False,
        light_mode=True
    )
    op.set_parameters(
        n=n_,
        nz=nz_,
        a=a_,
        b=b_,
        k=omega_,
        vz=vz_,
        m=m_,
        sigma=sigma_,
        precision=precision_,
        green_func_dir=green_func_dir_,
        num_threads=num_threads_,
        verbose=False
    )

    rhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    t1 = time.time()
    op.apply_kernel(u=sou_, output=rhs_)
    t2 = time.time()
    print("Operator application time = ", "{:6.2f}".format(t2 - t1), " s")

    # ----------------------------------------------
    # Initialize linear operator objects
    # ----------------------------------------------

    def func_matvec(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op.apply_kernel(u=v * psi_, output=u, adj=False, add=False)
        u = np.reshape(v - (omega_ ** 2) * u, newshape=(nz_ * n_,))
        u1 = mu_ * np.reshape(v[rec_locs_[:, 0], rec_locs_[:, 1]], newshape=(num_recs_,))

        out = np.zeros(shape=(nz_ * n_ + num_recs_,), dtype=precision_)
        out[0:nz_ * n_] = u
        out[nz_ * n_:] = u1

        return out

    def func_matvec_adj(v):

        v1 = np.reshape(v[0:nz_ * n_], newshape=(nz_, n_))
        u = v1 * 0
        op.apply_kernel(u=v1, output=u, adj=True, add=False)
        u = np.reshape(v1 - (omega_ ** 2) * u * psi_, newshape=(nz_ * n_,))

        v1 *= 0
        v1[rec_locs_[:, 0], rec_locs_[:, 1]] = mu_ * v[nz_ * n_:]
        v1 = np.reshape(v1, newshape=(nz_ * n_,))

        return u + v1

    linop_lse = LinearOperator(
        shape=(nz_ * n_ + num_recs_, nz_ * n_),
        matvec=func_matvec,
        rmatvec=func_matvec_adj,
        dtype=precision_
    )

    # ----------------------------------------------
    # Run solver iterations
    # ----------------------------------------------
    rhs_ = np.reshape(rhs_, newshape=(nz_ * n_,))
    rec_data_ = np.reshape(rec_data_, newshape=(num_recs_,))
    rhs1_ = np.zeros(shape=(nz_ * n_ + num_recs_,), dtype=precision_)
    rhs1_[0:nz_ * n_] = rhs_
    rhs1_[nz_ * n_:] = mu_ * rec_data_
    rhs_ = rhs1_

    if solver_name == "lsqr":

        print("----------------------------------------------")
        print("Solver: LSQR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsqr(
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_ + num_recs_, 1)),
            atol=tol_,
            btol=0,
            show=True,
            iter_lim=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_, n_))
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
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_ + num_recs_, 1)),
            atol=tol_,
            btol=0,
            show=True,
            maxiter=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_, n_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    plt.imshow(np.real(sol_), cmap="Greys")
    plt.show()

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------
    savefig_fname = (filepath5_ + "sol-" + solver_name +
                     "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_) + ".pdf")
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    np.savez(filepath4_ + "sol-" + solver_name +
             "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_) + ".npz", sol_)

    file_data = {}
    file_data["niter"] = total_iter
    file_data["tsolve"] = "{:4.2f}".format(tsolve)

    with open(
            filepath4_ + "stats-" + solver_name +
            "-" + "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_) + ".json", "w"
    ) as file:
        json.dump(file_data, file, indent=4)
