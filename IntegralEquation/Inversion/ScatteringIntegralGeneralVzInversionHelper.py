import os
import time
import numpy as np
from numpy import ndarray
from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


def update_wavefields_mp_helper_func(params):
    """
    Multiprocessing helper function for the class ScatteringIntegralGeneralVzInversion2d.
    This solves the Step 1 subproblem for a fixed k value and a fixed source. This should be
    called as multiprocessing function with parallelization happening over shots for a fixed
    k value.

    No checking of parameters is done.

    :param params:
        params[0]: (int) n
        params[1]: (int) nz
        params[2]: (int) a
        params[3]: (int) b
        params[4]: (float) k
        params[5]: (numpy ndarray of shape (nz, 1)) vz velocity
        params[6]: (int) decimation factor for helmholtz solves
        params[7]: (float) sigma for delta injection
        params[8]: (np.complex64 or np.complex128) precision for use in calculation
        params[9]: (str) green_func_dir to read Green's func from
        params[10]: (str) shared memory name for green's func
        params[11]: (str) shared memory name for output wavefields
        params[12]: (str) filename of true data
        params[13]: (int) source number for this thread
        params[14]: (str) source filename
        params[15]: (str) model pert filename
        params[16]: (str) previous iteration wavefield filename
        params[17]: (float) lambda weight
        params[18]: (float) mu weight
        params[19]: (int) max iterations for lsqr / lsmr
        params[20]: (float) tol for lsqr / lsmr
        params[21]: (bool) verbose flag for printing messages

    :return:
    """
    # Read all parameters
    n_ = int(params[0])
    nz_ = int(params[1])
    a_ = float(params[2])
    b_ = float(params[3])
    k_ = float(params[4])
    vz_ = params[5]
    m_ = int(params[6])
    sigma_ = float(params[7])
    precision_ = params[8]
    green_func_dir_ = str(params[9])
    sm_name_ = str(params[10])
    sm_wavefield_name_ = str(params[11])
    true_data_filename_ = str(params[12])
    num_source_ = int(params[13])
    source_filename_ = str(params[14])
    model_pert_filename_ = str(params[15])
    prev_iter_wavefield_filename_ = str(params[16])
    lambda_wt = float(params[17])
    mu_wt = float(params[18])
    max_iter_ = int(params[19])
    tol_ = float(params[20])
    verbose_ = bool(params[21])

    lambda_wt = np.sqrt(lambda_wt)
    mu_wt = np.sqrt(mu_wt)

    # ------------------------------------------------------
    # Create Lippmann-Schwinger operator
    # Attach to shared memory for Green's function

    op_ = TruncatedKernelGeneralVz2d(
        n=n_, nz=nz_, a=a_, b=b_, k=k_, vz=vz_, m=m_, sigma=sigma_, precision=precision_,
        green_func_dir=green_func_dir_, num_threads=1,
        no_mpi=True, verbose=False, light_mode=True
    )

    op_.set_parameters(
        n=n_, nz=nz_, a=a_, b=b_, k=k_, vz=vz_, m=m_, sigma=sigma_, precision=precision_,
        green_func_dir=green_func_dir_, num_threads=1,
        no_mpi=True, verbose=False
    )

    sm_ = SharedMemory(sm_name_)
    data_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_.buf)
    op_.greens_func = data_

    # ------------------------------------------------------
    # Get source, true perturbation in slowness squared, previous wavefield

    with np.load(source_filename_) as f:
        source_ = f["arr_0"]
        num_sources_ = source_.shape[0]
        source_ = source_[num_source_, :, :]

    with np.load(model_pert_filename_) as f:
        psi_ = f["arr_0"]

    with np.load(prev_iter_wavefield_filename_) as f:
        prev_iter_wavefield_ = f["arr_0"][num_source_, :, :]

    # ------------------------------------------------------
    # Attach to shared memory for output
    sm_updated_wavefield_ = SharedMemory(sm_wavefield_name_)
    updated_wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_updated_wavefield_.buf)

    # ------------------------------------------------------
    # Define linear operator objects
    # Compute rhs
    def func_matvec(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op_.apply_kernel(u=v * psi_, output=u, adj=False, add=False)
        return np.reshape(v - (k_ ** 2) * u, newshape=(nz_ * n_, 1))

    def func_matvec_adj(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op_.apply_kernel(u=v, output=u, adj=True, add=False)
        return np.reshape(v - (k_ ** 2) * u * psi_, newshape=(nz_ * n_, 1))

    linop_lse = LinearOperator(
        shape=(nz_ * n_, nz_ * n_),
        matvec=func_matvec,
        rmatvec=func_matvec_adj,
        dtype=precision_
    )

    rhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    start_t_ = time.time()
    op_.apply_kernel(u=source_, output=rhs_)
    end_t_ = time.time()
    print("Shot num = ", num_source_, ", Time to compute rhs: ", "{:4.2f}".format(end_t_ - start_t_), " s")

    # ------------------------------------------------------
    # Solve for solution
    start_t = time.time()
    if verbose_:
        sol, exitcode = gmres(
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_, 1)),
            maxiter=max_iter_,
            restart=max_iter_,
            atol=0,
            tol=tol_,
            callback=make_callback()
        )
        true_data_[num_source_, :, :] += np.reshape(sol, newshape=(nz_, n_))
        end_t = time.time()
        print(
            "Shot num = ", num_source_,
            ", Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s",
            ", Exitcode = ", exitcode
        )
    else:
        sol, exitcode = gmres(
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_, 1)),
            maxiter=max_iter_,
            restart=max_iter_,
            atol=0,
            tol=tol_
        )
        true_data_[num_source_, :, :] += np.reshape(sol, newshape=(nz_, n_))
        end_t = time.time()
        print(
            "Shot num = ", num_source_,
            ", Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s",
            ", Exitcode = ", exitcode
        )

    # ------------------------------------------------------
    # Release shared memory
    sm_.close()
    sm_true_data_.close()
