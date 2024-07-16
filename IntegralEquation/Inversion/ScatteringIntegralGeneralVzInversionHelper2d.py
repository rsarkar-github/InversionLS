import numpy as np
from numpy import ndarray
from scipy.sparse.linalg import LinearOperator, gmres, lsqr, lsmr, cg
import time
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from tqdm import tqdm
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d
from ...Utilities.LinearSolvers import gmres_counter, conjugate_gradient
from ...Utilities import TypeChecker


def check_iter_input(params):

    # Read all parameters
    model_pert_path_ = params[0]
    wavefield_filename_list_ = params[1]
    nz_ = params[2]
    n_ = params[3]
    num_sources_ = params[4]
    precision_ = params[5]
    precision_real_ = params[6]
    iter_num_ = params[7]

    # Check model pert
    model_pert_ = np.load(model_pert_path_)["arr_0"]

    TypeChecker.check_ndarray(
        x=model_pert_,
        shape=(nz_, n_),
        dtypes=(precision_real_,),
        nan_inf=True
    )

    # Check wavefields
    for path_ in wavefield_filename_list_:
        with np.load(path_) as f:
            wavefield_k_ = f["arr_0"]

        TypeChecker.check_ndarray(
            x=wavefield_k_,
            shape=(num_sources_, nz_, n_),
            dtypes=(precision_,),
            nan_inf=True
        )

    print("Checking Iter" + str(iter_num_) + " model pert and wavefields: OK")


def green_func_calculate_mp_helper_func(params):

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
    green_func_dir_ = params[9]

    TruncatedKernelGeneralVz2d(
        n=n_,
        nz=nz_,
        a=a_,
        b=b_,
        k=k_,
        vz=vz_,
        m=m_,
        sigma=sigma_,
        precision=precision_,
        green_func_dir=green_func_dir_,
        num_threads=1,
        no_mpi=True,
        verbose=False
    )


def true_data_calculate_mp_helper_func(params):

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
    sm_true_data_name_ = str(params[11])
    num_source_ = int(params[12])
    source_filename_ = str(params[13])
    true_pert_filename_ = str(params[14])
    max_iter_ = int(params[15])
    tol_ = float(params[16])
    verbose_ = bool(params[17])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_ = SharedMemory(sm_name_)
    data_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_.buf)
    op_.greens_func = data_

    # ------------------------------------------------------
    # Get source, and true perturbation in slowness squared

    with np.load(source_filename_) as f:
        source_ = f["arr_0"]
        num_sources_ = source_.shape[0]
        source_ = source_[num_source_, :, :]

    with np.load(true_pert_filename_) as f:
        psi_ = f["arr_0"]

    # ------------------------------------------------------
    # Attach to shared memory for output
    sm_true_data_ = SharedMemory(sm_true_data_name_)
    true_data_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_true_data_.buf)

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
    counter = gmres_counter()
    start_t = time.time()
    if verbose_:
        sol, exitcode = gmres(
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_, 1)),
            maxiter=max_iter_,
            restart=max_iter_,
            atol=0,
            tol=tol_,
            callback=counter
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


def compute_obj2(params):

    # ------------------------------------------------------
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
    precision_real_ = params[9]
    green_func_dir_ = str(params[10])
    num_k_values_ = int(params[11])
    num_sources_ = int(params[12])
    sm_obj2_name_ = str(params[13])
    sm_green_func_name_ = str(params[14])
    sm_source_name_ = str(params[15])
    sm_wavefield_name_ = str(params[16])
    sm_model_pert_name_ = str(params[17])
    num_source_ = int(params[18])
    num_k_ = int(params[19])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for obj2, source, wavefield, model_pert

    sm_obj2_ = SharedMemory(sm_obj2_name_)
    obj2_ = ndarray(shape=(num_k_values_, num_sources_), dtype=np.float64, buffer=sm_obj2_.buf)

    sm_source_ = SharedMemory(sm_source_name_)
    source_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_source_.buf)

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    sm_model_pert_ = SharedMemory(sm_model_pert_name_)
    psi_ = ndarray(shape=(nz_, n_), dtype=precision_real_, buffer=sm_model_pert_.buf)

    # ------------------------------------------------------
    # Compute obj2

    rhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    op_.apply_kernel(u=source_[num_source_, :, :], output=rhs_)

    lhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    op_.apply_kernel(u=wavefield_[num_source_, :, :] * psi_, output=lhs_, adj=False, add=False)
    lhs_ = wavefield_[num_source_, :, :] - (k_ ** 2) * lhs_

    obj2_[num_k_, num_source_] = np.linalg.norm(lhs_ - rhs_) ** 2.0

    # ------------------------------------------------------
    # Release shared memory
    sm_green_func_.close()
    sm_obj2_.close()
    sm_source_.close()
    sm_wavefield_.close()
    sm_model_pert_.close()


def compute_initial_wavefields(params):

    # ------------------------------------------------------
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
    precision_real_ = params[9]
    green_func_dir_ = str(params[10])
    num_sources_ = int(params[11])
    sm_green_func_name_ = str(params[12])
    sm_source_name_ = str(params[13])
    sm_wavefield_name_ = str(params[14])
    num_source_ = int(params[15])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for source, wavefield

    sm_source_ = SharedMemory(sm_source_name_)
    source_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_source_.buf)

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    # ------------------------------------------------------
    # Compute initial wavefiels

    op_.apply_kernel(u=source_[num_source_, :, :], output=wavefield_[num_source_, :, :])

    # ------------------------------------------------------
    # Release shared memory

    sm_green_func_.close()
    sm_source_.close()
    sm_wavefield_.close()


def update_wavefield(params):

    # ------------------------------------------------------
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
    precision_real_ = params[9]
    green_func_dir_ = str(params[10])
    num_sources_ = int(params[11])
    rec_locs_ = params[12]
    num_source_ = int(params[13])
    lambda_ = float(params[14])
    mu_ = float(params[15])
    sm_green_func_name_ = str(params[16])
    sm_source_name_ = str(params[17])
    sm_wavefield_name_ = str(params[18])
    sm_true_data_name_ = str(params[19])
    sm_model_pert_name_ = str(params[20])
    max_iter_ = int(params[21])
    solver_ = str(params[22])
    atol_ = float(params[23])
    btol_ = float(params[24])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for source, wavefield, true data, model_pert

    sm_source_ = SharedMemory(sm_source_name_)
    source_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_source_.buf)

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    sm_true_data_ = SharedMemory(sm_true_data_name_)
    true_data_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_true_data_.buf)

    sm_model_pert_ = SharedMemory(sm_model_pert_name_)
    psi_ = ndarray(shape=(nz_, n_), dtype=precision_real_, buffer=sm_model_pert_.buf)

    # ------------------------------------------------------
    # Inversion

    if lambda_ != 0.0 or mu_ != 0.0:

        # ------------------------------------------------------
        # Define linear operator objects
        # Compute rhs (scale to norm 1)

        num_recs_ = rec_locs_.shape[0]
        def func_matvec(v):
            v = np.reshape(v, newshape=(nz_, n_))
            u = v * 0
            op_.apply_kernel(u=v * psi_, output=u, adj=False, add=False)
            u = lambda_ * np.reshape(v - (k_ ** 2) * u, newshape=(nz_ * n_,))
            u1 = mu_ * np.reshape(v[rec_locs_[:, 0], rec_locs_[:, 1]], newshape=(num_recs_,))

            out = np.zeros(shape=(nz_ * n_ + num_recs_,), dtype=precision_)
            out[0:nz_ * n_] = u
            out[nz_ * n_:] = u1

            return out

        def func_matvec_adj(v):

            v1 = np.reshape(v[0:nz_ * n_], newshape=(nz_, n_))
            u = v1 * 0
            op_.apply_kernel(u=v1, output=u, adj=True, add=False)
            u = lambda_ * np.reshape(v1 - (k_ ** 2) * u * psi_, newshape=(nz_ * n_,))

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

        temp_ = np.zeros(shape=(nz_, n_), dtype=precision_)
        op_.apply_kernel(u=source_[num_source_, :, :], output=temp_)
        temp_ = lambda_ * np.reshape(temp_, newshape=(nz_ * n_,))

        temp1_ = true_data_[num_source_, :, :]
        temp1_ = np.reshape(mu_ * temp1_[rec_locs_[:, 0], rec_locs_[:, 1]], newshape=(num_recs_,))

        rhs_ = np.zeros(shape=(nz_ * n_ + num_recs_,), dtype=precision_)
        rhs_[0:nz_ * n_] = temp_
        rhs_[nz_ * n_:] = temp1_
        rhs_ -= func_matvec(
            v=np.reshape(wavefield_[num_source_, :, :], newshape=(nz_ * n_, 1))
        )

        rhs_scale_ = np.linalg.norm(rhs_)
        if rhs_scale_ < 1e-15:
            rhs_scale_ = 1.0
        rhs_ = rhs_ / rhs_scale_

        del temp_, temp1_

        # ------------------------------------------------------
        # Solve for solution

        if solver_ == "lsmr":
            start_t_ = time.time()
            sol_, istop_, itn_, normr_, normar_ = lsmr(
                linop_lse,
                rhs_,
                atol=atol_,
                btol=btol_,
                show=False,
                maxiter=max_iter_
            )[:5]

            wavefield_[num_source_, :, :] += np.reshape(rhs_scale_ * sol_, newshape=(nz_, n_))
            end_t_ = time.time()
            print(
                "Shot num = ", num_source_,
                ", Total time to solve: ", "{:4.2f}".format(end_t_ - start_t_), " s",
                ", istop = ", istop_,
                ", itn = ", itn_,
                ", normr_ = ", normr_,
                ", normar_ = ", normar_
            )

        if solver_ == "lsqr":
            start_t_ = time.time()
            sol_, istop_, itn_, normr_, _, _, _, normar_ = lsqr(
                linop_lse,
                rhs_,
                atol=atol_,
                btol=btol_,
                show=False,
                iter_lim=max_iter_
            )[:8]

            wavefield_[num_source_, :, :] += np.reshape(rhs_scale_ * sol_, newshape=(nz_, n_))
            end_t_ = time.time()
            print(
                "Shot num = ", num_source_,
                ", Total time to solve: ", "{:4.2f}".format(end_t_ - start_t_), " s",
                ", istop = ", istop_,
                ", itn = ", itn_,
                ", normr_ = ", normr_,
                ", normar_ = ", normar_
            )

    # ------------------------------------------------------
    # Release shared memory
    sm_green_func_.close()
    sm_source_.close()
    sm_wavefield_.close()
    sm_true_data_.close()
    sm_model_pert_.close()


def update_wavefield_cg(params):

    # ------------------------------------------------------
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
    precision_real_ = params[9]
    green_func_dir_ = str(params[10])
    num_sources_ = int(params[11])
    rec_locs_ = params[12]
    num_source_ = int(params[13])
    lambda_ = float(params[14])
    mu_ = float(params[15])
    sm_green_func_name_ = str(params[16])
    sm_source_name_ = str(params[17])
    sm_wavefield_name_ = str(params[18])
    sm_true_data_name_ = str(params[19])
    sm_model_pert_name_ = str(params[20])
    max_iter_ = int(params[21])
    solver_ = str(params[22])
    atol_ = float(params[23])
    btol_ = float(params[24])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for source, wavefield, true data, model_pert

    sm_source_ = SharedMemory(sm_source_name_)
    source_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_source_.buf)

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    sm_true_data_ = SharedMemory(sm_true_data_name_)
    true_data_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_true_data_.buf)

    sm_model_pert_ = SharedMemory(sm_model_pert_name_)
    psi_ = ndarray(shape=(nz_, n_), dtype=precision_real_, buffer=sm_model_pert_.buf)

    # ------------------------------------------------------
    # Inversion

    if lambda_ != 0.0 or mu_ != 0.0:

        # ------------------------------------------------------
        # Define linear operator objects
        # Compute rhs (scale to norm 1)

        num_recs_ = rec_locs_.shape[0]

        def func_normal_op(v):

            v = np.reshape(v, newshape=(nz_, n_))
            u = v * 0
            op_.apply_kernel(u=v * psi_, output=u, adj=False, add=False)
            u = v - (k_ ** 2) * u
            u *= lambda_

            out = v * 0
            op_.apply_kernel(u=u, output=out, adj=True, add=False)
            out = u - (k_ ** 2) * out * psi_
            out *= lambda_

            u *= 0
            u[rec_locs_[:, 0], rec_locs_[:, 1]] = v[rec_locs_[:, 0], rec_locs_[:, 1]]
            u *= (mu_ ** 2.0)

            out += u

            return np.reshape(out, newshape=(nz_ * n_,))

        def func_matvec_adj(v):

            v1 = np.reshape(v[0:nz_ * n_], newshape=(nz_, n_))
            u = v1 * 0
            op_.apply_kernel(u=v1, output=u, adj=True, add=False)
            u = lambda_ * np.reshape(v1 - (k_ ** 2) * u * psi_, newshape=(nz_ * n_,))

            v1 *= 0
            v1[rec_locs_[:, 0], rec_locs_[:, 1]] = mu_ * v[nz_ * n_:]
            v1 = np.reshape(v1, newshape=(nz_ * n_,))

            return u + v1

        linop_normal_op = LinearOperator(
            shape=(nz_ * n_, nz_ * n_),
            matvec=func_normal_op,
            dtype=precision_
        )

        temp_ = np.zeros(shape=(nz_, n_), dtype=precision_)
        op_.apply_kernel(u=source_[num_source_, :, :], output=temp_)
        temp_ = lambda_ * np.reshape(temp_, newshape=(nz_ * n_,))

        temp1_ = true_data_[num_source_, :, :]
        temp1_ = np.reshape(mu_ * temp1_[rec_locs_[:, 0], rec_locs_[:, 1]], newshape=(num_recs_,))

        temp2_ = np.zeros(shape=(nz_ * n_ + num_recs_,), dtype=precision_)
        temp2_[0:nz_ * n_] = temp_
        temp2_[nz_ * n_:] = temp1_

        rhs_ = func_matvec_adj(v=temp2_)
        rhs_ -= func_normal_op(
            v=np.reshape(wavefield_[num_source_, :, :], newshape=(nz_ * n_,))
        )

        rhs_scale_ = np.linalg.norm(rhs_)
        if rhs_scale_ < 1e-15:
            rhs_scale_ = 1.0
        rhs_ = rhs_ / rhs_scale_

        del temp_, temp1_, temp2_

        # ------------------------------------------------------
        # Solve for solution
        start_t_ = time.time()
        sol_, exit_code_ = cg(
                linop_normal_op,
                rhs_,
                atol=0,
                tol=btol_,
                maxiter=max_iter_
        )

        wavefield_[num_source_, :, :] += np.reshape(sol_ * rhs_scale_, newshape=(nz_, n_))
        end_t_ = time.time()
        print(
            "Shot num = ", num_source_,
            ", Total time to solve: ", "{:4.2f}".format(end_t_ - start_t_), " s",
            ", exit code = ", exit_code_
        )

    # ------------------------------------------------------
    # Release shared memory
    sm_green_func_.close()
    sm_source_.close()
    sm_wavefield_.close()
    sm_true_data_.close()
    sm_model_pert_.close()


def perform_inversion_update_pert(
        obj, iter_count,
        max_iter=100, tol=1e-5, mnorm=0.0, num_procs=1,
        multi_iter_flag=False
):
    """
    Perform inversion -- update wavefields.
    No parameter checking is performed.

    :param obj: ScatteringIntegralGeneralVzInversion2d
        Inversion object
    :param iter_count: int
        Iteration number
    :param max_iter: int
        Maximum number of iterations allowed by lsqr / lsmr
    :param tol: float
        tol for CG
    :param mnorm: float
        Weight to penalize update to pert
    :param num_procs: int
        Number of processors for multiprocessing while computing objective function
    :param multi_iter_flag: bool
        Multiple iteration flag (affects which initial model pert is loaded)
    """

    # ------------------------------------------------------
    # Load lambda array

    with np.load(obj.lambda_arr_filename(iter_count=iter_count)) as f:
        lambda_arr = f["arr_0"]

    # ------------------------------------------------------
    # Compute rhs

    with SharedMemoryManager() as smm:

        # Create shared memory for Green's function
        sm_greens_func = smm.SharedMemory(size=obj.num_bytes_greens_func())
        green_func = ndarray(
            shape=(obj.nz, obj.nz, 2 * obj.n - 1),
            dtype=obj.precision,
            buffer=sm_greens_func.buf
        )

        # Create shared memory for source
        sm_source = smm.SharedMemory(size=obj.num_bytes_true_data_per_k())
        source = ndarray(
            shape=(obj.num_sources, obj.nz, obj.n),
            dtype=obj.precision,
            buffer=sm_source.buf
        )

        # Create shared memory for wavefield
        sm_wavefield = smm.SharedMemory(size=obj.num_bytes_true_data_per_k())
        wavefield = ndarray(
            shape=(obj.num_sources, obj.nz, obj.n),
            dtype=obj.precision,
            buffer=sm_wavefield.buf
        )

        # Create shared memory for initial perturbation and load it
        sm_pert = smm.SharedMemory(size=obj.num_bytes_model_pert())
        pert = ndarray(shape=(obj.nz, obj.n), dtype=obj.precision_real, buffer=sm_pert.buf)
        pert *= 0
        if multi_iter_flag:
            model_pert_filename = obj.model_pert_filename(iter_count=iter_count)
        else:
            model_pert_filename = obj.model_pert_filename(iter_count=iter_count - 1)
        with np.load(model_pert_filename) as f:
            pert += f["arr_0"]

        # Create shared memory for rhs computation
        sm_rhs = smm.SharedMemory(size=obj.num_bytes_true_data_per_k())
        rhs = ndarray(
            shape=(obj.num_sources, obj.nz, obj.n),
            dtype=obj.precision,
            buffer=sm_rhs.buf
        )

        # ------------------------------------------------------
        # Compute rhs

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Computing rhs for inversion...")

        rhs_inv = np.zeros(shape=(obj.nz, obj.n), dtype=obj.precision)

        # Loop over k values
        for k in range(obj.num_k_values):

            print("\n---------------------------------------------")
            print("Starting k number ", k)

            if np.sum(lambda_arr[k, :]) > 0.0:

                # Load Green's func into shared memory
                green_func *= 0
                green_func_filename = obj.greens_func_filename(num_k=k)
                with np.load(green_func_filename) as f:
                    green_func += f["arr_0"]

                # Load source into shared memory
                source *= 0
                source_filename = obj.source_filename(num_k=k)
                with np.load(source_filename) as f:
                    source += f["arr_0"]

                # Load initial wavefield into shared memory
                wavefield *= 0
                wavefield_filename = obj.wavefield_filename(num_k=k, iter_count=iter_count)
                with np.load(wavefield_filename) as f:
                    wavefield += f["arr_0"]

                param_tuple_list = [
                    (
                        obj.n,
                        obj.nz,
                        obj.a,
                        obj.b,
                        obj.k_values[k],
                        obj.vz,
                        obj.m,
                        obj.sigma_greens_func,
                        obj.precision,
                        obj.precision_real,
                        obj.greens_func_filedir(num_k=k),
                        obj.num_sources,
                        i,
                        lambda_arr[k, i],
                        sm_greens_func.name,
                        sm_source.name,
                        sm_wavefield.name,
                        sm_pert.name,
                        sm_rhs.name
                    ) for i in range(obj.num_sources) if lambda_arr[k, i] != 0.0
                ]
                if len(param_tuple_list) >= 1:
                    with Pool(min(len(param_tuple_list), mp.cpu_count(), num_procs)) as pool:
                        max_ = len(param_tuple_list)

                        with tqdm(total=max_) as pbar:
                            for _ in pool.imap_unordered(compute_rhs_for_pert_update, param_tuple_list):
                                pbar.update()

                # Handle zero lambda values separately
                for i in range(obj.num_sources):
                    if lambda_arr[k, i] == 0.0:
                        rhs[i, :, :] = 0.0

                # Sum the result
                rhs_inv += np.sum(rhs, axis=0)

        # Take real part
        rhs_inv = np.real(rhs_inv).astype(obj.precision_real)

    # ------------------------------------------------------
    # Perform inversion

    with SharedMemoryManager() as smm:

        # Create shared memory for Green's function
        sm_greens_func = smm.SharedMemory(size=obj.num_bytes_greens_func())
        green_func = ndarray(
            shape=(obj.nz, obj.nz, 2 * obj.n - 1),
            dtype=obj.precision,
            buffer=sm_greens_func.buf
        )

        # Create shared memory for wavefield
        sm_wavefield = smm.SharedMemory(size=obj.num_bytes_true_data_per_k())
        wavefield = ndarray(
            shape=(obj.num_sources, obj.nz, obj.n),
            dtype=obj.precision,
            buffer=sm_wavefield.buf
        )

        # Create shared memory for sum computation
        sm_sumarr = smm.SharedMemory(size=obj.num_bytes_true_data_per_k())
        sumarr = ndarray(
            shape=(obj.num_sources, obj.nz, obj.n),
            dtype=obj.precision,
            buffer=sm_sumarr.buf
        )

        # ------------------------------------------------------
        # Define linear operator

        def zero_and_add(x_, f_):
            x_ *= 0
            x_ += f_

        def func_linop(v):

            start_tt_ = time.time()

            v = np.reshape(v, newshape=(obj.nz, obj.n))

            # Add mnorm term to output
            sum_accumulated = v * mnorm

            for k_ in range(obj.num_k_values):

                print("\n---------------------------------------------")
                print("Starting k number ", k_)

                if np.sum(lambda_arr[k_, :]) > 0.0:

                    # Load Green's func into shared memory
                    green_func_filename_ = obj.greens_func_filename(num_k=k_)
                    with np.load(green_func_filename_) as f_:
                        zero_and_add(green_func, f_["arr_0"])

                    # Load initial wavefield into shared memory
                    wavefield_filename_ = obj.wavefield_filename(num_k=k_, iter_count=iter_count)
                    with np.load(wavefield_filename_) as f_:
                        zero_and_add(wavefield, f_["arr_0"])

                    param_tuple_list_ = [
                        (
                            obj.n,
                            obj.nz,
                            obj.a,
                            obj.b,
                            obj.k_values[k_],
                            obj.vz,
                            obj.m,
                            obj.sigma_greens_func,
                            obj.precision,
                            obj.greens_func_filedir(num_k=k_),
                            obj.num_sources,
                            i,
                            lambda_arr[k_, i],
                            v,
                            sm_greens_func.name,
                            sm_wavefield.name,
                            sm_sumarr.name
                        ) for i in range(obj.num_sources) if lambda_arr[k_, i] != 0.0
                    ]

                    if len(param_tuple_list_) >= 1:
                        with Pool(min(len(param_tuple_list_), mp.cpu_count(), num_procs)) as pool_:
                            maxx_ = len(param_tuple_list_)

                            with tqdm(total=maxx_) as pbar_:
                                for _ in pool_.imap_unordered(compute_matvec_for_pert_update, param_tuple_list_):
                                    pbar_.update()

                    # Handle zero lambda values separately
                    for i in range(obj.num_sources):
                        if lambda_arr[k_, i] == 0.0:
                            sumarr[i, :, : ] = 0.0

                    # Sum the result
                    sum_accumulated += np.real(np.sum(sumarr, axis=0)).astype(obj.precision_real)

            end_tt_ = time.time()
            print(
                "Total time for operator application: ", "{:4.2f}".format(end_tt_ - start_tt_), " s"
            )

            # Return result
            return np.reshape(sum_accumulated, newshape=(obj.nz * obj.n,))

        linop_lse = LinearOperator(
            shape=(obj.nz * obj.n, obj.nz * obj.n),
            matvec=func_linop,
            dtype=obj.precision
        )

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Starting linear inversion...")

        rhs_inv_scale = np.linalg.norm(rhs_inv)
        if rhs_inv_scale < 1e-15:
            rhs_inv_scale = 1.0
        rhs_inv = rhs_inv / rhs_inv_scale

        start_t = time.time()
        sol, exit_code = cg(
            linop_lse,
            np.reshape(rhs_inv, newshape=(obj.nz * obj.n, )),
            atol=0,
            tol=tol,
            maxiter=max_iter
        )
        sol *= rhs_inv_scale
        end_t = time.time()

        print(
            "Total time for iteration: ", "{:4.2f}".format(end_t - start_t), " s",
            ", exit code = ", exit_code
        )

        print("---------------------------------------------")
        print("---------------------------------------------")
        print("Finished linear inversion...\n")

    return np.reshape(sol.astype(obj.precision_real), newshape=(obj.nz, obj.n)) + pert


def compute_rhs_for_pert_update(params):

    # ------------------------------------------------------
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
    precision_real_ = params[9]
    green_func_dir_ = str(params[10])
    num_sources_ = int(params[11])
    num_source_ = int(params[12])
    lambda_val = float(params[13])
    sm_green_func_name_ = str(params[14])
    sm_source_name_ = str(params[15])
    sm_wavefield_name_ = str(params[16])
    sm_model_pert_name_ = str(params[17])
    sm_rhs_name_ = str(params[18])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for source, wavefield, true data, model_pert

    sm_source_ = SharedMemory(sm_source_name_)
    source_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_source_.buf)

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    sm_rhs_ = SharedMemory(sm_rhs_name_)
    rhs_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_rhs_.buf)

    sm_model_pert_ = SharedMemory(sm_model_pert_name_)
    psi_ = ndarray(shape=(nz_, n_), dtype=precision_real_, buffer=sm_model_pert_.buf)

    # ------------------------------------------------------
    # Compute rhs

    if lambda_val == 0:
        rhs_[num_source_, :, :] *= 0

    else:
        source_[num_source_, :, :] += (k_ ** 2) * psi_ * wavefield_[num_source_, :, :]
        op_.apply_kernel(u=source_[num_source_, :, :], output=rhs_[num_source_, :, :], adj=False, add=False)
        rhs_[num_source_, :, :] *= (-1.0)
        rhs_[num_source_, :, :] += wavefield_[num_source_, :, :]

        temp_ = np.zeros(shape=(nz_, n_), dtype=precision_)
        op_.apply_kernel(u=rhs_[num_source_, :, :], output=temp_, adj=True, add=False)
        rhs_[num_source_, :, :] = temp_ * np.conjugate(wavefield_[num_source_, :, :])
        rhs_[num_source_, :, :] *= (k_ ** 2) * (lambda_val ** 2)

    # ------------------------------------------------------
    # Release shared memory

    sm_green_func_.close()
    sm_source_.close()
    sm_wavefield_.close()
    sm_model_pert_.close()
    sm_rhs_.close()


def compute_matvec_for_pert_update(params):

    # ------------------------------------------------------
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
    num_sources_ = int(params[10])
    num_source_ = int(params[11])
    lambda_val = float(params[12])
    v = params[13]
    sm_green_func_name_ = str(params[14])
    sm_wavefield_name_ = str(params[15])
    sm_sumarr_name_ = str(params[16])

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
        green_func_dir=green_func_dir_, num_threads=1, green_func_set=False,
        no_mpi=True, verbose=False
    )

    sm_green_func_ = SharedMemory(sm_green_func_name_)
    green_func_ = ndarray(shape=(nz_, nz_, 2 * n_ - 1), dtype=precision_, buffer=sm_green_func_.buf)
    op_.greens_func = green_func_

    # ------------------------------------------------------
    # Attach to shared memory for wavefield, sumarr

    sm_wavefield_ = SharedMemory(sm_wavefield_name_)
    wavefield_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_wavefield_.buf)

    sm_sumarr_ = SharedMemory(sm_sumarr_name_)
    sumarr_ = ndarray(shape=(num_sources_, nz_, n_), dtype=precision_, buffer=sm_sumarr_.buf)

    # ------------------------------------------------------
    # Compute the matvec product
    if lambda_val == 0:
        sumarr_[num_source_, :, :] *= 0

    else:
        temp_ = (k_ ** 4.0) * (lambda_val ** 2.0) * v * wavefield_[num_source_, :, :]
        temp_ = temp_.astype(precision_)
        op_.apply_kernel(
            u=temp_,
            output=sumarr_[num_source_, :, :],
            adj=False,
            add=False
        )
        op_.apply_kernel(
            u=sumarr_[num_source_, :, :],
            output=temp_,
            adj=True,
            add=False
        )
        sumarr_[num_source_, :, :] = temp_ * np.conjugate(wavefield_[num_source_, :, :])

    # ------------------------------------------------------
    # Release shared memory

    sm_green_func_.close()
    sm_wavefield_.close()
    sm_sumarr_.close()
