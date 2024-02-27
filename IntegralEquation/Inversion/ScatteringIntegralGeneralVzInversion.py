import numpy as np
from numpy import ndarray
import scipy as sp
from scipy.sparse.linalg import LinearOperator, gmres, lsqr, lsmr
import matplotlib.pyplot as plt
import os
import sys
import numba
import shutil
import json
import time
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from tqdm import tqdm
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d
from ...Utilities.JsonTools import update_json
from ...Utilities import TypeChecker
from ...Utilities.LinearSolvers import gmres_counter


def green_func_calculate_mp_helper_func(params):

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
    lhs_ = wavefield_ - (k_ ** 2) * lhs_

    obj2_[num_k_, num_source_] = np.linalg.norm(lhs_ - rhs_) ** 2.0

    # ------------------------------------------------------
    # Release shared memory
    sm_green_func_.close()
    sm_obj2_.close()
    sm_source_.close()
    sm_wavefield_.close()
    sm_model_pert_.close()


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
    rhs_ = rhs_ / rhs_scale_

    del temp_
    del temp1_

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


class ScatteringIntegralGeneralVzInversion2d:

    def __init__(self, basedir, restart=False, restart_code=None):
        """
        Always read parameters from .json file.
        Name of parameter file is basedir + "params.json"

        :param basedir: str
            Base directory for storing all results
        :param restart: bool
            Whether to run the class initialization in restart mode or not
        :param restart_code: int or None
            If restart is True, then restart_code decides from where to restart
        """

        TypeChecker.check(x=basedir, expected_type=(str,))
        TypeChecker.check(x=restart, expected_type=(bool,))
        if restart_code is not None:
            TypeChecker.check_int_bounds(x=restart_code, lb=0, ub=7)

        self._basedir = basedir
        self._restart = restart

        self._restart_code = None
        if self._restart is True:
            self._restart_code = restart_code

        # Read parameters from .json file
        self._param_file = os.path.join(basedir, "params.json")
        print("---------------------------------------------")
        print("---------------------------------------------")
        print("Reading parameter file:\n")
        with open(self._param_file) as file:
            self._params = json.load(file)

        # ----------------------------------------------------
        # Create some necessary class members from self._params

        # Geometry group
        self._n = int(self._params["geometry"]["n"])
        self._nz = int(self._params["geometry"]["nz"])
        self._a = float(self._params["geometry"]["a"])
        self._b = float(self._params["geometry"]["b"])
        self._scale_fac_inv = float(self._params["geometry"]["scale_fac_inv"])

        TypeChecker.check(x=self._n, expected_type=(int,))
        if self._n % 2 != 1 or self._n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=self._nz, expected_type=(int,))
        if self._nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check(x=self._a, expected_type=(float,))
        TypeChecker.check_float_strict_lower_bound(x=self._b, lb=self._a)
        TypeChecker.check_float_positive(x=self._scale_fac_inv)

        # Receiver locations
        self._rec_locs = list(self._params["rec_locs"])
        self._num_rec = len(self._rec_locs)

        TypeChecker.check_int_positive(x=self._num_rec)

        for i in range(self._num_rec):
            TypeChecker.check_int_bounds(x=len(self._rec_locs[i]), lb=2, ub=2)
            TypeChecker.check_int_bounds(x=self._rec_locs[i][0], lb=0, ub=self._nz - 1)
            TypeChecker.check_int_bounds(x=self._rec_locs[i][1], lb=0, ub=self._n - 1)

        # Precision for calculation of Green's functions
        TypeChecker.check(x=self._params["precision"], expected_type=(str,))
        if self._params["precision"] == "float":
            self._precision = np.complex64
            self._precision_real = np.float32
        elif self._params["precision"] == "double":
            self._precision = np.complex128
            self._precision_real = np.float64
        else:
            self._precision = np.complex64
            self._precision_real = np.float32

        # Green's function related group
        self._m = int(self._params["greens func"]["m"])
        self._sigma_greens_func = float(self._params["greens func"]["sigma"])

        file_name = self._params["greens func"]["vz file path"]
        with np.load(file_name) as f:
            self._vz = np.reshape(f["arr_0"], newshape=(self._nz, 1))

        TypeChecker.check_int_positive(self._m)
        TypeChecker.check_float_positive(x=self._sigma_greens_func)
        TypeChecker.check_ndarray(x=self._vz, shape=(self._nz, 1), dtypes=(np.float32,), lb=0.1)

        # Initialize state
        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Initializing state:\n")
        if "state" in self._params.keys():
            self._state = int(self._params["state"])
        else:
            self._state = 0
            update_json(filename=self._param_file, key="state", val=self._state)

        # This checks if restart_code can be applied, and resets self._state
        self.__check_restart_mode()
        print("state = ", self._state, ", ", self.__state_code(self._state))

        # ----------------------------------------------------
        # Initialize full list of members

        self._k_values = None
        self._num_k_values = None
        self._num_sources = None
        self._true_model_pert = None

        # If self._state <= 6, the next two fields are unchanged
        self._last_iter_num = None
        self._last_iter_step = None

        self.__run_initializer()

    @property
    def n(self):
        return self._n

    @property
    def nz(self):
        return self._nz

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def scale_fac_inv(self):
        return self._scale_fac_inv

    @property
    def precision(self):
        return self._precision

    @property
    def precision_real(self):
        return self._precision_real

    @property
    def num_rec(self):
        return self._num_rec

    @property
    def rec_locs(self):
        return self._rec_locs

    @property
    def vz(self):
        return self._vz

    @property
    def m(self):
        return self._m

    @property
    def sigma_greens_func(self):
        return self._sigma_greens_func

    @property
    def k_values(self):
        return self._k_values

    @property
    def num_k_values(self):
        return self._num_k_values

    @property
    def num_sources(self):
        return self._num_sources

    @property
    def state(self):
        return self._state

    @property
    def true_model_pert(self):
        return self._true_model_pert

    @property
    def last_iter_num(self):
        return self._last_iter_num

    @property
    def last_iter_step(self):
        return self._last_iter_step

    def add_k_values(self, k_values_list):
        """
        :param k_values_list: list of floats or numpy array
            list of k-values for inversion
        :return:
        """
        if self._state != 0:
            print(
                "\nOperation not allowed. Need self._state = 0, but obtained self._state = ", self._state
            )
            return

        if type(k_values_list) is list:
            self._k_values = np.squeeze(np.array(k_values_list, dtype=np.float32))

        if type(k_values_list) is np.ndarray:
            self._k_values = np.squeeze(k_values_list).astype(np.float32)

        self._num_k_values = self._k_values.shape[0]

        path = self.__k_values_filename()
        np.savez(path, self._k_values)

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def add_sources(self, num_sources, source_list):
        """
        :param num_sources: int
            Number of sources
        :param source_list: list of numpy arrays
            List of 3D numpy arrays of shape (num_sources, self._nz, self._n).
            Length of list = self._num_k_values
        :return:
        """
        if self._state != 1:
            print(
                "\nOperation not allowed. Need self._state = 1, but obtained self._state = ", self._state
            )
            return

        # Check parameters
        TypeChecker.check_int_positive(num_sources)

        if len(source_list) != self._num_k_values:
            print(
                "Length of list required (self._num_k_values) = ", self._num_k_values,
                ", but obtained = ", len(source_list)
            )
            return

        for i in range(self._num_k_values):
            TypeChecker.check_ndarray(
                x=source_list[i],
                shape=(num_sources, self._nz, self._n),
                dtypes=(self._precision,),
                nan_inf=True
            )

        # Write to file
        for i in range(self._num_k_values):
            path = self.__source_filename(num_k=i)
            np.savez_compressed(path, source_list[i])

        self._num_sources = num_sources

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def add_sources_gaussian(self, num_sources, amplitude_list, source_coords, std_list):
        """
        Automatically create Gaussian sources

        :param num_sources: int
            Number of sources.
        :param amplitude_list: 1D numpy array of complex numbers
            List of amplitudes. Must be of shape (self._num_k_values,).
        :param source_coords: 2D numpy array of floats
            Source coordinates (centers of Gaussians). Must be of shape (num_sources, 2).
        :param std_list: 1D numpy array of floats
            Standard deviation of Gaussian. Must be of shape (num_sources,).
        :return:
        """
        if self._state != 1:
            print(
                "\nOperation not allowed. Need self._state = 1, but obtained self._state = ", self._state
            )
            return

        # Check parameters
        TypeChecker.check_int_positive(num_sources)
        TypeChecker.check_ndarray(
            x=amplitude_list,
            shape=(self._num_k_values,),
            dtypes=(self._precision,),
            nan_inf=True
        )
        TypeChecker.check_ndarray(
            x=source_coords,
            shape=(num_sources, 2),
            dtypes=(np.float32, np.float64),
            nan_inf=True
        )
        TypeChecker.check_ndarray(
            x=std_list,
            shape=(num_sources,),
            dtypes=(np.float32, np.float64),
            nan_inf=True
        )

        # Create Gaussians
        zgrid = np.linspace(start=self._a, stop=self._b, num=self._nz, endpoint=True)
        xgrid = np.linspace(start=-0.5, stop=0.5, num=self._n, endpoint=True)
        zg, xg = np.meshgrid(zgrid, xgrid, indexing="ij")

        x = np.zeros(shape=(num_sources, self._nz, self._n), dtype=np.float32)

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Creating Gaussians...")
        print("\n")
        for source_num in range(num_sources):
            coord_z = source_coords[source_num, 0]
            coord_x = source_coords[source_num, 1]
            sou = np.exp(-0.5 * ((zg - coord_z) ** 2 + (xg - coord_x) ** 2) / (std_list[source_num] ** 2))
            x[source_num, :, :] += sou

        for kk in range(self._num_k_values):
            print("Creating sources for k number " + str(kk) + ", and writing sources to disk...")
            y = x * amplitude_list[kk]
            y = y.astype(self._precision)

            path = self.__source_filename(num_k=kk)
            np.savez_compressed(path, y)

        self._num_sources = num_sources

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def calculate_greens_func(self, num_procs):
        """
        Calculate Green's functions and write to disk
        :param num_procs: int
            Number of threads for multiprocessing.
        :return:
        """

        TypeChecker.check_int_positive(x=num_procs)

        if self._state != 2:
            print(
                "\nOperation not allowed. Need self._state = 2, but obtained self._state = ", self._state
            )
            return

        # Perform calculation
        param_tuple_list = [
            (
                self._n,
                self._nz,
                self._a,
                self._b,
                self._k_values[i],
                self._vz,
                self._m,
                self._sigma_greens_func,
                self._precision,
                self.greens_func_filedir(num_k=i)
            ) for i in range(self._num_k_values)
        ]

        with Pool(min(len(param_tuple_list), mp.cpu_count(), num_procs)) as pool:
            max_ = len(param_tuple_list)

            with tqdm(total=max_) as pbar:
                for _ in pool.imap_unordered(green_func_calculate_mp_helper_func, param_tuple_list):
                    pbar.update()

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def add_true_model_pert(self, model_pert):
        """
        Add true model perturbation
        :param model_pert: 2D numpy array of floats
            True model perturbation in slowness squared about background model. Must have shape (self._nz, self._n).
        :return:
        """
        if self._state != 3:
            print(
                "\nOperation not allowed. Need self._state = 3, but obtained self._state = ", self._state
            )
            return

        # Check parameters
        TypeChecker.check_ndarray(
            x=model_pert,
            shape=(self._nz, self._n),
            dtypes=(self._precision_real,),
            nan_inf=True
        )

        # Write to file
        path = self.__true_model_pert_filename()
        np.savez_compressed(path, model_pert)
        self._true_model_pert = model_pert

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def compute_true_data(self, num_procs, max_iter=2000, tol=1e-5, verbose=True):
        """
        Compute the true data.

        :param num_procs: int
            Number of threads for multiprocessing.
        :param max_iter: int
            Maximum number of iterations for GMRES solver.
        :param tol: float
            Tolerance to use for GMRES solver.
        :param verbose: bool
            Whether to print detailed messages during GMRES.
        :return:
        """

        if self._state != 4:
            print(
                "\nOperation not allowed. Need self._state = 4, but obtained self._state = ", self._state
            )
            return

        TypeChecker.check_int_positive(x=num_procs)
        TypeChecker.check_int_positive(x=max_iter)
        TypeChecker.check_float_positive(x=tol)
        TypeChecker.check(x=verbose, expected_type=(bool,))

        # Loop over k values
        for k in range(self._num_k_values):

            print("\n\n---------------------------------------------")
            print("---------------------------------------------")
            print("Starting k number ", k)

            # Create and load Green's function into shared memory
            with SharedMemoryManager() as smm:

                # Create shared memory and load Green's function into it
                sm = smm.SharedMemory(size=self.__num_bytes_greens_func())
                data = ndarray(shape=(self._nz, self._nz, 2 * self._n - 1), dtype=self._precision, buffer=sm.buf)
                data *= 0

                green_func_filename = self.__greens_func_filename(num_k=k)
                with np.load(green_func_filename) as f:
                    data += f["arr_0"]

                # Create shared memory for computed true data
                sm1 = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
                data1 = ndarray(shape=(self._num_sources, self._nz, self._n), dtype=self._precision, buffer=sm1.buf)
                data1 *= 0

                param_tuple_list = [
                    (
                        self._n,
                        self._nz,
                        self._a,
                        self._b,
                        self._k_values[k],
                        self._vz,
                        self._m,
                        self._sigma_greens_func,
                        self._precision,
                        self.greens_func_filedir(num_k=k),
                        sm.name,
                        sm1.name,
                        i,
                        self.__source_filename(num_k=k),
                        self.__true_model_pert_filename(),
                        max_iter,
                        tol,
                        verbose
                    ) for i in range(self._num_sources)
                ]

                with Pool(min(len(param_tuple_list), mp.cpu_count(), num_procs)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(true_data_calculate_mp_helper_func, param_tuple_list):
                            pbar.update()

                # Write computed data to disk
                np.savez_compressed(self.__true_data_filename(num_k=k), data1)

            print("\n\n")

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def get_true_wavefield(self, num_k, num_source):
        """
        Get true data. Follows 0 based indexing.

        :param num_k: int
            k value number
        :param num_source: int
            Source number

        :return: np.ndarray of shape (self._nz, self._n)
            True data for k value num_k, and source number num_source.
        """
        if self._state < 5:
            print(
                "\nOperation not allowed. Need self._state >= 5, but obtained self._state = ", self._state
            )
            return

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values - 1)
        TypeChecker.check_int_bounds(x=num_source, lb=0, ub=self._num_sources - 1)

        true_data_filename = self.__true_data_filename(num_k=num_k)

        with np.load(true_data_filename) as f:
            true_data = f["arr_0"][num_source, :, :]

        return true_data

    def set_initial_pert_wavefields(self, model_pert, wavefield_list):
        # TODO
        pass

    def set_zero_initial_pert_wavefields(self, num_procs=1):
        """
        Sets zero initial perturbation and wavefields

        :param num_procs: int
            number of processors for multiprocessing while computing objective function

        :return:
        """

        if self._state != 5:
            print(
                "\nOperation not allowed. Need self._state = 5, but obtained self._state = ", self._state
            )
            return

        # Set zero initial perturbation
        model_pert = np.zeros(shape=(self._nz, self._n), dtype=self._precision_real)
        path = self.__model_pert_filename(iter_count=-1)
        np.savez_compressed(path, model_pert)

        # Set zero initial wavefields
        for k in range(self._num_k_values):
            wavefield = np.zeros(
                shape=(self._num_sources, self._nz, self._n), dtype=self._precision
            )
            path = self.__wavefield_filename(iter_count=-1, num_k=k)
            np.savez_compressed(path, wavefield)

        # Compute objective functions
        self.__compute_obj1(iter_count=-1)
        self.__compute_obj2(iter_count=-1, iter_step=1, num_procs=num_procs)

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def perform_inversion_update_wavefield(
            self, iter_count=None,
            lambda_arr=None, mu_arr=None,
            max_iter=100, solver="lsmr", atol=1e-5, btol=1e-5,
            num_procs=1, clean=False
    ):

        # ------------------------------------------------------
        # Check inputs

        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return

        if iter_count is not None:
            TypeChecker.check_int_bounds(x=iter_count, lb=0, ub=self.__get_next_iter_num()[0])
        else:
            iter_count = self.__get_next_iter_num()[0]

        TypeChecker.check_int_positive(x=max_iter)
        TypeChecker.check(x=solver, expected_type=(str,))
        if solver not in ["lsqr", "lsmr"]:
            print("solver type not supported. Only solvers available are lsqr and lsmr.")
            return
        TypeChecker.check_float_bounds(x=atol, lb=1e-5, ub=1.0)
        TypeChecker.check_float_bounds(x=btol, lb=1e-5, ub=1.0)
        TypeChecker.check_int_positive(x=num_procs)
        TypeChecker.check(x=clean, expected_type=(bool,))

        if lambda_arr is not None:
            TypeChecker.check_ndarray(
                x=lambda_arr,
                shape=(self._num_k_values, self._num_sources),
                dtypes=(np.float32,),
                nan_inf=True,
                lb=0.0
            )
        else:
            lambda_arr = np.zeros(shape=(self._num_k_values, self._num_sources), dtype=np.float32) + 1.0

        if mu_arr is not None:
            TypeChecker.check_ndarray(
                x=mu_arr,
                shape=(self._num_k_values, self._num_sources),
                dtypes=(np.float32,),
                nan_inf=True,
                lb=0.0
            )
        else:
            mu_arr = np.zeros(shape=(self._num_k_values, self._num_sources), dtype=np.float32) + 1.0

        # ------------------------------------------------------
        # Clean directories if clean is True
        # Create directories if missing

        if clean:

            # Remove all future iteration directories including iter_count directory
            list_not_remove = ["iter" + str(i) for i in range(iter_count)]
            for item in os.listdir(os.path.join(self._basedir, "iterations/")):
                path = os.path.join(self._basedir, "iterations/", item)
                if os.path.isdir(path):
                    if item not in list_not_remove:
                        try:
                            shutil.rmtree(path)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))

        path = os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/")
        if not os.path.exists(path):
            os.makedirs(path)

        # ------------------------------------------------------
        # Perform update

        rec_locs = np.asarray(self._rec_locs, dtype=int)

        print("\n\n---------------------------------------------")
        print("Updating wave field, Iteration " + str(iter_count) + "...\n")

        # Create and load Green's function into shared memory
        with SharedMemoryManager() as smm:

            # Create shared memory for Green's function
            sm_greens_func = smm.SharedMemory(size=self.__num_bytes_greens_func())
            green_func = ndarray(
                shape=(self._nz, self._nz, 2 * self._n - 1),
                dtype=self._precision,
                buffer=sm_greens_func.buf
            )

            # Create shared memory for source
            sm_source = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
            source = ndarray(
                shape=(self._num_sources, self._nz, self._n),
                dtype=self._precision,
                buffer=sm_source.buf
            )

            # Create shared memory for wavefield
            sm_wavefield = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
            wavefield = ndarray(
                shape=(self._num_sources, self._nz, self._n),
                dtype=self._precision,
                buffer=sm_wavefield.buf
            )

            # Create shared memory for true data
            sm_true_data = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
            true_data = ndarray(
                shape=(self._num_sources, self._nz, self._n),
                dtype=self._precision,
                buffer=sm_true_data.buf
            )

            # Create shared memory for initial perturbation and load it
            sm_pert = smm.SharedMemory(size=self.__num_bytes_model_pert())
            pert = ndarray(shape=(self._nz, self._n), dtype=self._precision_real, buffer=sm_pert.buf)
            pert *= 0
            model_pert_filename = self.__model_pert_filename(iter_count=iter_count - 1)
            with np.load(model_pert_filename) as f:
                pert += f["arr_0"]

            # Loop over k values
            for k in range(self._num_k_values):

                print("\n\n---------------------------------------------")
                print("---------------------------------------------")
                print("Starting k number ", k)

                # Load Green's func into shared memory
                green_func *= 0
                green_func_filename = self.__greens_func_filename(num_k=k)
                with np.load(green_func_filename) as f:
                    green_func += f["arr_0"]

                # Load source into shared memory
                source *= 0
                source_filename = self.__source_filename(num_k=k)
                with np.load(source_filename) as f:
                    source += f["arr_0"]

                # Load initial wavefield into shared memory
                wavefield *= 0
                wavefield_filename = self.__wavefield_filename(num_k=k, iter_count=iter_count - 1)
                with np.load(wavefield_filename) as f:
                    wavefield += f["arr_0"]

                # Load true data into shared memory
                true_data *= 0
                true_data_filename = self.__true_data_filename(num_k=k)
                with np.load(true_data_filename) as f:
                    true_data += f["arr_0"]

                param_tuple_list = [
                    (
                        self._n,
                        self._nz,
                        self._a,
                        self._b,
                        self._k_values[k],
                        self._vz,
                        self._m,
                        self._sigma_greens_func,
                        self._precision,
                        self._precision_real,
                        self.greens_func_filedir(num_k=k),
                        self._num_sources,
                        rec_locs,
                        i,
                        np.sqrt(lambda_arr[k, i]),
                        np.sqrt(mu_arr[k, i]),
                        sm_greens_func.name,
                        sm_source.name,
                        sm_wavefield.name,
                        sm_true_data.name,
                        sm_pert.name,
                        max_iter,
                        solver,
                        atol,
                        btol
                    ) for i in range(self._num_sources)
                ]

                with Pool(min(len(param_tuple_list), mp.cpu_count(), num_procs)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(update_wavefield, param_tuple_list):
                            pbar.update()

                # Write computed wavefield to disk
                np.savez_compressed(self.__wavefield_filename(iter_count=iter_count, num_k=k), wavefield)

        # ------------------------------------------------------
        # Write lambda_arr and mu_arr to disk

        np.savez_compressed(self.__lambda_arr_filename(iter_count=iter_count), lambda_arr)
        np.savez_compressed(self.__mu_arr_filename(iter_count=iter_count), mu_arr)

        # ------------------------------------------------------
        # Compute objective functions

        self.__compute_obj1(iter_count=iter_count)
        self.__compute_obj2(iter_count=iter_count, iter_step=0, num_procs=num_procs)

        # ------------------------------------------------------
        # Update parameter file

        if clean or self._state == 6:
            self._last_iter_num = iter_count
            self._last_iter_step = 0
            update_json(filename=self._param_file, key="last iter num", val=self._last_iter_num)
            update_json(filename=self._param_file, key="last iter step", val=self._last_iter_step)

        else:
            if iter_count > self._last_iter_num:
                self._last_iter_num = iter_count
                self._last_iter_step = 0
                update_json(filename=self._param_file, key="last iter num", val=self._last_iter_num)
                update_json(filename=self._param_file, key="last iter step", val=self._last_iter_step)

        self._state = 7
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def get_inverted_wavefield(self, iter_count, num_k, num_source):
        """
        Get inverted data. Follows 0 based indexing.

        :param num_k: int
            k value number
        :param num_source: int
            Source number

        :return: np.ndarray of shape (self._nz, self._n)
            True data for iteration number iter_num, k value num_k, and source number num_source.
        """
        wavefield_filename = self.wavefield_filename(iter_count=iter_count, num_k=num_k)
        TypeChecker.check_int_bounds(x=num_source, lb=0, ub=self._num_sources)

        with np.load(wavefield_filename) as f:
            wavefield = f["arr_0"][num_source, :, :]

        return wavefield

    def print_params(self):

        print("\n")
        print("---------------------------------------------")
        print("---------------------------------------------")
        print("Printing contents of parameter file:")

        print("\n---------------------------------------------")
        print("Geometry related parameters\n")
        for key in self._params["geometry"].keys():
            print(key, ": ", self._params["geometry"][key])

        print("\n---------------------------------------------")
        print("Receiver locations related parameters\n")
        for i, j in enumerate(self._params["rec_locs"]):
            print("Receiver " + str(i), " location : ", j)

        print("\n---------------------------------------------")
        print("Precision related parameters\n")
        print("Precision : ", self._params["precision"])

        print("\n---------------------------------------------")
        print("State related parameters\n")
        print("State : ", self._state)
        print(self.__state_code(self._state))

        print("\n---------------------------------------------")
        print("Green's functions related parameters\n")
        for key in self._params["greens func"].keys():
            print(key, ": ", self._params["greens func"][key])

        print("\n---------------------------------------------")
        print("Iteration related parameters\n")
        if self._state >= 7:
            print("Last iteration number = ", self._last_iter_num)
            print("Last iteration step = ", self._last_iter_step)

    def k_values_filename(self):
        """
        :return: str
            k values filename
        """
        if self._state < 1:
            print(
                "\nOperation not allowed. Need self._state >= 1, but obtained self._state = ", self._state
            )
            return None

        return os.path.join(self._basedir, "k-values.npz")

    def source_filename(self, num_k):
        """
        :param num_k: int
            k value number

        :return: str
            Source filename for k value num_k
        """
        if self._state < 2:
            print(
                "\nOperation not allowed. Need self._state >= 2, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values - 1)
        return os.path.join(self._basedir, "sources/" + str(num_k) + ".npz")

    def greens_func_filename(self, num_k):
        """
        :param num_k: int
            k value number

        :return: str
            Green's func filename for k value num_k
        """
        if self._state < 3:
            print(
                "\nOperation not allowed. Need self._state >= 3, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values - 1)
        return os.path.join(self._basedir, "greens_func/" + str(num_k) + "/green_func.npz")

    def greens_func_filedir(self, num_k):
        """
        :param num_k: int
            k value number

        :return: str
            Green's func filedir for k value num_k
        """
        if self._state < 3:
            print(
                "\nOperation not allowed. Need self._state >= 3, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values - 1)
        return os.path.join(self._basedir, "greens_func/" + str(num_k))

    def true_model_pert_filename(self):
        """
        :return: str
            True model filename
        """
        if self._state < 4:
            print(
                "\nOperation not allowed. Need self._state >= 4, but obtained self._state = ", self._state
            )
            return None

        return os.path.join(self._basedir, "data/true_model_pert.npz")

    def true_data_filename(self, num_k):
        """
        :param num_k: int
            k value number

        :return: str
            True data filename for k value num_k
        """
        if self._state < 5:
            print(
                "\nOperation not allowed. Need self._state >= 5, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values-1)
        return os.path.join(self._basedir, "data/" + str(num_k) + ".npz")

    def model_pert_filename(self, iter_count):
        """
        :param iter_count: int
            Iteration number

        :return: str
            Model pert  filename after iteration
        """
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return None

        if self._state == 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=-1)

        if self._state > 6:
            last_iter_num_valid = self._last_iter_num - 1
            if self._last_iter_step == 1:
                last_iter_num_valid += 1
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=last_iter_num_valid)

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_model_pert.npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/model_pert_update.npz")

    def wavefield_filename(self, iter_count, num_k):
        """
        :param iter_count: int
            Iteration number
        :param num_k: int
            k value number

        :return: str
            Wave field filename after iteration
        """
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return None

        if self._state == 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=-1)

        if self._state > 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=self._last_iter_num)

        TypeChecker.check_int_bounds(x=num_k, lb=0, ub=self._num_k_values - 1)

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_wavefield_" + str(num_k) + ".npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/wavefield_" + str(num_k) + ".npz")

    def lambda_arr_filename(self, iter_count):
        """
        :param iter_count: int
            Iteration number

        :return: str
            Lambda array filename for iteration
        """
        if self._state < 7:
            print(
                "\nOperation not allowed. Need self._state >= 7, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=iter_count, lb=0, ub=self._last_iter_num)
        return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/lambda_arr.npz")

    def mu_arr_filename(self, iter_count):
        """
        :param iter_count: int
            Iteration number

        :return: str
            Mu array filename for iteration
        """
        if self._state < 7:
            print(
                "\nOperation not allowed. Need self._state >= 7, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=iter_count, lb=0, ub=self._last_iter_num)
        return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/mu_arr.npz")

    def obj1_filename(self, iter_count):
        """
        :param iter_count: int
            Iteration number

        :return: str
            Obj1 array filename for iteration
        """
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return None

        if self._state == 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=-1)
            return os.path.join(self._basedir, "iterations/initial_obj1_arr.npz")

        if self._state > 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=self._last_iter_num)
            if iter_count == -1:
                return os.path.join(self._basedir, "iterations/initial_obj1_arr.npz")
            else:
                return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj1_arr.npz")

    def obj2_filename(self, iter_count, iter_step):
        """
        :param iter_count: int
            Iteration number
        :param iter_step: int
            Iteration step

        :return: str
            Obj1 array filename for iteration
        """
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return None

        TypeChecker.check_int_bounds(x=iter_step, lb=0, ub=1)

        if self._state == 6:
            TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=-1)
            return os.path.join(self._basedir, "iterations/initial_obj2_arr.npz")

        if self._state > 6:

            if iter_step == 0:
                TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=self._last_iter_num)

                if iter_count == -1:
                    return os.path.join(self._basedir, "iterations/initial_obj2_arr.npz")
                else:
                    return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj2_step0_arr.npz")

            else:
                last_iter_num_valid = self._last_iter_num - 1
                if self._last_iter_step == 1:
                    last_iter_num_valid += 1
                TypeChecker.check_int_bounds(x=iter_count, lb=-1, ub=last_iter_num_valid)

                if iter_count == -1:
                    return os.path.join(self._basedir, "iterations/initial_obj2_arr.npz")
                else:
                    return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj2_step1_arr.npz")

    def __compute_obj1(self, iter_count):

        print("\n\n---------------------------------------------")
        print("Computing data residual...\n")

        obj1 = np.zeros(shape=(self._num_k_values, self._num_sources), dtype=np.float64)
        true_data = np.zeros(shape=(self._num_sources, self._nz, self._n), dtype=self._precision)
        iter_data = ndarray(shape=(self._num_sources, self._nz, self._n), dtype=self._precision)

        rec_locs = np.asarray(self._rec_locs, dtype=int)

        for k in numba.prange(self._num_k_values):

            true_data *= 0
            iter_data *= 0

            # Load true data file
            with np.load(self.__true_data_filename(num_k=k)) as f:
                true_data += f["arr_0"]

            # Load iteration data file
            with np.load(self.__wavefield_filename(iter_count=iter_count, num_k=k)) as f:
                iter_data += f["arr_0"]

            for num_src in range(self._num_sources):

                true_data_src = true_data[num_src, :, :]
                iter_data_src = iter_data[num_src, :, :]

                true_data_rcv = true_data_src[rec_locs[:, 0], rec_locs[:, 1]]
                iter_data_rcv = iter_data_src[rec_locs[:, 0], rec_locs[:, 1]]

                obj1[k, num_src] = np.linalg.norm(true_data_rcv - iter_data_rcv) ** 2.0

        # Write computed data to disk
        np.savez_compressed(self.__obj1_filename(iter_count=iter_count), obj1)

    def __compute_obj2(self, iter_count, iter_step, num_procs=1):

        print("\n\n---------------------------------------------")
        print("Computing LSE residual...\n")

        num_bytes_obj2_arr = self._num_k_values * self._num_sources * 8

        # Create and load Green's function into shared memory
        with SharedMemoryManager() as smm:

            # Create shared memory for obj2
            sm = smm.SharedMemory(size=num_bytes_obj2_arr)
            obj2 = ndarray(shape=(self._num_k_values, self._num_sources), dtype=np.float64, buffer=sm.buf)
            obj2 *= 0

            # Create shared memory for Green's function
            sm1 = smm.SharedMemory(size=self.__num_bytes_greens_func())
            green_func = ndarray(shape=(self._nz, self._nz, 2 * self._n - 1), dtype=self._precision, buffer=sm1.buf)

            # Create shared memory for source
            sm2 = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
            source = ndarray(shape=(self._num_sources, self._nz, self._n), dtype=self._precision, buffer=sm2.buf)

            # Create shared memory for wavefield
            sm3 = smm.SharedMemory(size=self.__num_bytes_true_data_per_k())
            wavefield = ndarray(shape=(self._num_sources, self._nz, self._n), dtype=self._precision, buffer=sm3.buf)

            # Create shared memory for perturbation and load it
            sm4 = smm.SharedMemory(size=self.__num_bytes_model_pert())
            pert = ndarray(shape=(self._nz, self._n), dtype=self._precision_real, buffer=sm4.buf)
            pert *= 0

            if iter_count == -1:
                model_pert_filename = self.__model_pert_filename(iter_count=iter_count)
            else:
                if iter_step == 1:
                    model_pert_filename = self.__model_pert_filename(iter_count=iter_count)
                else:
                    model_pert_filename = self.__model_pert_filename(iter_count=iter_count-1)

            with np.load(model_pert_filename) as f:
                pert += f["arr_0"]

            # Loop over k values
            for k in range(self._num_k_values):

                print("\n\n---------------------------------------------")
                print("---------------------------------------------")
                print("Starting k number ", k)

                # Load Green's func into shared memory
                green_func *= 0
                green_func_filename = self.__greens_func_filename(num_k=k)
                with np.load(green_func_filename) as f:
                    green_func += f["arr_0"]

                # Load source into shared memory
                source *= 0
                source_filename = self.__source_filename(num_k=k)
                with np.load(source_filename) as f:
                    source += f["arr_0"]

                # Load wavefield into shared memory
                wavefield *= 0
                wavefield_filename = self.__wavefield_filename(num_k=k, iter_count=iter_count)
                with np.load(wavefield_filename) as f:
                    wavefield += f["arr_0"]

                param_tuple_list = [
                    (
                        self._n,
                        self._nz,
                        self._a,
                        self._b,
                        self._k_values[k],
                        self._vz,
                        self._m,
                        self._sigma_greens_func,
                        self._precision,
                        self._precision_real,
                        self.greens_func_filedir(num_k=k),
                        self._num_k_values,
                        self._num_sources,
                        sm.name,
                        sm1.name,
                        sm2.name,
                        sm3.name,
                        sm4.name,
                        i,
                        k
                    ) for i in range(self._num_sources)
                ]

                with Pool(min(len(param_tuple_list), mp.cpu_count(), num_procs)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(compute_obj2, param_tuple_list):
                            pbar.update()

            # Write computed data to disk
            np.savez_compressed(self.__obj2_filename(iter_count=iter_count, iter_step=iter_step), obj2)

    def __get_next_iter_num(self):
        """
        Next iteration to run
        :return: int, int
            Next iteration num, Next iteration step
        """
        if self._state == 6:
            return 0, 0

        elif self._state == 7:
            if self._last_iter_step == 0:
                return self._last_iter_num, 1
            if self._last_iter_step == 1:
                return self._last_iter_num + 1, 0

    def __k_values_filename(self):
        return os.path.join(self._basedir, "k-values.npz")

    def __source_filename(self, num_k):
        return os.path.join(self._basedir, "sources/" + str(num_k) + ".npz")

    def __greens_func_filename(self, num_k):
        return os.path.join(self._basedir, "greens_func/" + str(num_k) + "/green_func.npz")

    def __greens_func_filedir(self, num_k):
        return os.path.join(self._basedir, "greens_func/" + str(num_k))

    def __true_model_pert_filename(self):
        return os.path.join(self._basedir, "data/true_model_pert.npz")

    def __true_data_filename(self, num_k):
        return os.path.join(self._basedir, "data/" + str(num_k) + ".npz")

    def __model_pert_filename(self, iter_count):
        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_model_pert.npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/model_pert_update.npz")

    def __wavefield_filename(self, iter_count, num_k):
        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_wavefield_" + str(num_k) + ".npz")
        else:
            return os.path.join(
                self._basedir, "iterations/iter" + str(iter_count) + "/wavefield_" + str(num_k) + ".npz"
            )

    def __lambda_arr_filename(self, iter_count):
        return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/lambda_arr.npz")

    def __mu_arr_filename(self, iter_count):
        return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/mu_arr.npz")

    def __obj1_filename(self, iter_count):
        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_obj1_arr.npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj1_arr.npz")

    def __obj2_filename(self, iter_count, iter_step):

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_obj2_arr.npz")
        else:
            if iter_step == 0:
                return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj2_step0_arr.npz")
            if iter_step == 1:
                return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/obj2_step1_arr.npz")

    def __num_bytes_greens_func(self):

        # Calculate num bytes for Green's function
        num_bytes = self._nz * self._nz * (2 * self._n - 1)
        if self._precision == np.complex64:
            num_bytes *= 8
        if self._precision == np.complex128:
            num_bytes *= 16

        return num_bytes

    def __num_bytes_true_data_per_k(self):

        # Calculate num bytes for Green's function
        num_bytes = self._num_sources * self._nz * self._n
        if self._precision == np.complex64:
            num_bytes *= 8
        if self._precision == np.complex128:
            num_bytes *= 16

        return num_bytes

    def __num_bytes_model_pert(self):

        # Calculate num bytes for model perturbation
        num_bytes = self._nz * self._n
        if self._precision_real == np.float32:
            num_bytes *= 4
        if self._precision_real == np.float64:
            num_bytes *= 8

        return num_bytes

    def __print_reset_state_msg(self):

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Resetting state:\n")
        print("state = ", self._state, ", ", self.__state_code(self._state))

    def __check_restart_mode(self):

        if self._restart is True:
            if self._restart_code is None:
                self._restart_code = self._state
            else:
                if self._restart_code > self._state:
                    raise ValueError(
                        "restart_code: " + str(self._restart_code) +
                        ", retrieved state: " + str(self._state) +
                        ". Does not satisfy restart_code <= retrieved state."
                    )
                else:
                    self._state = self._restart_code

        else:
            self._state = 0

        # Update .json file
        update_json(filename=self._param_file, key="state", val=self._state)

    def __run_initializer(self):

        # ----------------------------------------------------
        # Assume restart = False
        if self._restart is False:

            # Clean based on restart_code
            self.__clean(self._state)

            # Create directories
            dir_list = [
                "greens_func/",
                "iterations/",
                "sources/",
                "data/"
            ]
            for item in dir_list:
                path = self._basedir + item
                if not os.path.exists(path):
                    os.makedirs(path)
                elif len(os.listdir(path)) > 0:
                    raise ValueError(
                        "restart=False not allowed as " + os.path.abspath(path) + " is not empty."
                    )

        # Assume restart = True
        elif self._restart is True:

            print("\n\n---------------------------------------------")
            print("---------------------------------------------")
            print("Initializing class members:\n")

            # Clean based on self._state
            self.__clean(self._state)

            # Initialize class members
            if self._state >= 1:

                path = self.__k_values_filename()
                self._k_values = np.load(path)["arr_0"]
                self._num_k_values = self._k_values.shape[0]

                print("Checking k values array: OK")

            if self._state >= 2:

                path = self.__source_filename(num_k=0)
                x = np.load(path)["arr_0"]
                num_sources = x.shape[0]

                for i in range(self._num_k_values):
                    path = self.__source_filename(num_k=i)
                    x = np.load(path)["arr_0"]
                    TypeChecker.check_ndarray(
                        x=x,
                        shape=(num_sources, self._nz, self._n),
                        dtypes=(self._precision,),
                        nan_inf=True
                    )

                self._num_sources = num_sources

                print("Checking sources: OK")

            if self._state >= 3:

                for i in range(self._num_k_values):
                    path = self.__greens_func_filename(num_k=i)
                    x = np.load(path)["arr_0"]
                    if x.shape != (self._nz, self._nz, 2 * (self._n - 1) + 1):
                        raise ValueError("Green's function shape does not match.")

                print("Checking Green's functions shapes: OK")

            if self._state >= 4:

                path = self.__true_model_pert_filename()
                true_model_pert = np.load(path)["arr_0"]

                TypeChecker.check_ndarray(
                    x=true_model_pert,
                    shape=(self._nz, self._n),
                    dtypes=(self._precision_real,),
                    nan_inf=True
                )

                self._true_model_pert = true_model_pert

                print("Checking true model perturbation: OK")

            if self._state >= 5:

                for i in range(self._num_k_values):

                    path = self.__true_data_filename(num_k=i)
                    with np.load(path) as f:
                        true_data_k = f["arr_0"]

                    TypeChecker.check_ndarray(
                        x=true_data_k,
                        shape=(self._num_sources, self._nz, self._n),
                        dtypes=(self._precision,),
                        nan_inf=True
                    )

                print("Checking true computed data: OK")

            if self._state >= 6:

                # Check initial model pert
                path = self.__model_pert_filename(iter_count=-1)
                model_pert = np.load(path)["arr_0"]

                TypeChecker.check_ndarray(
                    x=model_pert,
                    shape=(self._nz, self._n),
                    dtypes=(self._precision_real,),
                    nan_inf=True
                )

                # Check initial wavefields
                for i in range(self._num_k_values):

                    path = self.__wavefield_filename(iter_count=-1, num_k=i)
                    with np.load(path) as f:
                        wavefield_k = f["arr_0"]

                    TypeChecker.check_ndarray(
                        x=wavefield_k,
                        shape=(self._num_sources, self._nz, self._n),
                        dtypes=(self._precision,),
                        nan_inf=True
                    )

                print("Checking initial model pert and wavefields: OK")

            if self._state >= 7:

                self._last_iter_num = int(self._params["last iter num"])
                self._last_iter_step = int(self._params["last iter step"])

                if self._last_iter_step == 1:

                    for iter_num in range(self._last_iter_num + 1):

                        # Check model pert
                        path = self.__model_pert_filename(iter_count=iter_num)
                        model_pert = np.load(path)["arr_0"]

                        TypeChecker.check_ndarray(
                            x=model_pert,
                            shape=(self._nz, self._n),
                            dtypes=(self._precision_real,),
                            nan_inf=True
                        )

                        # Check wavefields
                        for i in range(self._num_k_values):
                            path = self.__wavefield_filename(iter_count=iter_num, num_k=i)
                            with np.load(path) as f:
                                wavefield_k = f["arr_0"]

                            TypeChecker.check_ndarray(
                                x=wavefield_k,
                                shape=(self._num_sources, self._nz, self._n),
                                dtypes=(self._precision,),
                                nan_inf=True
                            )

                        print("Checking Iter" + str(iter_num) + " model pert and wavefields: OK")

                else:

                    for iter_num in range(self._last_iter_num):

                        # Check model pert
                        path = self.__model_pert_filename(iter_count=iter_num)
                        model_pert = np.load(path)["arr_0"]

                        TypeChecker.check_ndarray(
                            x=model_pert,
                            shape=(self._nz, self._n),
                            dtypes=(self._precision_real,),
                            nan_inf=True
                        )

                        # Check wavefields
                        for i in range(self._num_k_values):
                            path = self.__wavefield_filename(iter_count=iter_num, num_k=i)
                            with np.load(path) as f:
                                wavefield_k = f["arr_0"]

                            TypeChecker.check_ndarray(
                                x=wavefield_k,
                                shape=(self._num_sources, self._nz, self._n),
                                dtypes=(self._precision,),
                                nan_inf=True
                            )

                        print("Checking Iteration " + str(iter_num) + " model pert and wavefields: OK")

                    # Check wavefields
                    for i in range(self._num_k_values):
                        path = self.__wavefield_filename(iter_count=self._last_iter_num, num_k=i)
                        with np.load(path) as f:
                            wavefield_k = f["arr_0"]

                        TypeChecker.check_ndarray(
                            x=wavefield_k,
                            shape=(self._num_sources, self._nz, self._n),
                            dtypes=(self._precision,),
                            nan_inf=True
                        )

                    print("Checking Iteration " + str(self._last_iter_num) + " wavefields: OK")

    def __clean(self, state):

        dir_list = []
        if state in [0, 1]:
            dir_list = [
                "sources/",
                "greens_func/",
                "data/",
                "iterations/"
            ]
        if state == 2:
            dir_list = [
                "greens_func/",
                "data/",
                "iterations/"
            ]
        if state == 3:
            dir_list = [
                "data/",
                "iterations/"
            ]
        if state in [4, 5]:
            dir_list = [
                "iterations/"
            ]

        # Delete all contents in directories
        for item in dir_list:
            path = self._basedir + item
            try:
                shutil.rmtree(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        # Recreate deleted directories as empty directories
        for item in dir_list:
            path = self._basedir + item
            os.makedirs(path)

        if state == 0:
            path = self._basedir + "k-values.npz"
            try:
                os.remove(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    @staticmethod
    def __state_code(state):

        if state == 0:
            return "Empty object. Nothing created."
        elif state == 1:
            return "k values initialized."
        elif state == 2:
            return "Sources created."
        elif state == 3:
            return "Green's functions created."
        elif state == 4:
            return "True model perturbation set."
        elif state == 5:
            return "True data computed."
        elif state == 6:
            return "Initial perturbation and wave fields set."
        elif state == 7:
            return "Inversion performed."
