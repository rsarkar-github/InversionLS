import numpy as np
from numpy import ndarray
import scipy as sp
from scipy.sparse.linalg import LinearOperator, gmres, lsqr, lsmr
import matplotlib.pyplot as plt
import os
import sys
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

        # If self._state == 6, the next field is unchanged
        self._curr_iter_num = None

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
    def m(self):
        return self._m

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
    def curr_iter_num(self):
        return self._curr_iter_num

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

        path = self.k_values_filename()
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
            path = self.source_filename(i=i)
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

            path = self.source_filename(i=kk)
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
                self.greens_func_filedir(i=i)
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
        path = self.true_model_pert_filename()
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

                green_func_filename = self.greens_func_filename(i=k)
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
                        self.greens_func_filedir(i=k),
                        sm.name,
                        sm1.name,
                        i,
                        self.source_filename(i=k),
                        self.true_model_pert_filename(),
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
                np.savez_compressed(self.true_data_filename(i=k), data1)

            print("\n\n")

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def get_true_data(self, num_k, num_source):
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

        true_data_filename = self.true_data_filename(i=num_k)

        with np.load(true_data_filename) as f:
            true_data = f["arr_0"][num_source, :, :]

        return true_data

    def set_initial_pert_wavefields(self, model_pert, wavefield_list):
        # TODO
        pass

    def set_zero_initial_pert_wavefields(self):
        """
        Sets zero initial perturbation
        :return:
        """

        if self._state != 5:
            print(
                "\nOperation not allowed. Need self._state = 5, but obtained self._state = ", self._state
            )
            return

        # Set zero initial perturbation
        model_pert = np.zeros(shape=(self._nz, self._n), dtype=self._precision_real)
        path = self.model_pert_filename(iter_count=-1)
        np.savez_compressed(path, model_pert)

        # Set zero initial wavefields
        for k in range(self._num_k_values):
            wavefield = np.zeros(
                shape=(self._num_sources, self._nz, self._n), dtype=self._precision
            )
            path = self.wavefield_filename(iter_count=-1, i=k)
            np.savez_compressed(path, wavefield)

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def perform_inversion_update_wavefield(
            self, iter_num=None,
            lambda_arr=None, mu_arr=None,
            max_iter=100, solver="lsmr", tol=1e-5
    ):
        # TODO: fix this
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return

        TypeChecker.check_int_positive(x=max_iter)
        TypeChecker.check(x=solver, expected_type=(str,))
        if solver not in ["lsqr", "lsmr"]:
            print("solver type not supported. Only solvers available are lsqr and lsmr.")
            return

        if iter_num is None:

            next_iter_num, step_num = self.__get_next_iter_num()
            if step_num == 2:
                print("Iteration already performed.")
                return

            if not self.__check_input_file_availability(next_iter_num, step_num):
                print("Input files not available. Cannot proceed.")
                return

            # for k in range(self._num_k_values):



        # TODO
        else:
            pass

    def print_params(self):

        # TODO: Update as class develops

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

    def k_values_filename(self):
        return os.path.join(self._basedir, "k-values.npz")

    def source_filename(self, i):
        return os.path.join(self._basedir, "sources/" + str(i) + ".npz")

    def greens_func_filename(self, i):
        return os.path.join(self._basedir, "greens_func/" + str(i) + "/green_func.npz")

    def greens_func_filedir(self, i):
        return os.path.join(self._basedir, "greens_func/" + str(i))

    def true_model_pert_filename(self):
        return os.path.join(self._basedir, "data/true_model_pert.npz")

    def true_data_filename(self, i):
        return os.path.join(self._basedir, "data/" + str(i) + ".npz")

    def model_pert_filename(self, iter_count):

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_model_pert.npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/model_pert_update.npz")

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
        # TODO: Needs to be updated as class progresses
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

                path = self.k_values_filename()
                self._k_values = np.load(path)["arr_0"]
                self._num_k_values = self._k_values.shape[0]

                print("Checking k values array: OK")

            if self._state >= 2:

                path = self.source_filename(i=0)
                x = np.load(path)["arr_0"]
                num_sources = x.shape[0]

                for i in range(self._num_k_values):
                    path = self.source_filename(i=i)
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
                    path = self.greens_func_filename(i=i)
                    x = np.load(path)["arr_0"]
                    if x.shape != (self._nz, self._nz, 2 * (self._n - 1) + 1):
                        raise ValueError("Green's function shape does not match.")

                print("Checking Green's functions shapes: OK")

            if self._state >= 4:

                path = self.true_model_pert_filename()
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

                    path = self.true_data_filename(i=i)
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
                path = self.model_pert_filename(iter_count=-1)
                model_pert = np.load(path)["arr_0"]

                TypeChecker.check_ndarray(
                    x=model_pert,
                    shape=(self._nz, self._n),
                    dtypes=(self._precision_real,),
                    nan_inf=True
                )

                # Check initial wavefields
                for i in range(self._num_k_values):

                    path = self.wavefield_filename(iter_count=-1, i=i)
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
                pass

    def wavefield_filename(self, iter_count, i):

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_wavefield_" + str(i) + ".npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/wavefield_" + str(i) + ".npz")

    def __get_next_iter_num(self):
        """
        Next iteration to run
        :return: (int)
            Next iteration num
        """
        if self._state == 6:
            return 0

        elif self._state == 7:
            # TODO: fix this
            return self._curr_iter_num

    def __check_input_file_availability(self, next_iter_num, step_num):
        # TODO
        pass

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
            return "Inversion started."
import numpy as np
from numpy import ndarray
import scipy as sp
from scipy.sparse.linalg import LinearOperator, gmres, lsqr, lsmr
import matplotlib.pyplot as plt
import os
import sys
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

        # If self._state == 6, the next field is unchanged
        self._curr_iter_num = None

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
    def m(self):
        return self._m

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
    def curr_iter_num(self):
        return self._curr_iter_num

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

        path = self.k_values_filename()
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
            path = self.source_filename(i=i)
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

            path = self.source_filename(i=kk)
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
                self.greens_func_filedir(i=i)
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
        path = self.true_model_pert_filename()
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

                green_func_filename = self.greens_func_filename(i=k)
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
                        self.greens_func_filedir(i=k),
                        sm.name,
                        sm1.name,
                        i,
                        self.source_filename(i=k),
                        self.true_model_pert_filename(),
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
                np.savez_compressed(self.true_data_filename(i=k), data1)

            print("\n\n")

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def get_true_data(self, num_k, num_source):
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

        true_data_filename = self.true_data_filename(i=num_k)

        with np.load(true_data_filename) as f:
            true_data = f["arr_0"][num_source, :, :]

        return true_data

    def set_initial_pert_wavefields(self, model_pert, wavefield_list):
        # TODO
        pass

    def set_zero_initial_pert_wavefields(self):
        """
        Sets zero initial perturbation
        :return:
        """

        if self._state != 5:
            print(
                "\nOperation not allowed. Need self._state = 5, but obtained self._state = ", self._state
            )
            return

        # Set zero initial perturbation
        model_pert = np.zeros(shape=(self._nz, self._n), dtype=self._precision_real)
        path = self.model_pert_filename(iter_count=-1)
        np.savez_compressed(path, model_pert)

        # Set zero initial wavefields
        for k in range(self._num_k_values):
            wavefield = np.zeros(
                shape=(self._num_sources, self._nz, self._n), dtype=self._precision
            )
            path = self.wavefield_filename(iter_count=-1, i=k)
            np.savez_compressed(path, wavefield)

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def perform_inversion_update_wavefield(
            self, iter_num=None,
            lambda_arr=None, mu_arr=None,
            max_iter=100, solver="lsmr", tol=1e-5
    ):
        # TODO: fix this
        if self._state < 6:
            print(
                "\nOperation not allowed. Need self._state >= 6, but obtained self._state = ", self._state
            )
            return

        TypeChecker.check_int_positive(x=max_iter)
        TypeChecker.check(x=solver, expected_type=(str,))
        if solver not in ["lsqr", "lsmr"]:
            print("solver type not supported. Only solvers available are lsqr and lsmr.")
            return

        if iter_num is None:

            next_iter_num, step_num = self.__get_next_iter_num()
            if step_num == 2:
                print("Iteration already performed.")
                return

            if not self.__check_input_file_availability(next_iter_num, step_num):
                print("Input files not available. Cannot proceed.")
                return

            # for k in range(self._num_k_values):



        # TODO
        else:
            pass

    def print_params(self):

        # TODO: Update as class develops

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

    def k_values_filename(self):
        return os.path.join(self._basedir, "k-values.npz")

    def source_filename(self, i):
        return os.path.join(self._basedir, "sources/" + str(i) + ".npz")

    def greens_func_filename(self, i):
        return os.path.join(self._basedir, "greens_func/" + str(i) + "/green_func.npz")

    def greens_func_filedir(self, i):
        return os.path.join(self._basedir, "greens_func/" + str(i))

    def true_model_pert_filename(self):
        return os.path.join(self._basedir, "data/true_model_pert.npz")

    def true_data_filename(self, i):
        return os.path.join(self._basedir, "data/" + str(i) + ".npz")

    def model_pert_filename(self, iter_count):

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_model_pert.npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/model_pert_update.npz")

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
        # TODO: Needs to be updated as class progresses
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

                path = self.k_values_filename()
                self._k_values = np.load(path)["arr_0"]
                self._num_k_values = self._k_values.shape[0]

                print("Checking k values array: OK")

            if self._state >= 2:

                path = self.source_filename(i=0)
                x = np.load(path)["arr_0"]
                num_sources = x.shape[0]

                for i in range(self._num_k_values):
                    path = self.source_filename(i=i)
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
                    path = self.greens_func_filename(i=i)
                    x = np.load(path)["arr_0"]
                    if x.shape != (self._nz, self._nz, 2 * (self._n - 1) + 1):
                        raise ValueError("Green's function shape does not match.")

                print("Checking Green's functions shapes: OK")

            if self._state >= 4:

                path = self.true_model_pert_filename()
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

                    path = self.true_data_filename(i=i)
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
                path = self.model_pert_filename(iter_count=-1)
                model_pert = np.load(path)["arr_0"]

                TypeChecker.check_ndarray(
                    x=model_pert,
                    shape=(self._nz, self._n),
                    dtypes=(self._precision_real,),
                    nan_inf=True
                )

                # Check initial wavefields
                for i in range(self._num_k_values):

                    path = self.wavefield_filename(iter_count=-1, i=i)
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
                pass

    def wavefield_filename(self, iter_count, i):

        if iter_count == -1:
            return os.path.join(self._basedir, "iterations/initial_wavefield_" + str(i) + ".npz")
        else:
            return os.path.join(self._basedir, "iterations/iter" + str(iter_count) + "/wavefield_" + str(i) + ".npz")

    def __get_next_iter_num(self):
        """
        Next iteration to run
        :return: (int)
            Next iteration num
        """
        if self._state == 6:
            return 0

        elif self._state == 7:
            # TODO: fix this
            return self._curr_iter_num

    def __check_input_file_availability(self, next_iter_num, step_num):
        # TODO
        pass

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
            return "Inversion started."
