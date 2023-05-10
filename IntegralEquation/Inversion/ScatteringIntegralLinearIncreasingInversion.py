import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import shutil
import json
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d
from ...Utilities.JsonTools import update_json
from ...Utilities import TypeChecker


class ScatteringIntegralLinearIncreasingInversion2d:

    def __init__(self, basedir, restart=False, restart_code=None):
        """
        Always read parameters from .json file.
        Name of parameter file is basedir + "params.json"

        :param basedir: str
            Base directory for storing all results
        :param restart: bool
            Whether to run the class initialization in restart mode or not
        :param restart_code: int
            If restart is True, then restart_code decides from where to restart
        """

        self._basedir = basedir
        self._restart = restart

        self._restart_code = None
        if self._restart is True:
            self._restart_code = restart_code

        # Read parameters from .json file
        self._param_file = basedir + "params.json"
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

        # Precision for calculation of Green's functions
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
        self._m = int(self._params["green's func"]["m"])

        # Initialize state
        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Initializing state:\n")
        if "state" in self._params.keys():
            self._state = int(self._params["state"])
        else:
            self._state = 0
            update_json(filename=self._param_file, key="state", val=self._state)

        # This call checks if restart_code can be applied, and resets self._state
        self.__check_restart_mode()
        print("state = ", self._state, ", ", self.__state_code(self._state))

        # ----------------------------------------------------
        # Initialize full list of members

        self._k_values = None
        self._num_k_values = None

        self._num_sources = None
        self._source_list = None

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
    def precision(self):
        return self._precision

    @property
    def precision_real(self):
        return self._precision_real

    @property
    def m(self):
        return self._m

    @property
    def k_values(self):
        return self._k_values

    @property
    def num_k_values(self):
        return self._num_k_values

    @property
    def source_list(self):
        return self._source_list

    @property
    def num_sources(self):
        return self._num_sources

    @property
    def state(self):
        return self._state

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
            self._k_values = np.squeeze(k_values_list, dtype=np.float32)

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
            path = self.__source_filename(i=i)
            np.savez(path, source_list[i])

        self._num_sources = num_sources
        self._source_list = source_list

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
            shape=(num_sources, 2),
            dtypes=(np.float32, np.float64),
            nan_inf=True
        )

        # Create Gaussians
        source_list = []
        for i in range(self._num_k_values):
            # TODO
            x = np.zeros(shape=(num_sources, self._nz, self._n), dtype=self._precision)
            source_list.append(x)

        # Write to file
        for i in range(self._num_k_values):
            path = self.__source_filename(i=i)
            np.savez(path, source_list[i])

        self._num_sources = num_sources
        self._source_list = source_list

        self._state += 1
        update_json(filename=self._param_file, key="state", val=self._state)
        self.__print_reset_state_msg()

    def calculate_greens_func(self):
        """
        Calculate Green's functions and write to disk
        :return:
        """
        if self._state != 2:
            print(
                "\nOperation not allowed. Need self._state = 2, but obtained self._state = ", self._state
            )
            return

        linvel2d = TruncatedKernelLinearIncreasingVel2d(
            n=self._n,
            nz=self._nz,
            k=100,
            a=self._a,
            b=self._b,
            m=self._m,
            precision=self._precision,
            light_mode=True
        )

        # Perform calculation
        for i in range(self._num_k_values):

            # Print message
            print("\n\n---------------------------------------------")
            print("---------------------------------------------")
            print("Starting Green's function calculation for k = " + "{:6.3f}".format(self._k_values[i]) + "\n")

            linvel2d.set_parameters(
                n=self._n,
                nz=self._nz,
                k=float(self._k_values[i]),
                a=self._a,
                b=self._b,
                m=self._m,
                precision=self._precision,
                green_func=None
            )

            path = self.__greens_func_filename(i=i)
            linvel2d.write_green_func(green_func_file=path)

            self._state += 1
            update_json(filename=self._param_file, key="state", val=self._state)
            self.__print_reset_state_msg()

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
        print("Precision related parameters\n")
        print("Precision : ", self._params["precision"])

        print("\n---------------------------------------------")
        print("State related parameters\n")
        print("State : ", self._state)
        print(self.__state_code(self._state))

        print("\n---------------------------------------------")
        print("Green's functions related parameters\n")
        print("m: ", self._m)

    def __k_values_filename(self):
        return self._basedir + "k-values.npz"

    def __source_filename(self, i):
        return self._basedir + "sources/" + str(i) + ".npz"

    def __greens_func_filename(self, i):
        return self._basedir + "greens_func/" + str(i) + ".npz"

    def __state_code(self, state):

        if state == 0:
            return "Empty object. Nothing created."
        elif state == 1:
            return "k values initialized."
        elif state == 2:
            return "Sources created."
        elif state == 3:
            return "Green's functions created."
        elif state == 4:
            return "True data computed."
        elif state == 5:
            return "Initial perturbation and wave fields set."
        elif state == 6:
            return "Inversion started."

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

        for item in dir_list:
            path = self._basedir + item
            try:
                shutil.rmtree(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if state == 0:
            path = self._basedir + "k-values.npz"
            try:
                os.remove(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    def __print_reset_state_msg(self):

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Resetting state:\n")
        print("state = ", self._state, ", ", self.__state_code(self._state))

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

                path = self.__source_filename(i=0)
                x = np.load(path)["arr_0"]
                num_sources = x.shape[0]

                self._source_list = []
                for i in range(self._num_k_values):
                    path = self.__source_filename(i=i)
                    x = np.load(path)["arr_0"]
                    TypeChecker.check_ndarray(
                        x=x,
                        shape=(num_sources, self._nz, self._n),
                        dtypes=(self._precision,),
                        nan_inf=True
                    )
                    self._source_list.append(x)

                self._num_sources = num_sources

                print("Checking sources: OK")

            if self._state >= 3:

                for i in range(self._num_k_values):
                    path = self.__greens_func_filename(i=i)
                    x = np.load(path)["arr_0"]
                    if x.shape != (self._nz, self._nz, 2 * (self._n - 1) + 1):
                        raise ValueError("Green's function shape does not match.")

                print("Checking Green's functions shpes: OK")

            if self._state >= 4:
                pass

            if self._state >= 5:
                pass

            if self._state >= 6:
                pass


if __name__ == "__main__":

    basedir_ = "InversionLS/Expt/test0/"
    obj = ScatteringIntegralLinearIncreasingInversion2d(
        basedir=basedir_,
        restart=True,
        restart_code=None
    )
    obj.print_params()

    k_vals_list_ = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    obj.add_k_values(k_values_list=k_vals_list_)
    print(obj.k_values)
