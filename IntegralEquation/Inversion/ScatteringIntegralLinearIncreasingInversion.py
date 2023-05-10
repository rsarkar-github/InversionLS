import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import shutil
import json
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d
from ...Utilities.JsonTools import write_json, update_json
from ...Utilities import TypeChecker


class ScatteringIntegralLinearIncreasingInversion2d:

    def __init__(self, basedir, restart=False, restart_code=None):
        """
        If restart = True, read parameters from a parameter file in basedir.
        Otherwise, all parameters except restart must be provided.

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
        elif self._params["precision"] == "double":
            self._precision = np.complex128
        else:
            self._precision = np.complex64

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
            write_json(filename=self._param_file, key="state", val=self._state)
        print("state = ", self._state, ", ", self.__state_code(self._state))

        # This call checks if restart_code can be applied, and resets self._state
        self.__check_restart_mode()

        # ----------------------------------------------------
        # Initialize full list of members
        self._k_values = None
        self._num_k_values = None

        # ----------------------------------------------------
        # Assume restart = False
        if self._restart is False:

            # Clean based on restart_code
            self.__clean(self._state)

            # Create directories
            dir_list = [
                "green_func/",
                "iterations/",
                "sources/",
                "data/"
            ]
            for item in dir_list:
                path = basedir + item
                if not os.path.exists(path):
                    os.makedirs(path)
                elif len(os.listdir(path)) > 0:
                    raise ValueError(
                        "restart=False not allowed as " + os.path.abspath(path) + " is not empty."
                    )

        # Assume restart = True
        elif self._restart is True:

            # Clean based on restart_code
            self.__clean(self._state)

    @property
    def k_values(self):
        return self._k_values

    @property
    def num_k_values(self):
        return self._num_k_values

    @property
    def state(self):
        return self._state

    def add_k_values(self, k_values_list):
        """
        :param k_values_list: list of floats or numpy array
            list of k-values for inversion
        :return:
        """
        if self._state >= 1:
            print(
                "\nOperation not allowed. Need self._state < 1, but obtained self._state = ", self._state
            )
            return

        if type(k_values_list) is list:
            self._k_values = np.squeeze(np.array(k_values_list, dtype=np.float32))

        if type(k_values_list) is np.ndarray:
            self._k_values = np.squeeze(k_values_list, dtype=np.float32)

        self._num_k_values = self._k_values.shape[0]

        path = self._basedir + "k-values.npz"
        np.savez(path, self._k_values)
        update_json(filename=self._param_file, key="state", val=1)

    def add_sources(self, num_sources, source_list):
        raise NotImplementedError

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
        print("State : ", self._m)

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

    def __clean(self, state):

        dir_list = []
        if state in [0, 1]:
            dir_list = [
                "sources/",
                "green_func/",
                "data/",
                "iterations/"
            ]
        if state == 2:
            dir_list = [
                "green_func/",
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


if __name__ == "__main__":

    basedir_ = "Inversion-LS/Expt/test0/"
    obj = ScatteringIntegralLinearIncreasingInversion2d(
        basedir=basedir_,
        restart=True,
        restart_code=0
    )
    obj.print_params()

    # k_vals_list_ = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    # obj.add_k_values(k_values_list=k_vals_list_)
    # print(obj.k_values)
