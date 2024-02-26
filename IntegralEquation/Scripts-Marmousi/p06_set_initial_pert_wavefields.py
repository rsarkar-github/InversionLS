import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=5
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Set zero initial perturbation and wavefields...")
    print("\n")

    obj.set_zero_initial_pert_wavefields(num_procs=10)
