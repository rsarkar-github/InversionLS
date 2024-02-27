import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Perform update...")
    print("\n")

    obj.perform_inversion_update_wavefield(
        iter_count=0,
        lambda_arr=None,
        mu_arr=None,
        max_iter=10,
        solver="lsmr",
        atol=1e-5,
        btol=1e-5,
        num_procs=16,
        clean=True
    )
