import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=7
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Perform update...")
    print("\n")

    obj.perform_inversion_update_model_pert(
        iter_count=0,
        max_iter=20,
        tol=1e-5,
        num_procs=40,
        clean=False
    )