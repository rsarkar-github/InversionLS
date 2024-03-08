import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt
import multiprocessing as mp


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=6
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Perform update...")
    print("\n")

    lambda_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0
    mu_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0

    num_procs = min(obj.num_sources, mp.cpu_count(), 100)
    start_iter_num = 0

    for i in range(1):

        print("\n\n---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("Iteration ", start_iter_num + i, ", updating wavefields...")

        # Update wave field
        obj.perform_inversion_update_wavefield(
            iter_count=i,
            lambda_arr=lambda_arr,
            mu_arr=mu_arr,
            max_iter=40,
            solver="cg",
            atol=1e-5,
            btol=1e-5,
            num_procs=num_procs,
            clean=True
        )

        print("\n\n---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("Iteration ", start_iter_num + i, ", updating perturbation...")

        # Update pert
        obj.perform_inversion_update_model_pert(
            iter_count=i,
            max_iter=5,
            tol=1e-5,
            mnorm=0.0,
            use_bounds=True,
            num_procs=num_procs,
            clean=True
        )
