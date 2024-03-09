import sys
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import multiprocessing as mp


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    # Check arguments
    if len(sys.argv) < 2:
        raise ValueError("Program missing command line arguments.")

    num_iter = int(sys.argv[1])
    num_procs = min(obj.num_sources, mp.cpu_count(), 100)

    if num_iter == 0:

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Perform update...")
        print("\n")

        lambda_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0
        mu_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0

        print("\n\n---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("Iteration ", num_iter, ", updating wavefields...")

        # Update wave field
        obj.perform_inversion_update_wavefield(
            iter_count=num_iter,
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
        print("Iteration ", num_iter, ", updating perturbation...")

        # Update pert
        obj.perform_inversion_update_model_pert(
            iter_count=num_iter,
            max_iter=20,
            tol=1e-5,
            mnorm=0.001,
            use_bounds=True,
            num_procs=num_procs,
            clean=True
        )


    if num_iter == 1:

        print("\n\n---------------------------------------------")
        print("---------------------------------------------")
        print("Perform update...")
        print("\n")

        lambda_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0
        mu_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0

        print("\n\n---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("Iteration ", num_iter, ", updating wavefields...")

        # Update wave field
        obj.perform_inversion_update_wavefield(
            iter_count=num_iter,
            lambda_arr=lambda_arr,
            mu_arr=mu_arr,
            max_iter=1000,
            solver="lsmr",
            atol=1e-5,
            btol=1e-5,
            num_procs=num_procs,
            clean=True
        )

        # TODO: We are testing iteration 1 with no mnorm and no bounds
