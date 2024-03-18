from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import multiprocessing as mp


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=2
    )

    print(mp.cpu_count())

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Calculating Green's functions...")
    print("\n")
    num_procs = 4
    obj.calculate_greens_func(num_procs=num_procs)
