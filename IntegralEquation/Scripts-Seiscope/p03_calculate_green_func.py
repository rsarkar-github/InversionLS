from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/seiscope/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=2
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Calculating Green's functions...")
    print("\n")
    num_procs = 4
    obj.calculate_greens_func(num_procs=num_procs)
