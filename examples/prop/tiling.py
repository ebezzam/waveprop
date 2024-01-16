"""
Rectangular tiling as described in "Shifted Fresnel diffraction for computational holography".
But also applied with angular spectrum method.

This can be used to obtain a higher resolution image as input and output planes in simulation
have the same resolution. Tiling can be used to increase the resolution of the output plane,
by simulating multiple smaller output planes (off-axis and rescaling) and combining them.

"""

import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
from waveprop.util import rect2d, sample_points, plot2d, rect_tiling
from waveprop.fresnel import shifted_fresnel
import time
from waveprop.rs import angular_spectrum


@hydra.main(version_base=None, config_path="../configs", config_name="tiling")
def tiling(config):

    # simulation parameters
    N_in = config.sim.N_in  # number of grid points per size
    L_in = config.sim.L  # total size of grid, input and output
    n_tiles = config.sim.n_tiles  # number of tiles per side
    N_out = n_tiles * N_in
    diam = config.sim.diam  # diameter of aperture [m]
    d1 = L_in / N_in  # source-plane grid spacing
    if config.sim.same_size:
        d2 = L_in / N_out  # output-plane grid spacing
    else:
        d2 = d1
    wv = config.sim.wv  # wavelength
    dz_vals = config.sim.dz_vals  # distance [m]

    """ discretize aperture """
    x1, y1 = sample_points(N=N_in, delta=d1)
    u_in = rect2d(x1, y1, diam)
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")
    plt.savefig("1_input.png", dpi=config.plot.dpi)

    """ loop through distances """
    for i, dz in enumerate(dz_vals):

        print("\nPROPAGATION DISTANCE : {} m".format(dz))

        def prop_func_fresnel(out_shift):
            return shifted_fresnel(u_in, wv, d1, dz, d2=d2, out_shift=out_shift)[0]

        def prop_func_angular_spectrum(out_shift):
            return angular_spectrum(
                u_in=u_in, wv=wv, d1=d1, dz=dz, d2=d2, bandlimit=True, out_shift=out_shift
            )[0]

        def prop_func_angular_spectrum_pyffs(out_shift):
            return angular_spectrum(
                u_in=u_in,
                wv=wv,
                d1=d1,
                dz=dz,
                d2=d2,
                bandlimit=True,
                out_shift=out_shift,
                pyffs=True,
            )[0]

        start_time = time.time()
        u_out_fresnel, x2, y2 = rect_tiling(
            N_in=N_in, N_out=N_out, L=L_in, n_tiles=n_tiles, prop_func=prop_func_fresnel
        )
        print(f"Proc time (Fresnel) : {time.time() - start_time}")

        start_time = time.time()
        u_out_blas, x2, y2 = rect_tiling(
            N_in=N_in, N_out=N_out, L=L_in, n_tiles=n_tiles, prop_func=prop_func_angular_spectrum
        )
        print(f"Proc time (BLAS) : {time.time() - start_time}")

        start_time = time.time()
        u_out_pyffs, x2, y2 = rect_tiling(
            N_in=N_in,
            N_out=N_out,
            L=L_in,
            n_tiles=n_tiles,
            prop_func=prop_func_angular_spectrum_pyffs,
        )
        print(f"Proc time (BLAS, pyFFS) : {time.time() - start_time}")

        # plot
        fig, ax = plt.subplots(ncols=3)
        plot2d(
            x2.squeeze(),
            y2.squeeze(),
            np.abs(u_out_fresnel),
            title="Shifted Fresnel",
            ax=ax[0],
            colorbar=False,
        )

        plot2d(
            x2.squeeze(), y2.squeeze(), np.abs(u_out_blas), title="BLAS", ax=ax[1], colorbar=False
        )

        plot2d(
            x2.squeeze(),
            y2.squeeze(),
            np.abs(u_out_pyffs),
            title="pyFFS",
            ax=ax[2],
            colorbar=False,
        )
        fig.savefig(f"{i+2}_tiling_{dz}m.png", dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    tiling()
