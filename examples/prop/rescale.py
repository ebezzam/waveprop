"""
Rescale grid used in propagation models, i.e. to zoom-in or zoom-out.

Shifted Fresnel is not valid in near-field, and when zooming out (i.e.
output_scaling > 1.0).

"""

import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from waveprop.util import rect2d, sample_points, plot2d, bounding_box
from waveprop.fresnel import shifted_fresnel
from waveprop.rs import angular_spectrum
from waveprop.condition import fresnel_goodman, fresnel_saleh
import matplotlib


@hydra.main(version_base=None, config_path="../configs", config_name="rescale")
def rescale(config):

    # check config
    if config.sim.relative_shift is not None:
        assert config.sim.relative_shift < 1.0, "relative_shift must be less than 1.0"
        assert (
            config.sim.relative_shift >= 0.0
        ), "relative_shift must be greater than or equal to 0.0"

    # plotting parameters
    matplotlib.rc("font", **config.plot.font)

    # simulation parameters
    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    wv = config.sim.wv  # wavelength
    d1 = L / N  # source-plane grid spacing

    diam = config.sim.diam  # dimensions of rect aperture [m]
    dz_vals = config.sim.dz_vals  # distance [m]
    output_scaling = config.sim.output_scaling
    out_shift = d1 * N * config.sim.relative_shift
    N_out = None  # shifted Fresnel doesn't give this flexibility

    """ discretize aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = rect2d(x1, y1, diam)
    plot2d(x1, y1, u_in, title="Aperture")
    plt.savefig("1_input.png", dpi=config.plot.dpi)

    """ loop through distances """
    for i, dz in enumerate(dz_vals):

        print("\nPROPAGATION DISTANCE : {} m".format(dz))

        """ Shifted Fresnel """
        start_time = time.time()
        u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(
            u_in, wv, d1, dz, d2=output_scaling * d1, out_shift=out_shift
        )
        print("Fresnel : {} s".format(time.time() - start_time))
        print("-- ", end="")
        fresnel_goodman(wv, dz, x1, y1, x2_sfres, y2_sfres)
        print("-- ", end="")
        fresnel_saleh(wv, dz, x2_sfres, y2_sfres)

        """ Scaled BLAS """
        start_time = time.time()
        u_out_asm_scaled, x_asm_scaled, y_asm_scaled = angular_spectrum(
            u_in=u_in,
            wv=wv,
            d1=d1,
            dz=dz,
            bandlimit=True,
            N_out=N_out,
            d2=output_scaling * d1,
            out_shift=out_shift,
        )
        print("Angular spectrum : {} s".format(time.time() - start_time))
        u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True)

        """ pyFFS approach """
        start_time = time.time()
        u_out_asm_pyffs, x_asm_pyffs, y_asm_pyffs = angular_spectrum(
            u_in=u_in,
            wv=wv,
            d1=d1,
            dz=dz,
            bandlimit=True,
            N_out=N_out,
            d2=output_scaling * d1,
            out_shift=out_shift,
            pyffs=True,
        )
        print("Angular spectrum (pyFFS) : {} s".format(time.time() - start_time))

        """ Plot """
        fig, ax_2d = plt.subplots(ncols=4, figsize=(25, 5))
        plot2d(
            x2_sfres,
            y2_sfres,
            np.abs(u_out_sfres),
            title="Shifted Fresnel {} m".format(dz),
            ax=ax_2d[1],
        )
        plot2d(
            x_asm_scaled,
            y_asm_scaled,
            np.abs(u_out_asm_scaled),
            title="Scaled BLAS {} m".format(dz),
            ax=ax_2d[2],
        )
        plot2d(
            x_asm_pyffs,
            y_asm_pyffs,
            np.abs(u_out_asm_pyffs),
            title="BLAS pyFFS {} m".format(dz),
            ax=ax_2d[3],
        )
        plot2d(x_asm, y_asm, np.abs(u_out_asm), title="BLAS", ax=ax_2d[0])
        bounding_box(
            ax=ax_2d[0],
            start=[np.min(x_asm_scaled), np.min(y_asm_scaled)],
            stop=[np.max(x_asm_scaled), np.max(y_asm_scaled)],
            shift=output_scaling * d1 / 2,
            period=L,
            c="r",
            linestyle="--",
        )
        fig.savefig(f"{i + 2}_output_{dz}m.png", dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    rescale()
